/* The Brain, for the phone.
 *
 * This is the same model as soilsight/model/gp.py and the same next-spot rule as
 * soilsight/sampling/acquire.py, written again in JavaScript so a field visit needs no
 * signal. The Python code is the reference: tests/test_app.py runs both on the same
 * field and fails if they disagree. Change the maths in Python first, then here.
 *
 * Values are ln(reading / threshold): above 0 is unsafe.
 */
(function (root) {
  "use strict";

  var SQRT3 = Math.sqrt(3), SQRT1_2 = Math.SQRT1_2;

  // ---- normal distribution (erfc by Chebyshev series, accurate to about 1e-15)
  var COF = [-1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2, -9.561514786808631e-3,
    -9.46595344482036e-4, 3.66839497852761e-4, 4.2523324806907e-5, -2.0278578112534e-5, -1.624290004647e-6,
    1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8, 6.529054439e-9, 5.059343495e-9, -9.91364156e-10,
    -2.27365122e-10, 9.6467911e-11, 2.394038e-12, -6.886027e-12, 8.94487e-13, 3.13092e-13, -1.12708e-13,
    3.81e-16, 7.106e-15, -1.523e-15, -9.4e-17, 1.21e-16, -2.8e-17];

  function erfcPos(z) {
    var t = 2 / (2 + z), ty = 4 * t - 2, d = 0, dd = 0, tmp;
    for (var j = COF.length - 1; j > 0; j--) { tmp = d; d = ty * d - dd + COF[j]; dd = tmp; }
    return t * Math.exp(-z * z + 0.5 * (COF[0] + ty * d) - dd);
  }
  function ndtr(x) {            // P(standard normal < x)
    var z = -x * SQRT1_2;
    return 0.5 * (z >= 0 ? erfcPos(z) : 2 - erfcPos(-z));
  }

  function corr(d, length, kernel) {
    if (kernel === "rbf") { var q = d / length; return Math.exp(-0.5 * q * q); }
    var r = SQRT3 * d / length;
    return (1 + r) * Math.exp(-r);
  }

  // ---- small dense linear algebra (matrices are flat, row-major)
  function cholesky(a, n) {     // lower triangle L with L L^T = a; returns null if not positive definite
    var L = new Float64Array(n * n);
    for (var i = 0; i < n; i++) {
      for (var j = 0; j <= i; j++) {
        var s = a[i * n + j];
        for (var k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
        if (i === j) { if (s <= 0) return null; L[i * n + i] = Math.sqrt(s); }
        else L[i * n + j] = s / L[j * n + j];
      }
    }
    return L;
  }
  function cholSolve(L, n, b) { // solves (L L^T) x = b
    var y = new Float64Array(n), x = new Float64Array(n), i, k, s;
    for (i = 0; i < n; i++) { s = b[i]; for (k = 0; k < i; k++) s -= L[i * n + k] * y[k]; y[i] = s / L[i * n + i]; }
    for (i = n - 1; i >= 0; i--) { s = y[i]; for (k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
    return x;
  }
  function cholInverse(L, n) {
    var inv = new Float64Array(n * n), e = new Float64Array(n);
    for (var j = 0; j < n; j++) {
      e.fill(0); e[j] = 1;
      var col = cholSolve(L, n, e);
      for (var i = 0; i < n; i++) inv[i * n + j] = col[i];
    }
    return inv;
  }

  // ---- the Brain for one field package
  function create(pkg) {
    var g = pkg.grid, N = g.rows.length, m = pkg.model, cand = m.candidates, C = cand.signal_sd.length;
    var meanVar = m.mean_sd * m.mean_sd, kernel = m.kernel;
    var X = new Float64Array(N), Y = new Float64Array(N);
    for (var i = 0; i < N; i++) { X[i] = (g.cols[i] + 0.5) * g.cell_m; Y[i] = (g.ny - g.rows[i] - 0.5) * g.cell_m; }

    function fit(tests) {
      // tests: [{x, y, value, noise_sd, cores}]; returns the weighted candidate settings that explain them
      var n = tests.length;
      if (!n) throw new Error("need at least one test");
      var d = new Float64Array(n * n), i, j, c;
      for (i = 0; i < n; i++) for (j = 0; j < n; j++) d[i * n + j] = Math.hypot(tests[i].x - tests[j].x, tests[i].y - tests[j].y);
      var yMean = 0; for (i = 0; i < n; i++) yMean += tests[i].value; yMean /= n;
      var yc = new Float64Array(n); for (i = 0; i < n; i++) yc[i] = tests[i].value - yMean;
      var loglik = new Float64Array(C), alphas = new Array(C), chols = new Array(C), K = new Float64Array(n * n);
      for (c = 0; c < C; c++) {
        var sf2 = cand.signal_sd[c] * cand.signal_sd[c], len = cand.length_m[c], nug2 = cand.nugget_sd[c] * cand.nugget_sd[c];
        for (i = 0; i < n; i++) {
          for (j = 0; j < n; j++) K[i * n + j] = sf2 * corr(d[i * n + j], len, kernel) + meanVar;
          K[i * n + i] += tests[i].noise_sd * tests[i].noise_sd + nug2 / tests[i].cores + 1e-8;
        }
        var L = cholesky(K, n);
        if (!L) { loglik[c] = -Infinity; continue; }
        var alpha = cholSolve(L, n, yc), quad = 0, logdet = 0;
        for (i = 0; i < n; i++) { quad += alpha[i] * yc[i]; logdet += Math.log(L[i * n + i]); }
        loglik[c] = -0.5 * quad - logdet; alphas[c] = alpha; chols[c] = L;
      }
      var best = -Infinity; for (c = 0; c < C; c++) if (loglik[c] > best) best = loglik[c];
      var order = []; for (c = 0; c < C; c++) order.push(c);
      order.sort(function (a, b) { return loglik[b] - loglik[a] || a - b; });
      var comps = [], total = 0;
      for (var r = 0; r < Math.min(m.top_k, C); r++) {
        c = order[r];
        var w = Math.exp(loglik[c] - best);
        if (!(w > 1e-6)) continue;
        comps.push({ weight: w, signal_sd: cand.signal_sd[c], length_m: cand.length_m[c], nugget_sd: cand.nugget_sd[c],
                     alpha: alphas[c], kinv: cholInverse(chols[c], n) });
        total += w;
      }
      comps.forEach(function (q) { q.weight /= total; });
      return { n: n, tests: tests, yMean: yMean, comps: comps };
    }

    function top(post, k) {     // the k heaviest candidates, reweighted
      var comps = post.comps.slice().sort(function (a, b) { return b.weight - a.weight; }).slice(0, k), total = 0;
      comps.forEach(function (q) { total += q.weight; });
      return { n: post.n, tests: post.tests, yMean: post.yMean,
               comps: comps.map(function (q) { return Object.assign({}, q, { weight: q.weight / total }); }) };
    }

    function distToTests(post, cells) {   // (cells, n)
      var n = post.n, M = cells.length, out = new Float64Array(M * n);
      for (var z = 0; z < M; z++) for (var j = 0; j < n; j++)
        out[z * n + j] = Math.hypot(X[cells[z]] - post.tests[j].x, Y[cells[z]] - post.tests[j].y);
      return out;
    }

    function moments(post, q, dist, M) {  // mean and sd of every cell under one candidate
      var n = post.n, sf2 = q.signal_sd * q.signal_sd, mu = new Float64Array(M), sd = new Float64Array(M), ks = new Float64Array(n);
      for (var z = 0; z < M; z++) {
        var s = 0, j, k;
        for (j = 0; j < n; j++) { ks[j] = sf2 * corr(dist[z * n + j], q.length_m, kernel) + meanVar; s += ks[j] * q.alpha[j]; }
        var quad = 0;
        for (j = 0; j < n; j++) { var row = 0; for (k = 0; k < n; k++) row += q.kinv[j * n + k] * ks[k]; quad += ks[j] * row; }
        mu[z] = s + post.yMean;
        sd[z] = Math.sqrt(Math.max(sf2 + meanVar - quad, 1e-10));
      }
      return { mu: mu, sd: sd };
    }

    function predict(post, cells) {       // {mean, sd, p} for the given cells (default: all)
      if (!cells) { cells = new Int32Array(N); for (var i = 0; i < N; i++) cells[i] = i; }
      var M = cells.length, dist = distToTests(post, cells);
      var mean = new Float64Array(M), second = new Float64Array(M), p = new Float64Array(M);
      post.comps.forEach(function (q) {
        var mo = moments(post, q, dist, M);
        for (var z = 0; z < M; z++) {
          mean[z] += q.weight * mo.mu[z];
          second[z] += q.weight * (mo.sd[z] * mo.sd[z] + mo.mu[z] * mo.mu[z]);
          p[z] += q.weight * ndtr(mo.mu[z] / mo.sd[z]);
        }
      });
      var sd = new Float64Array(M);
      for (var z = 0; z < M; z++) sd[z] = Math.sqrt(Math.max(second[z] - mean[z] * mean[z], 1e-10));
      return { mean: mean, sd: sd, p: p };
    }

    function label(p) { return p < pkg.classes.lo ? "green" : p > pkg.classes.hi ? "red" : "yellow"; }

    var zoneDist = null;                  // pairwise distances inside each zone's sample, made once
    function zones(post) {                // one answer per zone: the zone's average, with its uncertainty
      var samples = pkg.zones.samples, n = post.n;
      if (!zoneDist) zoneDist = samples.map(function (cells) {
        var P = cells.length, d = new Float32Array(P * P);
        for (var a = 0; a < P; a++) for (var b = 0; b < P; b++) d[a * P + b] = Math.hypot(X[cells[a]] - X[cells[b]], Y[cells[a]] - Y[cells[b]]);
        return d;
      });
      return samples.map(function (cells, zi) {
        var P = cells.length, dist = distToTests(post, cells), dd = zoneDist[zi];
        var mean = 0, second = 0, p = 0;
        post.comps.forEach(function (q) {
          var sf2 = q.signal_sd * q.signal_sd, ks = new Float64Array(n), j, k, a;
          for (a = 0; a < P; a++) for (j = 0; j < n; j++) ks[j] += sf2 * corr(dist[a * n + j], q.length_m, kernel) + meanVar;
          for (j = 0; j < n; j++) ks[j] /= P;
          var kzz = 0; for (a = 0; a < P * P; a++) kzz += corr(dd[a], q.length_m, kernel);
          kzz = sf2 * kzz / (P * P) + meanVar;
          var mu = post.yMean, quad = 0;
          for (j = 0; j < n; j++) { mu += ks[j] * q.alpha[j]; var row = 0; for (k = 0; k < n; k++) row += q.kinv[j * n + k] * ks[k]; quad += ks[j] * row; }
          var sd = Math.sqrt(Math.max(kzz - quad, 1e-10));
          mean += q.weight * mu; second += q.weight * (sd * sd + mu * mu); p += q.weight * ndtr(mu / sd);
        });
        return { zone: zi, name: pkg.zones.names[zi], mean: mean, sd: Math.sqrt(Math.max(second - mean * mean, 1e-10)),
                 p: p, label: label(p) };
      });
    }

    function nextSpot(post) {
      // The spot whose test is expected to remove the most misclassified area; null when the budget is spent.
      var pr = pkg.protocol, n = post.n;
      if (n >= pr.budget) return null;
      var cs = pkg.plan.cand, ev = pkg.plan.eval, Cn = cs.length, M = ev.length, zq = m.quadrature.z, wq = m.quadrature.w;
      var p6 = top(post, m.acq_components), dCandObs = distToTests(p6, cs), dCellObs = distToTests(p6, ev);
      var dCandCell = new Float64Array(Cn * M), c, z, j, k;
      for (c = 0; c < Cn; c++) for (z = 0; z < M; z++) dCandCell[c * M + z] = Math.hypot(X[cs[c]] - X[ev[z]], Y[cs[c]] - Y[ev[z]]);
      var gains = new Float64Array(Cn), now = 0;
      p6.comps.forEach(function (q) {
        var sf2 = q.signal_sd * q.signal_sd, mo = moments(p6, q, dCellObs, M);
        var kzo = new Float64Array(M * n), t = new Float64Array(n), kco = new Float64Array(n);
        for (z = 0; z < M; z++) for (j = 0; j < n; j++) kzo[z * n + j] = sf2 * corr(dCellObs[z * n + j], q.length_m, kernel) + meanVar;
        var before = 0; for (z = 0; z < M; z++) before += ndtr(-Math.abs(mo.mu[z]) / mo.sd[z]); before /= M;
        var testVar = pr.kit_sd * pr.kit_sd + q.nugget_sd * q.nugget_sd / pr.cores;
        for (c = 0; c < Cn; c++) {
          var quad = 0;
          for (j = 0; j < n; j++) kco[j] = sf2 * corr(dCandObs[c * n + j], q.length_m, kernel) + meanVar;
          for (j = 0; j < n; j++) { var s = 0; for (k = 0; k < n; k++) s += kco[k] * q.kinv[k * n + j]; t[j] = s; quad += s * kco[j]; }
          var denom = Math.sqrt(Math.max(sf2 + meanVar - quad, 1e-10) + testVar), after = 0;
          for (z = 0; z < M; z++) {
            var cov = sf2 * corr(dCandCell[c * M + z], q.length_m, kernel) + meanVar;
            for (j = 0; j < n; j++) cov -= t[j] * kzo[z * n + j];
            var shift = cov / denom, sdNew = Math.sqrt(Math.max(mo.sd[z] * mo.sd[z] - shift * shift, 1e-10));
            for (k = 0; k < zq.length; k++) after += wq[k] * ndtr(-Math.abs(mo.mu[z] + shift * zq[k]) / sdNew);
          }
          gains[c] += q.weight * (before - after / M);
        }
        now += q.weight * before;
      });
      var sep = Math.max(pr.min_sep_m, 1e-9), bestC = -1, bestGain = -Infinity;
      for (c = 0; c < Cn; c++) {
        var near = false;
        for (j = 0; j < n; j++) if (Math.hypot(X[cs[c]] - post.tests[j].x, Y[cs[c]] - post.tests[j].y) < sep) { near = true; break; }
        if (!near && gains[c] > bestGain) { bestGain = gains[c]; bestC = c; }
      }
      return bestC < 0 ? null : { cell: cs[bestC], x: X[cs[bestC]], y: Y[cs[bestC]], gain: bestGain, now: now };
    }

    return { N: N, X: X, Y: Y, fit: fit, predict: predict, zones: zones, nextSpot: nextSpot, label: label };
  }

  var api = { create: create, ndtr: ndtr };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SoilBrain = api;
})(typeof self !== "undefined" ? self : this);
