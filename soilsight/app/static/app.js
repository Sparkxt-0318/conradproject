/* SoilSight field app.
 *
 * Everything the app knows about a visit is in its log: a list of entries, each chained to
 * the one before (see sha256.js and soilsight/app/record.py). The screen is always rebuilt
 * from the log, so what is exported is exactly what was shown.
 *
 * The field package and the log are kept in the phone's storage; nothing is sent anywhere.
 */
(function () {
  "use strict";

  var LAB_SD = 0.05;                 // an accredited-lab value is taken as this precise (sd of ln)
  var FAR_M = 15;                    // warn when a sample is recorded this far from its planned spot
  var COLOURS = { green: [46, 139, 87], yellow: [214, 158, 20], red: [192, 57, 43] };
  var $app = document.getElementById("app");
  var S = { pkg: null, brain: null, stored: [], gps: null, selected: null, entering: null, model: null, error: "" };

  // ---- storage

  function store(key, value) { try { localStorage.setItem(key, value); return true; } catch (e) { return false; } }
  function load(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }
  function logKey() { return "soilsight.log." + S.pkg.name + "|" + S.pkg.made; }

  function setPackage(pkg) {
    if (!pkg || pkg.format !== "soilsight-field/1") throw new Error("That file is not a SoilSight field package.");
    S.pkg = pkg;
    S.brain = SoilBrain.create(pkg);
    store("soilsight.pkg", JSON.stringify(pkg));
    var saved = load(logKey());
    S.stored = saved ? JSON.parse(saved) : [];
    S.selected = S.entering = null;
    recompute();
  }

  function append(type, data) {
    var prev = S.stored.length ? S.stored[S.stored.length - 1].hash : SoilRecord.GENESIS;
    S.stored.push(SoilRecord.makeEntry(S.stored.length + 1, new Date().toISOString(), type, data, prev));
    if (!store(logKey(), JSON.stringify(S.stored))) S.error = "This phone's storage is full: export the log now.";
  }

  // ---- the visit, rebuilt from the log

  function visit() {
    var v = { started: false, finished: false, threshold: S.pkg.threshold, unit: S.pkg.unit, spots: [], samples: {}, results: {} };
    S.pkg.plan.initial.forEach(function (cell, i) { v.spots.push({ id: spotId(i), cell: cell, source: "first" }); });
    S.stored.forEach(function (item) {
      var e = JSON.parse(item.body), d = e.data;
      if (e.type === "visit_started") { v.started = true; v.threshold = d.threshold; v.unit = d.unit; }
      else if (e.type === "sample_taken") v.samples[d.id] = d;
      else if (e.type === "result_entered") { if (d.void) delete v.results[d.id]; else v.results[d.id] = d; }
      else if (e.type === "spot_proposed") v.spots.push({ id: d.id, cell: d.cell, source: "loop", gain: d.gain });
      else if (e.type === "visit_finished") v.finished = true;
    });
    v.tests = [];
    v.spots.forEach(function (s) {
      var sample = v.samples[s.id], r = v.results[s.id];
      s.state = r ? "tested" : sample ? "sampled" : "planned";
      if (sample && r) v.tests.push({ id: s.id, x: sample.x, y: sample.y, value: Math.log(r.reading / v.threshold),
                                      noise_sd: r.noise_sd, cores: sample.cores, reading: r.reading });
    });
    return v;
  }
  function spotId(i) { return "S" + (i < 9 ? "0" : "") + (i + 1); }

  function recompute() {
    // Fit the map to the results so far and, when every sample has its result, ask for the next spot.
    S.model = null;
    if (!S.pkg) return;
    var v = visit();
    if (!v.tests.length) return;
    var post = S.brain.fit(v.tests);
    S.model = { post: post, map: S.brain.predict(post), zones: S.brain.zones(post) };
    var waiting = v.spots.some(function (s) { return s.state !== "tested"; });
    if (!waiting && !v.finished && v.tests.length < S.pkg.protocol.budget) {
      var next = S.brain.nextSpot(post);
      if (next) {
        append("spot_proposed", { id: spotId(v.spots.length), cell: next.cell, gain: next.gain, misclassified_now: next.now });
        S.selected = spotId(v.spots.length);
      }
    }
  }

  // ---- positions

  function cellXY(cell) { return { x: S.brain.X[cell], y: S.brain.Y[cell] }; }
  function toXY(lon, lat) { var t = S.pkg.geo.to_xy; return { x: t[0] * lon + t[1] * lat + t[2], y: t[3] * lon + t[4] * lat + t[5] }; }
  function compass(dx, dy) {
    var names = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"];
    return names[Math.round(((Math.atan2(dx, dy) * 180 / Math.PI) + 360) % 360 / 45) % 8];
  }
  function gpsFresh() { return S.gps && Date.now() - S.gps.at < 20000; }

  function startGps() {
    if (!S.pkg || !S.pkg.geo || !navigator.geolocation || S.watching) return;
    S.watching = navigator.geolocation.watchPosition(function (pos) {
      var p = toXY(pos.coords.longitude, pos.coords.latitude);
      S.gps = { x: p.x, y: p.y, lon: pos.coords.longitude, lat: pos.coords.latitude, acc: pos.coords.accuracy, at: Date.now() };
      render();
    }, function () { S.gps = null; render(); }, { enableHighAccuracy: true, maximumAge: 2000, timeout: 15000 });
  }

  // ---- actions

  var actions = {
    start: function () {
      var t = parseFloat(document.getElementById("threshold").value);
      if (!(t > 0)) { S.error = "Enter the threshold: the kit reading above which soil counts as unsafe."; return; }
      append("visit_started", { threshold: t, unit: S.pkg.unit, package: { name: S.pkg.name, made: S.pkg.made },
                                protocol: S.pkg.protocol });
      S.selected = spotId(0);
    },
    select: function (id) { S.selected = id; S.entering = null; },
    sampled: function (id, force) {
      var v = visit(), spot = v.spots.filter(function (s) { return s.id === id; })[0], planned = cellXY(spot.cell);
      var here = gpsFresh() ? S.gps : null;
      if (here && !force && Math.hypot(here.x - planned.x, here.y - planned.y) > FAR_M) { S.confirmFar = id; return; }
      S.confirmFar = null;
      var d = { id: id, cell: spot.cell, x: here ? here.x : planned.x, y: here ? here.y : planned.y,
                cores: S.pkg.protocol.cores, planned: spot.source === "first", gps: !!here };
      if (here) { d.lon = here.lon; d.lat = here.lat; d.gps_accuracy_m = here.acc; }
      append("sample_taken", d);
      var next = visit().spots.filter(function (s) { return s.state === "planned"; })[0];
      S.selected = next ? next.id : null;
    },
    cancelFar: function () { S.confirmFar = null; },
    enter: function (id) { S.entering = id; S.selected = id; },
    cancelEntry: function () { S.entering = null; },
    save: function (id) {
      var reading = parseFloat(document.getElementById("reading").value);
      if (!(reading > 0)) { S.error = "Enter the kit reading as a number above zero."; return; }
      var ph = parseFloat(document.getElementById("ph").value), nitrate = parseFloat(document.getElementById("nitrate").value);
      var lab = document.getElementById("lab").checked;
      var d = { id: id, reading: reading, noise_sd: lab ? LAB_SD : S.pkg.protocol.kit_sd, lab: lab };
      if (!isNaN(ph)) d.ph = ph;
      if (!isNaN(nitrate)) d.nitrate = nitrate;
      append("result_entered", d);
      S.entering = null;
      recompute();
      var v = visit(), next = v.spots.filter(function (s) { return s.state === "sampled"; })[0] || v.spots.filter(function (s) { return s.state === "planned"; })[0];
      S.selected = next ? next.id : null;
    },
    finish: function () { append("visit_finished", {}); },
    exportLog: function () {
      var blob = new Blob([exportLog()], { type: "application/json" }), a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "soilsight-log-" + S.pkg.name.replace(/[^\w-]+/g, "_") + "-" + new Date().toISOString().slice(0, 10) + ".json";
      document.body.appendChild(a); a.click(); a.remove();
    },
    forget: function () {
      if (!S.confirmForget) { S.confirmForget = true; return; }
      try { localStorage.removeItem("soilsight.pkg"); } catch (e) { /* nothing to remove */ }
      S.pkg = S.brain = S.model = null; S.stored = []; S.confirmForget = false;
    },
    keep: function () { S.confirmForget = false; }
  };

  function exportLog() { return JSON.stringify({ format: "soilsight-log/1", field: S.pkg.name, entries: S.stored }, null, 1); }

  $app.addEventListener("click", function (ev) {
    var el = ev.target.closest("[data-do]");
    if (!el) return;
    S.error = "";
    var name = el.getAttribute("data-do");
    actions[name](el.getAttribute("data-id"), el.hasAttribute("data-force"));
    render();
  });
  $app.addEventListener("change", function (ev) {
    if (ev.target.id !== "package-file" || !ev.target.files[0]) return;
    ev.target.files[0].text().then(function (text) { setPackage(JSON.parse(text)); startGps(); render(); })
      .catch(function (err) { S.error = err.message || "That file could not be read."; render(); });
  });

  // ---- screen

  function esc(t) { return String(t).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function pct(p) { return Math.round(p * 100) + "%"; }
  function fmt(x) { return (Math.round(x * 1000) / 1000).toString(); }

  function render() {
    var html = S.error ? '<div class="card warn" role="alert">' + esc(S.error) + "</div>" : "";
    if (!S.pkg) { $app.innerHTML = html + loadCard(); header(null); return; }
    var v = visit();
    header(v);
    if (!v.started) { $app.innerHTML = html + startCard(v); return; }
    html += '<div class="card" id="map-wrap"><canvas id="map"></canvas>' + legend() + "</div>";
    html += taskCard(v) + (S.model ? zonesCard() : "") + samplesCard(v) + endCard(v);
    $app.innerHTML = html;
    drawMap(v);
    var field = document.getElementById("reading");
    if (field) field.focus();
  }

  function header(v) {
    document.getElementById("field-name").textContent = S.pkg ? S.pkg.name : "SoilSight";
    document.getElementById("progress").textContent = v && v.started ? v.tests.length + " of " + S.pkg.protocol.budget + " tests" + (v.finished ? " · finished" : "") : "";
    var chip = document.getElementById("gps");
    chip.hidden = !(S.pkg && S.pkg.geo);
    chip.textContent = gpsFresh() ? "GPS ±" + Math.round(S.gps.acc) + " m" : "no GPS";
  }

  function loadCard() {
    return '<div class="card"><h2>Load a field</h2><p>Choose the field package made on the laptop ' +
      "(<code>field.soilsight.json</code>). It stays on this phone; the visit then needs no signal.</p>" +
      '<label for="package-file">Field package</label><input type="file" id="package-file" accept=".json,application/json"></div>';
  }

  function startCard(v) {
    var p = S.pkg.protocol;
    return '<div class="card"><h2>Start the visit</h2><p>' + esc(S.pkg.name) + ": " + p.budget + " tests, " + p.cores +
      " cores mixed into each. The first " + S.pkg.plan.initial.length + " spots are spread over the field; the rest are chosen as results come in.</p>" +
      '<label for="threshold">Unsafe above this kit reading (' + esc(S.pkg.unit) + ')</label>' +
      '<input type="number" id="threshold" inputmode="decimal" step="any" min="0" value="' + (v.threshold || "") + '">' +
      '<p class="muted">There is no built-in value. Use the one from your kit-to-lab comparison.</p>' +
      '<button class="primary wide" data-do="start">Start visit</button></div>';
  }

  function legend() {
    if (!S.model) return '<div class="legend">Spots to sample are numbered. The map is coloured once the first result is in.</div>';
    return '<div class="legend"><span><i class="dot" style="background:rgb(46,139,87)"></i>probably safe</span>' +
      '<span><i class="dot" style="background:rgb(214,158,20)"></i>not sure</span>' +
      '<span><i class="dot" style="background:rgb(192,57,43)"></i>probably unsafe</span><span>Dashed lines: zones A to ' +
      esc(S.pkg.zones.names[S.pkg.zones.names.length - 1]) + "</span></div>";
  }

  function where(spot) {
    if (!S.pkg.geo) return "";
    if (!gpsFresh()) return '<p class="where muted">Waiting for GPS. You can still record the sample at its planned position.</p>';
    var p = cellXY(spot.cell), dx = p.x - S.gps.x, dy = p.y - S.gps.y, d = Math.hypot(dx, dy);
    if (d > 2000) return '<p class="where muted">Your phone places you ' + Math.round(d / 1000) + " km from this field.</p>";
    return '<p class="where">' + (d < 3 ? "You are at the spot." : Math.round(d) + " m to the " + compass(dx, dy)) + "</p>";
  }

  function taskCard(v) {
    if (v.finished) return '<div class="card task"><p class="big">Visit finished</p><p>Export the log and hand it to the laptop for the Field Health Report.</p>' +
      '<button class="primary wide" data-do="exportLog">Export log</button></div>';
    var spot = v.spots.filter(function (s) { return s.id === (S.entering || S.selected); })[0];
    if (S.entering && spot) return entryCard(v, spot);
    if (!spot || spot.state === "tested") spot = v.spots.filter(function (s) { return s.state === "planned"; })[0] || v.spots.filter(function (s) { return s.state === "sampled"; })[0];
    if (!spot) return '<div class="card task"><p class="big">All ' + v.tests.length + " tests are in</p>" +
      "<p>The test budget is used. Finish the visit to close the log.</p>" +
      '<button class="primary wide" data-do="finish">Finish visit</button></div>';
    if (spot.state === "sampled") return '<div class="card task"><p class="big">' + spot.id + ": waiting for its result</p>" +
      "<p>Test the mixed sample, then enter the reading.</p>" +
      '<button class="primary wide" data-do="enter" data-id="' + spot.id + '">Enter result for ' + spot.id + "</button></div>";
    var why = spot.source === "loop" ? "<p>Chosen by the map: a test here is expected to settle " + (spot.gain * 100).toFixed(1) + "% of the field.</p>"
      : "<p>One of the first spots, spread evenly over the field.</p>";
    var far = S.confirmFar === spot.id;
    return '<div class="card task"><p class="big">Go to spot ' + spot.id + "</p>" + where(spot) + why +
      "<p class=\"muted\">Take " + S.pkg.protocol.cores + " cores within a few steps and mix them into one sample labelled " + spot.id + ".</p>" +
      (far ? '<p class="warn">You are more than ' + FAR_M + " m from this spot. Record the sample where you are standing?</p>" +
             '<div class="row"><button class="primary" data-do="sampled" data-force data-id="' + spot.id + '">Yes, record here</button>' +
             '<button data-do="cancelFar">No, keep walking</button></div>'
           : '<button class="primary wide" data-do="sampled" data-id="' + spot.id + '">Sample taken here</button>') + "</div>";
  }

  function entryCard(v, spot) {
    var old = v.results[spot.id] || {};
    return '<div class="card task"><p class="big">Result for ' + spot.id + "</p>" +
      '<label for="reading">Cadmium reading (' + esc(v.unit) + ')</label><input type="number" id="reading" inputmode="decimal" step="any" min="0" value="' + (old.reading || "") + '">' +
      '<div class="row"><div><label for="ph">pH (optional)</label><input type="number" id="ph" inputmode="decimal" step="any" value="' + (old.ph === undefined ? "" : old.ph) + '"></div>' +
      '<div><label for="nitrate">Nitrate (optional)</label><input type="number" id="nitrate" inputmode="decimal" step="any" value="' + (old.nitrate === undefined ? "" : old.nitrate) + '"></div></div>' +
      '<label class="check"><input type="checkbox" id="lab"' + (old.lab ? " checked" : "") + "> This value is from an accredited lab</label>" +
      '<div class="row"><button class="primary" data-do="save" data-id="' + spot.id + '">Save result</button><button data-do="cancelEntry">Cancel</button></div></div>';
  }

  function zonesCard() {
    var rows = S.model.zones.map(function (z) {
      var word = { green: "Probably safe", yellow: "Not sure", red: "Probably unsafe" }[z.label];
      return "<tr><td><strong>" + esc(z.name) + '</strong></td><td><span class="tag ' + z.label + '">' + word + '</span></td><td class="num">' + pct(z.p) + "</td></tr>";
    }).join("");
    return '<div class="card"><h2>Zones</h2><table><thead><tr><th>Zone</th><th>Answer</th><th>Chance unsafe</th></tr></thead><tbody>' + rows +
      '</tbody></table><p class="muted">A screening result, not an official classification. "Not sure" means more tests or a lab check are needed.</p></div>';
  }

  function samplesCard(v) {
    var items = v.spots.map(function (s) {
      var r = v.results[s.id], what, button = "";
      if (s.state === "tested") {
        what = fmt(r.reading) + " " + esc(v.unit) + (r.reading > v.threshold ? ' <span class="tag red">above</span>' : ' <span class="tag green">below</span>') + (r.lab ? " · lab" : "");
        if (!v.finished) button = '<button data-do="enter" data-id="' + s.id + '">Edit</button>';
      } else if (s.state === "sampled") {
        what = "sampled, waiting for result";
        if (!v.finished) button = '<button data-do="enter" data-id="' + s.id + '">Result</button>';
      } else {
        what = s.source === "loop" ? "to sample (chosen by the map)" : "to sample";
        if (!v.finished) button = '<button data-do="select" data-id="' + s.id + '">Go</button>';
      }
      return '<li><span class="id">' + s.id + '</span><span class="what">' + what + "</span>" + button + "</li>";
    }).join("");
    return '<div class="card"><h2>Samples</h2><ul class="list">' + items + "</ul></div>";
  }

  function endCard(v) {
    var head = S.stored.length ? S.stored[S.stored.length - 1].hash : "";
    var html = '<div class="card"><h2>Record</h2><p class="muted">' + S.stored.length + " entries. Last entry's fingerprint:</p><p class=\"hash\">" + head + "</p>";
    if (!v.finished) {
      var early = v.tests.length && v.spots.some(function (s) { return s.state !== "tested"; }) || (v.tests.length && v.tests.length < S.pkg.protocol.budget);
      html += '<div class="row"><button data-do="exportLog">Export log</button>' +
        (early ? '<button data-do="finish">Finish visit early</button>' : "") + "</div>";
    } else {
      html += S.confirmForget ? '<p class="warn">Remove this field from the phone? Export the log first; this cannot be undone.</p>' +
        '<div class="row"><button data-do="forget">Yes, remove it</button><button data-do="keep">Keep it</button></div>'
        : '<button class="wide" data-do="forget">Load another field</button>';
    }
    return html + "</div>";
  }

  // ---- map

  var photo = null;
  function drawMap(v) {
    var canvas = document.getElementById("map"), g = S.pkg.grid, wrap = canvas.parentNode;
    var PAD = 18, W = wrap.clientWidth, fw = W - 2 * PAD, fh = Math.round(fw * g.ny / g.nx), H = fh + 2 * PAD, dpr = window.devicePixelRatio || 1;
    canvas.style.height = H + "px"; canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    var ctx = canvas.getContext("2d"), s = fw / (g.nx * g.cell_m);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#dfe3dc"; ctx.fillRect(0, 0, W, H);
    if (S.pkg.image) {
      if (!photo) { photo = new Image(); photo.onload = function () { if (S.pkg) drawMap(visit()); }; photo.src = "data:image/jpeg;base64," + S.pkg.image.jpeg; }
      if (photo.complete && photo.naturalWidth) ctx.drawImage(photo, PAD, PAD, fw, fh);
    }
    var layer = document.createElement("canvas"); layer.width = g.nx; layer.height = g.ny;
    var lc = layer.getContext("2d"), img = lc.createImageData(g.nx, g.ny), i, o;
    for (i = 0; i < g.rows.length; i++) {
      o = (g.rows[i] * g.nx + g.cols[i]) * 4;
      var c = S.model ? COLOURS[S.brain.label(S.model.map.p[i])] : [255, 255, 255];
      img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = S.model ? (S.pkg.image ? 120 : 215) : (S.pkg.image ? 0 : 255);
    }
    lc.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(layer, PAD, PAD, fw, fh); ctx.imageSmoothingEnabled = true;

    function px(x, y) { return [PAD + x * s, PAD + fh - y * s]; }

    // zone borders and letters, so the zone table can be read against the map
    var zid = new Int16Array(g.nx * g.ny).fill(-1), sumX = [], sumY = [], count = [];
    for (i = 0; i < g.rows.length; i++) {
      var z = S.pkg.zones.ids[i];
      zid[g.rows[i] * g.nx + g.cols[i]] = z;
      sumX[z] = (sumX[z] || 0) + g.cols[i] + 0.5; sumY[z] = (sumY[z] || 0) + g.rows[i] + 0.5; count[z] = (count[z] || 0) + 1;
    }
    var cw = fw / g.nx, chh = fh / g.ny;
    ctx.beginPath();
    for (var r = 0; r < g.ny; r++) for (var c2 = 0; c2 < g.nx; c2++) {
      var here = zid[r * g.nx + c2], right = c2 + 1 < g.nx ? zid[r * g.nx + c2 + 1] : -1, below = r + 1 < g.ny ? zid[(r + 1) * g.nx + c2] : -1;
      if (here !== right && (here >= 0 || right >= 0)) { ctx.moveTo(PAD + (c2 + 1) * cw, PAD + r * chh); ctx.lineTo(PAD + (c2 + 1) * cw, PAD + (r + 1) * chh); }
      if (here !== below && (here >= 0 || below >= 0)) { ctx.moveTo(PAD + c2 * cw, PAD + (r + 1) * chh); ctx.lineTo(PAD + (c2 + 1) * cw, PAD + (r + 1) * chh); }
    }
    ctx.lineWidth = 1.5; ctx.strokeStyle = "rgba(20,20,20,.75)"; ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([]);
    ctx.font = "700 15px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    var minR = [], minC = [];
    for (i = 0; i < g.rows.length; i++) {
      var zz = S.pkg.zones.ids[i];
      minR[zz] = Math.min(minR[zz] === undefined ? 1e9 : minR[zz], g.rows[i]); minC[zz] = Math.min(minC[zz] === undefined ? 1e9 : minC[zz], g.cols[i]);
    }
    var corner = [], cornerD = [];              // the cell of each zone nearest its north-west corner: the letter goes there, clear of the spots
    for (i = 0; i < g.rows.length; i++) {
      var zc = S.pkg.zones.ids[i], dd = Math.pow(g.rows[i] - minR[zc], 2) + Math.pow(g.cols[i] - minC[zc], 2);
      if (cornerD[zc] === undefined || dd < cornerD[zc]) { cornerD[zc] = dd; corner[zc] = i; }
    }
    S.pkg.zones.names.forEach(function (name, zi) {
      if (!count[zi]) return;
      var inset = Math.max(12, 2.2 * cw), zx = PAD + g.cols[corner[zi]] * cw + inset, zy = PAD + g.rows[corner[zi]] * chh + inset;
      ctx.lineWidth = 4; ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.strokeText(name, zx, zy);
      ctx.fillStyle = "#111"; ctx.fillText(name, zx, zy);
    });
    ctx.font = "600 12px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    v.spots.forEach(function (spot) {
      var sample = v.samples[spot.id], pos = sample ? { x: sample.x, y: sample.y } : cellXY(spot.cell), p = px(pos.x, pos.y), r = v.results[spot.id];
      var current = spot.id === S.selected && !v.finished;
      ctx.beginPath(); ctx.arc(p[0], p[1], current ? 13 : 10, 0, 2 * Math.PI);
      ctx.fillStyle = spot.state === "tested" ? (r.reading > v.threshold ? "#8e1f14" : "#1d6b41") : spot.state === "sampled" ? "#6b7280" : "#ffffff";
      ctx.fill();
      ctx.lineWidth = current ? 4 : 2; ctx.strokeStyle = current ? "#1565c0" : "#111"; ctx.stroke();
      ctx.fillStyle = spot.state === "planned" ? "#111" : "#fff";
      ctx.fillText(spot.id.slice(1).replace(/^0/, ""), p[0], p[1] + 0.5);
    });
    if (gpsFresh()) {
      var u = px(S.gps.x, S.gps.y);
      ctx.beginPath(); ctx.arc(u[0], u[1], Math.max(S.gps.acc * s, 8), 0, 2 * Math.PI); ctx.fillStyle = "rgba(21,101,192,.18)"; ctx.fill();
      ctx.beginPath(); ctx.arc(u[0], u[1], 6, 0, 2 * Math.PI); ctx.fillStyle = "#1565c0"; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.stroke();
    }
    canvas.onclick = function (ev) {          // tap a spot to select it
      var rect = canvas.getBoundingClientRect(), mx = ev.clientX - rect.left, my = ev.clientY - rect.top, best = null, bd = 28;
      v.spots.forEach(function (spot) {
        var sample = v.samples[spot.id], pos = sample ? sample : cellXY(spot.cell), p = px(pos.x, pos.y), d = Math.hypot(p[0] - mx, p[1] - my);
        if (d < bd) { bd = d; best = spot.id; }
      });
      if (best && !v.finished) { S.selected = best; S.entering = null; render(); }
    };
  }
  window.addEventListener("resize", function () { if (S.pkg && document.getElementById("map")) drawMap(visit()); });

  // ---- start

  function boot() {
    var saved = load("soilsight.pkg");
    if (saved) {
      try { setPackage(JSON.parse(saved)); } catch (e) { S.pkg = null; }
    }
    if (S.pkg) { startGps(); render(); return; }
    render();
    fetch("package.json").then(function (r) { return r.ok ? r.json() : null; }).then(function (pkg) {   // a package served with the app, if any
      if (pkg && !S.pkg) { setPackage(pkg); startGps(); render(); }
    }).catch(function () { /* no package served: wait for a file */ });
  }

  if ("serviceWorker" in navigator && window.isSecureContext) navigator.serviceWorker.register("sw.js").catch(function () { /* runs online only */ });
  window.SoilSightApp = {        // for the laptop-side tests
    visit: function () { return S.pkg ? visit() : null; },
    zones: function () { return S.model ? S.model.zones : null; },
    exportLog: exportLog,
    position: function () { return S.gps ? { x: S.gps.x, y: S.gps.y, accuracy: S.gps.acc } : { x: NaN, y: NaN }; },
    setPosition: function (lon, lat, acc) { var p = toXY(lon, lat); S.gps = { x: p.x, y: p.y, lon: lon, lat: lat, acc: acc || 3, at: Date.now() }; render(); }
  };
  boot();
})();
