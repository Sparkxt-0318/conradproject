"""Stress test of the SoilSight adaptive-sampling loop on synthetic fields.

This is the evidence behind docs/software-plan.md. It needs only numpy.

    python experiments/stress_test.py methods      # 9 method variants x 12 scenarios (~6 min)
    python experiments/stress_test.py budget       # tests-per-field curve, zones, composites (~12 min)
    python experiments/stress_test.py validation   # power of the kit-vs-lab split-sample test
    python experiments/stress_test.py methods --scenarios S0,S7 --reps 10   # quick look

Everything is in ln(Cd / threshold): a cell is "unsafe" when its value is above 0.
The fields are made up. The assumptions that matter are listed in make_field().
"""
import argparse
import time
from math import comb, erf

import numpy as np

NX, NY, CELL = 100, 75, 2.0          # 200 m x 150 m field (3 ha) on a 2 m grid
N = NX * NY
_gx, _gy = np.meshgrid(np.arange(NX) * CELL, np.arange(NY) * CELL)
XY = np.c_[_gx.ravel(), _gy.ravel()]
ZONE = (_gx.ravel() // 67).astype(int) + 3 * (_gy.ravel() // 75).astype(int)  # six ~0.5 ha zones
CONST2 = 0.3 ** 2                    # prior variance of the field mean
C = 256                              # hyperparameter candidates tried per fit

SCENARIOS = {
    "S0": ("base", {}),
    "S1": ("drone layers useless", dict(R2=0.0)),
    "S2": ("drone layers strong", dict(R2=0.6)),
    "S3": ("short patches, 12 m", dict(rng_m=12)),
    "S4": ("long patches, 100 m", dict(rng_m=100)),
    "S5": ("kit error 10%", dict(sig=0.1)),
    "S6": ("kit error 40%", dict(sig=0.4)),
    "S7": ("unsafe area 10%", dict(prev=0.1)),
    "S8": ("unsafe area 50%", dict(prev=0.5)),
    "S9": ("strong contrast in field", dict(sd=0.8)),
    "S10": ("short-range noise 50%", dict(nug=0.5)),
    "S11": ("inlet plume dominant", dict(R2=0.6, plume_only=True)),
}


# ---------------------------------------------------------------- fields

def grf(rng, ell):
    """Smooth random field with patch size ~ell metres, zero mean, unit variance."""
    s = ell / CELL / np.sqrt(2)
    pad = 2
    w = rng.standard_normal((NY * pad, NX * pad))
    fy = np.fft.fftfreq(NY * pad)[:, None]
    fx = np.fft.fftfreq(NX * pad)[None, :]
    f = np.fft.ifft2(np.fft.fft2(w) * np.exp(-2 * np.pi ** 2 * s ** 2 * (fx ** 2 + fy ** 2)))
    f = f.real[:NY, :NX].ravel()
    return (f - f.mean()) / f.std()


def std(v):
    return (v - v.mean()) / v.std()


def block(y, b=5):
    """Mean over b x b cells (10 m blocks): the scale a farmer can act on."""
    Y = y.reshape(NY, NX)
    out = np.empty_like(Y)
    for i in range(0, NY, b):
        for j in range(0, NX, b):
            out[i:i + b, j:j + b] = Y[i:i + b, j:j + b].mean()
    return out.ravel()


def make_field(rng, R2=0.3, nug=0.2, rng_m=40, sd=0.4, prev=0.3, plume_only=False):
    """One synthetic field.

    R2      share of Cd variance explained by things the drone can see
    nug     share that is metre-scale noise no map can resolve
    rng_m   patch size of the remaining spatial pattern
    sd      within-field spread of ln(Cd) (0.4 is roughly a 40% CV)
    prev    share of the field above the threshold
    """
    A = grf(rng, 30)
    B = grf(rng, 15)
    elev = 0.5 * std(XY[:, 0]) * rng.choice([-1, 1]) + grf(rng, 60)
    G = np.c_[np.ones(N), XY]
    low = -(elev - G @ np.linalg.lstsq(G, elev, rcond=None)[0])
    gyy, gxx = np.gradient(elev.reshape(NY, NX))
    slope = np.hypot(gxx, gyy).ravel()
    d_in = np.hypot(*XY.T)                    # inlet at one corner
    d_out = np.hypot(*(XY - XY.max(0)).T)     # outlet at the opposite corner

    def nz():
        return 0.15 * rng.standard_normal(N)

    # The 11 layers of the plan: 5 colour, 3 terrain, 3 distance.
    layers = np.c_[A + 0.3 * B + nz(), -0.8 * A + 0.5 * B + nz(), 0.6 * A + 0.6 * B + nz(),
                   B + nz(), grf(rng, 10) + nz(), -low, low + 0.3 * grf(rng, 8), slope,
                   XY[:, 1], d_in, d_out]
    layers = (layers - layers.mean(0)) / layers.std(0)
    plume = std(np.exp(-d_in / 40.0))
    if plume_only:
        h = plume
    else:
        w = rng.dirichlet([1, 1, 1])
        h = std(w[0] * rng.choice([-1, 1]) * A + w[1] * std(low) + w[2] * plume)
    sp = std(0.8 * grf(rng, rng_m) + 0.6 * grf(rng, rng_m / 4))
    z = std(np.sqrt(R2) * h + np.sqrt(max(1 - R2 - nug, 0)) * sp + np.sqrt(nug) * rng.standard_normal(N))
    y = sd * (z - np.quantile(z, 1 - prev))
    U, S, _ = np.linalg.svd(layers, full_matrices=False)
    return dict(y=y, yb=block(y), pcs=U[:, :3] * np.sqrt(N), i_in=0, i_out=N - 1)


# ---------------------------------------------------------------- model

def hp_table(names):
    """Fixed table of candidate hyperparameters (drawn in the order given)."""
    hr = np.random.default_rng(123)
    lim = dict(sf=(0.1, 1.5), ls=(5, 300), lf=(0.3, 10), sn=(0.03, 1.0), sb=(0.03, 1.0))
    return {k: np.exp(hr.uniform(np.log(lim[k][0]), np.log(lim[k][1]), C)) for k in names}


def _lnp(x, med, s):
    return -0.5 * ((np.log(x) - np.log(med)) / s) ** 2


def fit_predict(HP, idx, yobs, F, model, mode, sig_known):
    """Gaussian process fit and prediction for every cell.

    model  'PC'  one kernel over position and 3 principal components (the PDF's model)
           'OK'  position only (ordinary kriging)
           'LIN' position kernel plus a linear trend on the principal components
    mode   'ML'  all settings fitted to the few samples by maximum likelihood
           'MAP' noise fixed to the known kit error, priors on the other settings
    """
    n = len(idx)
    Xs, Fs = XY[idx], F[idx]
    D2s = ((Xs[:, None] - Xs[None]) ** 2).sum(-1)
    ym = yobs.mean()
    yc = yobs - ym
    sf, ls = HP["sf"], HP["ls"]
    sn = np.full(C, sig_known) if mode == "MAP" else HP["sn"]
    K = sf[:, None, None] ** 2 * np.exp(-0.5 * D2s / ls[:, None, None] ** 2)
    if model == "PC":
        D2f = ((Fs[:, None] - Fs[None]) ** 2).sum(-1)
        K = K * np.exp(-0.5 * D2f / HP["lf"][:, None, None] ** 2)
    elif model == "LIN":
        K = K + HP["sb"][:, None, None] ** 2 * (Fs @ Fs.T / 3)
    K = K + CONST2 + (sn[:, None, None] ** 2 + 1e-8) * np.eye(n)
    Lc = np.linalg.cholesky(K)
    al = np.linalg.solve(K, np.broadcast_to(yc, (C, n))[..., None])[..., 0]
    lml = -0.5 * (al * yc).sum(1) - np.log(np.diagonal(Lc, axis1=1, axis2=2)).sum(1)
    if mode == "MAP":
        lml = lml + _lnp(sf, 0.4, 0.5) + _lnp(ls, 40, 0.7)
        if model == "PC":
            lml = lml + _lnp(HP["lf"], 2.0, 0.7)
        if model == "LIN":
            lml = lml + _lnp(HP["sb"], 0.2, 0.7)
    b = int(np.argmax(lml))
    d2s = ((XY[:, None] - Xs[None]) ** 2).sum(-1)
    ks = sf[b] ** 2 * np.exp(-0.5 * d2s / ls[b] ** 2)
    kss = np.full(N, sf[b] ** 2)
    if model == "PC":
        d2f = ((F[:, None] - Fs[None]) ** 2).sum(-1)
        ks = ks * np.exp(-0.5 * d2f / HP["lf"][b] ** 2)
    elif model == "LIN":
        ks = ks + HP["sb"][b] ** 2 * (F @ Fs.T) / 3
        kss = kss + HP["sb"][b] ** 2 * (F ** 2).sum(1) / 3
    ks = ks + CONST2
    kss = kss + CONST2
    mu = ks @ al[b] + ym
    v = np.linalg.solve(K[b], ks.T).T
    return mu, np.sqrt(np.maximum(kss - (v * ks).sum(1), 1e-10))


# ---------------------------------------------------------------- designs

def kmeans_pick(Z, k, rng, it=12):
    """k spots spread evenly over the space Z (spatial or feature coverage)."""
    c = Z[rng.choice(len(Z), k, replace=False)]
    for _ in range(it):
        a = ((Z[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (a == j).any():
                c[j] = Z[a == j].mean(0)
    return list(dict.fromkeys(((Z[:, None] - c[None]) ** 2).sum(-1).argmin(0).tolist()))


def clhs(F, k, rng, tries=400):
    """Conditioned Latin hypercube by random search (adequate for k <= 5)."""
    q = [np.quantile(F[:, j], np.linspace(0, 1, k + 1)) for j in range(F.shape[1])]
    best, bo = None, 1e9
    for _ in range(tries):
        s = rng.choice(N, k, replace=False)
        o = sum(np.abs(np.histogram(F[s, j], bins=q[j])[0] - 1).sum() for j in range(F.shape[1]))
        if o < bo:
            bo, best = o, s
    return best.tolist()


def observe(i, y, rng, sig, locerr=2.0, ncomp=1, composite_path=False):
    """Kit reading at cell i: GPS error moves the real spot, then kit noise is added.
    With ncomp > 1 several cores within a few metres are mixed into one test."""
    v = []
    for _ in range(ncomp):
        p = XY[i] + rng.standard_normal(2) * (locerr if ncomp == 1 else 4.0)
        cx = int(np.clip(round(p[0] / CELL), 0, NX - 1))
        cy = int(np.clip(round(p[1] / CELL), 0, NY - 1))
        v.append(y[cy * NX + cx])
    val = np.log(np.mean(np.exp(v))) if composite_path else v[0]
    return val + sig * rng.standard_normal()


_verf = np.vectorize(erf)


def Phi(x):
    return 0.5 * (1 + _verf(x / np.sqrt(2)))


# ---------------------------------------------------------------- experiment 1: methods

METHODS = {   # name: (design, next-spot rule, model, mode)
    "M1 random + GP-PC ML": ("random", None, "PC", "ML"),
    "M2 spatial-cover + OK MAP": ("spcov", None, "OK", "MAP"),
    "M3 PDF: cLHS+maxvar PC ML": ("init", "maxvar", "PC", "ML"),
    "M4 init+straddle PC ML": ("init", "straddle", "PC", "ML"),
    "M5 init+maxvar PC MAP": ("init", "maxvar", "PC", "MAP"),
    "M6 init+straddle PC MAP": ("init", "straddle", "PC", "MAP"),
    "M7 init+straddle LIN MAP": ("init", "straddle", "LIN", "MAP"),
    "M8 init+straddle OK MAP": ("init", "straddle", "OK", "MAP"),
    "M9 feature-cover LIN MAP": ("fscs", None, "LIN", "MAP"),
}


def metrics_methods(mu, s, fld):
    yb = fld["yb"]
    pred, tr = mu > 0, yb > 0
    fs = ((~pred) & tr).sum() / max(tr.sum(), 1)
    fu = (pred & ~tr).sum() / max((~tr).sum(), 1)
    p = Phi(mu / s)
    conf = (p > 0.9) | (p < 0.1)
    return dict(acc=(pred == tr).mean(), bal=1 - (fs + fu) / 2, fs=fs,
                rmse=np.sqrt(((mu - yb) ** 2).mean()), cov=(np.abs(yb - mu) < 1.645 * s).mean(),
                conf=conf.mean(), cacc=(pred == tr)[conf].mean() if conf.any() else np.nan)


def run_method(HP, fld, meth, rng, sig=0.22, nmax=15, n0=5, nug=0.2, sd=0.4):
    design, acq, model, mode = METHODS[meth]
    y, F = fld["y"], fld["pcs"]
    sk = np.sqrt(sig ** 2 + nug * sd ** 2)        # known kit error + assumed short-range noise
    if design == "random":
        idx = rng.choice(N, nmax, replace=False).tolist()
    elif design == "spcov":
        idx = kmeans_pick(XY / 50.0, nmax, rng)
    elif design == "fscs":
        idx = kmeans_pick(np.c_[XY / 50.0, F], nmax, rng)
    else:
        idx = clhs(F, n0 - 2, rng) + [fld["i_in"], fld["i_out"]]
    yo = [observe(i, y, rng, sig) for i in idx]
    stop = None
    while True:
        mu, s = fit_predict(HP, np.array(idx), np.array(yo), F, model, mode, sk)
        m = metrics_methods(mu, s, fld)
        if acq and stop is None and m["conf"] >= 0.9:     # the PDF's 90%/90% stop rule
            stop = (len(idx), m["acc"])
        if acq is None or len(idx) >= nmax:
            break
        a = s.copy() if acq == "maxvar" else 1.96 * s - np.abs(mu)
        a[idx] = -1e9
        j = int(a.argmax())
        idx.append(j)
        yo.append(observe(j, y, rng, sig))
    return m | dict(stop_n=stop[0] if stop else np.nan, stop_acc=stop[1] if stop else np.nan)


def exp_methods(scenarios, reps):
    HP = hp_table(["sf", "ls", "lf", "sn", "sb"])
    t0 = time.time()
    print("columns: accuracy, balanced accuracy, share of unsafe area missed (all vs 10 m block truth) |"
          " map error (ln), coverage of 90% ranges | share confidently classed, accuracy there |"
          " share of runs where the 90%/90% stop rule fired, tests used, accuracy at stop")
    for sid in scenarios:
        label, kw0 = SCENARIOS[sid]
        kw = dict(kw0)
        sig = kw.pop("sig", 0.22)
        res = {m: [] for m in METHODS}
        for r in range(reps):
            fld = make_field(np.random.default_rng(1000 + r), **kw)
            for m in METHODS:
                res[m].append(run_method(HP, fld, m, np.random.default_rng(5000 + r), sig=sig,
                                         nug=kw.get("nug", 0.2), sd=kw.get("sd", 0.4)))
        print(f"== {sid} {label}  ('all safe' accuracy = {1 - kw.get('prev', 0.3):.2f})  t={time.time() - t0:.0f}s")
        for m in METHODS:
            def g(k):
                a = np.array([x[k] for x in res[m]], float)
                return float("nan") if np.all(np.isnan(a)) else np.nanmean(a)
            fired = np.mean([not np.isnan(x["stop_n"]) for x in res[m]])
            print(f"  {m:27s} {g('acc'):.3f} {g('bal'):.3f} {g('fs'):.3f} | {g('rmse'):.3f} {g('cov'):.2f} |"
                  f" {g('conf'):.2f} {g('cacc'):.2f} | {fired:.2f} {g('stop_n'):4.1f} {g('stop_acc'):.2f}")


# ---------------------------------------------------------------- experiment 2: budget

def metrics_budget(mu, s, fld):
    yb = fld["yb"]
    pred, tr = mu > 0, yb > 0
    fs = ((~pred) & tr).sum() / max(tr.sum(), 1)
    fu = (pred & ~tr).sum() / max((~tr).sum(), 1)
    p = Phi(mu / s)
    green, red = p < 0.2, p > 0.8
    zt = np.array([fld["y"][ZONE == z].mean() for z in range(6)])
    zp = np.array([mu[ZONE == z].mean() for z in range(6)])
    return dict(acc=(pred == tr).mean(), bal=1 - (fs + fu) / 2, fs=fs,
                yellow=1 - green.mean() - red.mean(),
                fgreen=(tr & green).sum() / max(green.sum(), 1),
                caught=(tr & ~green).sum() / max(tr.sum(), 1),
                zacc=((zt > 0) == (zp > 0)).mean(), zrmse=np.sqrt(((zt - zp) ** 2).mean()),
                cov=(np.abs(yb - mu) < 1.645 * s).mean())


def run_budget(HP, fld, design, model, nmax, rng, sig=0.22, nug=0.2, sd=0.4, ncomp=1):
    """'static': all spots spread evenly. 'hybrid': 60% spread evenly, then straddle rule."""
    y, F = fld["y"], fld["pcs"]
    sk = np.sqrt(sig ** 2 + nug * sd ** 2 / ncomp)
    n0 = nmax if design == "static" else int(round(0.6 * nmax))
    idx = kmeans_pick(XY / 50.0, n0, rng)
    yo = [observe(i, y, rng, sig, ncomp=ncomp, composite_path=True) for i in idx]
    while True:
        mu, s = fit_predict(HP, np.array(idx), np.array(yo), F, model, "MAP", sk)
        if len(idx) >= nmax:
            break
        a = 1.96 * s - np.abs(mu)
        a[idx] = -1e9
        d = np.sqrt(((XY[:, None] - XY[idx][None]) ** 2).sum(-1)).min(1)
        a[d < 10] = -1e9                              # keep new spots 10 m from earlier ones
        j = int(a.argmax())
        idx.append(j)
        yo.append(observe(j, y, rng, sig, ncomp=ncomp, composite_path=True))
    return metrics_budget(mu, s, fld)


def exp_budget(scenarios, reps):
    HP = hp_table(["sf", "ls", "sb"])
    cfg = [("static", "OK", 1), ("hybrid", "OK", 1), ("hybrid", "LIN", 1), ("hybrid", "OK", 3)]
    t0 = time.time()
    print("columns: tests | block accuracy, balanced accuracy, unsafe area missed |"
          " share yellow, share of green that is really unsafe, share of unsafe area not green |"
          " zone accuracy, zone error (ln) | coverage of 90% ranges")
    for sid in scenarios:
        label, kw = SCENARIOS[sid]
        print(f"== {sid} {label}  ('all safe' accuracy = {1 - kw.get('prev', 0.3):.2f})  t={time.time() - t0:.0f}s")
        fkw = {k: v for k, v in kw.items() if k != "sig"}
        flds = [make_field(np.random.default_rng(1000 + r), **fkw) for r in range(reps)]
        for d, m, nc in cfg:
            for n in (15, 25, 40, 60):
                R = [run_budget(HP, flds[r], d, m, n, np.random.default_rng(7000 + r),
                                sig=kw.get("sig", 0.22), sd=kw.get("sd", 0.4), ncomp=nc) for r in range(reps)]

                def g(k):
                    return np.mean([x[k] for x in R])
                print(f"  {d:6s} {m:3s} cores/test={nc} n={n:2d} | {g('acc'):.3f} {g('bal'):.3f} {g('fs'):.3f} |"
                      f" {g('yellow'):.2f} {g('fgreen'):.3f} {g('caught'):.2f} |"
                      f" {g('zacc'):.2f} {g('zrmse'):.3f} | {g('cov'):.2f}")


# ---------------------------------------------------------------- experiment 3: validation power

def exp_validation():
    print("P(pass) = chance that at least 80% of n split samples land within 25% of the lab value")
    rng = np.random.default_rng(1)
    for sdk in (0.12, 0.18, 0.22, 0.30):
        e = np.exp(rng.standard_normal(400000) * sdk)
        p = ((e > 0.75) & (e < 1.25)).mean()
        out = []
        for n in (10, 20, 40):
            k0 = int(np.ceil(0.8 * n))
            pw = sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(k0, n + 1))
            out.append(f"n={n}: P(pass)={pw:.2f}, 95% CI +-{1.96 * np.sqrt(p * (1 - p) / n):.2f}")
        print(f"  kit error (sd of ln) {sdk:.2f}: true share within 25% = {p:.2f} | " + " ; ".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["methods", "budget", "validation"])
    ap.add_argument("--scenarios", default=None, help="comma-separated ids, e.g. S0,S7")
    ap.add_argument("--reps", type=int, default=None)
    a = ap.parse_args()
    if a.experiment == "methods":
        exp_methods((a.scenarios or ",".join(SCENARIOS)).split(","), a.reps or 40)
    elif a.experiment == "budget":
        exp_budget((a.scenarios or "S0,S7,S11,S9").split(","), a.reps or 24)
    else:
        exp_validation()
