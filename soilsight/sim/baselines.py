"""Reference sampling strategies the benchmark compares.

These are frozen baselines: the engineering plan's loop, plain kriging on evenly
spread spots, and the variants in between. New strategies are judged against them,
so their behaviour must not change. The production model lives elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from soilsight.sim.fields import Field, Grid, observe
from soilsight.sim.metrics import score

N_CANDIDATES = 256       # hyperparameter settings tried per fit
MEAN_PRIOR_VAR = 0.3 ** 2
_HP_RANGE = dict(sf=(0.1, 1.5), ls=(5, 300), lf=(0.3, 10), sn=(0.03, 1.0), sb=(0.03, 1.0))


def hp_table(names) -> dict:
    """Fixed table of candidate hyperparameters, drawn in the order given.

    sf signal sd, ls spatial length (m), lf length in layer space, sn noise sd, sb trend sd.
    """
    rng = np.random.default_rng(123)
    return {k: np.exp(rng.uniform(np.log(_HP_RANGE[k][0]), np.log(_HP_RANGE[k][1]), N_CANDIDATES))
            for k in names}


def _log_prior(x, median, s):
    return -0.5 * ((np.log(x) - np.log(median)) / s) ** 2


class GPFit(NamedTuple):
    mu: np.ndarray    # predicted ln(Cd / threshold) per cell
    sd: np.ndarray    # its standard deviation (without kit noise)
    hp: dict          # the hyperparameters that were picked


def fit_predict(hp: dict, grid: Grid, idx, yobs, feats, model: str, mode: str,
                noise_sd: float) -> GPFit:
    """Fit a Gaussian process to the samples and predict every cell.

    model  'PC'   one kernel over position and the 3 principal components (the PDF's model)
           'OK'   position only (ordinary kriging)
           'LIN'  position kernel plus a linear trend on the principal components
    mode   'ML'   every setting fitted to the samples by maximum likelihood
           'MAP'  noise fixed to noise_sd, priors on the other settings
    """
    idx = np.asarray(idx)
    yobs = np.asarray(yobs, float)
    n, xy = len(idx), grid.xy
    xs, fs = xy[idx], feats[idx]
    d2s = ((xs[:, None] - xs[None]) ** 2).sum(-1)
    ym = yobs.mean()
    yc = yobs - ym
    sf, ls = hp["sf"], hp["ls"]
    sn = np.full(N_CANDIDATES, noise_sd) if mode == "MAP" else hp["sn"]

    k = sf[:, None, None] ** 2 * np.exp(-0.5 * d2s / ls[:, None, None] ** 2)
    if model == "PC":
        d2f = ((fs[:, None] - fs[None]) ** 2).sum(-1)
        k = k * np.exp(-0.5 * d2f / hp["lf"][:, None, None] ** 2)
    elif model == "LIN":
        k = k + hp["sb"][:, None, None] ** 2 * (fs @ fs.T / 3)
    k = k + MEAN_PRIOR_VAR + (sn[:, None, None] ** 2 + 1e-8) * np.eye(n)

    chol = np.linalg.cholesky(k)
    alpha = np.linalg.solve(k, np.broadcast_to(yc, (N_CANDIDATES, n))[..., None])[..., 0]
    lml = -0.5 * (alpha * yc).sum(1) - np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(1)
    if mode == "MAP":
        lml = lml + _log_prior(sf, 0.4, 0.5) + _log_prior(ls, 40, 0.7)
        if model == "PC":
            lml = lml + _log_prior(hp["lf"], 2.0, 0.7)
        if model == "LIN":
            lml = lml + _log_prior(hp["sb"], 0.2, 0.7)
    b = int(np.argmax(lml))

    d2 = ((xy[:, None] - xs[None]) ** 2).sum(-1)
    ks = sf[b] ** 2 * np.exp(-0.5 * d2 / ls[b] ** 2)
    kss = np.full(grid.n, sf[b] ** 2)
    if model == "PC":
        d2f = ((feats[:, None] - fs[None]) ** 2).sum(-1)
        ks = ks * np.exp(-0.5 * d2f / hp["lf"][b] ** 2)
    elif model == "LIN":
        ks = ks + hp["sb"][b] ** 2 * (feats @ fs.T) / 3
        kss = kss + hp["sb"][b] ** 2 * (feats ** 2).sum(1) / 3
    ks = ks + MEAN_PRIOR_VAR
    kss = kss + MEAN_PRIOR_VAR
    mu = ks @ alpha[b] + ym
    v = np.linalg.solve(k[b], ks.T).T
    sd = np.sqrt(np.maximum(kss - (v * ks).sum(1), 1e-10))
    picked = {name: float(vals[b]) for name, vals in hp.items()} | {"sn": float(sn[b])}
    return GPFit(mu, sd, picked)


def kmeans_pick(z: np.ndarray, k: int, rng: np.random.Generator, iters: int = 12) -> list:
    """Up to k cells spread evenly over the space z (k-means centres, snapped to cells)."""
    c = z[rng.choice(len(z), k, replace=False)]
    for _ in range(iters):
        a = ((z[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (a == j).any():
                c[j] = z[a == j].mean(0)
    return list(dict.fromkeys(((z[:, None] - c[None]) ** 2).sum(-1).argmin(0).tolist()))


def clhs(feats: np.ndarray, k: int, rng: np.random.Generator, tries: int = 400) -> list:
    """Conditioned Latin hypercube sample of k cells, by random search (fine for k <= 5)."""
    q = [np.quantile(feats[:, j], np.linspace(0, 1, k + 1)) for j in range(feats.shape[1])]
    best, best_cost = None, 1e9
    for _ in range(tries):
        s = rng.choice(len(feats), k, replace=False)
        cost = sum(np.abs(np.histogram(feats[s, j], bins=q[j])[0] - 1).sum()
                   for j in range(feats.shape[1]))
        if cost < best_cost:
            best_cost, best = cost, s
    return best.tolist()


@dataclass(frozen=True)
class Strategy:
    """How spots are chosen and how the map is made.

    design       where the first spots go:
                 'random', 'spatial' (even spread), 'feature' (even spread in position
                 and layer space), 'clhs' (Latin hypercube plus inlet and outlet)
    acquisition  how later spots are chosen: None (all spots come from the design),
                 'maxvar' (most uncertain cell), 'straddle' (closest to the threshold
                 relative to its uncertainty)
    n_init       spots from the design when acquisition is set; a whole number, or a
                 fraction of the budget if below 1
    min_sep_m    keep each new spot at least this far from earlier ones
    cores        soil cores mixed into each test
    """

    design: str
    acquisition: str | None
    model: str
    mode: str
    n_init: float = 5
    min_sep_m: float = 0.0
    cores: int = 1


def run_strategy(hp: dict, fld: Field, strat: Strategy, rng: np.random.Generator,
                 kit_sd: float, budget: int = 15) -> dict:
    """Sample one field with one strategy and score the final map.

    Besides the scores, the result holds the cells sampled ('idx') and, for the
    PDF's stop rule (90% of cells classed with 90% confidence), the number of
    tests at which it first fired ('stop_n') and the accuracy then ('stop_acc').
    """
    grid, feats = fld.grid, fld.pcs
    xy = grid.xy
    # The model is told the kit error and the true amount of metre-scale noise.
    noise_sd = np.sqrt(kit_sd ** 2 + fld.nugget * fld.sd ** 2 / strat.cores)
    if strat.acquisition is None:
        n0 = budget
    else:
        n0 = int(round(strat.n_init * budget)) if strat.n_init < 1 else int(strat.n_init)

    if strat.design == "random":
        idx = rng.choice(grid.n, n0, replace=False).tolist()
    elif strat.design == "spatial":
        idx = kmeans_pick(xy / 50.0, n0, rng)
    elif strat.design == "feature":
        idx = kmeans_pick(np.c_[xy / 50.0, feats], n0, rng)
    elif strat.design == "clhs":
        idx = clhs(feats, n0 - 2, rng) + [fld.i_in, fld.i_out]
    else:
        raise ValueError(f"unknown design {strat.design!r}")
    yobs = [observe(fld, i, rng, kit_sd, cores=strat.cores) for i in idx]

    stop_n, stop_acc = np.nan, np.nan
    while True:
        fit = fit_predict(hp, grid, idx, yobs, feats, strat.model, strat.mode, noise_sd)
        m = score(fit.mu, fit.sd, fld)
        if strat.acquisition and np.isnan(stop_n) and m["conf"] >= 0.9:
            stop_n, stop_acc = len(idx), m["acc"]
        if strat.acquisition is None or len(idx) >= budget:
            break
        if strat.acquisition == "maxvar":
            a = fit.sd.copy()
        elif strat.acquisition == "straddle":
            a = 1.96 * fit.sd - np.abs(fit.mu)
        else:
            raise ValueError(f"unknown acquisition {strat.acquisition!r}")
        a[idx] = -1e9
        if strat.min_sep_m > 0:
            d = np.sqrt(((xy[:, None] - xy[idx][None]) ** 2).sum(-1)).min(1)
            a[d < strat.min_sep_m] = -1e9
        j = int(a.argmax())
        idx.append(j)
        yobs.append(observe(fld, j, rng, kit_sd, cores=strat.cores))
    return m | dict(stop_n=stop_n, stop_acc=stop_acc, idx=idx)


# Experiment "methods": nine ways to spend 15 tests.
METHODS = {
    "random + GP-PC ML": Strategy("random", None, "PC", "ML"),
    "spatial + OK MAP": Strategy("spatial", None, "OK", "MAP"),
    "PDF loop: cLHS + maxvar, GP-PC ML": Strategy("clhs", "maxvar", "PC", "ML"),
    "cLHS + straddle, GP-PC ML": Strategy("clhs", "straddle", "PC", "ML"),
    "cLHS + maxvar, GP-PC MAP": Strategy("clhs", "maxvar", "PC", "MAP"),
    "cLHS + straddle, GP-PC MAP": Strategy("clhs", "straddle", "PC", "MAP"),
    "cLHS + straddle, LIN MAP": Strategy("clhs", "straddle", "LIN", "MAP"),
    "cLHS + straddle, OK MAP": Strategy("clhs", "straddle", "OK", "MAP"),
    "feature + LIN MAP": Strategy("feature", None, "LIN", "MAP"),
}

# Experiment "budget": how accuracy grows with the number of tests.
BUDGET_STRATEGIES = {
    "static OK": Strategy("spatial", None, "OK", "MAP"),
    "hybrid OK": Strategy("spatial", "straddle", "OK", "MAP", n_init=0.6, min_sep_m=10),
    "hybrid LIN": Strategy("spatial", "straddle", "LIN", "MAP", n_init=0.6, min_sep_m=10),
    "hybrid OK, 3 cores": Strategy("spatial", "straddle", "OK", "MAP", n_init=0.6, min_sep_m=10, cores=3),
}
