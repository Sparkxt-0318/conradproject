"""Where the next test goes, and what it is worth.

The goal is a safe / unsafe map, so a test is valued by how much of the field it is
expected to stop being misclassified (after Marchant et al. 2013). For each candidate
spot the model imagines the possible readings there, updates the map for each, and
averages the share of the field that would still be on the wrong side of the threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from soilsight.model.gp import Posterior, _dist

_Z, _W = np.polynomial.hermite_e.hermegauss(5)   # imagined readings and their weights
_W = _W / _W.sum()


@dataclass(frozen=True)
class NextSpot:
    index: int               # index into the candidates
    gain: float              # expected drop in the misclassified share of the field
    misclassified_now: float  # expected misclassified share before the test
    gains: np.ndarray        # the gain of every candidate (minus infinity where not allowed)


def misclassified_share(post: Posterior, xy, feats=None) -> float:
    """Expected share of these cells that the current map puts on the wrong side."""
    mu, sd = post.component_moments(xy, feats)
    return float((post.weights[:, None] * ndtr(-np.abs(mu) / sd)).sum(0).mean())


def next_spot(post: Posterior, cand_xy, cells_xy, *, kit_sd: float, cores: int = 1,
              cand_feats=None, cells_feats=None, taken_xy=None, min_sep_m: float = 0.0,
              max_components: int = 6) -> NextSpot:
    """Pick the candidate whose test is expected to fix the most of the map.

    cand_xy   spots that could be tested
    cells_xy  cells the map is judged on (a thinned grid is enough)
    kit_sd, cores  the planned test: kit error and cores mixed into it
    taken_xy, min_sep_m  keep the new spot this far from earlier ones
    """
    post = post.thin(max_components)
    use = post.obs_feats is not None
    fc, fz = (cand_feats, cells_feats) if use else (None, None)
    d_cand_obs = _dist(cand_xy, post.obs_xy)
    d_cell_obs = _dist(cells_xy, post.obs_xy)
    d_cand_cell = _dist(cand_xy, cells_xy)
    mu_all, sd_all = post.component_moments(cells_xy, fz)

    gains = np.zeros(len(cand_xy))
    now = 0.0
    for c, mu, sd in zip(post.components, mu_all, sd_all):
        k_co = post._cov(c, d_cand_obs, fc, post.obs_feats)
        k_zo = post._cov(c, d_cell_obs, fz, post.obs_feats)
        cov = post._cov(c, d_cand_cell, fc, fz) - k_co @ c.k_inv @ k_zo.T      # (candidates, cells)
        var_cand = post._prior_var(c, fc) - np.einsum("ij,jk,ik->i", k_co, c.k_inv, k_co)
        test_var = kit_sd ** 2 + c.nugget_sd ** 2 / cores
        shift = cov / np.sqrt(np.maximum(var_cand, 1e-10) + test_var)[:, None]   # map moves by shift * z
        sd_new = np.sqrt(np.maximum(sd[None] ** 2 - shift ** 2, 1e-10))
        after = sum(w * ndtr(-np.abs(mu[None] + shift * z) / sd_new).mean(1) for z, w in zip(_Z, _W))
        before = ndtr(-np.abs(mu) / sd).mean()
        gains += c.weight * (before - after)
        now += c.weight * before

    if taken_xy is not None and len(taken_xy):
        gains[_dist(cand_xy, np.asarray(taken_xy, float)).min(1) < max(min_sep_m, 1e-9)] = -np.inf
    best = int(np.argmax(gains))
    return NextSpot(best, float(gains[best]), float(now), gains)
