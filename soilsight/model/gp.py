"""The soil map model: a Gaussian process that is honest about what it does not know.

Values are ln(reading / threshold), so a spot is unsafe when its value is above 0.

With 5 to 15 tests per field the model's own settings (how patchy the field is, how
noisy a metre of soil is, whether the drone layers matter) cannot be pinned down.
Picking one best guess makes the map overconfident. This model instead keeps many
candidate settings drawn from the priors, weights each by how well it explains the
tests, and averages their predictions. Disagreement between candidates shows up as
extra uncertainty on the map.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

SQRT3 = np.sqrt(3.0)


@dataclass(frozen=True)
class Priors:
    """What is assumed about a field before any test. Each pair is (median, sd of ln).

    These should come from fields already measured. The defaults are placeholders.
    """

    signal_sd: tuple = (0.4, 0.5)     # spread of ln(Cd) across the field
    length_m: tuple = (40.0, 0.7)     # patch size
    nugget_sd: tuple = (0.15, 0.5)    # metre-scale scatter of one core
    trend_sd: tuple = (0.2, 0.7)      # effect of the drone layers, when they matter
    trend_prob: float = 0.5           # chance that the drone layers matter at all
    mean_sd: float = 1.0              # how far the field average may sit from the tests' average


@dataclass(frozen=True)
class Prediction:
    mean: np.ndarray       # ln(Cd / threshold)
    sd: np.ndarray         # its standard deviation
    p_unsafe: np.ndarray   # probability that the value is above 0


@dataclass(frozen=True)
class _Component:
    """One candidate setting that survived, with everything needed to predict from it."""

    weight: float
    signal_sd: float
    length_m: float
    nugget_sd: float
    trend_sd: float
    alpha: np.ndarray      # K^-1 (y - mean)
    k_inv: np.ndarray      # K^-1


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1))


def _corr(d, length, kernel):
    if kernel == "rbf":
        return np.exp(-0.5 * (d / length) ** 2)
    r = SQRT3 * d / length
    return (1 + r) * np.exp(-r)


@dataclass(frozen=True)
class Posterior:
    """The fitted model. Predicts anywhere on the field."""

    kernel: str
    mean_var: float
    obs_xy: np.ndarray
    obs_feats: np.ndarray | None
    y_mean: float
    components: tuple = field(repr=False)

    def _cov(self, c: _Component, d, fa, fb):
        """Prior covariance between two sets of points, for one component."""
        k = c.signal_sd ** 2 * _corr(d, c.length_m, self.kernel) + self.mean_var
        if c.trend_sd > 0 and fa is not None:
            k = k + c.trend_sd ** 2 * (fa @ fb.T) / fa.shape[1]
        return k

    def _prior_var(self, c: _Component, feats):
        v = c.signal_sd ** 2 + self.mean_var
        if c.trend_sd > 0 and feats is not None:
            v = v + c.trend_sd ** 2 * (feats ** 2).sum(1) / feats.shape[1]
        return v

    def _use_feats(self, feats):
        if self.obs_feats is None:
            return None
        if feats is None:
            raise ValueError("this model was fitted with drone layers; pass feats to predict")
        return feats

    def top(self, k: int) -> "Posterior":
        """The same model cut down to its k heaviest components (faster, slightly cruder)."""
        comps = sorted(self.components, key=lambda c: -c.weight)[:k]
        total = sum(c.weight for c in comps)
        comps = tuple(_Component(c.weight / total, c.signal_sd, c.length_m, c.nugget_sd, c.trend_sd,
                                 c.alpha, c.k_inv) for c in comps)
        return Posterior(self.kernel, self.mean_var, self.obs_xy, self.obs_feats, self.y_mean, comps)

    def component_moments(self, xy, feats=None):
        """Mean and sd at each point under every component: arrays of shape (K, N)."""
        feats = self._use_feats(feats)
        d = _dist(xy, self.obs_xy)
        mu = np.empty((len(self.components), len(xy)))
        sd = np.empty_like(mu)
        for i, c in enumerate(self.components):
            ks = self._cov(c, d, feats, self.obs_feats)
            mu[i] = ks @ c.alpha + self.y_mean
            var = self._prior_var(c, feats) - np.einsum("ij,jk,ik->i", ks, c.k_inv, ks)
            sd[i] = np.sqrt(np.maximum(var, 1e-10))
        return mu, sd

    @property
    def weights(self) -> np.ndarray:
        return np.array([c.weight for c in self.components])

    def predict(self, xy, feats=None) -> Prediction:
        mu, sd = self.component_moments(xy, feats)
        w = self.weights[:, None]
        mean = (w * mu).sum(0)
        var = (w * (sd ** 2 + mu ** 2)).sum(0) - mean ** 2
        return Prediction(mean, np.sqrt(np.maximum(var, 1e-10)), (w * ndtr(mu / sd)).sum(0))

    def area_mean(self, xy, feats=None) -> Prediction:
        """Mean value over an area given by sample points xy: one number, with its uncertainty.

        Unlike averaging predict(), this accounts for neighbouring cells being wrong together.
        """
        feats = self._use_feats(feats)
        d_obs = _dist(xy, self.obs_xy)
        d_self = _dist(xy, xy)
        mu, var = [], []
        for c in self.components:
            ks = self._cov(c, d_obs, feats, self.obs_feats).mean(0)       # cov(area mean, tests)
            kzz = self._cov(c, d_self, feats, feats).mean()               # var of the area mean
            mu.append(ks @ c.alpha + self.y_mean)
            var.append(max(kzz - ks @ c.k_inv @ ks, 1e-10))
        mu, sd, w = np.array(mu), np.sqrt(var), self.weights
        mean = (w * mu).sum()
        total = (w * (sd ** 2 + mu ** 2)).sum() - mean ** 2
        return Prediction(np.array(mean), np.array(np.sqrt(max(total, 1e-10))), np.array((w * ndtr(mu / sd)).sum()))


class SoilGP:
    """Fits the model to a field's tests.

    kernel        'matern32' (soil-like, rough) or 'rbf' (very smooth)
    n_candidates  settings drawn from the priors
    top_k         how many of them are kept for prediction
    """

    def __init__(self, priors: Priors = Priors(), kernel: str = "matern32",
                 n_candidates: int = 512, top_k: int = 24, seed: int = 0):
        if kernel not in ("matern32", "rbf"):
            raise ValueError(f"unknown kernel {kernel!r}")
        self.priors, self.kernel, self.top_k = priors, kernel, top_k
        rng = np.random.default_rng(seed)

        def draw(median_sd):
            return np.exp(np.log(median_sd[0]) + median_sd[1] * rng.standard_normal(n_candidates))

        self.signal_sd = draw(priors.signal_sd)
        self.length_m = draw(priors.length_m)
        self.nugget_sd = draw(priors.nugget_sd)
        self.trend_sd = np.where(rng.random(n_candidates) < priors.trend_prob, draw(priors.trend_sd), 0.0)

    def fit(self, obs_xy, y, noise_sd, obs_feats=None, cores=1) -> Posterior:
        """obs_xy (n, 2) metres; y (n,) ln(reading / threshold); noise_sd one number or one per
        test (a lab result has a smaller one than a kit result); obs_feats (n, p) standardised
        drone layers at the tests, or None to ignore the layers; cores mixed into each test
        (one number or one per test), which averages out metre-scale scatter."""
        obs_xy = np.asarray(obs_xy, float).reshape(-1, 2)
        y = np.asarray(y, float)
        n = len(y)
        if n == 0:
            raise ValueError("need at least one test")
        noise_var = np.broadcast_to(np.asarray(noise_sd, float) ** 2, (n,))
        cores = np.broadcast_to(np.asarray(cores, float), (n,))
        trend_sd = self.trend_sd if obs_feats is not None else np.zeros_like(self.trend_sd)
        if obs_feats is not None:
            obs_feats = np.asarray(obs_feats, float).reshape(n, -1)

        d = _dist(obs_xy, obs_xy)
        y_mean = float(y.mean())
        yc = y - y_mean
        k = self.signal_sd[:, None, None] ** 2 * _corr(d[None], self.length_m[:, None, None], self.kernel)
        if obs_feats is not None:
            k = k + trend_sd[:, None, None] ** 2 * (obs_feats @ obs_feats.T / obs_feats.shape[1])
        k = k + self.priors.mean_sd ** 2 + (noise_var[None] + self.nugget_sd[:, None] ** 2 / cores[None] + 1e-8)[:, :, None] * np.eye(n)

        chol = np.linalg.cholesky(k)
        alpha = np.linalg.solve(k, np.broadcast_to(yc, (len(k), n))[..., None])[..., 0]
        log_lik = -0.5 * (alpha * yc).sum(1) - np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(1)
        w = np.exp(log_lik - log_lik.max())
        keep = np.argsort(-w)[:self.top_k]
        keep = keep[w[keep] > 1e-6]
        w_keep = w[keep] / w[keep].sum()
        comps = tuple(_Component(float(wi), float(self.signal_sd[i]), float(self.length_m[i]),
                                 float(self.nugget_sd[i]), float(trend_sd[i]), alpha[i], np.linalg.inv(k[i]))
                      for wi, i in zip(w_keep, keep))
        return Posterior(self.kernel, self.priors.mean_sd ** 2, obs_xy, obs_feats, y_mean, comps)
