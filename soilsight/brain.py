"""The Brain: decides where to test, updates the map after each result, and reports.

One Brain serves one field visit:

    brain = Brain(xy, protocol=Protocol(budget=15), fixed_xy=[inlet, outlet])
    for spot in brain.initial_spots():
        brain.add_result(spot, reading)          # ln(reading / threshold)
    while (nxt := brain.next_spot()) is not None:
        brain.add_result(nxt.xy, reading)
    cells = brain.map()                          # per-cell mean, sd, P(unsafe)
    zones = brain.zones(zone_ids)                # one answer per management zone
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soilsight.model.classify import zone_answers
from soilsight.model.gp import Posterior, Prediction, Priors, SoilGP
from soilsight.sampling.acquire import next_spot
from soilsight.sampling.design import spread_spots


@dataclass(frozen=True)
class Protocol:
    """How a field visit is run."""

    budget: int = 15           # tests per field
    cores: int = 3             # soil cores mixed into each test
    kit_sd: float = 0.22       # kit error, sd of ln(reading / truth), from the lab comparison
    init_frac: float = 0.6     # share of the budget spread evenly before the loop starts
    min_sep_m: float = 10.0    # keep tests at least this far apart
    cand_spacing_m: float = 10.0   # spacing of the spots the loop may choose from
    eval_spacing_m: float = 6.0    # spacing of the cells a candidate test is judged on


@dataclass(frozen=True)
class Spot:
    xy: np.ndarray             # where to test, metres
    gain: float                # expected drop in the misclassified share of the field
    misclassified_now: float   # expected misclassified share before this test


def _thin(xy: np.ndarray, spacing: float) -> np.ndarray:
    """Indices of cells on a lattice of about the given spacing."""
    key = np.round(xy / spacing).astype(int)
    centre = key * spacing
    order = np.argsort(((xy - centre) ** 2).sum(1), kind="stable")
    _, first = np.unique(key[order], axis=0, return_index=True)
    return np.sort(order[first])


class Brain:
    """xy: (N, 2) cell centres in metres. feats: (N, p) standardised drone layers, or None
    to map from position alone. fixed_xy: spots that must be tested (inlet, outlet)."""

    def __init__(self, xy, feats=None, fixed_xy=None, protocol: Protocol = Protocol(),
                 priors: Priors = Priors(), kernel: str = "matern32", seed: int = 0, center_on_tests: bool = False,
                 select: str = "resample"):
        self.xy = np.asarray(xy, float)
        self.feats = None if feats is None else np.asarray(feats, float)
        self.fixed_xy = fixed_xy
        self.protocol = protocol
        self._gp = SoilGP(priors, kernel, seed=seed, center_on_tests=center_on_tests, select=select)
        self._rng = np.random.default_rng(seed)
        self._cand = _thin(self.xy, protocol.cand_spacing_m)
        self._eval = _thin(self.xy, protocol.eval_spacing_m)
        self._obs_xy, self._y, self._noise, self._cores, self._obs_feats = [], [], [], [], []
        self._post = None

    @classmethod
    def from_package(cls, package: dict) -> "Brain":
        """The Brain for a field package (see soilsight.app.package), using the package's own
        candidate settings and spot lists so it agrees with the phone app to rounding."""
        from soilsight.app.package import cells_xy
        p, m = package["protocol"], package["model"]
        brain = cls(cells_xy(package), protocol=Protocol(budget=p["budget"], cores=p["cores"], kit_sd=p["kit_sd"],
                                                         init_frac=p["init_frac"], min_sep_m=p["min_sep_m"]),
                    kernel=m["kernel"])
        brain._gp = SoilGP.from_candidates(m["candidates"], m["kernel"], m["top_k"], m["mean_sd"])
        brain._cand = np.asarray(package["plan"]["cand"], int)
        brain._eval = np.asarray(package["plan"]["eval"], int)
        brain._initial = np.asarray(package["plan"]["initial"], int)
        return brain

    # ---- where to test

    def initial_spots(self) -> np.ndarray:
        """The evenly spread first tests, as (n, 2) positions."""
        if getattr(self, "_initial", None) is not None:
            return self.xy[self._initial]
        n = max(int(round(self.protocol.init_frac * self.protocol.budget)), 2)
        return self.xy[spread_spots(self.xy, min(n, self.protocol.budget), self._rng, self.fixed_xy)]

    def next_spot(self) -> Spot | None:
        """The most useful next test, or None when the budget is spent."""
        if self.n_tests >= self.protocol.budget:
            return None
        p = self.protocol
        f = self.feats
        pick = next_spot(self.posterior(), self.xy[self._cand], self.xy[self._eval], kit_sd=p.kit_sd,
                         cores=p.cores, cand_feats=None if f is None else f[self._cand],
                         cells_feats=None if f is None else f[self._eval],
                         taken_xy=np.array(self._obs_xy), min_sep_m=p.min_sep_m)
        return Spot(self.xy[self._cand[pick.index]], pick.gain, pick.misclassified_now)

    # ---- results

    @property
    def n_tests(self) -> int:
        return len(self._y)

    def nearest_cell(self, xy) -> int:
        return int(((self.xy - np.asarray(xy, float)) ** 2).sum(1).argmin())

    def add_result(self, xy, value: float, noise_sd: float | None = None, cores: int | None = None) -> None:
        """Record one test. value is ln(reading / threshold). noise_sd defaults to the kit
        error; pass a smaller one for an accredited-lab result. cores defaults to the protocol's."""
        self._obs_xy.append(np.asarray(xy, float))
        self._y.append(float(value))
        self._noise.append(self.protocol.kit_sd if noise_sd is None else float(noise_sd))
        self._cores.append(self.protocol.cores if cores is None else int(cores))
        if self.feats is not None:
            self._obs_feats.append(self.feats[self.nearest_cell(xy)])
        self._post = None

    # ---- answers

    def posterior(self) -> Posterior:
        if self._post is None:
            if not self._y:
                raise ValueError("no test results yet")
            self._post = self._gp.fit(np.array(self._obs_xy), np.array(self._y), np.array(self._noise),
                                      np.array(self._obs_feats) if self.feats is not None else None,
                                      cores=np.array(self._cores))
        return self._post

    def map(self) -> Prediction:
        """Mean, sd and P(unsafe) for every cell."""
        return self.posterior().predict(self.xy, self.feats)

    def zones(self, zone_ids, lo: float = 0.2, hi: float = 0.8) -> list:
        """One green / yellow / red answer per management zone."""
        return zone_answers(self.posterior(), self.xy, zone_ids, self.feats, lo, hi)
