"""Turning the model's probabilities into the answers a farmer reads."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soilsight.model.gp import Posterior

GREEN, YELLOW, RED = 0, 1, 2
CLASS_NAMES = ("green", "yellow", "red")


def classify(p_unsafe, lo: float = 0.2, hi: float = 0.8) -> np.ndarray:
    """Green when probably safe, red when probably unsafe, yellow when the tests cannot say."""
    p = np.asarray(p_unsafe)
    return np.where(p < lo, GREEN, np.where(p > hi, RED, YELLOW))


@dataclass(frozen=True)
class ZoneAnswer:
    zone: int
    mean: float        # ln(Cd / threshold), averaged over the zone
    sd: float
    p_unsafe: float    # probability that the zone's average is above the threshold
    label: str         # green / yellow / red


def zone_answers(post: Posterior, xy, zone_ids, feats=None, lo: float = 0.2, hi: float = 0.8,
                 max_points: int = 300) -> list:
    """One answer per management zone. Zones larger than max_points cells are thinned evenly."""
    zone_ids = np.asarray(zone_ids)
    out = []
    for z in np.unique(zone_ids):
        cells = np.flatnonzero(zone_ids == z)
        if len(cells) > max_points:
            cells = cells[np.linspace(0, len(cells) - 1, max_points).astype(int)]
        m = post.area_mean(xy[cells], None if feats is None else feats[cells])
        p = float(m.p_unsafe)
        out.append(ZoneAnswer(int(z), float(m.mean), float(m.sd), p, CLASS_NAMES[int(classify(p, lo, hi))]))
    return out
