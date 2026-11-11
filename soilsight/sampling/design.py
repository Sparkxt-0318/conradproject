"""Where the first tests go: spread evenly over the field."""
from __future__ import annotations

import numpy as np


def spread_spots(xy: np.ndarray, n: int, rng: np.random.Generator, fixed_xy=None,
                 iters: int = 20) -> np.ndarray:
    """Pick n cells that cover the field evenly; returns their indices into xy.

    fixed_xy are spots that must be tested (the irrigation inlet and outlet). They count
    towards n, and the remaining spots fill the gaps around them.
    """
    fixed = np.empty((0, 2)) if fixed_xy is None else np.asarray(fixed_xy, float).reshape(-1, 2)
    n_free = n - len(fixed)
    if n_free < 0:
        raise ValueError(f"{len(fixed)} fixed spots do not fit in a design of {n}")
    centres = np.vstack([fixed, xy[rng.choice(len(xy), n_free, replace=False)]])
    for _ in range(iters):
        nearest = ((xy[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
        for j in range(len(fixed), n):                 # fixed centres never move
            if (nearest == j).any():
                centres[j] = xy[nearest == j].mean(0)
    idx = ((xy[:, None] - centres[None]) ** 2).sum(-1).argmin(0)
    return np.array(list(dict.fromkeys(idx.tolist())))
