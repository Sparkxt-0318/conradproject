"""Terrain layers from the stitched surface model.

A paddy is nearly flat, so what matters is the few centimetres left after the field's
overall tilt is removed. Stitching without ground control points can bow the surface by
more than that, which is why these layers carry their own reliability check.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from soilsight.layers.grid import FieldGrid, block_stat, read_on_grid, smooth

TERRAIN_LAYERS = {
    "elevation": "height in metres after removing the field's overall tilt",
    "low_spot": "how far a cell sits below its surroundings within 10 m, in metres; positive in hollows",
    "slope": "steepness of the smoothed surface, in percent",
}


def fit_plane(z: np.ndarray, x: np.ndarray, y: np.ndarray, use: np.ndarray) -> np.ndarray:
    """The best-fitting tilted plane through z over the cells in use, evaluated everywhere."""
    a = np.c_[np.ones(use.sum()), x[use], y[use]]
    coef = np.linalg.lstsq(a, z[use], rcond=None)[0]
    return coef[0] + coef[1] * x + coef[2] * y


def _disk_mean(a: np.ndarray, radius_cells: float) -> np.ndarray:
    r = int(np.ceil(radius_cells))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2 <= radius_cells ** 2).astype("float64")
    valid = np.isfinite(a)
    num = ndimage.convolve(np.where(valid, a, 0).astype("float64"), disk, mode="nearest")
    den = ndimage.convolve(valid.astype("float64"), disk, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def terrain_layers(dsm, grid: FieldGrid, smooth_m: float = 3.0, radius_m: float = 10.0, factor: int = 5) -> tuple:
    """Terrain layers per grid cell, and numbers describing how far to trust them.

    Returns (layers, stats) where stats holds relief_m (5th to 95th percentile of the smoothed,
    tilt-free surface) and roughness_m (typical height scatter inside one cell: soil clods
    plus stitching noise).
    """
    fine = read_on_grid(dsm, grid, (1,), factor)[0].astype("float64")
    z = block_stat(fine, factor)
    rough = block_stat(fine, factor, "sd")
    x, y = grid.centres()
    use = grid.mask & np.isfinite(z)
    if use.sum() < 10:
        raise ValueError("the surface model covers almost none of the field")
    plane = fit_plane(z, x - x.mean(), y - y.mean(), use)
    sigma = smooth_m / 2 / grid.cell
    flat = smooth(np.where(use, z - plane, np.nan), sigma)
    full = smooth(np.where(use, z, np.nan), sigma)
    filled = np.where(np.isfinite(full), full, np.nanmean(full))
    gy, gx = np.gradient(filled, grid.cell)
    slope = np.where(use, np.hypot(gx, gy) * 100, np.nan)
    low = _disk_mean(flat, radius_m / grid.cell) - flat
    stats = dict(relief_m=float(np.nanpercentile(flat, 95) - np.nanpercentile(flat, 5)),
                 roughness_m=float(np.nanmedian(rough[use])))
    return {"elevation": flat, "low_spot": low, "slope": slope}, stats
