"""The field grid: square cells in the map's projection, and which of them are in the field."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy import ndimage


@dataclass(frozen=True)
class FieldGrid:
    crs: str
    west: float            # left edge, map units (metres)
    north: float           # top edge
    cell: float            # cell size, metres
    mask: np.ndarray       # (ny, nx) True for cells inside the field

    @property
    def shape(self) -> tuple:
        return self.mask.shape

    @property
    def transform(self):
        return from_origin(self.west, self.north, self.cell, self.cell)

    def fine_transform(self, factor: int):
        return from_origin(self.west, self.north, self.cell / factor, self.cell / factor)

    def centres(self) -> tuple:
        """Map coordinates of every cell centre: two (ny, nx) arrays (x east, y north)."""
        ny, nx = self.shape
        x = self.west + (np.arange(nx) + 0.5) * self.cell
        y = self.north - (np.arange(ny) + 0.5) * self.cell
        return np.meshgrid(x, y)

    @property
    def origin(self) -> tuple:
        """Lower-left corner; local coordinates are measured from here."""
        return self.west, self.north - self.shape[0] * self.cell


def _snap(bounds, cell):
    w, s, e, n = bounds
    return np.floor(w / cell) * cell, np.floor(s / cell) * cell, np.ceil(e / cell) * cell, np.ceil(n / cell) * cell


def _geom_bounds(geom):
    pts = np.array([p[:2] for ring in (geom["coordinates"] if geom["type"] == "Polygon"
                    else [r for poly in geom["coordinates"] for r in poly]) for p in ring])
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def grid_from_boundary(boundary: dict, crs, cell: float) -> FieldGrid:
    """Grid covering a boundary polygon; a cell is in the field when its centre is."""
    w, s, e, n = _snap(_geom_bounds(boundary), cell)
    shape = int(round((n - s) / cell)), int(round((e - w) / cell))
    mask = rasterize([(boundary, 1)], out_shape=shape, transform=from_origin(w, n, cell, cell), fill=0,
                     dtype="uint8").astype(bool)
    return FieldGrid(str(crs), float(w), float(n), float(cell), mask)


def grid_from_raster(path, cell: float, margin_m: float = 2.0) -> FieldGrid:
    """Grid covering everything a raster has data for, pulled in from the ragged edge by
    margin_m. Used when no boundary was drawn."""
    with rasterio.open(path) as src:
        w, s, e, n = _snap(src.bounds, cell)
        shape = int(round((n - s) / cell)), int(round((e - w) / cell))
        valid = src.dataset_mask().astype("float32")
        cover = np.zeros(shape, "float32")
        reproject(valid, cover, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=from_origin(w, n, cell, cell), dst_crs=src.crs, resampling=Resampling.average)
        crs = src.crs
    mask = cover >= 254.0
    steps = int(np.ceil(margin_m / cell))
    if steps:
        mask = ndimage.binary_erosion(mask, iterations=steps)
    return FieldGrid(str(crs), float(w), float(n), float(cell), mask)


def read_on_grid(path, grid: FieldGrid, bands, factor: int = 1, alpha_band: int | None = None) -> np.ndarray:
    """Average a raster's bands onto the grid (or a grid factor times finer). Cells with no
    data come back as NaN. Returns (len(bands), ny * factor, nx * factor)."""
    ny, nx = grid.shape
    out = np.full((len(bands), ny * factor, nx * factor), np.nan, "float32")
    with rasterio.open(path) as src:
        kwargs = {}
        if alpha_band and src.count >= alpha_band:
            kwargs["src_alpha"] = alpha_band
        for i, b in enumerate(bands):
            reproject(rasterio.band(src, b), out[i], dst_transform=grid.fine_transform(factor), dst_crs=grid.crs,
                      resampling=Resampling.average, dst_nodata=np.nan, src_nodata=src.nodata, **kwargs)
    return out


def block_stat(fine: np.ndarray, factor: int, stat: str = "mean", min_valid: float = 0.5) -> np.ndarray:
    """Collapse each factor x factor block of a fine array to one cell, ignoring NaN.
    Blocks with less than min_valid of their pixels present become NaN."""
    ny, nx = fine.shape[0] // factor, fine.shape[1] // factor
    blocks = fine[:ny * factor, :nx * factor].reshape(ny, factor, nx, factor).transpose(0, 2, 1, 3).reshape(ny, nx, -1)
    valid = np.isfinite(blocks)
    count = valid.sum(-1)
    total = np.where(valid, blocks, 0).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        if stat == "mean":
            out = mean
        elif stat == "sd":
            out = np.sqrt(np.where(valid, (blocks - mean[..., None]) ** 2, 0).sum(-1) / count)
        else:
            raise ValueError(f"unknown stat {stat!r}")
    out[count < min_valid * factor * factor] = np.nan
    return out


def smooth(a: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaN instead of spreading it."""
    if sigma_cells <= 0:
        return a
    valid = np.isfinite(a)
    num = ndimage.gaussian_filter(np.where(valid, a, 0).astype("float64"), sigma_cells, mode="nearest")
    den = ndimage.gaussian_filter(valid.astype("float64"), sigma_cells, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~valid] = np.nan
    return out
