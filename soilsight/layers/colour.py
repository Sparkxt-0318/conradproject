"""Soil colour layers from the orthophoto.

The photo's numbers are first turned back into light (undoing the sRGB curve) and, when a
grey card was on the ground, scaled so the card reads its known reflectance. Indices follow
the soil-colour literature (Escadafal 1989; Mathieu et al. 1998); excess green is from
Woebbecke et al. 1995 and marks straw, weeds and stubble rather than soil.
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask

from soilsight.layers.grid import FieldGrid, block_stat, read_on_grid

COLOUR_LAYERS = {
    "brightness": "overall soil brightness, sqrt((R² + G² + B²) / 3); dark soil is wetter or richer in organic matter",
    "redness": "log10 of the redness index R² / (B·G³); iron oxides",
    "colour_index": "(R − G) / (R + G); red against green",
    "saturation": "(R − B) / (R + B); how strongly coloured the soil is",
    "excess_green": "(2G − R − B) / (R + G + B); straw, weeds and stubble",
}


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    """8-bit sRGB values (0-255) to linear light (0-1)."""
    c = np.asarray(v, "float64") / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def grey_card_gains(orthophoto, card: dict, reflectance: float = 0.18) -> np.ndarray:
    """Per-channel scale that makes the grey card read its known reflectance."""
    with rasterio.open(orthophoto) as src:
        pixels, _ = rio_mask(src, [card], crop=True, indexes=[1, 2, 3], filled=False)
    lin = srgb_to_linear(pixels.astype("float64"))
    means = np.array([np.ma.masked_array(lin[i], pixels[i].mask).mean() for i in range(3)], float)
    if not np.all(np.isfinite(means)) or (means <= 0).any():
        raise ValueError("the grey card polygon covers no usable orthophoto pixels")
    return reflectance / means


def colour_indices(rgb: np.ndarray) -> dict:
    """The five indices from linear R, G, B arrays (any shape, stacked on axis 0)."""
    r, g, b = (np.maximum(c, 1e-4) for c in rgb)
    return {
        "brightness": np.sqrt((r ** 2 + g ** 2 + b ** 2) / 3),
        "redness": np.log10(r ** 2 / (b * g ** 3)),
        "colour_index": (r - g) / (r + g),
        "saturation": (r - b) / (r + b),
        "excess_green": (2 * g - r - b) / (r + g + b),
    }


def colour_layers(orthophoto, grid: FieldGrid, gains=None, factor: int = 5) -> dict:
    """Colour indices per grid cell. The orthophoto is averaged to cells factor times finer
    than the grid, converted to light, scaled by gains, then averaged to the grid."""
    with rasterio.open(orthophoto) as src:
        alpha = 4 if src.count >= 4 else None
    fine = srgb_to_linear(read_on_grid(orthophoto, grid, (1, 2, 3), factor, alpha_band=alpha))
    if gains is not None:
        fine = fine * np.asarray(gains, float)[:, None, None]
    rgb = np.stack([block_stat(fine[i], factor) for i in range(3)])
    out = colour_indices(rgb)
    missing = ~np.isfinite(rgb).all(0)
    for a in out.values():
        a[missing] = np.nan
    return out
