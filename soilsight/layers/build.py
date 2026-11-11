"""Building, saving and loading a field's map layers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import rasterio

from soilsight.field import FieldDir
from soilsight.layers.annotations import Annotations, load_annotations
from soilsight.layers.colour import COLOUR_LAYERS, colour_layers, grey_card_gains
from soilsight.layers.distance import DISTANCE_LAYERS, distance_to_lines, distance_to_point
from soilsight.layers.grid import FieldGrid, grid_from_boundary, grid_from_raster, smooth
from soilsight.layers.terrain import TERRAIN_LAYERS, terrain_layers
from soilsight.report.checks import FAIL, INFO, OK, WARN, Check

DESCRIPTIONS = COLOUR_LAYERS | TERRAIN_LAYERS | DISTANCE_LAYERS
GROUP = ({k: "colour" for k in COLOUR_LAYERS} | {k: "terrain" for k in TERRAIN_LAYERS}
         | {k: "distance" for k in DISTANCE_LAYERS})


@dataclass(frozen=True)
class LayerStack:
    grid: FieldGrid
    names: tuple
    data: np.ndarray                 # (layers, ny, nx), NaN outside the field or where there is no data
    quality: dict = field(default_factory=dict)   # layer name -> {"status": ..., "note": ...}

    def layer(self, name: str) -> np.ndarray:
        return self.data[self.names.index(name)]

    def brain_inputs(self, names=None) -> tuple:
        """What the Brain needs: (xy, feats, cells).

        xy     (N, 2) cell centres in metres from the grid's lower-left corner
        feats  (N, p) the chosen layers, each scaled to zero mean and unit spread
        cells  (N, 2) row and column of each of those cells in the grid

        Only cells inside the field with every chosen layer present are returned.
        """
        names = tuple(names) if names is not None else self.names
        stack = np.stack([self.layer(n) for n in names]) if names else np.empty((0,) + self.grid.shape)
        keep = self.grid.mask & np.isfinite(stack).all(0)
        x, y = self.grid.centres()
        x0, y0 = self.grid.origin
        feats = stack[:, keep].T.astype("float64")
        sd = feats.std(0)
        feats = (feats - feats.mean(0)) / np.where(sd > 0, sd, 1)
        return np.c_[x[keep] - x0, y[keep] - y0], feats, np.argwhere(keep)


def build_layers(fd: FieldDir) -> tuple:
    """Make every layer the field's inputs allow. Returns (LayerStack, checks)."""
    cfg = fd.config()
    with rasterio.open(fd.orthophoto) as src:
        crs = src.crs
    ann = load_annotations(fd.annotations, crs)
    checks = [Check("layers", "Annotations", WARN, "problem", p) for p in ann.problems]

    if ann.boundary is not None:
        grid = grid_from_boundary(ann.boundary, crs, cfg.grid_m)
        checks.append(Check("layers", "Field boundary", OK, "drawn"))
    else:
        grid = grid_from_raster(fd.orthophoto, cfg.grid_m)
        checks.append(Check("layers", "Field boundary", WARN, "not drawn",
                            "using everything the orthophoto covers; draw a boundary to keep roads and trees out"))
    if not grid.mask.any():
        raise ValueError("the field grid is empty: check that the boundary lies on the orthophoto")
    area_ha = grid.mask.sum() * grid.cell ** 2 / 1e4
    checks.append(Check("layers", "Field area", INFO, f"{area_ha:.2f} ha", f"{int(grid.mask.sum())} cells of {grid.cell:g} m"))

    sigma = cfg.smooth_m / 2 / grid.cell
    layers, quality = {}, {}

    # colour
    gains = None
    if ann.grey_card is not None:
        gains = grey_card_gains(fd.orthophoto, ann.grey_card, ann.grey_reflectance)
        note = f"scaled to a grey card of reflectance {ann.grey_reflectance:g}"
        checks.append(Check("layers", "Colour calibration", OK, "grey card",
                            "channel gains " + ", ".join(f"{g:.2f}" for g in gains)))
    else:
        note = "no grey card: values are not comparable between flights or fields"
        checks.append(Check("layers", "Colour calibration", WARN, "none", note))
    for name, a in colour_layers(fd.orthophoto, grid, gains).items():
        layers[name] = smooth(np.where(grid.mask, a, np.nan), sigma)
        quality[name] = dict(status=OK if gains is not None else WARN, note=note)

    # terrain
    from soilsight.photogrammetry.odm import used_ground_control
    terrain, stats = terrain_layers(fd.dsm, grid, cfg.smooth_m)
    gcp = used_ground_control(fd)
    relief_cm, rough_cm = stats["relief_m"] * 100, stats["roughness_m"] * 100
    if gcp and stats["relief_m"] >= 0.10:
        t_status, t_note = OK, f"relief {relief_cm:.0f} cm, tied to ground control points"
    elif gcp:
        t_status, t_note = WARN, f"relief only {relief_cm:.0f} cm: close to what stitching can resolve"
    else:
        t_status = WARN
        t_note = (f"relief {relief_cm:.0f} cm with no ground control points: stitching alone can bow the "
                  "surface by this much, so hollows and rises may not be real")
    checks.append(Check("layers", "Terrain relief", t_status, f"{relief_cm:.0f} cm",
                        t_note + f"; height scatter inside one cell {rough_cm:.1f} cm"))
    for name, a in terrain.items():
        layers[name] = np.where(grid.mask, a, np.nan)
        quality[name] = dict(status=t_status, note=t_note)

    # distance
    x, y = grid.centres()
    wanted = (("dist_inlet", ann.inlet, "inlet"), ("dist_outlet", ann.outlet, "outlet"),
              ("dist_ditch", ann.ditches or None, "ditch"))
    absent = []
    for name, geom, label in wanted:
        if geom is None:
            absent.append(label)
            continue
        d = distance_to_lines(x, y, geom) if name == "dist_ditch" else distance_to_point(x, y, geom)
        layers[name] = np.where(grid.mask, d, np.nan)
        quality[name] = dict(status=OK, note="from the marked " + label)
    if absent:
        checks.append(Check("layers", "Distance layers", WARN, f"{3 - len(absent)} of 3",
                            f"not marked in annotations.geojson: {', '.join(absent)}"))
    else:
        checks.append(Check("layers", "Distance layers", OK, "3 of 3"))

    names = tuple(n for n in DESCRIPTIONS if n in layers)
    data = np.stack([layers[n] for n in names]).astype("float32")
    cover = float(np.isfinite(data[:len(COLOUR_LAYERS)]).all(0)[grid.mask].mean())
    checks.append(Check("layers", "Field covered by the orthophoto", OK if cover >= 0.99 else WARN if cover >= 0.9 else FAIL,
                        f"{cover:.0%}", "" if cover >= 0.99 else "part of the field has no imagery"))
    checks.append(Check("layers", "Layers made", INFO, f"{len(names)} of {len(DESCRIPTIONS)}", ", ".join(names)))
    return LayerStack(grid, names, data, quality), checks


def save_layers(stack: LayerStack, fd: FieldDir) -> None:
    """Write layers.tif (one band per layer) and layers.json (what each band means)."""
    fd.layers_dir.mkdir(parents=True, exist_ok=True)
    g = stack.grid
    ny, nx = g.shape
    with rasterio.open(fd.layers_tif, "w", driver="GTiff", height=ny, width=nx, count=len(stack.names) + 1,
                       dtype="float32", crs=g.crs, transform=g.transform, nodata=np.nan, compress="deflate") as dst:
        for i, name in enumerate(stack.names, 1):
            dst.write(stack.data[i - 1], i)
            dst.set_band_description(i, name)
        dst.write(g.mask.astype("float32"), len(stack.names) + 1)
        dst.set_band_description(len(stack.names) + 1, "in_field")
    x0, y0 = g.origin
    meta = dict(crs=g.crs, cell_m=g.cell, west=g.west, north=g.north, origin=[x0, y0], shape=[ny, nx],
                cells_in_field=int(g.mask.sum()),
                layers=[dict(name=n, group=GROUP[n], description=DESCRIPTIONS[n], **stack.quality.get(n, {}))
                        for n in stack.names])
    fd.layers_json.write_text(json.dumps(meta, indent=1) + "\n")


def load_layers(fd: FieldDir) -> LayerStack:
    meta = json.loads(fd.layers_json.read_text())
    with rasterio.open(fd.layers_tif) as src:
        data = src.read().astype("float32")
    names = tuple(layer["name"] for layer in meta["layers"])
    grid = FieldGrid(meta["crs"], meta["west"], meta["north"], meta["cell_m"], data[-1] > 0.5)
    quality = {layer["name"]: dict(status=layer.get("status", OK), note=layer.get("note", "")) for layer in meta["layers"]}
    return LayerStack(grid, names, data[:-1], quality)
