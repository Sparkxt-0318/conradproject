"""The field package: one JSON file holding everything the phone app needs for a visit.

The phone has no signal in the field, so nothing can be fetched later. The package carries
the field grid, a picture of the field, the zones, the test protocol, the first spots to
sample, and the model's candidate settings. With those the app's own copy of the Brain
(JavaScript) gives the same answers as the Python Brain, which the tests check.
"""
from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone

import numpy as np

from soilsight.brain import Brain, Protocol

FORMAT = "soilsight-field/1"
MAX_CELLS = 20000          # the phone redraws the whole map after every result
ZONE_SAMPLES = 300         # cells used to average over one zone
_Z, _W = np.polynomial.hermite_e.hermegauss(5)


def cells_xy(package: dict) -> np.ndarray:
    """(N, 2) cell centres in metres from the grid's lower-left corner."""
    g = package["grid"]
    rows, cols = np.asarray(g["rows"]), np.asarray(g["cols"])
    return np.c_[(cols + 0.5) * g["cell_m"], (g["ny"] - rows - 0.5) * g["cell_m"]]


def auto_zones(xy: np.ndarray, target_ha: float = 0.5, lo: int = 2, hi: int = 8, seed: int = 0) -> np.ndarray:
    """Split the field into compact zones of about target_ha each. Returns a zone number per
    cell; zones are numbered from the north-west."""
    cell_area = _cell_area(xy)
    k = int(np.clip(round(len(xy) * cell_area / 1e4 / target_ha), lo, hi))
    rng = np.random.default_rng(seed)
    centres = xy[rng.choice(len(xy), k, replace=False)].astype(float)
    for _ in range(30):
        nearest = ((xy[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (nearest == j).any():
                centres[j] = xy[nearest == j].mean(0)
    nearest = ((xy[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
    order = np.lexsort((centres[:, 0], -np.round(centres[:, 1] / 20)))      # north first, then west to east
    rank = np.empty(k, int)
    rank[order] = np.arange(k)
    return rank[nearest]


def _cell_area(xy: np.ndarray) -> float:
    xs = np.unique(xy[:, 0])
    return float(np.min(np.diff(xs)) ** 2) if len(xs) > 1 else 1.0


def _zone_names(n: int) -> list:
    return [chr(ord("A") + i) for i in range(n)]


def build_package(name: str, rows, cols, nx: int, ny: int, cell_m: float, *, protocol: Protocol = Protocol(),
                  zone_ids=None, fixed_cells=(), threshold: float | None = None, unit: str = "mg/kg",
                  to_lonlat=None, image_jpeg: bytes | None = None, seed: int = 0, lo: float = 0.2, hi: float = 0.8) -> dict:
    """Assemble a package from a grid of cells.

    rows, cols     where each field cell sits in an ny x nx raster (row 0 is the north edge)
    fixed_cells    cells that must be sampled first (inlet, outlet)
    to_lonlat      six numbers a..f with lon = a*x + b*y + c and lat = d*x + e*y + f, for
                   x, y in metres from the grid's lower-left corner; None if not georeferenced
    image_jpeg     a picture of the grid rectangle, north up
    """
    rows, cols = np.asarray(rows, int), np.asarray(cols, int)
    grid = dict(nx=int(nx), ny=int(ny), cell_m=float(cell_m), rows=rows.tolist(), cols=cols.tolist())
    xy = cells_xy(dict(grid=grid))
    brain = Brain(xy, fixed_xy=xy[list(fixed_cells)] if len(fixed_cells) else None, protocol=protocol, seed=seed)
    initial = [brain.nearest_cell(p) for p in brain.initial_spots()]
    zone_ids = auto_zones(xy) if zone_ids is None else np.asarray(zone_ids, int)
    samples = []
    for z in np.unique(zone_ids):
        cells = np.flatnonzero(zone_ids == z)
        if len(cells) > ZONE_SAMPLES:
            cells = cells[np.linspace(0, len(cells) - 1, ZONE_SAMPLES).astype(int)]
        samples.append(cells.tolist())
    cand = brain._gp.candidates()
    geo = None
    if to_lonlat is not None:
        a, b, c, d, e, f = (float(v) for v in to_lonlat)
        inv = np.linalg.inv(np.array([[a, b], [d, e]]))
        geo = dict(to_lonlat=[a, b, c, d, e, f],
                   to_xy=[inv[0, 0], inv[0, 1], -(inv[0, 0] * c + inv[0, 1] * f),
                          inv[1, 0], inv[1, 1], -(inv[1, 0] * c + inv[1, 1] * f)])
    return dict(
        format=FORMAT, name=name, made=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        grid=grid, geo=geo,
        image=None if image_jpeg is None else dict(jpeg=base64.b64encode(image_jpeg).decode()),
        zones=dict(ids=zone_ids.tolist(), names=_zone_names(len(samples)), samples=samples),
        protocol=dict(budget=protocol.budget, cores=protocol.cores, kit_sd=protocol.kit_sd,
                      init_frac=protocol.init_frac, min_sep_m=protocol.min_sep_m),
        threshold=threshold, unit=unit, classes=dict(lo=lo, hi=hi),
        model=dict(kernel=brain._gp.kernel, mean_sd=brain._gp.priors.mean_sd, top_k=brain._gp.top_k, acq_components=6,
                   candidates=dict(signal_sd=cand["signal_sd"], length_m=cand["length_m"], nugget_sd=cand["nugget_sd"]),
                   quadrature=dict(z=_Z.tolist(), w=(_W / _W.sum()).tolist())),
        plan=dict(initial=initial, cand=brain._cand.tolist(), eval=brain._eval.tolist(), fixed=[int(c) for c in fixed_cells]),
    )


def package_from_sim(fld, name: str = "Simulated field", protocol: Protocol = Protocol(), threshold: float | None = 1.0,
                     seed: int = 0, corner_lonlat=None) -> dict:
    """A package for a synthetic field (soilsight.sim), for tests and demonstrations.

    corner_lonlat places the field's south-west corner on the Earth, so the app's GPS
    guidance can be exercised; None leaves the package without a position.
    """
    g = fld.grid
    n = np.arange(g.n)
    rows, cols = g.ny - 1 - n // g.nx, n % g.nx          # the sim's cell 0 is the south-west corner
    to_lonlat = None
    if corner_lonlat is not None:
        from soilsight.geo import metres_per_degree
        m_lon, m_lat = metres_per_degree(corner_lonlat[1])
        to_lonlat = (1 / m_lon, 0.0, corner_lonlat[0], 0.0, 1 / m_lat, corner_lonlat[1])
    return build_package(name, rows, cols, g.nx, g.ny, g.cell, protocol=protocol, zone_ids=g.zone,
                         fixed_cells=(fld.i_in, fld.i_out), threshold=threshold, seed=seed, to_lonlat=to_lonlat)


def package_from_field(fd) -> dict:
    """A package for a scouted field folder: its grid, orthophoto, boundary and settings."""
    import rasterio
    from PIL import Image
    from rasterio.warp import transform

    from soilsight.layers import load_annotations, load_layers
    from soilsight.layers.grid import FieldGrid, read_on_grid

    cfg = fd.config()
    stack = load_layers(fd)
    g = stack.grid
    factor = 1
    while g.mask[::factor, ::factor].sum() > MAX_CELLS:        # a coarser grid for big fields
        factor += 1
    mask = g.mask[::factor, ::factor]
    cell = g.cell * factor
    ny, nx = mask.shape
    coarse = FieldGrid(g.crs, g.west, g.north, cell, mask)
    rows, cols = np.nonzero(mask)

    # lon/lat of the grid: fit a plane through its corners and centre (exact enough over one field)
    x0, y0 = g.west, g.north - ny * cell
    pts = np.array([(0, 0), (nx * cell, 0), (0, ny * cell), (nx * cell, ny * cell), (nx * cell / 2, ny * cell / 2)], float)
    lon, lat = transform(g.crs, "EPSG:4326", (pts[:, 0] + x0).tolist(), (pts[:, 1] + y0).tolist())
    design = np.c_[pts, np.ones(len(pts))]
    (a, b, c), (d, e, f) = np.linalg.lstsq(design, np.array(lon), rcond=None)[0], np.linalg.lstsq(design, np.array(lat), rcond=None)[0]

    with rasterio.open(fd.orthophoto) as src:
        alpha = 4 if src.count >= 4 else None
    px = max(1, min(8, int(np.ceil(900 / max(nx, ny)))))
    rgb = read_on_grid(fd.orthophoto, coarse, (1, 2, 3), px, alpha_band=alpha)
    picture = np.where(np.isfinite(rgb), rgb, 235).clip(0, 255).astype("uint8").transpose(1, 2, 0)
    buf = io.BytesIO()
    Image.fromarray(picture).save(buf, "JPEG", quality=80)

    ann = load_annotations(fd.annotations, g.crs)
    xy = np.c_[(cols + 0.5) * cell, (ny - rows - 0.5) * cell]
    fixed = [int(((xy - (np.array(p) - (x0, y0))) ** 2).sum(1).argmin()) for p in (ann.inlet, ann.outlet) if p is not None]
    return build_package(cfg.name or fd.root.name, rows, cols, nx, ny, cell,
                         protocol=Protocol(budget=cfg.budget, cores=cfg.cores, kit_sd=cfg.kit_sd),
                         fixed_cells=fixed, threshold=cfg.threshold, unit=cfg.unit, to_lonlat=(a, b, c, d, e, f),
                         image_jpeg=buf.getvalue())


def save_package(package: dict, path) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(package, separators=(",", ":")))


def load_package(path) -> dict:
    from pathlib import Path
    package = json.loads(Path(path).read_text())
    if package.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file")
    return package
