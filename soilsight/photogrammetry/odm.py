"""Stitching with OpenDroneMap through a NodeODM server.

Start a server on the processing laptop with:

    docker run -d -p 3000:3000 opendronemap/nodeodm

Nothing here re-implements photogrammetry. This module chooses OpenDroneMap's options from
the camera profile, sends the photos and their positions, and keeps the three outputs the
rest of the pipeline needs.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from soilsight.field import FieldDir
from soilsight.ingest.sensors import SensorProfile
from soilsight.report.checks import FAIL, INFO, OK, WARN, Check, graded

# where NodeODM puts each output inside a finished task, and where it is kept in the field folder
ASSETS = {
    "odm_orthophoto/odm_orthophoto.tif": "odm_orthophoto.tif",
    "odm_dem/dsm.tif": "dsm.tif",
    "odm_report/stats.json": "stats.json",
}


def odm_options(profile: SensorProfile | None, gsd_cm: float | None = None, fast: bool = False) -> dict:
    """OpenDroneMap options for this camera.

    gsd_cm  ground pixel size of the photos, if known: the orthophoto is not made finer than it
    fast    lower quality, for a quick check that a photo set stitches at all
    """
    resolution = max(gsd_cm or 2.0, 2.0)
    opts = {
        "dsm": True,                                   # the surface model the terrain layers need
        "orthophoto-resolution": 5.0 if fast else resolution,
        "dem-resolution": 10.0 if fast else max(2 * resolution, 5.0),
        "feature-quality": "medium" if fast else "high",
        "pc-quality": "low" if fast else "medium",
        "skip-3dmodel": True,                          # no textured mesh: the map layers do not use it
    }
    if profile is not None and profile.shutter == "rolling" and profile.readout_ms:
        opts["rolling-shutter"] = True
        opts["rolling-shutter-readout"] = float(profile.readout_ms)
    return opts


def stitch(field: FieldDir, photos, options: dict, node_url: str = "http://localhost:3000",
           log=None, node=None) -> dict:
    """Send the photos and geo.txt to NodeODM, wait, and keep the outputs under field.odm.

    Returns the task's final info as a dict. node lets a test pass in a stand-in server.
    """
    if node is None:
        from pyodm import Node
        node = Node.from_url(node_url)
    files = [str(p) for p in photos]
    for extra in (field.geo_txt, field.gcp_list):
        if extra.exists():
            files.append(str(extra))
    log = log or (lambda msg: None)
    log(f"sending {len(photos)} photos to the stitcher with options {options}")
    task = node.create_task(files, options)
    last = [-10.0]

    def progress(info):
        pct = float(getattr(info, "progress", 0) or 0)
        if pct - last[0] >= 10:
            last[0] = pct
            log(f"  stitching {pct:.0f}%")

    task.wait_for_completion(status_callback=progress)
    with tempfile.TemporaryDirectory() as tmp:
        task.download_assets(tmp)
        present = {src: dst for src, dst in ASSETS.items() if (Path(tmp) / src).exists()}
        essential = [src for src in ASSETS if src not in present and not src.endswith("stats.json")]
        if essential:
            raise RuntimeError(f"stitching finished without {', '.join(essential)}")
        # Replace the outputs as a set, so files from an earlier run never mix with new ones.
        field.odm.mkdir(parents=True, exist_ok=True)
        for dst in ASSETS.values():
            (field.odm / dst).unlink(missing_ok=True)
        for src, dst in present.items():
            shutil.copy2(Path(tmp) / src, field.odm / dst)
    info = task.info()
    return {"uuid": getattr(info, "uuid", ""), "processing_time_s": (getattr(info, "processing_time", 0) or 0) / 1000,
            "images": getattr(info, "images_count", len(photos)), "options": options}


def _get(d, *path, default=None):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def stitching_checks(field: FieldDir, n_photos: int | None = None) -> list:
    """Quality checks on the stitched outputs, from the rasters and OpenDroneMap's stats.json."""
    import rasterio

    checks = []
    for label, path in (("Orthophoto", field.orthophoto), ("Surface model", field.dsm)):
        if not path.exists():
            checks.append(Check("stitching", label, FAIL, "missing", f"expected {path}"))
    if any(c.status == FAIL for c in checks):
        return checks

    with rasterio.open(field.orthophoto) as src:
        res = abs(src.transform.a)
        projected = src.crs is not None and src.crs.is_projected
        checks.append(Check("stitching", "Orthophoto pixel size", INFO, f"{res * 100:.1f} cm",
                            f"{src.width} x {src.height} px"))
        checks.append(Check("stitching", "Map projection", OK if projected else FAIL,
                            str(src.crs.to_string() if src.crs else "none"),
                            "" if projected else "the orthophoto must be in a metric projection such as UTM"))
    with rasterio.open(field.dsm) as src:
        checks.append(Check("stitching", "Surface model pixel size", INFO, f"{abs(src.transform.a) * 100:.1f} cm"))

    if not field.odm_stats.exists():
        checks.append(Check("stitching", "Stitching statistics", WARN, "not available",
                            "no stats.json, so image count and alignment error were not checked"))
        return checks
    stats = json.loads(field.odm_stats.read_text())
    used = _get(stats, "reconstruction_statistics", "reconstructed_shots_count")
    total = _get(stats, "reconstruction_statistics", "initial_shots_count") or n_photos
    if used is not None and total:
        share = used / total
        checks.append(Check("stitching", "Photos used", graded(share, 0.95, 0.80), f"{used} of {total}",
                            "" if share >= 0.95 else "photos left out leave holes or weak spots in the map"))
    err = _get(stats, "reconstruction_statistics", "reprojection_error_pixels")
    if err is not None:
        checks.append(Check("stitching", "Alignment error", graded(float(err), 1.0, 2.0, False), f"{float(err):.2f} px",
                            "how far matched points land from where the camera model puts them"))
    gps = _get(stats, "gps_errors", "average_error")
    if gps is not None:
        checks.append(Check("stitching", "Position error", INFO, f"{float(gps):.2f} m",
                            "difference between photo positions and the stitched solution; not map accuracy"))
    gcp = _get(stats, "gcp_errors", "average_error")
    has_gcp = used_ground_control(field)
    checks.append(Check("stitching", "Ground control points", OK if has_gcp else WARN,
                        (f"used, error {float(gcp):.2f} m" if gcp is not None else "used") if has_gcp else "none",
                        "" if has_gcp else "without them the surface model can bow by decimetres across the field"))
    return checks


def used_ground_control(field: FieldDir) -> bool:
    """Whether the stitched outputs were tied to surveyed ground control points."""
    if not field.odm_stats.exists():
        return field.gcp_list.exists()
    stats = json.loads(field.odm_stats.read_text())
    return bool(_get(stats, "reconstruction_statistics", "has_gcp")) or bool(_get(stats, "gcp_errors", "average_error"))
