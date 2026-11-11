"""Scout processing: from a folder of drone photos to map layers and a quality report.

    flight checks  ->  stitching (OpenDroneMap)  ->  layers  ->  report

Each step's checks go into one report. A failed check stops the pipeline before the next
slow step, and the report says why.
"""
from __future__ import annotations

from dataclasses import dataclass

from soilsight.field import FieldDir
from soilsight.ingest.sensors import SensorProfile
from soilsight.ingest import collect_shots, flight_checks, flight_geometry, get_profile, profile_from_exif, write_geo_txt
from soilsight.layers import build_layers, save_layers
from soilsight.photogrammetry.odm import measured_gsd_m, odm_options, stitch, stitching_checks
from soilsight.report.checks import FAIL, INFO, WARN, Check
from soilsight.report.quality import write_quality


@dataclass(frozen=True)
class ScoutResult:
    overall: str          # ok, warn or fail
    checks: list
    stack: object         # the LayerStack, or None if the pipeline stopped early
    stopped: str


def _profile(cfg, shots):
    if cfg.camera:
        return SensorProfile(**({"name": "custom"} | cfg.camera))
    if cfg.sensor:
        return get_profile(cfg.sensor)
    first = next((s for s in shots if s.width_px), None)
    return profile_from_exif(first.width_px, first.height_px, first.focal_35mm, first.model) if first else None


def check_flight(fd: FieldDir) -> tuple:
    """Read the photo folder and judge the flight. Returns (shots, profile, checks)."""
    cfg = fd.config()
    shots, missing = collect_shots(fd.photos, fd.sidecar)
    profile = _profile(cfg, shots)
    checks = flight_checks(shots, missing, profile, cfg.flight_height_m)
    if profile is not None:
        checks.insert(0, Check("flight", "Camera", INFO, profile.name,
                               profile.description + ("" if cfg.sensor or cfg.camera else
                                                      "; set sensor in field.toml for exact geometry")))
    else:
        checks.insert(0, Check("flight", "Camera", INFO, "unknown",
                               "the photos do not describe the camera; add a [camera] table to field.toml"))
    return shots, profile, checks


def run(fd: FieldDir, node_url: str = "http://localhost:3000", skip_stitch: bool = False, fast: bool = False,
        force: bool = False, log=print, node=None) -> ScoutResult:
    """Run the whole pipeline on a field folder and write its report.

    skip_stitch  use the orthophoto and surface model already in odm/ (for example exported
                 from WebODM) instead of stitching
    force        carry on past failed flight checks
    """
    checks, stack, stopped = [], None, ""

    def finish():
        doc = write_quality(fd, checks, stack, stopped)
        log(f"report: {fd.quality_html}")
        return ScoutResult(doc["overall"], checks, stack, stopped)

    shots, profile = [], None
    if fd.photos.is_dir():
        log("reading photo positions")
        shots, profile, flight = check_flight(fd)
        checks += flight
        if any(c.status == FAIL for c in flight) and not (force or skip_stitch):
            stopped = "The flight checks failed, so the photos were not stitched. Use --force to stitch anyway."
            return finish()
    elif not skip_stitch:
        raise FileNotFoundError(f"no photos folder at {fd.photos}")
    else:
        checks.append(Check("flight", "Photos", INFO, "not supplied", "flight checks skipped"))

    if not skip_stitch:
        fd.work.mkdir(parents=True, exist_ok=True)
        if fd.sidecar.exists():
            write_geo_txt(shots, fd.geo_txt)       # positions from the capture computer
        gsd_cm = None
        if profile is not None and len(shots) >= 3:
            height = flight_geometry(shots, fd.config().flight_height_m).height_m
            gsd_cm = profile.gsd_m(height) * 100 if height == height else None
        info = stitch(fd, [fd.photos / s.file for s in shots], odm_options(profile, gsd_cm, fast), node_url, log, node)
        checks.append(Check("stitching", "Stitching time", INFO, f"{info['processing_time_s'] / 60:.0f} min"))

    log("checking the stitched outputs")
    stitched = stitching_checks(fd, len(shots) or None)
    checks += stitched
    if any(c.status == FAIL for c in stitched):
        stopped = "The stitched outputs are missing or unusable, so no layers were made."
        return finish()

    # Photos that do not record their height leave overlap unchecked. The stitched pixel size
    # gives the height after the fact, so the flight can still be judged.
    gsd = measured_gsd_m(fd)
    unchecked = [c for c in checks if c.name == "Overlap, blur and smear" and "height" in c.detail]
    if unchecked and gsd and profile is not None:
        height = gsd / profile.gsd_m(1.0)
        redo = [c for c in flight_checks(shots, [], profile, height)
                if c.name not in ("Photos with a position", "Photo count", "Flight lines")]
        # Judged after the fact, a failed check is a warning: the photos did stitch.
        redo = [Check(c.stage, c.name, WARN, c.value, c.detail + "; below the limit, but the photos did stitch")
                if c.status == FAIL else c for c in redo]
        at = checks.index(unchecked[0])
        checks[at:at + 1] = [Check("flight", "Flying height", INFO, f"{height:.0f} m",
                                   "worked out from the stitched pixel size, because the photos do not record it")] + redo

    log("building layers")
    stack, layer_checks = build_layers(fd)
    checks += layer_checks
    save_layers(stack, fd)
    log(f"layers: {fd.layers_tif}")
    return finish()
