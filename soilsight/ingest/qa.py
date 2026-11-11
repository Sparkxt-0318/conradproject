"""Was the flight good enough to stitch? Judged from photo positions and the camera profile.

This runs in seconds, before any stitching, so a bad flight can be re-flown the same day.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soilsight.geo import metres_per_degree
from soilsight.ingest.sensors import SensorProfile
from soilsight.report.checks import FAIL, INFO, OK, WARN, Check, graded

def local_xy(shots) -> np.ndarray:
    """Photo positions as metres east and north of their centre (fine for a single field)."""
    lat = np.array([s.lat for s in shots])
    lon = np.array([s.lon for s in shots])
    m_lon, m_lat = metres_per_degree(lat.mean())
    x = (lon - lon.mean()) * m_lon
    y = (lat - lat.mean()) * m_lat
    return np.c_[x, y]


@dataclass(frozen=True)
class FlightGeometry:
    n_shots: int
    n_lines: int
    spacing_m: float            # between photos along a flight line
    line_spacing_m: float       # between neighbouring flight lines (NaN with a single line)
    speed_ms: float             # NaN when photos carry no usable time
    height_m: float             # above ground (NaN when unknown)
    height_sd_m: float


def flight_geometry(shots, height_m: float | None = None) -> FlightGeometry:
    """Flight lines, spacing and speed from the order and positions of the photos.

    height_m is the planned height above ground, used when the photos do not record it.
    """
    timed = all(s.time is not None for s in shots)
    order = sorted(range(len(shots)), key=(lambda i: (shots[i].time, shots[i].file)) if timed
                   else (lambda i: shots[i].file))
    shots = [shots[i] for i in order]
    xy = local_xy(shots)
    step = np.diff(xy, axis=0)
    length = np.hypot(*step.T)
    moving = length > 0.05

    # Flight axis: the direction most steps share, ignoring which way along it they go.
    ang = np.arctan2(step[:, 1], step[:, 0])
    axis = 0.5 * np.arctan2((length * np.sin(2 * ang))[moving].sum(), (length * np.cos(2 * ang))[moving].sum())
    off_axis = np.abs((ang - axis + np.pi / 2) % np.pi - np.pi / 2)
    on_line = moving & (off_axis < np.radians(20))
    spacing = float(np.median(length[on_line])) if on_line.any() else float("nan")

    # A new flight line starts at every turn or reversal.
    along = np.cos(ang - axis)
    line_of = np.zeros(len(shots), int)
    for i in range(1, len(shots)):
        prev_ok = on_line[i - 1]
        same_way = i < 2 or not on_line[i - 2] or along[i - 1] * along[i - 2] > 0
        line_of[i] = line_of[i - 1] + (0 if prev_ok and same_way else 1)
    across = xy @ np.array([-np.sin(axis), np.cos(axis)])
    centres = sorted(float(np.median(across[line_of == k])) for k in np.unique(line_of) if (line_of == k).sum() >= 3)
    gaps = np.diff(centres)
    gaps = gaps[gaps > 0.25 * spacing] if len(gaps) and spacing == spacing else gaps
    line_spacing = float(np.median(gaps)) if len(gaps) else float("nan")

    # Speed: within each flight line, distance over time between every pair of photos, then the
    # middle value of them all. Pairs far apart smooth out whole-second time stamps; taking the
    # middle value shrugs off a hover before the first line, a corner, or a skipped photo.
    speed = float("nan")
    if timed:
        t = np.array([(s.time - shots[0].time).total_seconds() for s in shots])
        s_along = xy @ np.array([np.cos(axis), np.sin(axis)])
        rates = []
        for k in np.unique(line_of):
            i = np.flatnonzero(line_of == k)
            if len(i) >= 3:
                a, b = np.triu_indices(len(i), 1)
                dt = t[i[b]] - t[i[a]]
                ok = dt > 0
                rates.append(np.abs(s_along[i[b]] - s_along[i[a]])[ok] / dt[ok])
        if rates and sum(len(r) for r in rates):
            speed = float(np.median(np.concatenate(rates)))

    agl = [s.agl_m for s in shots if s.agl_m is not None]
    if agl:
        height, height_sd = float(np.median(agl)), float(np.std(agl))
    else:
        alt = [s.alt_m for s in shots if s.alt_m is not None]
        height = float(height_m) if height_m else float("nan")
        height_sd = float(np.std(alt)) if alt else float("nan")
    return FlightGeometry(len(shots), max(len(centres), 1), spacing, line_spacing, speed, height, height_sd)


def flight_checks(shots, missing, profile: SensorProfile | None, height_m: float | None = None) -> list:
    """The flight's quality checks. missing: photos with no position at all."""
    total = len(shots) + len(missing)
    checks = [Check("flight", "Photos with a position", FAIL if missing else OK, f"{len(shots)} of {total}",
                    f"no position for: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}" if missing else "")]
    if len(shots) < 3:
        return checks + [Check("flight", "Photo count", FAIL, str(len(shots)), "too few photos to stitch")]
    g = flight_geometry(shots, height_m)
    checks.append(Check("flight", "Photo count", OK if g.n_shots >= 20 else WARN, str(g.n_shots),
                        "" if g.n_shots >= 20 else "small sets often fail to stitch"))
    checks.append(Check("flight", "Flight lines", INFO, str(g.n_lines)))

    known = profile is not None and g.height_m == g.height_m
    if not known:
        why = "the camera is not described" if profile is None else "flying height unknown (set flight_height_m in field.toml)"
        checks.append(Check("flight", "Overlap, blur and smear", WARN, "not checked", why))
        return checks

    gsd = profile.gsd_m(g.height_m)
    long_side, short_side = profile.footprint_m(g.height_m)
    checks.append(Check("flight", "Ground pixel size", INFO, f"{gsd * 100:.1f} cm",
                        f"at {g.height_m:.0f} m above ground; one photo covers {long_side:.0f} x {short_side:.0f} m"))
    if g.height_sd_m == g.height_sd_m:
        rel = g.height_sd_m / g.height_m
        checks.append(Check("flight", "Height steadiness", graded(rel, 0.03, 0.08, False), f"±{g.height_sd_m:.1f} m",
                            "height changes alter the pixel size from photo to photo" if rel > 0.03 else ""))

    forward = 1 - g.spacing_m / short_side
    checks.append(Check("flight", "Forward overlap", graded(forward, 0.75, 0.60), f"{forward:.0%}",
                        f"photos {g.spacing_m:.1f} m apart; assumes the photo's short side points along the "
                        "flight line; target 80% on bare soil"))
    if g.line_spacing_m == g.line_spacing_m:
        side = 1 - g.line_spacing_m / long_side
        checks.append(Check("flight", "Side overlap", graded(side, 0.60, 0.45), f"{side:.0%}",
                            f"flight lines {g.line_spacing_m:.1f} m apart; target 70%"))
    else:
        checks.append(Check("flight", "Side overlap", WARN, "not checked", "only one flight line found"))

    if g.speed_ms == g.speed_ms:
        checks.append(Check("flight", "Ground speed", INFO, f"{g.speed_ms:.1f} m/s"))
        exposures = [s.exposure_s for s in shots if s.exposure_s]
        if exposures:
            blur = profile.blur_px(g.speed_ms, g.height_m, float(np.median(exposures)))
            checks.append(Check("flight", "Motion blur", graded(blur, 1.0, 2.0, False), f"{blur:.1f} px",
                                f"at 1/{1 / np.median(exposures):.0f} s exposure"))
        smear = profile.smear_px(g.speed_ms, g.height_m)
        if profile.shutter == "global":
            checks.append(Check("flight", "Rolling-shutter smear", OK, "none", "global shutter"))
        elif smear != smear:
            checks.append(Check("flight", "Rolling-shutter smear", WARN, "unknown",
                                "the camera's readout time is not in its profile"))
        else:
            checks.append(Check("flight", "Rolling-shutter smear", OK if smear <= 1 else WARN, f"{smear:.0f} px",
                                "" if smear <= 1 else f"stitching will correct for it using a {profile.readout_ms:.0f} ms "
                                "readout; a wrong readout value bends the map"))
    else:
        checks.append(Check("flight", "Speed, blur and smear", WARN, "not checked", "photos carry no usable time"))
    return checks
