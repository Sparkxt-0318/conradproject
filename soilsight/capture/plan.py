"""Flight settings for a camera, and a survey mission over a field.

Everything follows from the camera profile: how much ground one photo covers at a height,
how far apart photos and flight lines must be for the wanted overlap, and how fast the drone
may fly before the picture smears.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soilsight.geo import metres_per_degree
from soilsight.ingest.sensors import SensorProfile

CMD_WAYPOINT, CMD_RTL, CMD_TAKEOFF, CMD_CHANGE_SPEED, CMD_CAM_TRIGG_DIST = 16, 20, 22, 178, 206
FRAME_GLOBAL, FRAME_RELATIVE_ALT = 0, 3


@dataclass(frozen=True)
class FlightPlan:
    height_m: float
    gsd_m: float               # ground size of one pixel
    footprint_m: tuple         # one photo on the ground: (across the flight line, along it)
    photo_spacing_m: float     # along a flight line
    line_spacing_m: float      # between flight lines
    speed_ms: float
    interval_s: float          # time between photos at that speed
    blur_px: float             # motion blur during one exposure
    smear_px: float            # rolling-shutter smear (0 for a global shutter)
    max_speed_ms: float        # fastest speed that keeps blur, smear and the photo rate within limits
    limited_by: str            # which of those sets max_speed_ms

    def area_ha_per_min(self) -> float:
        return self.speed_ms * self.line_spacing_m * 60 / 1e4


def flight_plan(profile: SensorProfile, height_m: float, forward: float = 0.8, side: float = 0.7,
                speed_ms: float | None = None, exposure_s: float = 1 / 1000, min_interval_s: float = 1.0,
                max_px: float = 1.0, margin: float = 0.03, top_speed_ms: float = 5.0) -> FlightPlan:
    """Settings for this camera at this height. The camera's long side lies across the flight line.

    forward, side    overlap between consecutive photos and between flight lines
    speed_ms         ground speed; by default the fastest the camera allows, up to top_speed_ms
    min_interval_s   the shortest time the camera needs between photos (an assumption until
                     measured on the real camera)
    max_px           the most blur or smear to accept, in pixels
    margin           photos are planned this much closer together than the overlap strictly
                     needs, because a trigger always fires a little after its planned point
    """
    if not (0 < forward < 1 and 0 < side < 1):
        raise ValueError("overlaps must be between 0 and 1")
    gsd = profile.gsd_m(height_m)
    across, along = profile.footprint_m(height_m)
    spacing, lines = along * (1 - forward) * (1 - margin), across * (1 - side)
    limits = {"motion blur": max_px * gsd / exposure_s, "photo rate": spacing / min_interval_s}
    if profile.shutter == "rolling" and profile.readout_ms:
        limits["rolling-shutter smear"] = max_px * gsd / (profile.readout_ms / 1000.0)
    limited_by = min(limits, key=limits.get)
    if speed_ms is None:
        speed_ms = min(top_speed_ms, limits[limited_by])
    smear = profile.smear_px(speed_ms, height_m)
    return FlightPlan(height_m, gsd, (across, along), spacing, lines, speed_ms, spacing / speed_ms,
                      profile.blur_px(speed_ms, height_m, exposure_s), smear, limits[limited_by], limited_by)


# ---- survey lines over a field

def _local(lonlat: np.ndarray) -> tuple:
    origin = lonlat.mean(0)
    scale = np.array(metres_per_degree(origin[1]))
    return (lonlat - origin) * scale, origin, scale


def survey_lines(polygon_xy, line_spacing_m: float, overshoot_m: float = 0.0, angle_deg: float | None = None) -> list:
    """Back-and-forth flight lines covering a polygon (metres). Returns [(start, end), ...] in
    flying order.

    Lines run along the polygon side that needs the fewest lines, unless angle_deg (measured
    anticlockwise from east) is given. overshoot_m extends each line past the field so the
    drone is at speed and level before the first photo.
    """
    poly = np.asarray(polygon_xy, float)
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    if angle_deg is None:
        best = None
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            d = b - a
            if np.hypot(*d) < 1e-6:
                continue
            ang = np.arctan2(d[1], d[0])
            extent = np.ptp(poly @ np.array([-np.sin(ang), np.cos(ang)]))
            if best is None or extent < best[0] - 1e-9:
                best = (extent, ang)
        ang = best[1]
    else:
        ang = np.radians(angle_deg)
    rot = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])   # lines become horizontal
    p = poly @ rot.T
    lo, hi = p[:, 1].min(), p[:, 1].max()
    # A field a hair wider than n line spacings still gets n lines: with the usual side overlap
    # the outer photos reach well past the boundary.
    n = max(int(np.ceil((hi - lo) / line_spacing_m - 0.1)), 1)
    ys = (lo + hi) / 2 + (np.arange(n) - (n - 1) / 2) * line_spacing_m
    lines = []
    for k, y in enumerate(ys):
        xs = []
        for a, b in zip(p, np.roll(p, -1, axis=0)):
            if (a[1] - y) * (b[1] - y) <= 0 and a[1] != b[1]:
                xs.append(a[0] + (y - a[1]) / (b[1] - a[1]) * (b[0] - a[0]))
        if len(xs) < 2:
            continue
        x0, x1 = min(xs) - overshoot_m, max(xs) + overshoot_m
        start, end = np.array([x0, y]) @ rot, np.array([x1, y]) @ rot
        lines.append((start, end) if k % 2 == 0 else (end, start))
    return lines


def boundary_lonlat(geojson_path) -> np.ndarray:
    """The field boundary's outer ring from an annotations.geojson, as (n, 2) lon/lat."""
    for feat in json.loads(Path(geojson_path).read_text()).get("features", []):
        if (feat.get("properties") or {}).get("role") == "boundary":
            geom = feat["geometry"]
            ring = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0]
            return np.asarray(ring, float)[:, :2]
    raise ValueError(f"{geojson_path} has no feature with role 'boundary'")


def survey_mission(boundary, plan: FlightPlan, autopilot_trigger: bool = False, angle_deg: float | None = None) -> list:
    """Mission items for a survey of the field: take off, fly the lines, return.

    boundary is (n, 2) lon/lat. Each item is a dict with command, frame, params (4 numbers),
    lat, lon and alt. With autopilot_trigger the flight controller is told to trigger the camera
    every photo_spacing_m; otherwise the capture computer spaces the photos itself.
    """
    xy, origin, scale = _local(np.asarray(boundary, float))
    lines = survey_lines(xy, plan.line_spacing_m, overshoot_m=plan.footprint_m[1] / 2, angle_deg=angle_deg)
    if not lines:
        raise ValueError("the boundary is too small for a flight line")
    to_ll = lambda p: tuple(p / scale + origin)                       # (lon, lat)
    h = plan.height_m
    first = to_ll(lines[0][0])

    def item(command, lon=0.0, lat=0.0, alt=0.0, params=(0, 0, 0, 0), frame=FRAME_RELATIVE_ALT):
        return dict(command=command, frame=frame, params=tuple(float(v) for v in params), lat=float(lat), lon=float(lon), alt=float(alt))

    items = [item(CMD_WAYPOINT, first[0], first[1], 0.0, frame=FRAME_GLOBAL),          # home placeholder
             item(CMD_TAKEOFF, alt=h),
             item(CMD_CHANGE_SPEED, params=(1, plan.speed_ms, -1, 0)),
             item(CMD_WAYPOINT, first[0], first[1], h)]
    if autopilot_trigger:
        items.append(item(CMD_CAM_TRIGG_DIST, params=(plan.photo_spacing_m, 0, 1, 0)))
    for k, (start, end) in enumerate(lines):
        if k:
            items.append(item(CMD_WAYPOINT, *to_ll(start), h))
        items.append(item(CMD_WAYPOINT, *to_ll(end), h))
    if autopilot_trigger:
        items.append(item(CMD_CAM_TRIGG_DIST, params=(0, 0, 0, 0)))
    items.append(item(CMD_RTL))
    return items


def write_waypoints(items, path) -> None:
    """Save mission items as a "QGC WPL 110" file, which Mission Planner, QGroundControl and
    MAVProxy all load."""
    rows = ["QGC WPL 110"]
    for i, it in enumerate(items):
        p = it["params"]
        rows.append("\t".join(str(v) for v in (i, 1 if i == 0 else 0, it["frame"], it["command"],
                                               f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}", f"{p[3]:.6f}",
                                               f"{it['lat']:.8f}", f"{it['lon']:.8f}", f"{it['alt']:.3f}", 1)))
    Path(path).write_text("\n".join(rows) + "\n")
