"""Where and when each photo was taken.

Three sources, in order of preference:

  capture.csv   written by the drone's capture computer next to the photos
  flight.bin    the flight controller's log of its own camera triggers
  EXIF          GPS tags inside the photos (any off-the-shelf drone)

The result is written as an OpenDroneMap geo.txt, so the photos themselves are never modified.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

PHOTO_SUFFIXES = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
SIDECAR_COLUMNS = ("file", "time_utc", "lat", "lon", "alt_m", "agl_m", "yaw_deg", "pitch_deg", "roll_deg",
                   "h_acc_m", "v_acc_m", "exposure_s")
_GPS_IFD, _EXIF_IFD = 0x8825, 0x8769


@dataclass(frozen=True)
class Shot:
    """One photo. Angles in degrees; alt_m above sea level; agl_m above the ground."""

    file: str
    lat: float
    lon: float
    alt_m: float | None = None
    agl_m: float | None = None
    time: datetime | None = None
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    h_acc_m: float | None = None
    v_acc_m: float | None = None
    exposure_s: float | None = None
    width_px: int | None = None
    height_px: int | None = None
    focal_35mm: float | None = None
    model: str = ""
    source: str = "exif"


def list_photos(photo_dir) -> list:
    return sorted(p for p in Path(photo_dir).iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)


_XMP_HEIGHT = re.compile(rb'RelativeAltitude(?:="|>)\s*([+-]?\d+(?:\.\d+)?)')


def _xmp_height(path) -> float | None:
    """Height above the take-off point, which DJI drones write into the photo's XMP block."""
    with open(path, "rb") as fh:
        m = _XMP_HEIGHT.search(fh.read(262144))
    return float(m.group(1)) if m else None


def _degrees(value, ref) -> float:
    d, m, s = (float(v) for v in value)
    deg = d + m / 60 + s / 3600
    return -deg if str(ref).upper() in ("S", "W") else deg


def read_exif(path) -> Shot | None:
    """Position and time from one photo's EXIF, or None if it has no GPS position."""
    path = Path(path)
    with Image.open(path) as im:
        exif = im.getexif()
        width, height = im.size
    gps = exif.get_ifd(_GPS_IFD)
    if 2 not in gps or 4 not in gps:
        return None
    sub = exif.get_ifd(_EXIF_IFD)
    alt = None
    if 6 in gps:
        alt = float(gps[6]) * (-1 if gps.get(5) in (1, b"\x01") else 1)
    when = None
    stamp = sub.get(0x9003) or exif.get(0x0132)        # DateTimeOriginal, else DateTime
    if stamp:
        try:
            when = datetime.strptime(str(stamp).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
            frac = str(sub.get(0x9291, "")).strip("\x00 ")
            if frac.isdigit():
                when = when.replace(microsecond=int(float("0." + frac) * 1e6))
        except ValueError:
            when = None
    exposure = sub.get(0x829A)
    focal35 = sub.get(0xA405)
    return Shot(file=path.name, lat=_degrees(gps[2], gps.get(1, "N")), lon=_degrees(gps[4], gps.get(3, "E")),
                alt_m=alt, agl_m=_xmp_height(path), time=when, exposure_s=float(exposure) if exposure else None,
                width_px=width, height_px=height, focal_35mm=float(focal35) if focal35 else None,
                model=str(exif.get(0x0110, "")).strip("\x00 "), source="exif")


def _num(row, key):
    v = (row.get(key) or "").strip()
    return float(v) if v else None


def read_sidecar(csv_path) -> list:
    """Shots from a capture.csv. Required columns: file, lat, lon. Optional: the rest of
    SIDECAR_COLUMNS. time_utc is ISO 8601 (2026-11-11T03:04:05.250Z)."""
    shots = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"file", "lat", "lon"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{csv_path}: missing column(s) {', '.join(sorted(missing))}")
        for row in reader:
            if not (row.get("lat") or "").strip() or not (row.get("lon") or "").strip():
                continue          # the capture computer lists a photo it could not place; treat it as unplaced
            when = None
            stamp = (row.get("time_utc") or "").strip()
            if stamp:
                when = datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None)
            shots.append(Shot(file=row["file"].strip(), lat=float(row["lat"]), lon=float(row["lon"]),
                              alt_m=_num(row, "alt_m"), agl_m=_num(row, "agl_m"), time=when,
                              yaw_deg=_num(row, "yaw_deg"), pitch_deg=_num(row, "pitch_deg"),
                              roll_deg=_num(row, "roll_deg"), h_acc_m=_num(row, "h_acc_m"),
                              v_acc_m=_num(row, "v_acc_m"), exposure_s=_num(row, "exposure_s"), source="sidecar"))
    return shots


def collect_shots(photo_dir, sidecar=None, flight_log=None) -> tuple:
    """Every photo's position: from the sidecar where it lists the photo, else from the flight
    log's camera triggers, else from EXIF.

    Returns (shots, photos without any position). Image size and camera model always come
    from the photo itself. A sidecar line with no position counts as no position. Raises
    ValueError if a flight log is given but its triggers cannot be paired with the photos.
    """
    listed = {s.file: s for s in read_sidecar(sidecar)} if sidecar and Path(sidecar).exists() else {}
    if not listed and flight_log and Path(flight_log).exists():
        from soilsight.ingest.flightlog import shots_from_flight_log
        listed = {s.file: s for s in shots_from_flight_log(photo_dir, flight_log)}
    shots, missing = [], []
    for path in list_photos(photo_dir):
        from_exif = read_exif(path)
        if path.name in listed:
            s = listed[path.name]
            with Image.open(path) as im:
                width, height = im.size
            extra = dict(width_px=width, height_px=height)
            if from_exif:
                extra |= dict(focal_35mm=from_exif.focal_35mm, model=from_exif.model,
                              exposure_s=s.exposure_s or from_exif.exposure_s)
            shots.append(Shot(**{**s.__dict__, **extra}))
        elif from_exif:
            shots.append(from_exif)
        else:
            missing.append(path.name)
    return shots, missing


def write_geo_txt(shots, path) -> None:
    """OpenDroneMap geo.txt: a projection line, then
    image lon lat alt [yaw pitch roll [horizontal accuracy  vertical accuracy]].

    OpenDroneMap reads the angles only from lines with at least 7 fields and the accuracies
    only from lines with 9, so the optional columns are written as complete groups. Unknown
    angles are written as 0 when accuracies follow them."""
    def has(*names):
        return all(getattr(s, n) is not None for s in shots for n in names)

    with_acc = has("h_acc_m", "v_acc_m")
    with_angles = with_acc or has("yaw_deg", "pitch_deg", "roll_deg")
    lines = ["EPSG:4326"]
    for s in shots:
        parts = [s.file, f"{s.lon:.8f}", f"{s.lat:.8f}", f"{s.alt_m if s.alt_m is not None else 0:.3f}"]
        if with_angles:
            parts += [f"{v if v is not None else 0:.2f}" for v in (s.yaw_deg, s.pitch_deg, s.roll_deg)]
        if with_acc:
            parts += [f"{s.h_acc_m:.2f}", f"{s.v_acc_m:.2f}"]
        lines.append(" ".join(parts))
    Path(path).write_text("\n".join(lines) + "\n")
