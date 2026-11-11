"""Photo positions from the flight controller's own log.

When the flight controller triggers the camera, ArduPilot writes one CAM line per trigger to
its log (the .bin file on its SD card). For a camera with no capture computer, such as an
action camera on a trigger cable, those lines are the photo positions: the first photo
belongs to the first line, and so on.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from soilsight.ingest.geotag import Shot, list_photos

GPS_EPOCH = datetime(1980, 1, 6)
GPS_LEAP_SECONDS = 18      # GPS time is ahead of UTC by this much (unchanged since 2017)


def read_camera_log(path) -> list:
    """Every camera trigger in an ArduPilot .bin log, as Shots with an empty file name."""
    from pymavlink import mavutil
    log = mavutil.mavlink_connection(str(path))
    shots = []
    while True:
        m = log.recv_match(type="CAM", blocking=False)
        if m is None:
            break
        d = m.to_dict()
        when = None
        if d.get("GPSWeek") and d.get("GPSTime") is not None:
            when = GPS_EPOCH + timedelta(weeks=int(d["GPSWeek"]), milliseconds=int(d["GPSTime"]) - GPS_LEAP_SECONDS * 1000)
        shots.append(Shot(file="", lat=float(d["Lat"]), lon=float(d["Lng"]), alt_m=float(d["Alt"]),
                          agl_m=float(d["RelAlt"]) if "RelAlt" in d else None, time=when,
                          yaw_deg=float(d["Y"]) if "Y" in d else None, pitch_deg=float(d["P"]) if "P" in d else None,
                          roll_deg=float(d["R"]) if "R" in d else None, source="flight log"))
    return shots


def shots_from_flight_log(photo_dir, log_path) -> list:
    """Pair the photos (in name order) with the log's camera triggers (in time order).

    The counts must match: a photo the camera missed, or one extra, would shift every later
    photo onto the wrong position, so that is an error and not something to guess around.
    """
    photos = list_photos(photo_dir)
    triggers = read_camera_log(log_path)
    if len(photos) != len(triggers):
        raise ValueError(f"{len(photos)} photos but {len(triggers)} camera triggers in {Path(log_path).name}: "
                         "they cannot be paired by order")
    return [Shot(**{**t.__dict__, "file": p.name}) for p, t in zip(photos, triggers)]
