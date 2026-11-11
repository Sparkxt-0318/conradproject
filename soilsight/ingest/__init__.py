"""Reading a photo folder: where each photo was taken, with which camera, and how well."""

from soilsight.ingest.geotag import Shot, collect_shots, list_photos, read_exif, read_sidecar, write_geo_txt
from soilsight.ingest.flightlog import read_camera_log, shots_from_flight_log
from soilsight.ingest.qa import FlightGeometry, flight_checks, flight_geometry, local_xy
from soilsight.ingest.sensors import SensorProfile, builtin_profiles, get_profile, profile_from_exif

__all__ = ["FlightGeometry", "SensorProfile", "Shot", "builtin_profiles", "collect_shots", "flight_checks",
           "flight_geometry", "get_profile", "list_photos", "local_xy", "profile_from_exif", "read_camera_log",
           "read_exif", "read_sidecar", "shots_from_flight_log", "write_geo_txt"]
