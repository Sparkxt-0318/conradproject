"""One field visit is one folder. This module knows where everything in it lives.

    <field>/
      field.toml            optional settings (camera, flying height, grid size)
      photos/               the drone photos
      capture.csv           optional: positions written by the capture computer
      flight.bin            optional: the flight controller's log (positions of its camera triggers)
      annotations.geojson   optional: field boundary, inlet, outlet, ditches, grey card
      work/geo.txt          photo positions handed to the stitcher
      odm/                  stitched orthophoto and surface model
      layers/               map layers on the field grid
      report/               quality report
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FieldConfig:
    name: str = ""
    sensor: str = ""                    # a profile name from sensors.toml; empty: derive from the photos
    camera: dict | None = None          # or describe the camera here: a [camera] table with the
                                        # fields of a sensors.toml entry (width_px, height_px, ...)
    flight_height_m: float | None = None  # above ground; used when photos do not record it
    grid_m: float = 1.0                 # cell size of the map layers
    smooth_m: float = 3.0               # layers are smoothed over about this distance


@dataclass(frozen=True)
class FieldDir:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root))

    photos = property(lambda self: self.root / "photos")
    sidecar = property(lambda self: self.root / "capture.csv")
    annotations = property(lambda self: self.root / "annotations.geojson")
    config_path = property(lambda self: self.root / "field.toml")
    work = property(lambda self: self.root / "work")
    geo_txt = property(lambda self: self.root / "work" / "geo.txt")
    gcp_list = property(lambda self: self.root / "gcp_list.txt")
    flight_log = property(lambda self: self.root / "flight.bin")
    odm = property(lambda self: self.root / "odm")
    orthophoto = property(lambda self: self.root / "odm" / "odm_orthophoto.tif")
    dsm = property(lambda self: self.root / "odm" / "dsm.tif")
    odm_stats = property(lambda self: self.root / "odm" / "stats.json")
    layers_dir = property(lambda self: self.root / "layers")
    layers_tif = property(lambda self: self.root / "layers" / "layers.tif")
    layers_json = property(lambda self: self.root / "layers" / "layers.json")
    report_dir = property(lambda self: self.root / "report")
    quality_json = property(lambda self: self.root / "report" / "quality.json")
    quality_html = property(lambda self: self.root / "report" / "quality.html")

    def config(self) -> FieldConfig:
        if not self.config_path.exists():
            return FieldConfig(name=self.root.name)
        raw = tomllib.loads(self.config_path.read_text())
        unknown = set(raw) - set(FieldConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{self.config_path}: unknown setting(s) {', '.join(sorted(unknown))}")
        return FieldConfig(**({"name": self.root.name} | raw))
