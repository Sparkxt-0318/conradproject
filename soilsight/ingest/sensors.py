"""Camera profiles: what one photo covers on the ground, and how much it smears."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib import resources


@dataclass(frozen=True)
class SensorProfile:
    name: str
    width_px: int
    height_px: int
    pixel_um: float           # pixel pitch at this image size
    focal_mm: float
    shutter: str = "rolling"  # "global" or "rolling"
    readout_ms: float | None = None   # rolling shutter only; None means unknown
    description: str = ""

    def __post_init__(self):
        if self.shutter not in ("global", "rolling"):
            raise ValueError(f"shutter must be 'global' or 'rolling', not {self.shutter!r}")

    def gsd_m(self, height_m: float) -> float:
        """Ground size of one pixel at this flying height."""
        return height_m * self.pixel_um * 1e-6 / (self.focal_mm * 1e-3)

    def footprint_m(self, height_m: float) -> tuple:
        """Ground size of one photo: (long side, short side)."""
        g = self.gsd_m(height_m)
        return g * max(self.width_px, self.height_px), g * min(self.width_px, self.height_px)

    def smear_px(self, speed_ms: float, height_m: float) -> float:
        """Rolling-shutter smear at this speed, in pixels. Zero for a global shutter."""
        if self.shutter == "global":
            return 0.0
        if self.readout_ms is None:
            return float("nan")
        return speed_ms * self.readout_ms * 1e-3 / self.gsd_m(height_m)

    def blur_px(self, speed_ms: float, height_m: float, exposure_s: float) -> float:
        """Motion blur during one exposure, in pixels."""
        return speed_ms * exposure_s / self.gsd_m(height_m)


def builtin_profiles() -> dict:
    text = resources.files("soilsight.ingest").joinpath("sensors.toml").read_text()
    return {name: SensorProfile(name=name, **spec) for name, spec in tomllib.loads(text).items()}


def get_profile(name: str) -> SensorProfile:
    profiles = builtin_profiles()
    if name not in profiles:
        raise KeyError(f"unknown sensor {name!r}; known: {', '.join(sorted(profiles))}")
    return profiles[name]


def profile_from_exif(width_px: int, height_px: int, focal_35mm: float | None,
                      model: str = "") -> SensorProfile | None:
    """A profile for an unknown camera, from the 35 mm-equivalent focal length in its photos.

    Shutter type and readout time cannot be read from a photo, so the result is marked as a
    rolling shutter with unknown readout. Returns None when the photos carry no focal length.
    """
    if not focal_35mm:
        return None
    diag_px = (width_px ** 2 + height_px ** 2) ** 0.5
    pixel_um = 43.27 / diag_px * 1e3          # as if the sensor were 35 mm film
    return SensorProfile(name=model or "from-exif", width_px=width_px, height_px=height_px,
                         pixel_um=pixel_um, focal_mm=float(focal_35mm), shutter="rolling",
                         readout_ms=None, description="derived from EXIF; 35 mm-equivalent geometry")
