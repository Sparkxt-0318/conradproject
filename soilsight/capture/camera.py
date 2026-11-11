"""Cameras the capture computer can drive.

Every camera does one thing: take a photo to a file and say when its exposure was centred,
on the capture computer's monotonic clock. That instant is what the photo's position is
looked up for.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from soilsight.ingest.sensors import SensorProfile


@dataclass(frozen=True)
class Exposure:
    mid_mono: float        # centre of the exposure, monotonic seconds
    exposure_s: float


class FakeCamera:
    """Stands in for a camera in simulation and tests: writes a small grey JPEG.

    latency_s models a camera that exposes a little after it is asked to.
    """

    name = "fake"

    def __init__(self, profile: SensorProfile | None = None, exposure_s: float = 1 / 1000, latency_s: float = 0.0,
                 clock=time.monotonic, size=(64, 48)):
        self.profile, self.exposure_s, self.latency_s, self.clock, self.size = profile, exposure_s, latency_s, clock, size
        self.count = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def capture(self, path) -> Exposure:
        from PIL import Image
        shade = 80 + (self.count * 37) % 120
        Image.new("RGB", self.size, (shade, shade - 20, shade - 40)).save(path, "JPEG")
        self.count += 1
        return Exposure(self.clock() + self.latency_s + self.exposure_s / 2, self.exposure_s)


class PiCamera:
    """A Raspberry Pi camera through picamera2, set up for mapping: fixed exposure, fixed
    white balance and, on lenses that can move, focus locked at infinity.

    NOT YET RUN ON HARDWARE. It follows the picamera2 manual; bench-test it before flying.

    exposure_s     shutter time; 1/1000 s or shorter keeps blur under a pixel at survey speed
    gain           analogue gain; raise it before lengthening the exposure
    colour_gains   (red, blue) white-balance gains, fixed so soil colour is comparable between
                   photos; read a pair off a grey card in the day's light
    """

    name = "pi"

    def __init__(self, profile: SensorProfile, exposure_s: float = 1 / 1000, gain: float = 1.0,
                 colour_gains: tuple = (2.0, 1.6), quality: int = 93):
        self.profile, self.exposure_s, self.gain, self.colour_gains, self.quality = profile, exposure_s, gain, colour_gains, quality
        self._cam = None

    def start(self) -> None:
        from picamera2 import Picamera2          # only present on the Pi
        cam = Picamera2()
        cam.configure(cam.create_still_configuration(main={"size": (self.profile.width_px, self.profile.height_px)},
                                                     buffer_count=2))
        cam.options["quality"] = self.quality
        controls = {"AeEnable": False, "ExposureTime": int(self.exposure_s * 1e6), "AnalogueGain": float(self.gain),
                    "AwbEnable": False, "ColourGains": tuple(float(g) for g in self.colour_gains)}
        if "AfMode" in cam.camera_controls:      # a lens that can move: lock it at infinity
            controls |= {"AfMode": 0, "LensPosition": 0.0}
        cam.set_controls(controls)
        cam.start()
        time.sleep(1.0)                          # let the fixed settings take hold
        self._cam = cam

    def stop(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam.close()
            self._cam = None

    def capture(self, path) -> Exposure:
        request = self._cam.capture_request()
        try:
            meta = request.get_metadata()
            request.save("main", str(Path(path)))
        finally:
            request.release()
        exposure = meta.get("ExposureTime", self.exposure_s * 1e6) / 1e6
        # SensorTimestamp is when the first row began exposing, in nanoseconds since boot. A rolling
        # shutter's rows follow over the readout time, so the frame's centre is half of that later.
        readout = (self.profile.readout_ms or 0.0) / 1000.0 if self.profile.shutter == "rolling" else 0.0
        return Exposure(meta["SensorTimestamp"] / 1e9 + exposure / 2 + readout / 2, exposure)
