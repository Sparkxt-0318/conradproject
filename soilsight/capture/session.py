"""One capture flight: take photos, and write where each was taken to capture.csv.

The capture computer's whole job in the air is this loop:

    read flight-controller messages -> is a photo due? -> take it -> note its exposure time
    -> once the controller has reported a position from after that instant, interpolate the
       photo's position and append one line to capture.csv

A line is written and forced to disk as soon as it is known, so a power cut loses at most
the last photo or two. Photos already in the folder are kept: numbering carries on.
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

from soilsight.capture.camera import Exposure
from soilsight.capture.telemetry import State, Telemetry
from soilsight.field import FieldDir
from soilsight.ingest.geotag import SIDECAR_COLUMNS

MIN_FIX = 3   # a 3D GPS fix


@dataclass(frozen=True)
class Shot:
    file: str
    exposure: Exposure
    state: State | None        # None if the position never became known


class Session:
    """camera, trigger and telemetry are the three parts; clock is injectable for tests.

    settle_s   how long to wait for a position sample from after the exposure before
               falling back to extrapolating from the last one
    """

    def __init__(self, field: FieldDir, camera, trigger, telemetry: Telemetry | None = None,
                 clock=time.monotonic, settle_s: float = 1.0):
        self.field, self.camera, self.trigger = field, camera, trigger
        self.telemetry = telemetry or Telemetry()
        self.clock, self.settle_s = clock, settle_s
        self.shots: list = []          # finished, in order
        self._pending: list = []       # (file, Exposure, taken_mono) waiting for their position
        self.skipped_no_fix = 0        # photo points passed over because the GPS fix was not good enough
        field.photos.mkdir(parents=True, exist_ok=True)
        existing = [int(p.stem.split("_")[-1]) for p in field.photos.glob("img_*.jpg") if p.stem.split("_")[-1].isdigit()]
        self._next = max(existing, default=0) + 1
        new_file = not field.sidecar.exists()
        self._csv = open(field.sidecar, "a", newline="")
        self._writer = csv.writer(self._csv)
        if new_file:
            self._writer.writerow(SIDECAR_COLUMNS)
            self._flush()

    # ---- inputs

    def handle(self, msg_type: str, fields: dict, recv_mono: float) -> None:
        """One message from the flight controller."""
        if msg_type == "COMMAND_LONG":
            self.trigger.command(fields["command"], tuple(fields.get(f"param{i}", 0) for i in range(1, 8)))
        else:
            self.telemetry.feed(msg_type, fields, recv_mono)

    # ---- the loop body

    def step(self) -> Shot | None:
        """Call often (50 times a second is plenty). Takes a photo when one is due and writes
        out any photo whose position has become known. Returns the photo taken, if any."""
        now = self.clock()
        self._settle(now)
        state = self.telemetry.state_at(now)
        if state is None or not self.trigger.due(state, now):
            return None
        if state.fix_type is not None and state.fix_type < MIN_FIX:
            self.skipped_no_fix += 1
            self.trigger.fired(state, now)      # do not retry every loop while the fix is bad
            return None
        name = f"img_{self._next:06d}.jpg"
        exposure = self.camera.capture(self.field.photos / name)
        self._next += 1
        self.trigger.fired(state, now)
        self._pending.append((name, exposure, now))
        return Shot(name, exposure, None)

    def _settle(self, now: float, force: bool = False) -> None:
        while self._pending:
            name, exposure, taken = self._pending[0]
            if not (force or self.telemetry.covers(exposure.mid_mono) or now - taken > self.settle_s):
                break
            self._pending.pop(0)
            shot = Shot(name, exposure, self.telemetry.state_at(exposure.mid_mono))
            self.shots.append(shot)
            self._write(shot)

    def _write(self, shot: Shot) -> None:
        s = shot.state
        if s is None:      # keep the photo listed, so the gap is visible downstream
            row = [shot.file] + [""] * (len(SIDECAR_COLUMNS) - 2) + [f"{shot.exposure.exposure_s:.6f}"]
        else:
            def num(v, places):
                return "" if v is None else f"{v:.{places}f}"
            stamp = s.time_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" if s.time_utc else ""
            row = [shot.file, stamp, f"{s.lat:.8f}", f"{s.lon:.8f}", num(s.alt_m, 3), num(s.agl_m, 3),
                   num(s.yaw_deg, 2), num(s.pitch_deg, 2), num(s.roll_deg, 2), num(s.h_acc_m, 3), num(s.v_acc_m, 3),
                   f"{shot.exposure.exposure_s:.6f}"]
        self._writer.writerow(row)
        self._flush()

    def _flush(self) -> None:
        self._csv.flush()
        os.fsync(self._csv.fileno())

    @property
    def unplaced(self) -> int:
        """Photos whose position never became known."""
        return sum(s.state is None for s in self.shots)

    @property
    def extrapolated(self) -> int:
        """Photos placed by carrying on from the last sample instead of between two samples."""
        return sum(s.state is not None and not s.state.exact for s in self.shots)

    def close(self) -> None:
        """Write out photos still waiting for a position, and close the file."""
        if not self._csv.closed:
            self._settle(self.clock(), force=True)
            self._csv.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
