"""Where the drone was at the instant a photo was taken.

The flight controller sends its position about ten times a second, each sample stamped with
the controller's own clock (milliseconds since it booted). A photo is stamped with the
capture computer's clock. This module learns the offset between the two clocks from the
messages themselves, keeps a few seconds of samples, and interpolates between the samples
on either side of the photo.

Clock offset: every message arrives a little after it was stamped, never before, so the
smallest "arrival time minus stamp" seen recently is the best estimate of the offset. What
remains is the link's fixed delay (a few milliseconds on a serial cable, which is a few
centimetres of travel at survey speed).

Nothing here talks to hardware: messages are fed in as plain dictionaries.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from soilsight.geo import metres_per_degree


@dataclass(frozen=True)
class State:
    """The drone at one instant."""

    lat: float
    lon: float
    alt_m: float                 # above sea level
    agl_m: float                 # above the take-off point
    speed_ms: float              # over the ground
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    h_acc_m: float | None = None
    v_acc_m: float | None = None
    fix_type: int | None = None   # 3 = 3D fix, 4 = differential, 5 = RTK float, 6 = RTK fixed
    time_utc: datetime | None = None
    gap_s: float = 0.0           # distance in time to the nearest position sample used
    exact: bool = True           # False when extrapolated past the newest sample


def _lerp(a, b, t):
    return a + (b - a) * t


def _lerp_angle(a, b, t):
    return (a + ((b - a + 180.0) % 360.0 - 180.0) * t) % 360.0


class Telemetry:
    """Feed it flight-controller messages; ask it where the drone was at any recent instant."""

    def __init__(self, history_s: float = 20.0, offset_window_s: float = 30.0, max_gap_s: float = 0.6):
        self.history_s, self.offset_window_s, self.max_gap_s = history_s, offset_window_s, max_gap_s
        self._pos = deque()        # (boot_ms, lat, lon, alt, agl, vn, ve)
        self._att = deque()        # (boot_ms, roll, pitch, yaw) in degrees
        self._offsets = deque()    # (recv_mono, recv_mono - boot_s)
        self._boot_epoch_s = None  # UTC seconds at controller boot
        self.h_acc_m = self.v_acc_m = self.fix_type = None
        self.satellites = None

    # ---- feeding

    def feed(self, msg_type: str, f: dict, recv_mono: float) -> None:
        """One decoded MAVLink message, its fields as a dict, and when it arrived (monotonic seconds)."""
        if "time_boot_ms" in f and msg_type in ("GLOBAL_POSITION_INT", "ATTITUDE", "SYSTEM_TIME"):
            self._offsets.append((recv_mono, recv_mono - f["time_boot_ms"] / 1000.0))
            while self._offsets and self._offsets[0][0] < recv_mono - self.offset_window_s:
                self._offsets.popleft()
        if msg_type == "GLOBAL_POSITION_INT":
            self._pos.append((f["time_boot_ms"], f["lat"] / 1e7, f["lon"] / 1e7, f["alt"] / 1000.0,
                              f["relative_alt"] / 1000.0, f["vx"] / 100.0, f["vy"] / 100.0))
            self._trim(self._pos)
        elif msg_type == "ATTITUDE":
            self._att.append((f["time_boot_ms"], np.degrees(f["roll"]), np.degrees(f["pitch"]),
                              np.degrees(f["yaw"]) % 360.0))
            self._trim(self._att)
        elif msg_type == "GPS_RAW_INT":
            self.fix_type = f.get("fix_type")
            self.satellites = f.get("satellites_visible")
            # 0 means "not reported"; older links send no accuracy at all
            self.h_acc_m = f["h_acc"] / 1000.0 if f.get("h_acc") else None
            self.v_acc_m = f["v_acc"] / 1000.0 if f.get("v_acc") else None
        elif msg_type == "SYSTEM_TIME":
            if f["time_unix_usec"] > 1_000_000_000_000_000:        # a real date, i.e. GPS time is known
                self._boot_epoch_s = f["time_unix_usec"] / 1e6 - f["time_boot_ms"] / 1000.0

    def _trim(self, q) -> None:
        while len(q) > 2 and q[0][0] < q[-1][0] - self.history_s * 1000:
            q.popleft()

    # ---- clocks

    @property
    def clock_offset_s(self) -> float | None:
        """Capture-computer time minus flight-controller time, or None before any message."""
        return min(o for _, o in self._offsets) if self._offsets else None

    def boot_ms_at(self, mono: float) -> float | None:
        off = self.clock_offset_s
        return None if off is None else (mono - off) * 1000.0

    @property
    def newest_boot_ms(self) -> float | None:
        return self._pos[-1][0] if self._pos else None

    def covers(self, mono: float) -> bool:
        """True once a position sample newer than this instant has arrived."""
        t = self.boot_ms_at(mono)
        return t is not None and self._pos and self._pos[-1][0] >= t

    # ---- asking

    def state_at(self, mono: float) -> State | None:
        """The drone's state at a capture-computer instant. None when there is no position
        within max_gap_s of it."""
        t = self.boot_ms_at(mono)
        if t is None or not self._pos:
            return None
        times = [p[0] for p in self._pos]
        i = bisect_left(times, t)
        if 0 < i < len(times):
            a, b = self._pos[i - 1], self._pos[i]
            w = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
            lat, lon, alt, agl, vn, ve = (_lerp(a[k], b[k], w) for k in range(1, 7))
            gap, exact = min(t - a[0], b[0] - t) / 1000.0, True
        else:
            p = self._pos[-1] if i else self._pos[0]
            dt = (t - p[0]) / 1000.0
            if abs(dt) > self.max_gap_s:
                return None
            _, lat, lon, alt, agl, vn, ve = p                      # carry on at the last known velocity
            m_lon, m_lat = metres_per_degree(lat)
            lat += vn * dt / m_lat
            lon += ve * dt / m_lon
            gap, exact = abs(dt), False
        if gap > self.max_gap_s:
            return None

        roll = pitch = yaw = None
        if self._att:
            at = [a[0] for a in self._att]
            j = bisect_left(at, t)
            if 0 < j < len(at):
                a, b = self._att[j - 1], self._att[j]
                w = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
                roll, pitch, yaw = _lerp(a[1], b[1], w), _lerp(a[2], b[2], w), _lerp_angle(a[3], b[3], w)
            else:
                a = self._att[-1] if j else self._att[0]
                if abs(t - a[0]) <= self.max_gap_s * 1000:
                    roll, pitch, yaw = a[1], a[2], a[3]
        utc = None
        if self._boot_epoch_s is not None:
            utc = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=self._boot_epoch_s + t / 1000.0)
        return State(lat=lat, lon=lon, alt_m=alt, agl_m=agl, speed_ms=float(np.hypot(vn, ve)), yaw_deg=yaw,
                     pitch_deg=pitch, roll_deg=roll, h_acc_m=self.h_acc_m, v_acc_m=self.v_acc_m,
                     fix_type=self.fix_type, time_utc=utc, gap_s=gap, exact=exact)


def distance_m(a: State, b: State) -> float:
    """Ground distance between two states (flat-earth; fine over one field)."""
    m_lon, m_lat = metres_per_degree(a.lat)
    return float(np.hypot((b.lon - a.lon) * m_lon, (b.lat - a.lat) * m_lat))
