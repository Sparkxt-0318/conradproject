"""A stand-in drone for the capture tests: flies a path and emits the flight controller's
messages on a simulated clock, so a whole flight runs in milliseconds and the true position
at any instant is known exactly."""
from datetime import datetime

import numpy as np

from soilsight.geo import metres_per_degree


class SimClock:
    def __init__(self, start=5000.0):
        self.now = start

    def __call__(self):
        return self.now


class FakeVehicle:
    """Flies straight between the points of path_xy (metres east, north) at a steady speed.

    clock_offset_s  capture-computer clock minus flight-controller clock
    latency_s       how long a message takes to arrive; jitter_s adds a random extra delay
    """

    def __init__(self, path_xy, speed=5.0, height=40.0, lat0=28.2, lon0=113.0, rate_hz=10.0, clock_offset_s=4000.0,
                 latency_s=0.0, jitter_s=0.0, start_boot_s=1000.0, fix_type=3, with_time=True, seed=0):
        self.path = np.asarray(path_xy, float)
        seg = np.diff(self.path, axis=0)
        self.seg_len = np.hypot(*seg.T)
        self.cum = np.r_[0, np.cumsum(self.seg_len)]
        self.speed, self.height, self.lat0, self.lon0 = speed, height, lat0, lon0
        self.rate, self.offset, self.latency, self.jitter = rate_hz, clock_offset_s, latency_s, jitter_s
        self.t0, self.fix_type, self.with_time = start_boot_s, fix_type, with_time
        self.rng = np.random.default_rng(seed)
        self.utc0 = datetime(2026, 11, 11, 3, 0, 0)        # UTC at start_boot_s
        self._k = 0
        self.duration = self.cum[-1] / speed

    def xy_at(self, boot_s):
        d = np.clip((boot_s - self.t0) * self.speed, 0, self.cum[-1])
        i = min(np.searchsorted(self.cum, d, side="right") - 1, len(self.seg_len) - 1)
        w = (d - self.cum[i]) / self.seg_len[i]
        p = self.path[i] + w * (self.path[i + 1] - self.path[i])
        moving = 0 < (boot_s - self.t0) * self.speed < self.cum[-1]
        v = (self.path[i + 1] - self.path[i]) / self.seg_len[i] * (self.speed if moving else 0.0)
        return p, v

    def lonlat(self, xy):
        m_lon, m_lat = metres_per_degree(self.lat0)
        return (self.lon0 + xy[0] / m_lon, self.lat0 + xy[1] / m_lat)

    def true_xy_at_mono(self, mono):
        return self.xy_at(mono - self.offset)[0]

    def messages_until(self, mono):
        """Every message that has arrived by this capture-computer instant, in arrival order."""
        out = []
        while True:
            boot = self.t0 - 2.0 + self._k / self.rate
            recv = boot + self.offset + self.latency + (self.rng.uniform(0, self.jitter) if self.jitter else 0.0)
            if boot + self.offset + self.latency + self.jitter > mono and recv > mono:
                break
            p, v = self.xy_at(boot)
            lon, lat = self.lonlat(p)
            ms = int(round(boot * 1000))
            yaw = np.arctan2(v[0], v[1]) if np.hypot(*v) > 0 else 0.0
            out.append((recv, "GLOBAL_POSITION_INT", dict(time_boot_ms=ms, lat=int(round(lat * 1e7)), lon=int(round(lon * 1e7)),
                        alt=int((60 + self.height) * 1000), relative_alt=int(self.height * 1000),
                        vx=int(round(v[1] * 100)), vy=int(round(v[0] * 100)), vz=0, hdg=0)))
            out.append((recv, "ATTITUDE", dict(time_boot_ms=ms, roll=0.01, pitch=-0.02, yaw=float(yaw))))
            if self._k % int(self.rate / 2) == 0:
                out.append((recv, "GPS_RAW_INT", dict(fix_type=self.fix_type, h_acc=800, v_acc=1200, satellites_visible=14)))
            if self._k % int(self.rate) == 0:
                unix = (self.utc0 - datetime(1970, 1, 1)).total_seconds() + (boot - self.t0) if self.with_time else 0.0
                out.append((recv, "SYSTEM_TIME", dict(time_unix_usec=int(unix * 1e6), time_boot_ms=ms)))
            self._k += 1
        return [(t, f, r) for r, t, f in sorted(out, key=lambda m: m[0])]


def grid_path(lines=4, length=60.0, line_spacing=10.0, lead=10.0):
    """A back-and-forth survey path, with a lead-in before the first line."""
    pts = [(-lead, 0.0)]
    for k in range(lines):
        y = k * line_spacing
        pts += [(0.0, y), (length, y)] if k % 2 == 0 else [(length, y), (0.0, y)]
    return pts


def fly(session, vehicle, clock, extra_s=3.0, dt=0.005):
    """Run the capture loop against the stand-in drone for the whole flight."""
    end = vehicle.t0 + vehicle.offset + vehicle.duration + extra_s
    clock.now = vehicle.t0 + vehicle.offset - 1.0
    while clock.now < end:
        for msg_type, fields, recv in vehicle.messages_until(clock.now):
            session.handle(msg_type, fields, recv)
        session.step()
        clock.now += dt
    session.close()
