"""Bench test for a real camera, on a desk or a pole, before it goes on the drone.

It answers the two questions the flight plan can only assume:

  How fast can this camera and computer really take photos?   (the plan assumes one a second)
  Do the photos stay sharp when the mount is shaken?           (a lens that floats will blur)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def sharpness(path) -> float:
    """How much fine detail a photo holds: the spread of its second differences, on the
    brightness of the centre half. Only meaningful compared with other photos of the same scene."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L"), float)
    h, w = g.shape
    g = g[h // 4: h - h // 4, w // 4: w - w // 4]
    lap = g[1:-1, 2:] + g[1:-1, :-2] + g[2:, 1:-1] + g[:-2, 1:-1] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


@dataclass(frozen=True)
class BenchResult:
    photos: int
    interval_median_s: float     # time between photos, as achieved
    interval_worst_s: float
    late: int                    # photos that came more than 20% later than asked
    sharpness: tuple             # one score per photo
    soft: tuple                  # photos much less sharp than the sharpest tenth

    @property
    def sustainable_interval_s(self) -> float:
        """An interval this camera kept up with, with a little room: for the flight plan's min_interval_s."""
        return round(self.interval_worst_s * 1.1, 2)


def bench_camera(camera, out_dir, count: int = 30, interval_s: float = 0.0, clock=time.monotonic,
                 sleep=time.sleep, log=None) -> BenchResult:
    """Take count photos, interval_s apart (0: as fast as the camera goes), and measure.

    Point the camera at a detailed, well-lit scene that does not change. For the vibration
    test, tap or shake the mount during the run and look at which photos come out soft.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    camera.start()
    mids, files = [], []
    try:
        next_at = clock()
        for i in range(count):
            wait = next_at - clock()
            if wait > 0:
                sleep(wait)
            path = out / f"bench_{i + 1:04d}.jpg"
            mids.append(camera.capture(path).mid_mono)
            files.append(path)
            next_at = (next_at + interval_s) if interval_s else clock()
            if log:
                log(f"photo {i + 1} of {count}")
    finally:
        camera.stop()
    gaps = np.diff(mids)
    scores = tuple(sharpness(p) for p in files)
    reference = float(np.percentile(scores, 90)) if scores else 0.0
    soft = tuple(p.name for p, s in zip(files, scores) if reference > 0 and s < 0.5 * reference)
    late = int((gaps > 1.2 * interval_s).sum()) if interval_s else 0
    return BenchResult(len(files), float(np.median(gaps)) if len(gaps) else float("nan"),
                       float(gaps.max()) if len(gaps) else float("nan"), late, scores, soft)
