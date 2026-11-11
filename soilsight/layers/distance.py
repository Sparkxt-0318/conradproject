"""Distance layers: how far each cell is from the points and lines a person marked."""
from __future__ import annotations

import numpy as np

DISTANCE_LAYERS = {
    "dist_inlet": "metres to the irrigation inlet",
    "dist_outlet": "metres to the outlet",
    "dist_ditch": "metres to the nearest ditch",
}


def distance_to_point(x: np.ndarray, y: np.ndarray, point) -> np.ndarray:
    return np.hypot(x - point[0], y - point[1])


def distance_to_lines(x: np.ndarray, y: np.ndarray, lines) -> np.ndarray:
    """Distance from each (x, y) to the nearest segment of any polyline."""
    best = np.full(np.shape(x), np.inf)
    for line in lines:
        for (ax, ay), (bx, by) in zip(line[:-1], line[1:]):
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            t = np.clip(((x - ax) * dx + (y - ay) * dy) / length2, 0, 1) if length2 > 0 else 0.0
            best = np.minimum(best, np.hypot(x - (ax + t * dx), y - (ay + t * dy)))
    return best
