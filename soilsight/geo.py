"""Small-area geometry on the Earth: metres per degree at a latitude."""
from __future__ import annotations

import numpy as np


def metres_per_degree(lat_deg: float) -> tuple:
    """(metres per degree of longitude, metres per degree of latitude) on the WGS84 ellipsoid.

    Good to a few parts in a million near the given latitude, which is far better than a field
    survey needs. A single constant for all latitudes would be about 0.5% off at 28 degrees north.
    """
    p = np.radians(lat_deg)
    m_lat = 111132.92 - 559.82 * np.cos(2 * p) + 1.175 * np.cos(4 * p) - 0.0023 * np.cos(6 * p)
    m_lon = 111412.84 * np.cos(p) - 93.5 * np.cos(3 * p) + 0.118 * np.cos(5 * p)
    return float(m_lon), float(m_lat)
