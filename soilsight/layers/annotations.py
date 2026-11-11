"""What a person marks on the map: the field boundary, inlet, outlet, ditches, grey card.

One GeoJSON file in longitude/latitude (as QGIS, QField or geojson.io save it). Each feature
carries properties.role, one of:

    boundary    Polygon          the field; layers are made inside it
    inlet       Point            where irrigation water enters
    outlet      Point            where it leaves
    ditch       LineString       a ditch or drain (any number)
    grey_card   Polygon          a grey reference card on the ground; properties.reflectance
                                 gives its reflectance (default 0.18)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rasterio.warp import transform_geom

ROLES = {"boundary": ("Polygon", "MultiPolygon"), "inlet": ("Point",), "outlet": ("Point",),
         "ditch": ("LineString", "MultiLineString"), "grey_card": ("Polygon",)}


@dataclass(frozen=True)
class Annotations:
    """Geometries in the map's projection (metres)."""

    boundary: dict | None = None
    inlet: tuple | None = None
    outlet: tuple | None = None
    ditches: tuple = ()            # each an (n, 2) array of vertices
    grey_card: dict | None = None
    grey_reflectance: float = 0.18
    problems: tuple = field(default_factory=tuple)


def load_annotations(path, crs) -> Annotations:
    """Read the GeoJSON and move it into the map's projection. A missing file gives empty
    annotations; features with an unknown role or wrong shape are listed under problems."""
    path = Path(path)
    if not path.exists():
        return Annotations()
    out, ditches, problems = {}, [], []
    for i, feat in enumerate(json.loads(path.read_text()).get("features", [])):
        role = (feat.get("properties") or {}).get("role")
        geom = feat.get("geometry") or {}
        if role not in ROLES:
            problems.append(f"feature {i}: unknown role {role!r}")
            continue
        if geom.get("type") not in ROLES[role]:
            problems.append(f"feature {i}: a {role} must be a {' or '.join(ROLES[role])}, not {geom.get('type')}")
            continue
        g = transform_geom("EPSG:4326", crs, geom)
        if role in ("inlet", "outlet"):
            out[role] = tuple(g["coordinates"][:2])
        elif role == "ditch":
            lines = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
            ditches += [np.asarray(line, float)[:, :2] for line in lines]
        elif role == "grey_card":
            out["grey_card"] = g
            out["grey_reflectance"] = float(feat["properties"].get("reflectance", 0.18))
        else:
            out[role] = g
    return Annotations(ditches=tuple(ditches), problems=tuple(problems), **out)
