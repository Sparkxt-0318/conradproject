"""Small synthetic photos, rasters and annotations with known answers, for the scout tests."""
import json
from datetime import datetime, timedelta

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

CRS = "EPSG:32650"                      # UTM zone 50N
WEST, SOUTH = 500000.0, 3100000.0       # lower-left corner of the test field
FIELD_W, FIELD_H = 60.0, 40.0           # metres
ILLUMINATION = np.array([0.70, 0.75, 0.80])   # scene light per channel (a bluish cast)
SOIL = np.array([0.20, 0.15, 0.10])     # true reflectance of plain soil
RED_SOIL = np.array([0.30, 0.13, 0.08])
STRAW = np.array([0.20, 0.30, 0.10])
RED_PATCH = (10, 10, 25, 25)            # x0, y0, x1, y1 in metres from the corner
STRAW_PATCH = (40, 25, 55, 35)
CARD = (2.0, 36.0, 3.5, 37.5)           # grey card, reflectance 0.18
HOLLOW = (42.0, 12.0)                   # centre of a 15 cm deep hollow
TILT = 0.005                            # 0.5% slope to the east


def to_lonlat(x, y):
    lon, lat = warp_transform(CRS, "EPSG:4326", [WEST + x], [SOUTH + y])
    return lon[0], lat[0]


def linear_to_srgb8(lin):
    lin = np.clip(lin, 0, 1)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.round(c * 255).astype("uint8")


def write_rasters(fd, margin=5.0, ortho_px=0.1, dsm_px=0.2, noise=0.01, seed=0):
    """Orthophoto and surface model covering the field plus a margin."""
    fd.odm.mkdir(parents=True, exist_ok=True)
    w, n = WEST - margin, SOUTH + FIELD_H + margin
    nx, ny = int((FIELD_W + 2 * margin) / ortho_px), int((FIELD_H + 2 * margin) / ortho_px)
    x = (np.arange(nx) + 0.5) * ortho_px - margin
    y = FIELD_H + margin - (np.arange(ny) + 0.5) * ortho_px
    xx, yy = np.meshgrid(x, y)
    refl = np.broadcast_to(SOIL[:, None, None], (3, ny, nx)).copy()
    for (x0, y0, x1, y1), colour in ((RED_PATCH, RED_SOIL), (STRAW_PATCH, STRAW), (CARD, np.full(3, 0.18))):
        inside = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
        refl[:, inside] = colour[:, None]
    rgb = linear_to_srgb8(refl * ILLUMINATION[:, None, None])
    alpha = np.full((ny, nx), 255, "uint8")
    with rasterio.open(fd.orthophoto, "w", driver="GTiff", width=nx, height=ny, count=4, dtype="uint8", crs=CRS,
                       transform=from_origin(w, n, ortho_px, ortho_px)) as dst:
        dst.write(rgb, [1, 2, 3])
        dst.write(alpha, 4)

    nx, ny = int((FIELD_W + 2 * margin) / dsm_px), int((FIELD_H + 2 * margin) / dsm_px)
    x = (np.arange(nx) + 0.5) * dsm_px - margin
    y = FIELD_H + margin - (np.arange(ny) + 0.5) * dsm_px
    xx, yy = np.meshgrid(x, y)
    z = 30 + TILT * xx - 0.15 * np.exp(-((xx - HOLLOW[0]) ** 2 + (yy - HOLLOW[1]) ** 2) / (2 * 4.0 ** 2))
    z = z + noise * np.random.default_rng(seed).standard_normal(z.shape)
    with rasterio.open(fd.dsm, "w", driver="GTiff", width=nx, height=ny, count=1, dtype="float32", crs=CRS,
                       transform=from_origin(w, n, dsm_px, dsm_px), nodata=-9999.0) as dst:
        dst.write(z.astype("float32"), 1)


def write_stats(fd, used=40, total=40, gcp=False):
    stats = {"reconstruction_statistics": {"initial_shots_count": total, "reconstructed_shots_count": used,
                                           "reprojection_error_pixels": 0.8, "has_gcp": gcp},
             "gps_errors": {"average_error": 1.2}, "gcp_errors": {"average_error": 0.03} if gcp else {},
             "odm_processing_statistics": {"average_gsd": 2.0}}
    fd.odm.mkdir(parents=True, exist_ok=True)
    fd.odm_stats.write_text(json.dumps(stats))


def _poly(x0, y0, x1, y1):
    ring = [to_lonlat(*p) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
    return {"type": "Polygon", "coordinates": [[list(p) for p in ring]]}


def write_annotations(fd, boundary=True, inlet=True, outlet=True, ditch=True, card=True, extra=()):
    feats = []
    if boundary:
        feats.append({"type": "Feature", "properties": {"role": "boundary"}, "geometry": _poly(0, 0, FIELD_W, FIELD_H)})
    if inlet:
        feats.append({"type": "Feature", "properties": {"role": "inlet"},
                      "geometry": {"type": "Point", "coordinates": list(to_lonlat(0, 0))}})
    if outlet:
        feats.append({"type": "Feature", "properties": {"role": "outlet"},
                      "geometry": {"type": "Point", "coordinates": list(to_lonlat(FIELD_W, FIELD_H))}})
    if ditch:   # along the southern edge
        feats.append({"type": "Feature", "properties": {"role": "ditch"},
                      "geometry": {"type": "LineString", "coordinates": [list(to_lonlat(0, 0)), list(to_lonlat(FIELD_W, 0))]}})
    if card:
        feats.append({"type": "Feature", "properties": {"role": "grey_card", "reflectance": 0.18},
                      "geometry": _poly(CARD[0] + 0.2, CARD[1] + 0.2, CARD[2] - 0.2, CARD[3] - 0.2)})
    feats += list(extra)
    fd.root.mkdir(parents=True, exist_ok=True)
    fd.annotations.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def _dms(deg):
    deg = abs(deg)
    d = int(deg)
    m = int((deg - d) * 60)
    return (float(d), float(m), round((deg - d - m / 60) * 3600, 6))


def write_photo(path, lat=None, lon=None, alt=None, when=None, size=(64, 48), exposure=None, focal35=None,
                model="TestCam", xmp_height=None):
    """A tiny JPEG with the given EXIF. lat=None writes a photo with no position."""
    im = Image.new("RGB", size, (120, 100, 80))
    exif = im.getexif()
    exif[0x0110] = model
    sub = {}
    if when is not None:
        sub[0x9003] = when.strftime("%Y:%m:%d %H:%M:%S")
        sub[0x9291] = f"{when.microsecond // 1000:03d}"
    if exposure:
        sub[0x829A] = exposure
    if focal35:
        sub[0xA405] = focal35
    if sub:
        exif[0x8769] = sub
    if lat is not None:
        gps = {1: "N" if lat >= 0 else "S", 2: _dms(lat), 3: "E" if lon >= 0 else "W", 4: _dms(lon)}
        if alt is not None:
            gps[5] = b"\x00"
            gps[6] = float(alt)
        exif[0x8825] = gps
    im.save(path, "JPEG", exif=exif)
    if xmp_height is not None:
        with open(path, "ab") as fh:
            fh.write(f'<x:xmpmeta drone-dji:RelativeAltitude="{xmp_height:+.2f}"/>'.encode())


def lawnmower(lines=5, per_line=20, spacing=5.0, line_spacing=10.0, speed=5.0, lat0=28.2, lon0=113.0):
    """Positions and times of a back-and-forth grid flight: list of (name, lat, lon, time)."""
    out, t = [], datetime(2026, 11, 11, 3, 0, 0)
    for line in range(lines):
        ys = np.arange(per_line) * spacing
        for y in (ys if line % 2 == 0 else ys[::-1]):
            out.append((f"img_{len(out):04d}.jpg", lat0 + y / 111320.0,
                        lon0 + line * line_spacing / (111320.0 * np.cos(np.radians(lat0))), t))
            t += timedelta(seconds=spacing / speed)
        t += timedelta(seconds=6)
    return out
