from datetime import datetime

import pytest

from scout_helpers import lawnmower, write_photo
from soilsight.ingest import (Shot, collect_shots, flight_checks, flight_geometry, get_profile, builtin_profiles,
                              profile_from_exif, read_exif, read_sidecar, write_geo_txt)
from soilsight.ingest.sensors import SensorProfile


# ---- camera profiles

def test_global_shutter_profile_geometry():
    p = get_profile("pi_gs_6mm")
    assert p.gsd_m(40) * 100 == pytest.approx(2.3, abs=0.01)
    long_side, short_side = p.footprint_m(40)
    assert (long_side, short_side) == (pytest.approx(33.5, abs=0.1), pytest.approx(25.0, abs=0.1))
    assert p.smear_px(5, 40) == 0


def test_rolling_shutter_smear_matches_the_plan():
    full, binned = get_profile("pi_cm3_full"), get_profile("pi_cm3_binned")
    assert full.gsd_m(40) * 100 == pytest.approx(1.18, abs=0.01)
    assert full.smear_px(5, 40) == pytest.approx(29, abs=1)       # 34 cm at 5 m/s
    assert binned.smear_px(5, 40) == pytest.approx(3.6, abs=0.3)
    assert full.blur_px(5, 40, 1 / 1000) == pytest.approx(0.42, abs=0.02)


def test_profiles_are_validated():
    assert set(builtin_profiles()) == {"pi_gs_6mm", "pi_cm3_binned", "pi_cm3_full"}
    with pytest.raises(KeyError, match="known: pi_cm3_binned"):
        get_profile("gopro")
    with pytest.raises(ValueError, match="shutter"):
        SensorProfile("x", 10, 10, 1.0, 1.0, shutter="leaf")


def test_profile_from_exif():
    p = profile_from_exif(4000, 3000, 20.0, "FC300S")
    assert p.name == "FC300S" and p.readout_ms is None
    assert p.gsd_m(40) * 100 == pytest.approx(1.73, abs=0.02)     # 43.27 mm / 5000 px, 20 mm lens
    assert p.smear_px(5, 40) != p.smear_px(5, 40)                  # NaN: readout unknown
    assert profile_from_exif(4000, 3000, None) is None


# ---- reading positions

def test_exif_round_trip(tmp_path):
    when = datetime(2026, 11, 11, 3, 4, 5, 250000)
    write_photo(tmp_path / "a.jpg", 28.123456, 113.654321, 85.5, when, exposure=1 / 800, focal35=24, model="Cam X")
    s = read_exif(tmp_path / "a.jpg")
    assert s.lat == pytest.approx(28.123456, abs=1e-6) and s.lon == pytest.approx(113.654321, abs=1e-6)
    assert s.alt_m == pytest.approx(85.5) and s.time == when
    assert s.exposure_s == pytest.approx(1 / 800) and s.focal_35mm == 24 and s.model == "Cam X"
    assert (s.width_px, s.height_px, s.source, s.agl_m) == (64, 48, "exif", None)


def test_exif_southern_and_western_positions_are_negative(tmp_path):
    write_photo(tmp_path / "a.jpg", -33.5, -70.25, 500)
    s = read_exif(tmp_path / "a.jpg")
    assert s.lat == pytest.approx(-33.5) and s.lon == pytest.approx(-70.25)


def test_photo_without_position(tmp_path):
    write_photo(tmp_path / "a.jpg")
    assert read_exif(tmp_path / "a.jpg") is None


def test_dji_height_above_ground_is_read(tmp_path):
    write_photo(tmp_path / "a.jpg", 28.0, 113.0, 90, xmp_height=41.2)
    assert read_exif(tmp_path / "a.jpg").agl_m == pytest.approx(41.2)


def test_sidecar(tmp_path):
    csv = tmp_path / "capture.csv"
    csv.write_text("file,time_utc,lat,lon,alt_m,agl_m,yaw_deg,pitch_deg,roll_deg,h_acc_m,v_acc_m,exposure_s\n"
                   "a.jpg,2026-11-11T03:04:05.250Z,28.1,113.2,80,40,90,-1,2,0.02,0.03,0.001\n"
                   "b.jpg,,28.2,113.3,,,,,,,,\n")
    a, b = read_sidecar(csv)
    assert (a.lat, a.lon, a.alt_m, a.agl_m, a.yaw_deg, a.h_acc_m, a.source) == (28.1, 113.2, 80, 40, 90, 0.02, "sidecar")
    assert a.time == datetime(2026, 11, 11, 3, 4, 5, 250000)
    assert b.time is None and b.alt_m is None
    csv.write_text("file,lat\na.jpg,1\n")
    with pytest.raises(ValueError, match="missing column.*lon"):
        read_sidecar(csv)


def test_collect_prefers_the_sidecar_and_lists_photos_without_position(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    write_photo(photos / "a.jpg", 28.0, 113.0, 80, focal35=24)      # EXIF says 28.0
    write_photo(photos / "b.jpg", 28.1, 113.1, 80)
    write_photo(photos / "c.jpg")                                    # nowhere
    (tmp_path / "capture.csv").write_text("file,lat,lon,alt_m\na.jpg,28.5,113.5,81\n")
    shots, missing = collect_shots(photos, tmp_path / "capture.csv")
    by = {s.file: s for s in shots}
    assert missing == ["c.jpg"]
    assert by["a.jpg"].lat == 28.5 and by["a.jpg"].source == "sidecar"
    assert by["a.jpg"].width_px == 64 and by["a.jpg"].focal_35mm == 24     # image facts still from the photo
    assert by["b.jpg"].source == "exif"


def test_geo_txt_writes_complete_column_groups(tmp_path):
    out = tmp_path / "geo.txt"
    bare = [Shot("a.jpg", 28.1, 113.2, 80.0), Shot("b.jpg", 28.2, 113.3)]
    write_geo_txt(bare, out)
    assert out.read_text().splitlines() == ["EPSG:4326", "a.jpg 113.20000000 28.10000000 80.000",
                                            "b.jpg 113.30000000 28.20000000 0.000"]
    angles = [Shot("a.jpg", 28.1, 113.2, 80.0, yaw_deg=90, pitch_deg=-1, roll_deg=2)]
    write_geo_txt(angles, out)
    assert out.read_text().splitlines()[1].split()[4:] == ["90.00", "-1.00", "2.00"]
    accuracy_only = [Shot("a.jpg", 28.1, 113.2, 80.0, h_acc_m=0.02, v_acc_m=0.03)]
    write_geo_txt(accuracy_only, out)             # 9 fields, unknown angles as zeros
    assert out.read_text().splitlines()[1].split()[4:] == ["0.00", "0.00", "0.00", "0.02", "0.03"]


# ---- judging the flight

def _shots(agl=40.0, exposure=1 / 1000, **kw):
    return [Shot(name, lat, lon, alt_m=80, agl_m=agl, time=t, exposure_s=exposure) for name, lat, lon, t in lawnmower(**kw)]


def test_flight_geometry_of_a_grid_flight():
    g = flight_geometry(_shots())
    assert (g.n_shots, g.n_lines) == (100, 5)
    assert g.spacing_m == pytest.approx(5, abs=0.01) and g.line_spacing_m == pytest.approx(10, abs=0.01)
    assert g.speed_ms == pytest.approx(5, abs=0.01) and g.height_m == 40


def test_speed_ignores_skipped_photos_and_turns():
    shots = _shots()
    gappy = [s for i, s in enumerate(shots) if i % 20 not in (7, 8, 9)]      # three photos lost on every line
    g = flight_geometry(gappy)
    assert g.speed_ms == pytest.approx(5, abs=0.05) and g.spacing_m == pytest.approx(5, abs=0.01)


def test_flight_geometry_orders_by_time_not_name():
    shots = _shots()
    shuffled = [Shot(f"z{99 - i:03d}.jpg", s.lat, s.lon, s.alt_m, s.agl_m, s.time) for i, s in enumerate(shots)]
    g = flight_geometry(shuffled[::-1])
    assert g.n_lines == 5 and g.spacing_m == pytest.approx(5, abs=0.01)


def _status(checks):
    return {c.name: c.status for c in checks}


def test_a_good_flight_passes():
    st = _status(flight_checks(_shots(), [], get_profile("pi_gs_6mm")))
    assert st["Forward overlap"] == st["Side overlap"] == st["Motion blur"] == st["Rolling-shutter smear"] == "ok"
    assert "fail" not in st.values() and "warn" not in st.values()


def test_sparse_photos_fail_overlap():
    st = _status(flight_checks(_shots(spacing=12.0, line_spacing=22.0, per_line=10), [], get_profile("pi_gs_6mm")))
    assert st["Forward overlap"] == "fail" and st["Side overlap"] == "fail"


def test_rolling_shutter_and_slow_exposure_are_flagged():
    checks = flight_checks(_shots(exposure=1 / 100), [], get_profile("pi_cm3_full"))
    st = _status(checks)
    assert st["Rolling-shutter smear"] == "warn" and st["Motion blur"] == "fail"
    smear = next(c for c in checks if c.name == "Rolling-shutter smear")
    assert "68 ms" in smear.detail and smear.value == "29 px"


def test_photos_without_position_fail_the_flight():
    checks = flight_checks(_shots(), ["x.jpg", "y.jpg"], get_profile("pi_gs_6mm"))
    first = checks[0]
    assert first.status == "fail" and first.value == "100 of 102" and "x.jpg" in first.detail


def test_unknown_height_is_reported_not_guessed():
    shots = [Shot(s.file, s.lat, s.lon, s.alt_m, None, s.time) for s in _shots()]
    st = _status(flight_checks(shots, [], get_profile("pi_gs_6mm")))
    assert st["Overlap, blur and smear"] == "warn" and "Forward overlap" not in st
    st = _status(flight_checks(shots, [], get_profile("pi_gs_6mm"), height_m=40))     # planned height given
    assert st["Forward overlap"] == "ok"


def test_too_few_photos():
    st = _status(flight_checks(_shots()[:2], [], get_profile("pi_gs_6mm")))
    assert st["Photo count"] == "fail"
