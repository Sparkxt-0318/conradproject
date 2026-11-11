import numpy as np
import pytest

from capture_helpers import FakeVehicle, SimClock, fly, grid_path
from soilsight.capture import (AutopilotTrigger, DistanceTrigger, FakeCamera, IntervalTrigger, Session, State, Telemetry,
                               distance_m, flight_plan, survey_lines, survey_mission, write_waypoints)
from soilsight.capture.plan import boundary_lonlat
from soilsight.field import FieldDir
from soilsight.geo import metres_per_degree
from soilsight.ingest import flight_checks, get_profile, read_sidecar
from soilsight.scout import check_flight


M_LON, M_LAT = metres_per_degree(28.0)


def test_metres_per_degree():
    assert metres_per_degree(0.0) == pytest.approx((111319.5, 110574.3), abs=1.0)        # equator
    assert M_LAT > 110574.3 and M_LON < 111319.5 * np.cos(np.radians(28.0)) * 1.002      # lat degrees lengthen polewards
    # checked against a map projection: 100 m east and north of a point at 28 N
    from rasterio.warp import transform
    x, y = transform("EPSG:4326", "EPSG:32649", [113.0, 113.0 + 100 / M_LON, 113.0], [28.0, 28.0, 28.0 + 100 / M_LAT])
    scale = 0.9996 * (1 + ((x[0] - 500000) / 6378137) ** 2 / 2)                          # UTM scale factor here
    assert np.hypot(x[1] - x[0], y[1] - y[0]) / scale == pytest.approx(100, abs=0.02)
    assert np.hypot(x[2] - x[0], y[2] - y[0]) / scale == pytest.approx(100, abs=0.02)


def _pos(boot_ms, lat=28.0, lon=113.0, agl=40.0, vn=0.0, ve=0.0):
    return dict(time_boot_ms=boot_ms, lat=int(lat * 1e7), lon=int(lon * 1e7), alt=int((60 + agl) * 1000),
                relative_alt=int(agl * 1000), vx=int(vn * 100), vy=int(ve * 100), vz=0, hdg=0)


# ---- telemetry

def test_clock_offset_is_the_smallest_delay_seen():
    t = Telemetry()
    assert t.clock_offset_s is None and t.state_at(10.0) is None
    for boot_ms, delay in ((1000, 0.05), (1100, 0.02), (1200, 0.08), (1300, 0.03)):
        t.feed("GLOBAL_POSITION_INT", _pos(boot_ms), 500.0 + boot_ms / 1000 + delay)
    assert t.clock_offset_s == pytest.approx(500.02)
    assert t.boot_ms_at(501.52) == pytest.approx(1500)


def test_position_is_interpolated_between_samples():
    t = Telemetry()
    t.feed("GLOBAL_POSITION_INT", _pos(1000, lat=28.0, ve=5.0), 101.0)
    t.feed("GLOBAL_POSITION_INT", _pos(1100, lat=28.0001, ve=5.0), 101.1)
    s = t.state_at(101.025)
    assert s.lat == pytest.approx(28.000025, abs=1e-9) and s.exact and s.gap_s == pytest.approx(0.025)
    assert s.speed_ms == pytest.approx(5.0) and s.agl_m == 40.0 and s.alt_m == 100.0
    assert t.covers(101.05) and not t.covers(101.2)


def test_position_is_extrapolated_briefly_then_given_up():
    t = Telemetry()
    t.feed("GLOBAL_POSITION_INT", _pos(1000, vn=5.0), 101.0)
    t.feed("GLOBAL_POSITION_INT", _pos(1100, vn=5.0), 101.1)
    s = t.state_at(101.3)                          # 0.2 s past the newest sample
    assert not s.exact and s.lat == pytest.approx(28.0 + 5.0 * 0.2 / M_LAT, abs=1e-8)
    assert t.state_at(102.0) is None               # too long without a sample


def test_attitude_wraps_through_north_and_time_comes_from_the_controller():
    t = Telemetry()
    for boot_ms, yaw in ((1000, np.radians(350)), (1100, np.radians(10))):
        t.feed("GLOBAL_POSITION_INT", _pos(boot_ms), boot_ms / 1000 + 100)
        t.feed("ATTITUDE", dict(time_boot_ms=boot_ms, roll=0.0, pitch=0.1, yaw=yaw), boot_ms / 1000 + 100)
    yaw = t.state_at(101.05).yaw_deg
    assert min(yaw, 360 - yaw) < 1e-6                                          # north, not 180
    assert t.state_at(101.05).pitch_deg == pytest.approx(np.degrees(0.1))
    assert t.state_at(101.05).time_utc is None
    t.feed("SYSTEM_TIME", dict(time_unix_usec=0, time_boot_ms=1100), 101.1)    # no GPS time yet
    assert t.state_at(101.05).time_utc is None
    t.feed("SYSTEM_TIME", dict(time_unix_usec=1_794_366_000_000_000, time_boot_ms=1100), 101.1)
    utc = t.state_at(101.05).time_utc
    assert (utc.year, utc.microsecond) == (2026, 950000)                       # 50 ms before the stamped instant


def test_gps_quality_is_carried():
    t = Telemetry()
    t.feed("GLOBAL_POSITION_INT", _pos(1000), 101.0)
    t.feed("GPS_RAW_INT", dict(fix_type=6, h_acc=14, v_acc=0, satellites_visible=20), 101.0)
    s = t.state_at(101.0)
    assert (s.fix_type, s.h_acc_m, s.v_acc_m) == (6, 0.014, None)


def test_distance_between_states():
    a = State(28.0, 113.0, 100, 40, 0)
    b = State(28.0 + 30 / M_LAT, 113.0 + 40 / M_LON, 100, 40, 0)
    assert distance_m(a, b) == pytest.approx(50, abs=0.01)


# ---- triggers

def test_distance_trigger():
    trig = DistanceTrigger(5.0, min_agl_m=20)
    low = State(28.0, 113.0, 70, 10, 0)
    here = State(28.0, 113.0, 100, 40, 0)
    assert not trig.due(low, 0) and trig.due(here, 0)             # first photo once high enough
    trig.fired(here, 0)
    near = State(28.0 + 4.9 / M_LAT, 113.0, 100, 40, 5)
    far = State(28.0 + 5.1 / M_LAT, 113.0, 100, 40, 5)
    assert not trig.due(near, 1) and trig.due(far, 1)
    with pytest.raises(ValueError):
        DistanceTrigger(0)


def test_interval_trigger():
    trig = IntervalTrigger(1.0)
    s = State(28.0, 113.0, 100, 40, 0)
    assert trig.due(s, 10.0)
    trig.fired(s, 10.0)
    assert not trig.due(s, 10.9) and trig.due(s, 11.0)


def test_autopilot_trigger_counts_camera_commands():
    trig = AutopilotTrigger()
    s = State(28.0, 113.0, 100, 40, 0)
    assert not trig.due(s, 0)
    trig.command(203, (0, 0, 0, 0, 1, 0, 0))       # DO_DIGICAM_CONTROL, shoot
    trig.command(203, (0, 0, 0, 0, 0, 0, 0))       # same command, not a shot
    trig.command(511, (33, 100000, 0, 0, 0, 0, 0))  # unrelated
    assert trig.due(s, 0)
    trig.fired(s, 0)
    assert not trig.due(s, 0)


# ---- a whole flight

def _flight(tmp_path, camera_latency=0.0, **vehicle):
    clock = SimClock()
    v = FakeVehicle(grid_path(), **vehicle)
    fd = FieldDir(tmp_path / "field")
    cam = FakeCamera(get_profile("pi_gs_6mm"), clock=clock, latency_s=camera_latency)
    session = Session(fd, cam, DistanceTrigger(5.0), clock=clock)
    fly(session, v, clock)
    return fd, session, v


def _errors(session, vehicle):
    out = []
    for shot in session.shots:
        true = vehicle.true_xy_at_mono(shot.exposure.mid_mono)
        lon, lat = vehicle.lonlat(true)
        out.append(distance_m(shot.state, State(lat, lon, 0, 0, 0)))
    return np.array(out)


def test_every_photo_gets_a_position(tmp_path):
    fd, session, v = _flight(tmp_path)
    rows = read_sidecar(fd.sidecar)
    photos = sorted(p.name for p in fd.photos.iterdir())
    assert len(photos) == len(rows) == len(session.shots) > 50
    assert [r.file for r in rows] == photos
    assert all(r.lat and r.lon and r.agl_m == 40.0 and r.time is not None and r.yaw_deg is not None for r in rows)
    assert all(s.state.exact for s in session.shots)                # interpolated, never extrapolated
    assert rows[0].h_acc_m == 0.8 and rows[0].exposure_s == pytest.approx(0.001)


def test_positions_are_right_to_centimetres_on_a_clean_link(tmp_path):
    _, session, v = _flight(tmp_path)
    err = _errors(session, v)
    straight = err[[abs(s.state.speed_ms - 5.0) < 0.01 for s in session.shots]]
    assert np.median(straight) < 0.02
    assert np.percentile(err, 95) < 0.6        # corners: a straight line between samples cuts them


def test_link_delay_and_jitter_cost_centimetres_not_metres(tmp_path):
    _, session, v = _flight(tmp_path, latency_s=0.02, jitter_s=0.05)
    err = _errors(session, v)
    straight = err[[abs(s.state.speed_ms - 5.0) < 0.01 for s in session.shots]]
    # The jitter is filtered out; the fixed 20 ms delay stays, worth 10 cm at 5 m/s.
    assert 0.05 < np.median(straight) < 0.20


def test_a_slow_camera_is_still_placed_where_it_exposed(tmp_path):
    _, session, v = _flight(tmp_path, camera_latency=0.15)          # exposes 15 cm... 0.75 m after being asked
    straight = _errors(session, v)[[abs(s.state.speed_ms - 5.0) < 0.01 for s in session.shots]]
    assert np.median(straight) < 0.02


def test_photos_are_evenly_spaced_and_pass_the_flight_checks(tmp_path):
    fd, session, v = _flight(tmp_path)
    fd.config_path.write_text('sensor = "pi_gs_6mm"\n')
    shots, profile, checks = check_flight(fd)
    st = {c.name: c for c in checks}
    assert st["Photos with a position"].status == "ok"
    assert st["Forward overlap"].status == "ok" and int(st["Forward overlap"].value.rstrip("%")) >= 79
    assert st["Side overlap"].status == "ok" and st["Rolling-shutter smear"].value == "none"
    assert st["Ground speed"].value == "5.0 m/s" and "fail" not in {c.status for c in checks}


def test_no_photos_below_the_minimum_height_or_without_a_fix(tmp_path):
    clock = SimClock()
    low = Session(FieldDir(tmp_path / "low"), FakeCamera(clock=clock), DistanceTrigger(5.0, min_agl_m=20), clock=clock)
    fly(low, FakeVehicle(grid_path(lines=1), height=10.0), clock)
    assert low.shots == [] and not list(low.field.photos.iterdir())

    clock = SimClock()
    nofix = Session(FieldDir(tmp_path / "nofix"), FakeCamera(clock=clock), DistanceTrigger(5.0), clock=clock)
    fly(nofix, FakeVehicle(grid_path(lines=1), fix_type=1), clock)
    assert nofix.shots == [] and nofix.skipped_no_fix > 5


def test_rows_reach_the_disk_as_the_flight_goes(tmp_path):
    clock = SimClock()
    v = FakeVehicle(grid_path(lines=1))
    fd = FieldDir(tmp_path / "field")
    session = Session(fd, FakeCamera(clock=clock), DistanceTrigger(5.0), clock=clock)
    clock.now = v.t0 + v.offset - 1.0
    while len(session.shots) < 4:                   # stop mid-flight, as a power cut would
        for msg in v.messages_until(clock.now):
            session.handle(*msg)
        session.step()
        clock.now += 0.005
    on_disk = fd.sidecar.read_text().splitlines()   # read without closing the session
    assert len(on_disk) == 1 + 4 and on_disk[1].startswith("img_000001.jpg,2026-11-11T")


def test_a_second_flight_continues_the_numbering(tmp_path):
    fd, first, _ = _flight(tmp_path)
    n = len(first.shots)
    clock = SimClock()
    again = Session(fd, FakeCamera(clock=clock), DistanceTrigger(5.0), clock=clock)
    fly(again, FakeVehicle(grid_path(lines=1)), clock)
    names = sorted(p.name for p in fd.photos.iterdir())
    assert len(names) == n + len(again.shots) and names[n] == f"img_{n + 1:06d}.jpg"
    assert len(read_sidecar(fd.sidecar)) == len(names)
    assert fd.sidecar.read_text().count("file,time_utc") == 1      # one header


def test_a_photo_with_no_position_is_listed_not_hidden(tmp_path):
    clock = SimClock()
    fd = FieldDir(tmp_path / "field")
    slow = FakeCamera(clock=clock, latency_s=2.0)   # exposes two seconds after being asked
    session = Session(fd, slow, IntervalTrigger(1.0), clock=clock, settle_s=0.5)
    session.handle("GLOBAL_POSITION_INT", _pos(1000), clock.now)
    assert session.step() is not None               # photo taken, then the link goes silent
    clock.now += 5.0
    session.close()
    line = fd.sidecar.read_text().splitlines()[1]
    assert line.startswith("img_000001.jpg,,,,") and session.shots[0].state is None and session.unplaced == 1


def test_a_photo_just_after_the_last_sample_is_placed_by_carrying_on(tmp_path):
    clock = SimClock()
    session = Session(FieldDir(tmp_path / "f"), FakeCamera(clock=clock), IntervalTrigger(1.0), clock=clock, settle_s=0.5)
    session.handle("GLOBAL_POSITION_INT", _pos(1000, vn=5.0), clock.now)
    clock.now += 0.2                                # link goes silent, photo 0.2 s later
    assert session.step() is not None
    clock.now += 5.0
    session.close()
    s = session.shots[0].state
    assert s is not None and not s.exact and session.extrapolated == 1
    assert s.lat == pytest.approx(28.0 + 5.0 * 0.2005 / M_LAT, abs=1e-8)


def test_autopilot_trigger_shoots_on_command(tmp_path):
    clock = SimClock()
    session = Session(FieldDir(tmp_path / "f"), FakeCamera(clock=clock), AutopilotTrigger(), clock=clock)
    session.handle("GLOBAL_POSITION_INT", _pos(1000), clock.now)
    assert session.step() is None
    session.handle("COMMAND_LONG", dict(command=203, param1=0, param2=0, param3=0, param4=0, param5=1, param6=0, param7=0), clock.now)
    assert session.step() is not None and session.step() is None


# ---- planning

def test_flight_plan_for_the_global_shutter_camera():
    p = flight_plan(get_profile("pi_gs_6mm"), 40)
    assert p.gsd_m * 100 == pytest.approx(2.3, abs=0.01)
    assert p.photo_spacing_m == pytest.approx(5.0, abs=0.05) and p.line_spacing_m == pytest.approx(10.05, abs=0.05)
    assert p.interval_s == pytest.approx(1.0, abs=0.01) and p.smear_px == 0 and p.blur_px < 0.3
    assert p.limited_by == "photo rate" and p.max_speed_ms == pytest.approx(5.0, abs=0.05)
    assert p.area_ha_per_min() == pytest.approx(0.30, abs=0.01)


def test_rolling_shutter_limits_speed():
    full = flight_plan(get_profile("pi_cm3_full"), 40)
    binned = flight_plan(get_profile("pi_cm3_binned"), 40)
    assert full.limited_by == binned.limited_by == "rolling-shutter smear"
    assert full.max_speed_ms == pytest.approx(0.17, abs=0.01) and binned.max_speed_ms == pytest.approx(1.39, abs=0.02)
    assert full.smear_px == pytest.approx(29, abs=1)
    with pytest.raises(ValueError, match="overlaps"):
        flight_plan(get_profile("pi_gs_6mm"), 40, forward=1.2)


def test_survey_lines_cover_a_rectangle_along_its_long_side():
    rect = [(0, 0), (60, 0), (60, 40), (0, 40), (0, 0)]
    lines = survey_lines(rect, 10.0, overshoot_m=5.0)
    assert len(lines) == 4
    ys = [start[1] for start, _ in lines]
    assert ys == pytest.approx([5, 15, 25, 35])                     # centred, 10 m apart
    assert lines[0][0] == pytest.approx((-5, 5)) and lines[0][1] == pytest.approx((65, 5))
    assert lines[1][0] == pytest.approx((65, 15))                   # back the other way
    north_south = survey_lines(rect, 10.0, angle_deg=90)
    assert len(north_south) == 6 and abs(north_south[0][0][0] - north_south[0][1][0]) < 1e-9


def test_survey_lines_follow_a_slanted_field():
    a = np.radians(30)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rect = np.array([(0, 0), (80, 0), (80, 30), (0, 30)]) @ rot.T
    lines = survey_lines(rect, 10.0)
    assert len(lines) == 3
    d = lines[0][1] - lines[0][0]
    assert np.degrees(np.arctan2(d[1], d[0])) % 180 == pytest.approx(30, abs=0.01)
    assert np.hypot(*d) == pytest.approx(80, abs=0.01)


def test_survey_mission_and_waypoint_file(stitched_field, tmp_path):
    plan = flight_plan(get_profile("pi_gs_6mm"), 40)
    boundary = boundary_lonlat(stitched_field.annotations)
    items = survey_mission(boundary, plan)
    cmds = [it["command"] for it in items]
    assert cmds[:4] == [16, 22, 178, 16] and cmds[-1] == 20 and 206 not in cmds
    assert items[1]["alt"] == 40 and items[2]["params"][1] == 5.0
    assert cmds.count(16) == 2 * 4 + 1                              # home, then both ends of 4 lines over a 60 x 40 m field
    with_trigger = survey_mission(boundary, plan, autopilot_trigger=True)
    trig = [it for it in with_trigger if it["command"] == 206]
    assert [round(t["params"][0], 1) for t in trig] == [5.0, 0.0]   # on before the lines, off after
    out = tmp_path / "survey.waypoints"
    write_waypoints(items, out)
    rows = out.read_text().splitlines()
    assert rows[0] == "QGC WPL 110" and len(rows) == len(items) + 1
    first = rows[1].split("\t")
    assert (first[0], first[1], first[3]) == ("0", "1", "16") and len(first) == 12
    with pytest.raises(ValueError, match="no feature with role 'boundary'"):
        (tmp_path / "empty.geojson").write_text('{"type": "FeatureCollection", "features": []}')
        boundary_lonlat(tmp_path / "empty.geojson")
