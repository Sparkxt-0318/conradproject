"""Command line: soilsight sim benchmark <experiment> [options]."""
from __future__ import annotations

import argparse
import sys

from soilsight.sim import benchmark
from soilsight.sim.fields import SCENARIOS


def _csv(kind):
    return lambda s: tuple(kind(x) for x in s.split(","))


def _benchmark(a) -> int:
    kwargs = {}
    if a.experiment == "validation":
        if a.scenarios or a.reps or a.budgets:
            print("validation takes no --scenarios, --reps or --budgets", file=sys.stderr)
            return 2
        reps = None
    else:
        d = benchmark.DEFAULTS[a.experiment]
        unknown = [s for s in (a.scenarios or ()) if s not in SCENARIOS]
        if unknown:
            print(f"unknown scenario(s) {', '.join(unknown)}; choose from {', '.join(SCENARIOS)}", file=sys.stderr)
            return 2
        if a.budgets and a.experiment != "budget":
            print("--budgets only applies to the budget experiment", file=sys.stderr)
            return 2
        reps = a.reps or d["reps"]
        kwargs = dict(scenarios=a.scenarios or d["scenarios"], reps=reps, jobs=a.jobs or benchmark.default_jobs(),
                      log=lambda msg: print(msg, file=sys.stderr, flush=True))
        if a.experiment == "budget":
            kwargs["budgets"] = a.budgets or d["budgets"]
        if a.write_expected and (a.scenarios or a.reps or a.budgets):
            print("--write-expected needs the default scenarios, reps and budgets", file=sys.stderr)
            return 2

    expected = None
    if a.check:
        expected = benchmark.load_expected(a.experiment)
        if reps != expected["reps"]:
            print(f"--check needs --reps {expected['reps']} (the reference run); got {reps}", file=sys.stderr)
            return 2

    rows = benchmark.RUNNERS[a.experiment](**kwargs)
    print(benchmark.format_table(a.experiment, rows))
    if a.experiment == "brain":
        print("\ngate (single-core tests):")
        for passed, text in benchmark.gate(rows):
            print(f"  {'PASS' if passed else 'FAIL'}  {text}")

    if a.write_expected:
        benchmark.save_expected(a.experiment, rows, reps)
        print(f"\nreference results for '{a.experiment}' rewritten", file=sys.stderr)
    if expected is not None:
        problems = benchmark.compare(rows, expected["rows"])
        if problems:
            print(f"\nCHECK FAILED: {len(problems)} value(s) differ from the reference by more than "
                  f"{benchmark.CHECK_TOL}:", file=sys.stderr)
            for p in problems[:20]:
                print("  " + p, file=sys.stderr)
            return 1
        print(f"\ncheck passed: {len(rows)} rows match the reference within {benchmark.CHECK_TOL}", file=sys.stderr)
    return 0


def _scout_check(a) -> int:
    from soilsight.field import FieldDir
    from soilsight.report.checks import worst
    from soilsight.scout import check_flight

    _, _, checks = check_flight(FieldDir(a.field))
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"{c.status:5s} {c.name:{width}s}  {c.value}" + (f"  ({c.detail})" if c.detail else ""))
    return 1 if worst(checks) == "fail" else 0


def _scout_run(a) -> int:
    from soilsight.field import FieldDir
    from soilsight.scout import run

    result = run(FieldDir(a.field), node_url=a.node, skip_stitch=a.skip_stitch, fast=a.fast, force=a.force,
                 log=lambda msg: print(msg, file=sys.stderr, flush=True))
    for c in result.checks:
        if c.status in ("warn", "fail"):
            print(f"{c.status:5s} {c.stage}: {c.name}: {c.value}" + (f"  ({c.detail})" if c.detail else ""))
    print(f"overall: {result.overall}" + (f"  {result.stopped}" if result.stopped else ""))
    return 1 if result.overall == "fail" else 0


def _capture_profile(a):
    from soilsight.field import FieldDir
    from soilsight.ingest import get_profile
    from soilsight.ingest.sensors import SensorProfile

    if a.sensor:
        return get_profile(a.sensor)
    field = getattr(a, "field", None)
    if field:
        cfg = FieldDir(field).config()
        if cfg.camera:
            return SensorProfile(**({"name": "custom"} | cfg.camera))
        if cfg.sensor:
            return get_profile(cfg.sensor)
    raise SystemExit("name the camera with --sensor (or set it in the field's field.toml)")


def _capture_plan(a) -> int:
    from soilsight.capture.plan import boundary_lonlat, flight_plan, survey_mission, write_waypoints

    profile = _capture_profile(a)
    p = flight_plan(profile, a.height, a.forward, a.side, a.speed, a.exposure)
    print(f"camera            {profile.name}")
    print(f"flying height     {p.height_m:g} m")
    print(f"ground pixel      {p.gsd_m * 100:.1f} cm")
    print(f"one photo covers  {p.footprint_m[0]:.0f} m across the flight line, {p.footprint_m[1]:.0f} m along it")
    print(f"photo every       {p.photo_spacing_m:.1f} m  ({p.interval_s:.2f} s at {p.speed_ms:.1f} m/s) for {a.forward:.0%} forward overlap")
    print(f"flight lines      {p.line_spacing_m:.1f} m apart for {a.side:.0%} side overlap")
    print(f"motion blur       {p.blur_px:.1f} px at 1/{1 / a.exposure:.0f} s")
    print(f"shutter smear     " + ("none (global shutter)" if profile.shutter == "global" else
                                   "unknown (no readout time in the profile)" if p.smear_px != p.smear_px else f"{p.smear_px:.0f} px"))
    print(f"fastest speed     {p.max_speed_ms:.1f} m/s, limited by {p.limited_by}")
    print(f"covers            {p.area_ha_per_min():.2f} ha a minute on the lines")
    too_fast = p.speed_ms > p.max_speed_ms * 1.02
    if too_fast:
        print(f"\n{p.speed_ms:.1f} m/s is too fast for this camera at this height: slow to {p.max_speed_ms:.1f} m/s or fly higher")
    if a.mission:
        if not a.boundary:
            print("--mission needs --boundary", file=sys.stderr)
            return 2
        items = survey_mission(boundary_lonlat(a.boundary), p, autopilot_trigger=a.autopilot_trigger, angle_deg=a.angle)
        write_waypoints(items, a.mission)
        lines = sum(it["command"] == 16 for it in items) // 2
        print(f"\nmission with {lines} flight lines written to {a.mission}")
    return 1 if too_fast else 0


def _capture_run(a) -> int:
    import signal

    from soilsight.capture.camera import FakeCamera, PiCamera
    from soilsight.capture.link import MavLink, run_capture
    from soilsight.capture.plan import flight_plan
    from soilsight.capture.session import Session
    from soilsight.capture.trigger import AutopilotTrigger, DistanceTrigger, IntervalTrigger
    from soilsight.field import FieldDir

    profile = _capture_profile(a)
    if a.trigger == "autopilot":
        trigger = AutopilotTrigger()
    elif a.trigger == "interval":
        trigger = IntervalTrigger(a.interval, a.min_height)
    else:
        spacing = a.spacing or flight_plan(profile, a.height, a.forward).photo_spacing_m
        trigger = DistanceTrigger(spacing, a.min_height)
    camera = FakeCamera(profile, a.exposure) if a.camera == "fake" else PiCamera(profile, a.exposure, a.gain)
    stopping = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.append(1))
    say = lambda msg: print(msg, file=sys.stderr, flush=True)
    link = MavLink(a.connect, a.baud)
    session = Session(FieldDir(a.field), camera, trigger)
    say(f"capturing to {a.field}; stop with Ctrl-C")
    taken = run_capture(session, link, stop=lambda: bool(stopping), log=say)
    link.close()
    print(f"{taken} photos; {session.unplaced} without a position; {session.skipped_no_fix} skipped for lack of a GPS fix")
    return 1 if session.unplaced else 0


def _capture_bench(a) -> int:
    from soilsight.capture.bench import bench_camera
    from soilsight.capture.camera import FakeCamera, PiCamera

    profile = _capture_profile(a)
    camera = FakeCamera(profile, a.exposure) if a.camera == "fake" else PiCamera(profile, a.exposure, a.gain)
    r = bench_camera(camera, a.out, a.count, a.interval, log=lambda msg: print(msg, file=sys.stderr, flush=True))
    print(f"photos                 {r.photos}")
    print(f"time between photos    {r.interval_median_s:.2f} s typical, {r.interval_worst_s:.2f} s at worst")
    if a.interval:
        print(f"late photos            {r.late} (more than 20% after their time)")
    else:
        print(f"safe photo interval    {r.sustainable_interval_s:.2f} s (the flight plan assumes 1.00 s)")
    print(f"soft photos            {len(r.soft)}" + (f": {', '.join(r.soft[:8])}{'...' if len(r.soft) > 8 else ''}" if r.soft else ""))
    print("\nA soft photo has under half the fine detail of the sharpest ones. With a fixed scene and a still camera")
    print("there should be none; soft photos while the mount is tapped mean the lens or mount moves.")
    return 1 if r.soft or r.late else 0


def _capture_simulate(a) -> int:
    import numpy as np

    from soilsight.capture.plan import boundary_lonlat, flight_plan
    from soilsight.capture.sitl import simulate_flight
    from soilsight.field import FieldDir
    from soilsight.geo import metres_per_degree

    profile = _capture_profile(a)
    plan = flight_plan(profile, a.height, a.forward, a.side, a.speed)
    if a.boundary:
        boundary = boundary_lonlat(a.boundary)
    else:                                         # a 60 x 40 m field in Hunan
        m_lon, m_lat = metres_per_degree(28.2)
        boundary = np.array([(113.0 + x / m_lon, 28.2 + y / m_lat) for x, y in ((0, 0), (60, 0), (60, 40), (0, 40), (0, 0))])
    fd = FieldDir(a.field)
    fd.root.mkdir(parents=True, exist_ok=True)
    if not fd.config_path.exists() and a.sensor:
        fd.config_path.write_text(f'name = "Simulated survey"\nsensor = "{a.sensor}"\n')
    r = simulate_flight(fd, a.sitl, a.defaults, boundary, plan, profile, autopilot_trigger=a.autopilot_trigger,
                        log=lambda msg: print(msg, file=sys.stderr, flush=True))
    print(f"photos                 {r.photos}")
    print(f"without a position     {r.unplaced}")
    print(f"placed by carrying on  {r.extrapolated}")
    print(f"flight time            {r.flight_s:.0f} s")
    print(f"position error         {r.error_median_m * 100:.0f} cm typical, under {r.error_p95_m * 100:.0f} cm for 19 photos in 20 "
          "(against the simulator's true position)")
    if r.log_triggers is not None:
        print(f"triggers in flight log {r.log_triggers}")
    print(f"\nnow judge the flight:  soilsight scout check {a.field}")
    return 1 if r.unplaced or not r.photos else 0


def _app_package(a) -> int:
    from soilsight.app.package import package_from_field, save_package
    from soilsight.field import FieldDir

    fd = FieldDir(a.field)
    if not fd.layers_json.exists():
        print(f"{fd.root} has no map layers yet: run 'soilsight scout run {a.field}' first", file=sys.stderr)
        return 2
    package = package_from_field(fd)
    out = a.out or fd.package
    save_package(package, out)
    p = package["protocol"]
    print(f"field package written to {out}")
    print(f"  {len(package['grid']['rows'])} cells of {package['grid']['cell_m']:g} m, {len(package['zones']['names'])} zones, "
          f"{p['budget']} tests of {p['cores']} cores, first {len(package['plan']['initial'])} spots planned")
    if package["threshold"] is None:
        print("  no threshold set: the app will ask for one at the start of the visit (or set threshold in field.toml)")
    return 0


def _app_serve(a) -> int:
    from soilsight.app.server import make_server

    server = make_server(a.host, a.port, a.package)
    where = "localhost" if a.host in ("127.0.0.1", "localhost") else a.host
    print(f"field app at http://{where}:{a.port}/" + (f" with {a.package}" if a.package else "") + "; stop with Ctrl-C", file=sys.stderr)
    if a.host not in ("127.0.0.1", "localhost"):
        print("a phone opening this address gets no GPS and no offline start: both need an https address", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def _app_report(a) -> int:
    import json
    import shutil

    from soilsight.app.package import load_package
    from soilsight.app.record import RecordError, read_log
    from soilsight.app.visit import replay
    from soilsight.field import FieldDir
    from soilsight.report.health import WORD, write_health_report, write_results_geojson

    fd = FieldDir(a.field)
    package_path = a.package or fd.package
    try:
        package = load_package(package_path)
    except FileNotFoundError:
        print(f"no field package at {package_path}: make one with 'soilsight app package {a.field}'", file=sys.stderr)
        return 2
    try:
        entries = read_log(a.log)
    except RecordError as err:
        print(f"the log cannot be trusted: {err}", file=sys.stderr)
        return 1
    started = next((e.data.get("package") for e in entries if e.type == "visit_started"), None)
    if started and (started.get("name"), started.get("made")) != (package["name"], package["made"]):
        print(f"this log belongs to another field package ({started.get('name')}, made {started.get('made')})", file=sys.stderr)
        return 1
    visit = replay(package, entries)
    if not visit.tests:
        print("the log holds no test results yet", file=sys.stderr)
        return 1
    fd.visit_log.parent.mkdir(parents=True, exist_ok=True)
    if fd.visit_log.resolve() != type(fd.visit_log)(a.log).resolve():
        shutil.copy(a.log, fd.visit_log)
    checks = json.loads(fd.quality_json.read_text())["checks"] if fd.quality_json.exists() else None
    doc = write_health_report(fd.health_html, visit, checks)
    points = write_results_geojson(fd.results_geojson, visit)
    print(f"log verified: {len(entries)} entries, {len(visit.tests)} tests" + ("" if visit.finished else " (visit not marked finished)"))
    for z in doc["zones"]:
        print(f"  zone {z['zone']}: {WORD[z['label']].lower():16s} chance unsafe {z['p_unsafe'] * 100:3.0f}%")
    print(f"report: {fd.health_html}")
    if points:
        print(f"test points for QGIS: {fd.results_geojson}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="soilsight", description="SoilSight tools")
    sub = ap.add_subparsers(dest="command", required=True)
    sim = sub.add_parser("sim", help="synthetic fields and the benchmark").add_subparsers(dest="sim_command", required=True)
    b = sim.add_parser("benchmark", help="run a benchmark experiment and print its table",
                       description="Run a benchmark experiment on synthetic fields. "
                                   "On four cores methods, budget and brain each take a few minutes "
                                   "(brain about 6); validation takes a second.")
    b.add_argument("experiment", choices=sorted(benchmark.RUNNERS))
    b.add_argument("--scenarios", type=_csv(str), help="comma-separated ids, e.g. S0,S7 (default: the reference set)")
    b.add_argument("--reps", type=int, help="fields per scenario (default: the reference number)")
    b.add_argument("--budgets", type=_csv(int), help="budget experiment only: tests per field, e.g. 15,25")
    b.add_argument("--jobs", type=int, help="worker processes (default: all cores but one); results do not depend on it")
    b.add_argument("--check", action="store_true", help="fail if results differ from the stored reference")
    b.add_argument("--write-expected", action="store_true", help="overwrite the stored reference with this run")
    b.set_defaults(func=_benchmark)

    scout = sub.add_parser("scout", help="from drone photos to map layers").add_subparsers(dest="scout_command", required=True)
    c = scout.add_parser("check", help="judge a flight from its photo positions, before stitching")
    c.add_argument("field", help="the field folder (holds photos/)")
    c.set_defaults(func=_scout_check)
    r = scout.add_parser("run", help="stitch the photos, build the layers and write the quality report",
                         description="Turn a field folder's photos into map layers and a quality report. "
                                     "Stitching needs a NodeODM server: docker run -d -p 3000:3000 opendronemap/nodeodm")
    r.add_argument("field", help="the field folder (holds photos/)")
    r.add_argument("--node", default="http://localhost:3000", help="NodeODM address (default: %(default)s)")
    r.add_argument("--skip-stitch", action="store_true", help="use the orthophoto and surface model already in odm/")
    r.add_argument("--fast", action="store_true", help="lower stitching quality, for a quick trial")
    r.add_argument("--force", action="store_true", help="stitch even if the flight checks fail")
    r.set_defaults(func=_scout_run)
    cap = sub.add_parser("capture", help="the capture computer on the drone").add_subparsers(dest="capture_command", required=True)

    def flight_args(p, speed=True):
        p.add_argument("--sensor", help="camera profile: pi_gs_6mm, pi_cm3_binned or pi_cm3_full")
        p.add_argument("--height", type=float, default=40.0, help="flying height above ground, metres (default: %(default)s)")
        p.add_argument("--forward", type=float, default=0.8, help="overlap between consecutive photos (default: %(default)s)")
        p.add_argument("--side", type=float, default=0.7, help="overlap between flight lines (default: %(default)s)")
        p.add_argument("--exposure", type=float, default=0.001, help="shutter time, seconds (default: %(default)s)")
        if speed:
            p.add_argument("--speed", type=float, help="ground speed, m/s (default: the fastest the camera allows, up to 5)")

    pl = cap.add_parser("plan", help="flight settings for a camera, and optionally a survey mission file")
    flight_args(pl)
    pl.add_argument("--boundary", help="annotations.geojson holding the field boundary")
    pl.add_argument("--mission", help="write a survey mission (QGC WPL file) here")
    pl.add_argument("--angle", type=float, help="direction of the flight lines, degrees anticlockwise from east (default: along the field)")
    pl.add_argument("--autopilot-trigger", action="store_true", help="have the flight controller trigger the camera")
    pl.set_defaults(func=_capture_plan)

    ru = cap.add_parser("run", help="take photos in flight and record where each was taken",
                        description="Runs on the capture computer. Photos go to FIELD/photos and their positions to FIELD/capture.csv.")
    ru.add_argument("field", help="the field folder to capture into")
    ru.add_argument("--connect", default="/dev/serial0", help="MAVLink address of the flight controller (default: %(default)s)")
    ru.add_argument("--baud", type=int, default=115200)
    ru.add_argument("--camera", choices=("pi", "fake"), default="pi")
    ru.add_argument("--trigger", choices=("distance", "interval", "autopilot"), default="distance")
    ru.add_argument("--spacing", type=float, help="metres between photos (default: from the camera, height and overlap)")
    ru.add_argument("--interval", type=float, default=1.0, help="seconds between photos, for --trigger interval")
    ru.add_argument("--min-height", type=float, default=20.0, help="take no photos below this height, metres (default: %(default)s)")
    ru.add_argument("--gain", type=float, default=1.0, help="camera analogue gain (default: %(default)s)")
    flight_args(ru, speed=False)
    ru.set_defaults(func=_capture_run)

    be = cap.add_parser("bench", help="bench-test a real camera: photo rate and sharpness",
                        description="Take a run of photos with the camera on a desk or pole and measure how fast it can "
                                    "shoot and whether the photos stay sharp. Run it once still and once while tapping the mount.")
    be.add_argument("out", help="folder for the test photos")
    be.add_argument("--sensor", help="camera profile: pi_gs_6mm, pi_cm3_binned or pi_cm3_full")
    be.add_argument("--camera", choices=("pi", "fake"), default="pi")
    be.add_argument("--count", type=int, default=30, help="photos to take (default: %(default)s)")
    be.add_argument("--interval", type=float, default=0.0, help="seconds between photos; 0 means as fast as possible (default)")
    be.add_argument("--exposure", type=float, default=0.001, help="shutter time, seconds (default: %(default)s)")
    be.add_argument("--gain", type=float, default=1.0, help="camera analogue gain (default: %(default)s)")
    be.set_defaults(func=_capture_bench)

    si = cap.add_parser("simulate", help="fly a survey in ArduPilot's simulator and capture it",
                        description="A rehearsal: the real ArduCopter code flies the survey in simulation while the capture "
                                    "software records it with a stand-in camera. Takes about four minutes of real time.")
    si.add_argument("field", help="the field folder to capture into")
    si.add_argument("--sitl", required=True, help="path to the arducopter simulator binary (build/sitl/bin/arducopter)")
    si.add_argument("--defaults", required=True, help="path to ArduPilot's copter.parm default parameters")
    si.add_argument("--boundary", help="annotations.geojson holding the field boundary (default: a 60 x 40 m test field)")
    si.add_argument("--autopilot-trigger", action="store_true", help="have the flight controller trigger the camera")
    flight_args(si)
    si.set_defaults(func=_capture_simulate)

    app = sub.add_parser("app", help="the phone app for a field visit, and its report").add_subparsers(dest="app_command", required=True)
    pk = app.add_parser("package", help="make the file a phone needs for a visit to a scouted field")
    pk.add_argument("field", help="the field folder (must hold map layers)")
    pk.add_argument("-o", "--out", help="where to write it (default: FIELD/visit/field.soilsight.json)")
    pk.set_defaults(func=_app_package)
    sv = app.add_parser("serve", help="serve the app on this computer, to try it")
    sv.add_argument("--package", help="a field package for the app to load by itself")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to let other devices on the network reach it")
    sv.set_defaults(func=_app_serve)
    rp = app.add_parser("report", help="check a visit's log and make the Field Health Report")
    rp.add_argument("field", help="the field folder")
    rp.add_argument("log", help="the log file exported from the app")
    rp.add_argument("--package", help="the field package the visit used (default: FIELD/visit/field.soilsight.json)")
    rp.set_defaults(func=_app_report)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
