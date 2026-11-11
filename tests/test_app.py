import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

from soilsight.app.package import auto_zones, cells_xy, load_package, package_from_sim, save_package
from soilsight.brain import Brain, Protocol
from soilsight.sim.fields import Grid, make_field, observe

NODE = shutil.which("node")
JS = Path(__file__).parent / "js"
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


@pytest.fixture(scope="module")
def sim_field():
    return make_field(np.random.default_rng(5), Grid(60, 40, 2.0))


@pytest.fixture(scope="module")
def package(sim_field):
    return package_from_sim(sim_field, protocol=Protocol(budget=12, cores=3))


# ---- the field package

def test_package_describes_the_field(sim_field, package):
    g = sim_field.grid
    assert package["format"] == "soilsight-field/1" and len(package["grid"]["rows"]) == g.n
    assert np.allclose(cells_xy(package), g.xy + g.cell / 2)             # same cells, measured from the grid corner
    plan = package["plan"]
    assert len(plan["initial"]) == 7 and plan["initial"][:2] == [sim_field.i_in, sim_field.i_out] == plan["fixed"]
    assert len(package["zones"]["samples"]) == 6 and package["zones"]["names"] == list("ABCDEF")
    assert all(len(s) <= 300 for s in package["zones"]["samples"])
    assert len(package["model"]["candidates"]["length_m"]) == 512 and package["threshold"] == 1.0
    assert sum(package["model"]["quadrature"]["w"]) == pytest.approx(1)


def test_package_round_trip_and_format_check(package, tmp_path):
    path = tmp_path / "f.soilsight.json"
    save_package(package, path)
    assert load_package(path) == json.loads(json.dumps(package))
    path.write_text('{"format": "something-else"}')
    with pytest.raises(ValueError, match="not a soilsight-field/1 file"):
        load_package(path)


def test_brain_from_package_matches_a_fresh_brain(sim_field, package):
    g = sim_field.grid
    fresh = Brain(cells_xy(package), fixed_xy=cells_xy(package)[[sim_field.i_in, sim_field.i_out]],
                  protocol=Protocol(budget=12, cores=3))
    packed = Brain.from_package(package)
    assert np.array_equal(packed.initial_spots(), fresh.initial_spots())
    for xy in packed.initial_spots():
        v = float(sim_field.y[packed.nearest_cell(xy)])
        fresh.add_result(xy, v)
        packed.add_result(xy, v)
    assert np.allclose(packed.map().p_unsafe, fresh.map().p_unsafe, atol=1e-12)
    assert np.array_equal(packed.next_spot().xy, fresh.next_spot().xy)


def test_auto_zones_are_compact_and_ordered():
    xy = np.array([(x + 0.5, y + 0.5) for y in range(100) for x in range(200)], float)      # 2 ha
    z = auto_zones(xy)
    assert sorted(np.unique(z)) == [0, 1, 2, 3]                       # about 0.5 ha each
    share = np.bincount(z) / len(z)
    assert share.min() > 0.15
    centres = np.array([xy[z == k].mean(0) for k in range(4)])
    assert centres[0][1] >= centres[3][1]                             # numbered from the north
    assert len(np.unique(auto_zones(xy[:2000]))) == 2                 # never fewer than two


# ---- the phone's Brain against the reference

@needs_node
def test_javascript_brain_matches_python(sim_field, package, tmp_path):
    """The whole loop, step by step: same weights, same map, same zone answers, same next spot."""
    rng = np.random.default_rng(11)
    brain = Brain.from_package(package)
    xy = cells_xy(package)
    steps, expected, tests = [], [], []

    def add(cell_xy):
        cell = brain.nearest_cell(cell_xy)
        x, y = cell_xy + rng.normal(0, 1.5, 2)                        # sampled a little off the planned spot
        value = observe(sim_field, cell, rng, 0.22, cores=3)
        brain.add_result((x, y), value)
        tests.append(dict(x=float(x), y=float(y), value=float(value), noise_sd=0.22, cores=3))

    def snapshot():
        post, pred = brain.posterior(), brain.map()
        zones = [post.area_mean(xy[c]) for c in package["zones"]["samples"]]
        nxt = brain.next_spot()
        steps.append(list(tests))
        expected.append(dict(weights=np.sort(post.weights)[::-1], mean=pred.mean, sd=pred.sd, p=pred.p_unsafe,
                             zone_p=[float(z.p_unsafe) for z in zones], zone_mean=[float(z.mean) for z in zones],
                             next=None if nxt is None else (brain.nearest_cell(nxt.xy), nxt.gain, nxt.misclassified_now)))
        return nxt

    for spot in brain.initial_spots():
        add(spot)
    while (nxt := snapshot()) is not None:
        add(nxt.xy)
    assert len(steps) == 6 and expected[-1]["next"] is None            # 7 first spots, 5 from the loop, then done

    probe = np.linspace(-8, 8, 161)
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps(dict(package=package, steps=steps, ndtr=probe.tolist())))
    out = json.loads(subprocess.run([NODE, str(JS / "run_brain.js"), str(scenario)], capture_output=True, text=True, check=True).stdout)

    assert np.allclose(out["ndtr"], ndtr(probe), rtol=1e-12, atol=1e-300)
    for got, want in zip(out["steps"], expected):
        assert np.allclose(sorted(got["weights"], reverse=True), want["weights"], atol=1e-9)
        assert np.allclose(got["mean"], want["mean"], atol=1e-8) and np.allclose(got["sd"], want["sd"], atol=1e-8)
        assert np.allclose(got["p"], want["p"], atol=1e-8)
        assert np.allclose([z["p"] for z in got["zones"]], want["zone_p"], atol=1e-6)
        assert np.allclose([z["mean"] for z in got["zones"]], want["zone_mean"], atol=1e-6)
        if want["next"] is None:
            assert got["next"] is None
        else:
            assert got["next"]["cell"] == want["next"][0]
            assert got["next"]["gain"] == pytest.approx(want["next"][1], abs=1e-9)
            assert got["next"]["now"] == pytest.approx(want["next"][2], abs=1e-9)
    assert max(s["ms"] for s in out["steps"]) < 5000


# ---- the measurement log

from soilsight.app.record import GENESIS, LogWriter, RecordError, read_log, verify, write_log


def _small_log():
    w = LogWriter()
    w.add("visit_started", dict(threshold=0.05, unit="mg/kg"), "2026-11-11T03:00:00Z")
    w.add("sample_taken", dict(id="S01", cell=3, x=1.5, y=2.5, cores=3), "2026-11-11T03:05:00Z")
    w.add("result_entered", dict(id="S01", reading=0.031, ph=5.4), "2026-11-11T03:20:00Z")
    return w


def test_log_verifies_and_round_trips(tmp_path):
    w = _small_log()
    entries = verify(w.stored)
    assert [e.type for e in entries] == ["visit_started", "sample_taken", "result_entered"]
    assert entries[2].data == dict(id="S01", reading=0.031, ph=5.4) and entries[0].seq == 1
    assert json.loads(w.stored[0]["body"])["prev"] == GENESIS and json.loads(w.stored[1]["body"])["prev"] == w.stored[0]["hash"]
    path = tmp_path / "log.json"
    write_log(w.stored, path, "North paddy")
    assert read_log(path) == entries
    path.write_text('{"format": "other"}')
    with pytest.raises(RecordError, match="not a soilsight-log/1 file"):
        read_log(path)


def test_any_tampering_is_caught():
    stored = _small_log().stored
    edited = [dict(s) for s in stored]
    edited[2]["body"] = edited[2]["body"].replace("0.031", "0.013")          # a reading changed afterwards
    with pytest.raises(RecordError, match="entry 3 was altered"):
        verify(edited)
    with pytest.raises(RecordError, match="entry 2 does not follow entry 1"):
        verify([stored[0], stored[2]])                                        # an entry removed
    with pytest.raises(RecordError, match="does not follow"):
        verify([stored[1], stored[0], stored[2]])                             # reordered
    rehashed = [dict(s) for s in stored]                                      # edited and re-hashed: the next link breaks
    import hashlib
    rehashed[1]["body"] = rehashed[1]["body"].replace('"cores":3', '"cores":1')
    rehashed[1]["hash"] = hashlib.sha256(rehashed[1]["body"].encode()).hexdigest()
    with pytest.raises(RecordError, match="entry 3 does not follow entry 2"):
        verify(rehashed)
    assert verify([]) == []


@needs_node
def test_phone_hashing_and_log_entries_match_python(tmp_path):
    import hashlib
    strings = ["", "abc", "a" * 55, "a" * 56, "a" * 64, "a" * 1000, "田间 pH 5.4 µg/L ✓", '{"seq":1,"data":{"x":0.1}}']
    events = [dict(time="2026-11-11T03:00:00.000Z", type="visit_started", data=dict(threshold=0.05, unit="mg/kg")),
              dict(time="2026-11-11T03:05:00.000Z", type="sample_taken", data=dict(id="S01", cell=3, x=1.25, y=2.5, cores=3)),
              dict(time="2026-11-11T03:20:00.000Z", type="result_entered", data=dict(id="S01", reading=0.031, ph=5.4, note="稻田"))]
    src = tmp_path / "in.json"
    src.write_text(json.dumps(dict(strings=strings, events=events), ensure_ascii=False))
    out = json.loads(subprocess.run([NODE, str(JS / "run_record.js"), str(src)], capture_output=True, text=True, check=True).stdout)
    assert out["hashes"] == [hashlib.sha256(s.encode()).hexdigest() for s in strings]
    entries = verify(out["entries"])                                          # a log written by the phone verifies here
    assert [e.type for e in entries] == ["visit_started", "sample_taken", "result_entered"]
    assert entries[2].data["note"] == "稻田" and entries[1].data["x"] == 1.25


# ---- replaying a visit and reporting on it

import threading
import urllib.request

from soilsight.app.package import package_from_field
from soilsight.app.server import make_server
from soilsight.app.visit import replay
from soilsight.cli import main
from soilsight.field import FieldDir
from soilsight.report.health import lab_candidates, map_png, write_health_report, write_results_geojson

PLAYWRIGHT = NODE is not None and subprocess.run([NODE, "-e", "require('playwright')"], capture_output=True).returncode == 0
needs_browser = pytest.mark.skipif(not PLAYWRIGHT, reason="Node.js with Playwright is not installed")


def simulated_log(package, fld, threshold=0.05, seed=3, lonlat=False):
    """A whole visit written as the phone would write it, with readings from a synthetic field."""
    rng = np.random.default_rng(seed)
    brain = Brain.from_package(package)
    w = LogWriter()
    clock = iter(f"2026-11-11T03:{m:02d}:00.000Z" for m in range(60))
    w.add("visit_started", dict(threshold=threshold, unit="mg/kg", package=dict(name=package["name"], made=package["made"])), next(clock))
    n = 0

    def test(xy, planned):
        nonlocal n
        n += 1
        sid, cell = f"S{n:02d}", brain.nearest_cell(xy)
        x, y = (float(v) for v in xy + rng.normal(0, 1.0, 2))
        d = dict(id=sid, cell=cell, x=x, y=y, cores=3, planned=planned)
        if lonlat:
            a, b, c, dd, e, f = package["geo"]["to_lonlat"]
            d |= dict(lon=a * x + b * y + c, lat=dd * x + e * y + f, gps_accuracy_m=2.5)
        w.add("sample_taken", d, next(clock))
        reading = threshold * float(np.exp(observe(fld, cell, rng, 0.22, cores=3)))
        w.add("result_entered", dict(id=sid, reading=reading, ph=5.4, noise_sd=0.22, lab=False), next(clock))
        brain.add_result((x, y), float(np.log(reading / threshold)))

    for spot in brain.initial_spots():
        test(spot, True)
    while (nxt := brain.next_spot()) is not None:
        w.add("spot_proposed", dict(id=f"S{n + 1:02d}", cell=brain.nearest_cell(nxt.xy), gain=nxt.gain), next(clock))
        test(nxt.xy, False)
    w.add("visit_finished", {}, next(clock))
    return w, brain


@pytest.fixture(scope="module")
def finished_visit(sim_field, package):
    w, brain = simulated_log(package, sim_field)
    return replay(package, verify(w.stored)), brain, w


def test_replay_rebuilds_the_visit(finished_visit, package):
    visit, brain, w = finished_visit
    assert len(visit.tests) == 12 and visit.finished and visit.threshold == 0.05 and len(visit.proposals) == 5
    assert visit.head == w.stored[-1]["hash"] and visit.started.startswith("2026-11-11T03:00")
    t = visit.tests[0]
    assert t.id == "S01" and t.cores == 3 and t.ph == 5.4 and t.value == pytest.approx(np.log(t.reading / 0.05))
    assert np.allclose(visit.map().p_unsafe, brain.map().p_unsafe, atol=1e-12)       # same map as the Brain that ran the visit
    assert visit.next_spot() is None
    zones = visit.zones()
    assert [z.zone for z in zones] == list(range(6)) and all(z.label in ("green", "yellow", "red") for z in zones)


def test_later_results_replace_earlier_ones(package):
    w = LogWriter()
    w.add("visit_started", dict(threshold=0.05, unit="mg/kg"), "t0")
    w.add("sample_taken", dict(id="S01", cell=5, x=10.0, y=10.0, cores=3), "t1")
    w.add("sample_taken", dict(id="S02", cell=900, x=60.0, y=40.0, cores=3), "t2")
    w.add("result_entered", dict(id="S01", reading=0.5, noise_sd=0.22), "t3")
    assert [t.reading for t in replay(package, verify(w.stored)).tests] == [0.5]      # S02 has no result yet
    w.add("result_entered", dict(id="S01", reading=0.05, noise_sd=0.05, lab=True), "t4")   # corrected by a lab value
    v = replay(package, verify(w.stored))
    assert (v.tests[0].reading, v.tests[0].lab, v.tests[0].noise_sd, v.tests[0].value) == (0.05, True, 0.05, 0.0)
    w.add("result_entered", dict(id="S01", void=True), "t5")
    v = replay(package, verify(w.stored))
    assert v.tests == [] and not v.finished and "S01" in v.samples


def test_results_need_a_threshold(package):
    w = LogWriter()
    w.add("sample_taken", dict(id="S01", cell=5, x=10.0, y=10.0), "t1")
    w.add("result_entered", dict(id="S01", reading=0.5), "t2")
    no_threshold = dict(package, threshold=None)
    with pytest.raises(ValueError, match="no threshold"):
        replay(no_threshold, verify(w.stored))


def test_field_health_report(finished_visit, tmp_path):
    visit, _, _ = finished_visit
    doc = write_health_report(tmp_path / "health.html", visit)
    page = (tmp_path / "health.html").read_text()
    assert "Field Health Report" in page and "screening result, not an official classification" in page
    assert "Zone by zone" in page and "Send to an accredited lab" in page and visit.head in page
    assert page.count("data:image/png;base64,") == 1 and "<script" not in page and "@media print" in page
    assert len(doc["zones"]) == 6 and sum(z["tests"] for z in doc["zones"]) == 12
    assert sum(doc["field_share"].values()) == pytest.approx(1) and 0 < doc["expected_wrong_share"] < 0.5
    assert doc == json.loads((tmp_path / "health.json").read_text())
    counts = {k: sum(z["label"] == k for z in doc["zones"]) for k in ("green", "yellow", "red")}
    assert f"{len(doc['zones'])} zones:" in page and all(f"{n} " in page for n in counts.values() if n)
    assert doc["tests"] == 12 and doc["finished"] is True


def test_report_carries_scout_warnings_and_unfinished_visits(finished_visit, package, tmp_path):
    visit, _, w = finished_visit
    checks = [dict(name="Ground control points", status="warn", value="none"), dict(name="Photo count", status="ok", value="80")]
    write_health_report(tmp_path / "a.html", visit, checks)
    page = (tmp_path / "a.html").read_text()
    assert "Ground control points (none)" in page and "Photo count" not in page
    unfinished = replay(package, verify(w.stored[:-1]))
    write_health_report(tmp_path / "b.html", unfinished)
    assert "not marked finished" in (tmp_path / "b.html").read_text()
    with pytest.raises(ValueError, match="no test results"):
        write_health_report(tmp_path / "c.html", replay(package, verify(w.stored[:1])))


def test_lab_candidates_are_one_in_ten_plus_the_highest_reading(finished_visit):
    visit, brain, _ = finished_visit
    p = brain.map().p_unsafe
    picks = lab_candidates(visit, p)
    doubt = sorted(visit.tests, key=lambda t: abs(p[t.cell] - 0.5))[:2]             # 12 tests: two by doubt
    top = max(visit.tests, key=lambda t: t.reading)
    assert [t.id for t, _ in picks][:2] == [t.id for t in doubt]
    assert top.id in [t.id for t, _ in picks] and 2 <= len(picks) <= 3
    assert map_png(visit.package, p, visit)[:8] == b"\x89PNG\r\n\x1a\n"


def test_results_geojson_needs_positions(sim_field, tmp_path):
    placed = package_from_sim(sim_field, protocol=Protocol(budget=12, cores=3), corner_lonlat=(113.0, 28.2))
    w, _ = simulated_log(placed, sim_field, lonlat=True)
    visit = replay(placed, verify(w.stored))
    assert write_results_geojson(tmp_path / "r.geojson", visit) == 12
    feat = json.loads((tmp_path / "r.geojson").read_text())["features"][0]
    assert feat["geometry"]["coordinates"][0] == pytest.approx(113.0, abs=0.01) and feat["properties"]["id"] == "S01"
    w2, _ = simulated_log(package_from_sim(sim_field, protocol=Protocol(budget=12, cores=3)), sim_field)
    assert write_results_geojson(tmp_path / "none.geojson", replay(placed, verify(w2.stored))) == 0      # no GPS positions recorded


# ---- a real scouted field

@pytest.fixture(scope="module")
def sample_field(tmp_path_factory):
    from soilsight.scout import run
    fd = FieldDir(tmp_path_factory.mktemp("sample") / "field")
    shutil.copytree(Path(__file__).resolve().parents[1] / "data" / "sample_field", fd.root)
    fd.config_path.write_text('name = "Sample"\nthreshold = 0.05\nbudget = 10\n')
    run(fd, skip_stitch=True, log=lambda m: None)
    return fd


def test_package_from_a_scouted_field(sample_field):
    pkg = package_from_field(sample_field)
    assert pkg["name"] == "Sample" and pkg["threshold"] == 0.05 and pkg["protocol"]["budget"] == 10
    assert len(pkg["grid"]["rows"]) == 3500 and pkg["grid"]["cell_m"] == 1.0 and len(pkg["zones"]["names"]) == 2
    assert len(pkg["plan"]["initial"]) == 6 and pkg["plan"]["initial"][:2] == pkg["plan"]["fixed"]     # inlet and outlet first
    import base64, io
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(pkg["image"]["jpeg"])))
    assert im.size[0] / im.size[1] == pytest.approx(pkg["grid"]["nx"] / pkg["grid"]["ny"], rel=0.01)
    a, b, c, d, e, f = pkg["geo"]["to_lonlat"]
    t = pkg["geo"]["to_xy"]
    lon, lat = a * 30 + b * 20 + c, d * 30 + e * 20 + f
    assert (t[0] * lon + t[1] * lat + t[2], t[3] * lon + t[4] * lat + t[5]) == pytest.approx((30, 20), abs=1e-4)
    assert lon == pytest.approx(-91.994, abs=0.002) and lat == pytest.approx(46.8426, abs=0.002)        # Brighton Beach, Minnesota
    # 30 m east on the grid is 30 m on the ground
    from soilsight.geo import metres_per_degree
    assert (a * 30) * metres_per_degree(lat)[0] == pytest.approx(30, abs=0.1)


def test_cli_package_and_report(sample_field, sim_field, tmp_path, capsys):
    assert main(["app", "package", str(tmp_path / "nothing")]) == 2
    assert "no map layers yet" in capsys.readouterr().err
    assert main(["app", "package", str(sample_field.root)]) == 0
    assert "10 tests of 3 cores" in capsys.readouterr().out and sample_field.package.exists()

    from soilsight.app.package import load_package
    pkg = load_package(sample_field.package)
    w = LogWriter()
    brain = Brain.from_package(pkg)
    w.add("visit_started", dict(threshold=0.05, unit="mg/kg", package=dict(name=pkg["name"], made=pkg["made"])), "2026-11-11T03:00:00Z")
    for i, xy in enumerate(brain.initial_spots()):
        w.add("sample_taken", dict(id=f"S{i + 1:02d}", cell=brain.nearest_cell(xy), x=float(xy[0]), y=float(xy[1]), cores=3,
                                   lon=pkg["geo"]["to_lonlat"][2], lat=pkg["geo"]["to_lonlat"][5]), "2026-11-11T03:05:00Z")
        w.add("result_entered", dict(id=f"S{i + 1:02d}", reading=0.03 + 0.01 * i, noise_sd=0.22), "2026-11-11T03:30:00Z")
    log = tmp_path / "log.json"
    write_log(w.stored, log, pkg["name"])
    assert main(["app", "report", str(sample_field.root), str(log)]) == 0
    out = capsys.readouterr().out
    assert "log verified: 13 entries, 6 tests (visit not marked finished)" in out and "zone A:" in out
    assert sample_field.health_html.exists() and sample_field.visit_log.exists() and sample_field.results_geojson.exists()
    page = sample_field.health_html.read_text()
    assert "Ground control points (none)" in page and "data:image/jpeg;base64," in page            # scout warnings and the photo

    doc = json.loads(log.read_text())
    assert '"reading":0.03' in doc["entries"][2]["body"]
    doc["entries"][2]["body"] = doc["entries"][2]["body"].replace('"reading":0.03', '"reading":0.3')     # a reading edited afterwards
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    assert main(["app", "report", str(sample_field.root), str(bad)]) == 1
    assert "cannot be trusted" in capsys.readouterr().err

    other = LogWriter()
    other.add("visit_started", dict(threshold=0.05, unit="mg/kg", package=dict(name="Elsewhere", made="2020")), "t")
    write_log(other.stored, bad)
    assert main(["app", "report", str(sample_field.root), str(bad)]) == 1
    assert "belongs to another field package" in capsys.readouterr().err


# ---- the app itself

@pytest.fixture()
def app_server(sim_field, tmp_path):
    placed = package_from_sim(sim_field, "Demo paddy", Protocol(budget=10, cores=3), threshold=None, corner_lonlat=(113.0, 28.2))
    path = tmp_path / "field.soilsight.json"
    save_package(placed, path)
    server = make_server(port=0, package_path=path)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://localhost:{server.server_address[1]}/", placed
    server.shutdown()


def test_server_serves_the_app_and_the_package(app_server):
    url, placed = app_server
    page = urllib.request.urlopen(url).read().decode()
    assert "SoilSight field visit" in page and "brain.js" in page
    assert json.loads(urllib.request.urlopen(url + "package.json").read())["name"] == "Demo paddy"
    for name in ("app.js", "brain.js", "sha256.js", "style.css", "sw.js", "manifest.webmanifest", "icon.svg"):
        assert urllib.request.urlopen(url + name).status == 200
    bare = make_server(port=0)
    threading.Thread(target=bare.serve_forever, daemon=True).start()
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"http://localhost:{bare.server_address[1]}/package.json")
    bare.shutdown()


@needs_browser
def test_a_whole_visit_on_a_phone_with_no_signal(app_server, sim_field, tmp_path):
    """The phase 5 gate. An emulated phone loads the app, the network is cut, and a person's
    taps, walking and typing take the visit from the first spot to the exported log."""
    url, placed = app_server
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps(dict(truth=sim_field.y.tolist(), threshold=0.05, kit_sd=0.22, seed=7)))
    done = subprocess.run([NODE, str(JS / "run_visit.js"), url, str(truth), str(tmp_path)], capture_output=True, text=True, timeout=240)
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout)

    assert out["offlineStart"] == "unreachable"                              # the network really was off
    assert [e for e in out["errors"] if "ERR_FAILED" not in e and "ERR_INTERNET_DISCONNECTED" not in e] == []
    assert out["tests"] == out["spots"] == 10                                # the whole budget, 6 spread and 4 from the loop
    assert any("km from this field" in g for g in out["guidance"])           # before the phone knew where it was
    assert any(g in ("You are at the spot.",) or " m to the " in g for g in out["guidance"])

    entries = read_log(out["log"])                                           # the phone's log verifies
    kinds = [e.type for e in entries]
    assert kinds[0] == "visit_started" and kinds[-1] == "visit_finished" and kinds.count("spot_proposed") == 4
    assert kinds.count("sample_taken") == 10 and kinds.count("result_entered") == 10
    visit = replay(placed, entries)
    assert len(visit.tests) == 10 and all(t.lon is not None for t in visit.tests)
    for phone, ref in zip(out["zones"], visit.zones()):                      # and the reference Brain agrees with the phone
        assert phone["label"] == ref.label and phone["p"] == pytest.approx(ref.p_unsafe, abs=1e-6)
    doc = write_health_report(tmp_path / "health.html", visit)
    assert doc["tests"] == 10 and (tmp_path / "health.html").exists()
