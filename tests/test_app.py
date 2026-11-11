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
