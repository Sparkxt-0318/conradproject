import copy

import pytest

from soilsight.cli import main
from soilsight.sim import benchmark as bm
from soilsight.sim.baselines import BUDGET_STRATEGIES, METHODS
from soilsight.sim.fields import SCENARIOS

# Accuracy of each strategy on scenario S0, two fields, 80 x 60 m grid. A change
# here means the baselines or the field generator no longer behave as they did.
GOLDEN_METHODS = {
    "random + GP-PC ML": 0.750417,
    "spatial + OK MAP": 0.80625,
    "PDF loop: cLHS + maxvar, GP-PC ML": 0.659583,
    "cLHS + straddle, GP-PC ML": 0.711667,
    "cLHS + maxvar, GP-PC MAP": 0.73875,
    "cLHS + straddle, GP-PC MAP": 0.745833,
    "cLHS + straddle, LIN MAP": 0.725,
    "cLHS + straddle, OK MAP": 0.692917,
    "feature + LIN MAP": 0.825417,
}
GOLDEN_BUDGET = {
    "static OK": 0.819167,
    "hybrid OK": 0.725417,
    "hybrid LIN": 0.7075,
    "hybrid OK, 3 cores": 0.770833,
}


@pytest.fixture(scope="module")
def small_methods(small_grid):
    return bm.run_methods(scenarios=("S0",), reps=2, grid=small_grid)


def test_methods_golden_values(small_methods):
    assert [r["strategy"] for r in small_methods] == list(METHODS)
    for r in small_methods:
        assert r["scenario"] == "S0" and r["budget"] == 15
        assert r["acc"] == pytest.approx(GOLDEN_METHODS[r["strategy"]], abs=1e-4)


def test_budget_golden_values(small_grid):
    rows = bm.run_budget(scenarios=("S0",), reps=2, budgets=(15,), grid=small_grid)
    assert [r["strategy"] for r in rows] == list(BUDGET_STRATEGIES)
    for r in rows:
        assert r["acc"] == pytest.approx(GOLDEN_BUDGET[r["strategy"]], abs=1e-4)


def test_static_strategies_never_fire_the_stop_rule(small_methods):
    for r in small_methods:
        if METHODS[r["strategy"]].acquisition is None:
            assert r["stop_fired"] == 0


def test_compare_accepts_itself_and_flags_changes(small_methods):
    assert bm.compare(small_methods, small_methods) == []
    changed = copy.deepcopy(small_methods)
    changed[2]["acc"] += 0.02
    problems = bm.compare(changed, small_methods)
    assert len(problems) == 1 and "acc" in problems[0] and "PDF loop" in problems[0]
    assert bm.compare(changed, small_methods, tol=0.05) == []


def test_compare_checks_a_subset_but_not_unknown_rows(small_methods):
    assert bm.compare(small_methods[:3], small_methods) == []
    assert "no reference row" in bm.compare(small_methods, small_methods[:3])[0]


def test_compare_treats_nan_and_missing_alike():
    nan = float("nan")
    ref = [dict(scenario="S0", strategy="a", budget=15, stop_n=None, acc=0.7)]
    assert bm.compare([dict(scenario="S0", strategy="a", budget=15, stop_n=nan, acc=0.7)], ref) == []
    assert len(bm.compare([dict(scenario="S0", strategy="a", budget=15, stop_n=7.0, acc=0.7)], ref)) == 1
    assert len(bm.compare([dict(scenario="S0", strategy="a", budget=15, stop_n=nan, acc=nan)], ref)) == 1


def test_expected_round_trip(tmp_path, small_methods):
    path = tmp_path / "methods.json"
    bm.save_expected("methods", small_methods, 2, path)
    import json
    stored = json.loads(path.read_text())
    assert stored["reps"] == 2 and len(stored["rows"]) == len(small_methods)
    assert stored["rows"][0]["stop_n"] is None           # NaN is stored as null
    assert bm.compare(small_methods, stored["rows"]) == []


def test_format_table_lists_strategies_and_explains_columns(small_methods):
    text = bm.format_table("methods", small_methods)
    for name in METHODS:
        assert name in text
    assert "share of the unsafe area called safe" in text
    assert "'all safe' accuracy 0.70" in text


# ---- the stored reference results are the numbers quoted in docs/software-plan.md

def _row(rows, scenario, strategy, budget=15):
    return next(r for r in rows if (r["scenario"], r["strategy"], r["budget"]) == (scenario, strategy, budget))


def test_methods_reference_holds_the_plans_headline_numbers():
    ref = bm.load_expected("methods")
    assert ref["reps"] == 40 and len(ref["rows"]) == len(SCENARIOS) * len(METHODS)
    rows = ref["rows"]
    pdf = _row(rows, "S0", "PDF loop: cLHS + maxvar, GP-PC ML")
    even = _row(rows, "S0", "spatial + OK MAP")
    assert pdf["acc"] == pytest.approx(0.742, abs=6e-4)
    assert even["acc"] == pytest.approx(0.786, abs=6e-4)
    assert pdf["cov"] == pytest.approx(0.72, abs=6e-3)          # "90%" ranges hold the truth 72% of the time
    assert pdf["stop_fired"] == pytest.approx(0.12, abs=6e-3)   # stop rule fires early
    assert _row(rows, "S2", "cLHS + straddle, GP-PC MAP")["acc"] == pytest.approx(0.787, abs=6e-4)
    assert _row(rows, "S7", "cLHS + straddle, GP-PC MAP")["bal"] == pytest.approx(0.665, abs=6e-4)


def test_budget_reference_holds_the_plans_headline_numbers():
    ref = bm.load_expected("budget")
    assert ref["reps"] == 24 and len(ref["rows"]) == 4 * len(BUDGET_STRATEGIES) * 4
    rows = ref["rows"]
    assert _row(rows, "S0", "hybrid OK")["acc"] == pytest.approx(0.758, abs=6e-4)
    assert _row(rows, "S0", "hybrid OK, 3 cores")["acc"] == pytest.approx(0.786, abs=6e-4)
    assert _row(rows, "S0", "hybrid OK", 60)["acc"] == pytest.approx(0.855, abs=6e-4)
    assert _row(rows, "S7", "static OK", 60)["bal"] == pytest.approx(0.706, abs=6e-4)
    zone = [r["zacc"] for r in rows if r["budget"] == 15]
    assert min(zone) >= 0.855 and max(zone) <= 0.985           # "86-98% of zones correct at 15 tests"


def test_validation_reproduces_its_reference():
    rows = bm.run_validation()
    assert bm.compare(rows, bm.load_expected("validation")["rows"]) == []
    r = _row(rows, "kit sd 0.22", "split samples", 20)
    assert r["within_25pct"] == pytest.approx(0.75, abs=0.005)
    assert r["p_pass"] == pytest.approx(0.41, abs=0.01)


# ---- command line

def test_cli_validation_check_passes(capsys):
    assert main(["sim", "benchmark", "validation", "--check"]) == 0
    out = capsys.readouterr()
    assert "kit sd 0.22" in out.out and "check passed" in out.err


def test_cli_rejects_bad_requests(capsys):
    assert main(["sim", "benchmark", "methods", "--scenarios", "S99"]) == 2
    assert "unknown scenario" in capsys.readouterr().err
    assert main(["sim", "benchmark", "methods", "--reps", "3", "--check"]) == 2
    assert "--reps 40" in capsys.readouterr().err
    assert main(["sim", "benchmark", "methods", "--budgets", "15"]) == 2
    assert main(["sim", "benchmark", "validation", "--reps", "3"]) == 2


def test_cli_will_not_overwrite_the_reference_from_a_partial_run(capsys):
    assert main(["sim", "benchmark", "methods", "--scenarios", "S0", "--reps", "1", "--write-expected"]) == 2
    assert "--write-expected needs the default" in capsys.readouterr().err
