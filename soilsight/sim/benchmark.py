"""The benchmark: three experiments, their reference results, and a check against them.

    methods     nine strategies, 15 tests per field, twelve scenarios
    budget      accuracy against the number of tests, zones, mixed cores
    validation  how often a kit-vs-lab split-sample test would pass

Every experiment returns a list of rows (plain dicts), one per scenario, strategy
and budget, holding means over the repeated fields.
"""
from __future__ import annotations

import json
import time
from importlib import resources
from math import comb

import numpy as np

from soilsight.sim.baselines import BUDGET_STRATEGIES, METHODS, hp_table, run_strategy
from soilsight.sim.fields import DEFAULT_GRID, SCENARIOS, Grid, make_field

DEFAULTS = {
    "methods": dict(scenarios=tuple(SCENARIOS), reps=40),
    "budget": dict(scenarios=("S0", "S7", "S11", "S9"), reps=24, budgets=(15, 25, 40, 60)),
}
KEY = ("scenario", "strategy", "budget")
CHECK_TOL = 0.005   # means are reported to 3 decimals; allows for BLAS differences

_COLUMNS = {
    "methods": ["acc", "bal", "missed", "rmse", "cov", "conf", "cacc", "stop_fired", "stop_n", "stop_acc"],
    "budget": ["acc", "bal", "missed", "yellow", "fgreen", "caught", "zacc", "zrmse", "cov"],
    "validation": ["within_25pct", "p_pass", "ci95"],
}
LEGEND = {
    "acc": "share of the field classed correctly (against 10 m block truth)",
    "bal": "balanced accuracy",
    "missed": "share of the unsafe area called safe",
    "rmse": "map error, ln units",
    "cov": "share of truth inside the 90% ranges (should be 0.90)",
    "conf": "share of the field classed with 90% confidence",
    "cacc": "accuracy inside that confident share",
    "stop_fired": "share of runs where the 90%/90% stop rule fired",
    "stop_n": "tests used when it fired",
    "stop_acc": "accuracy when it fired",
    "yellow": "share of the field in the 'not sure' class",
    "fgreen": "share of the green area that is really unsafe",
    "caught": "share of the unsafe area that is not green",
    "zacc": "share of the six zones classed correctly",
    "zrmse": "error of zone means, ln units",
    "within_25pct": "true share of kit results within 25% of the lab",
    "p_pass": "chance that at least 80% of the split samples land within 25%",
    "ci95": "95% confidence half-width on the measured share",
}


def _mean(runs, key):
    a = np.array([r[key] for r in runs], float)
    return float("nan") if np.all(np.isnan(a)) else float(np.nanmean(a))


def _summarise(runs) -> dict:
    out = {k: _mean(runs, k) for k in runs[0] if k != "idx"}
    out["stop_fired"] = float(np.mean([not np.isnan(r["stop_n"]) for r in runs]))
    return out


def _experiment(strategies, hp, scenarios, reps, budgets, seed, grid, log):
    rows, t0 = [], time.time()
    for sid in scenarios:
        sc = SCENARIOS[sid]
        fields = [make_field(np.random.default_rng(1000 + r), grid, **sc.field) for r in range(reps)]
        for name, strat in strategies.items():
            for budget in budgets:
                runs = [run_strategy(hp, fields[r], strat, np.random.default_rng(seed + r),
                                     sc.kit_sd, budget) for r in range(reps)]
                rows.append(dict(scenario=sid, strategy=name, budget=budget) | _summarise(runs))
        if log:
            log(f"  {sid} ({sc.label}) done, {time.time() - t0:.0f}s elapsed")
    return rows


def run_methods(scenarios=None, reps=None, grid: Grid = DEFAULT_GRID, log=None) -> list:
    d = DEFAULTS["methods"]
    hp = hp_table(["sf", "ls", "lf", "sn", "sb"])
    return _experiment(METHODS, hp, scenarios or d["scenarios"], reps or d["reps"], (15,), 5000, grid, log)


def run_budget(scenarios=None, reps=None, budgets=None, grid: Grid = DEFAULT_GRID, log=None) -> list:
    d = DEFAULTS["budget"]
    hp = hp_table(["sf", "ls", "sb"])
    return _experiment(BUDGET_STRATEGIES, hp, scenarios or d["scenarios"], reps or d["reps"],
                       budgets or d["budgets"], 7000, grid, log)


def run_validation() -> list:
    """Power of the split-sample test "at least 80% of kit results within 25% of the lab".

    The scenario column holds the kit error (sd of ln), the budget column the number
    of split samples.
    """
    rng = np.random.default_rng(1)
    rows = []
    for kit_sd in (0.12, 0.18, 0.22, 0.30):
        ratio = np.exp(rng.standard_normal(400000) * kit_sd)
        p = float(((ratio > 0.75) & (ratio < 1.25)).mean())
        for n in (10, 20, 40):
            k0 = int(np.ceil(0.8 * n))
            p_pass = sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(k0, n + 1))
            rows.append(dict(scenario=f"kit sd {kit_sd:.2f}", strategy="split samples", budget=n,
                             within_25pct=p, p_pass=float(p_pass),
                             ci95=float(1.96 * np.sqrt(p * (1 - p) / n))))
    return rows


RUNNERS = {"methods": run_methods, "budget": run_budget, "validation": run_validation}


def format_table(experiment: str, rows: list) -> str:
    cols = _COLUMNS[experiment]
    width = max(len(r["strategy"]) for r in rows)
    lines = [f"{'scenario':12s} {'strategy':{width}s} {'n':>3s} " + " ".join(f"{c:>10s}" for c in cols)]
    last = None
    for r in rows:
        if last is not None and r["scenario"] != last:
            lines.append("")
        last = r["scenario"]
        label = r["scenario"]
        if label in SCENARIOS:
            label = f"{label} "
        lines.append(f"{label:12s} {r['strategy']:{width}s} {r['budget']:3d} "
                     + " ".join(f"{r[c]:10.3f}" for c in cols))
    lines.append("")
    lines += [f"{c:>12s}  {LEGEND[c]}" for c in cols]
    if experiment != "validation":
        used = sorted({r["scenario"] for r in rows}, key=lambda s: list(SCENARIOS).index(s))
        lines.append("")
        lines += [f"{s:>12s}  {SCENARIOS[s].label}; 'all safe' accuracy "
                  f"{1 - SCENARIOS[s].field.get('prevalence', 0.3):.2f}" for s in used]
    return "\n".join(lines)


def _expected_path(experiment: str):
    return resources.files("soilsight.sim").joinpath("expected", f"{experiment}.json")


def load_expected(experiment: str) -> dict:
    return json.loads(_expected_path(experiment).read_text())


def save_expected(experiment: str, rows: list, reps, path=None) -> None:
    clean = [{k: (None if isinstance(v, float) and np.isnan(v) else (round(v, 6) if isinstance(v, float) else v))
              for k, v in r.items()} for r in rows]
    text = json.dumps(dict(experiment=experiment, reps=reps, rows=clean), indent=1)
    (path or _expected_path(experiment)).write_text(text + "\n")


def compare(rows: list, expected_rows: list, tol: float = CHECK_TOL) -> list:
    """Differences between a run and the reference rows; an empty list means it reproduces.

    Only rows present in the run are compared, so a subset of scenarios can be checked.
    """
    ref = {tuple(r[k] for k in KEY): r for r in expected_rows}
    problems = []
    for r in rows:
        key = tuple(r[k] for k in KEY)
        if key not in ref:
            problems.append(f"{key}: no reference row")
            continue
        for name, want in ref[key].items():
            if name in KEY:
                continue
            got = r.get(name)
            got_nan = got is None or np.isnan(got)
            want_nan = want is None or np.isnan(want)   # NaN is stored as null in the JSON
            if want_nan or got_nan:
                if want_nan != got_nan:
                    problems.append(f"{key} {name}: got {got}, reference {want}")
            elif abs(got - want) > tol:
                problems.append(f"{key} {name}: got {got:.4f}, reference {want:.4f}")
    return problems
