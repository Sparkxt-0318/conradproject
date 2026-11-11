"""The benchmark: four experiments, their reference results, and a check against them.

    methods     nine reference strategies, 15 tests per field, twelve scenarios
    budget      accuracy against the number of tests, zones, mixed cores
    validation  how often a kit-vs-lab split-sample test would pass
    brain       the production Brain against the two reference strategies it must beat

Every experiment returns a list of rows (plain dicts), one per scenario, strategy
and budget, holding means over the repeated fields.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from importlib import resources
from math import comb

import numpy as np

from soilsight.sim.baselines import BUDGET_STRATEGIES, METHODS, Strategy, hp_table, run_strategy
from soilsight.sim.brain_runner import BrainSetup, run_brain
from soilsight.sim.fields import DEFAULT_GRID, SCENARIOS, Grid, make_field

PDF_LOOP = "PDF loop: cLHS + maxvar, GP-PC ML"
EVEN_SPREAD = "spatial + OK MAP"
BRAIN = "Brain"
BRAIN_STRATEGIES = {
    PDF_LOOP: METHODS[PDF_LOOP],
    EVEN_SPREAD: METHODS[EVEN_SPREAD],
    BRAIN: BrainSetup(cores=1),
    "spatial + OK MAP, 3 cores": Strategy("spatial", None, "OK", "MAP", cores=3),
    "Brain, 3 cores": BrainSetup(cores=3),
}

# strategies, hyperparameter table for the reference strategies, seed for the sampling noise
EXPERIMENTS = {
    "methods": (METHODS, ("sf", "ls", "lf", "sn", "sb"), 5000),
    "budget": (BUDGET_STRATEGIES, ("sf", "ls", "sb"), 7000),
    "brain": (BRAIN_STRATEGIES, ("sf", "ls", "lf", "sn", "sb"), 5000),
}
DEFAULTS = {
    "methods": dict(scenarios=tuple(SCENARIOS), reps=40),
    "budget": dict(scenarios=("S0", "S7", "S11", "S9"), reps=24, budgets=(15, 25, 40, 60)),
    "brain": dict(scenarios=tuple(SCENARIOS), reps=40),
}
PAIRED = ("acc", "bal", "missed")   # per-field values kept for paired comparisons
KEY = ("scenario", "strategy", "budget")
CHECK_TOL = 0.005   # means are reported to 3 decimals; allows for BLAS differences

_COLUMNS = {
    "methods": ["acc", "bal", "missed", "rmse", "cov", "conf", "cacc", "stop_fired", "stop_n", "stop_acc"],
    "budget": ["acc", "bal", "missed", "yellow", "fgreen", "caught", "zacc", "zrmse", "cov"],
    "brain": ["acc", "bal", "missed", "rmse", "cov", "yellow", "fgreen", "zone_acc", "zone_yellow", "zone_fgreen"],
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
    "zone_acc": "share of zones answered on the right side (Brain only)",
    "zone_yellow": "share of zones answered 'not sure' (Brain only)",
    "zone_fgreen": "share of green zones that are really unsafe (Brain only)",
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
    out["_runs"] = {k: [float(r[k]) for r in runs] for k in PAIRED}
    return out


@lru_cache(maxsize=4)
def _field(sid: str, r: int, grid: Grid):
    return make_field(np.random.default_rng(1000 + r), grid, **SCENARIOS[sid].field)


@lru_cache(maxsize=None)
def _hp(names: tuple):
    return hp_table(names)


def _run_one(task) -> dict:
    """One strategy on one field. Self-contained so it can run in a worker process."""
    experiment, sid, name, budget, r, grid = task
    strategies, hp_names, seed = EXPERIMENTS[experiment]
    spec, sc = strategies[name], SCENARIOS[sid]
    fld, rng = _field(sid, r, grid), np.random.default_rng(seed + r)
    if isinstance(spec, BrainSetup):
        out = run_brain(fld, rng, sc.kit_sd, budget, cores=spec.cores, use_layers=spec.use_layers, **spec.options)
    else:
        out = run_strategy(_hp(hp_names), fld, spec, rng, sc.kit_sd, budget)
    return {k: v for k, v in out.items() if k != "idx"}


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _experiment(experiment, scenarios, reps, budgets, grid, log, jobs) -> list:
    """Run every strategy of an experiment. Each run has its own seed, so the result does
    not depend on jobs."""
    strategies = EXPERIMENTS[experiment][0]
    rows, t0 = [], time.time()
    # Workers start fresh with single-threaded linear algebra: several processes each
    # running a multi-threaded library on the same cores is slower than either alone.
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    pool = None
    if jobs > 1:
        os.environ.update({k: "1" for k in _THREAD_VARS})
        pool = ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn"))
    try:
        for sid in scenarios:
            tasks = [(experiment, sid, name, budget, r, grid)
                     for r in range(reps) for name in strategies for budget in budgets]
            results = list(pool.map(_run_one, tasks, chunksize=4) if pool else map(_run_one, tasks))
            by_cell = {}
            for task, res in zip(tasks, results):
                by_cell.setdefault((task[2], task[3]), []).append(res)
            for name in strategies:
                for budget in budgets:
                    rows.append(dict(scenario=sid, strategy=name, budget=budget) | _summarise(by_cell[(name, budget)]))
            if log:
                log(f"  {sid} ({SCENARIOS[sid].label}) done, {time.time() - t0:.0f}s elapsed")
    finally:
        if pool:
            pool.shutdown()
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return rows


def run_methods(scenarios=None, reps=None, grid: Grid = DEFAULT_GRID, log=None, jobs: int = 1) -> list:
    d = DEFAULTS["methods"]
    return _experiment("methods", scenarios or d["scenarios"], reps or d["reps"], (15,), grid, log, jobs)


def run_budget(scenarios=None, reps=None, budgets=None, grid: Grid = DEFAULT_GRID, log=None, jobs: int = 1) -> list:
    d = DEFAULTS["budget"]
    return _experiment("budget", scenarios or d["scenarios"], reps or d["reps"],
                       budgets or d["budgets"], grid, log, jobs)


def run_brain_experiment(scenarios=None, reps=None, grid: Grid = DEFAULT_GRID, log=None, jobs: int = 1) -> list:
    """The Brain against the PDF loop and the even-spread baseline, on the same fields and
    with the same sampling noise as the methods experiment."""
    d = DEFAULTS["brain"]
    return _experiment("brain", scenarios or d["scenarios"], reps or d["reps"], (15,), grid, log, jobs)


def paired_difference(rows, a: str, b: str, metric: str) -> tuple:
    """Mean of (a - b) over every field of every scenario, and its standard error."""
    pick = lambda name: np.concatenate([r["_runs"][metric] for r in rows if r["strategy"] == name])
    d = pick(a) - pick(b)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("inf")
    return float(d.mean()), float(se)


def gate(rows) -> list:
    """The phase 2 gate, judged on single-core tests: the Brain must beat the PDF loop and the
    even-spread baseline on accuracy, and its 90% ranges must hold the truth 85-95% of the time.

    "Beats" means the mean paired difference in accuracy is above zero by more than 1.96
    standard errors. Returns (passed, description) pairs.
    """
    out = []
    for other in (PDF_LOOP, EVEN_SPREAD):
        d, se = paired_difference(rows, BRAIN, other, "acc")
        extra = ", ".join(f"{m} {paired_difference(rows, BRAIN, other, m)[0]:+.3f}"
                          f"±{paired_difference(rows, BRAIN, other, m)[1]:.3f}" for m in ("bal", "missed"))
        out.append((bool(d - 1.96 * se > 0), f"accuracy vs '{other}': {d:+.3f} ± {se:.3f} (standard error); {extra}"))
    cov = [r["cov"] for r in rows if r["strategy"] == BRAIN]
    outside = [f"{r['scenario']} {r['cov']:.2f}" for r in rows if r["strategy"] == BRAIN and not 0.85 <= r["cov"] <= 0.95]
    out.append((bool(0.85 <= np.mean(cov) <= 0.95),
                f"90% ranges hold the truth {np.mean(cov):.3f} of the time on average"
                + (f"; outside 0.85-0.95 in {', '.join(outside)}" if outside else "; inside 0.85-0.95 in every scenario")))
    return out


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


RUNNERS = {"methods": run_methods, "budget": run_budget, "validation": run_validation,
           "brain": run_brain_experiment}


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
                     + " ".join(f"{r.get(c, float('nan')):10.3f}" for c in cols))
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
              for k, v in r.items() if not k.startswith("_")} for r in rows]
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
            if name in KEY or name.startswith("_"):
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
