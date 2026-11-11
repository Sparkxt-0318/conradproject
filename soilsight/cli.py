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
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
