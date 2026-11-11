# SoilSight

Software for SoilSight: a drone maps a drained rice field, the software picks where to test the soil, a strip kit tests each spot, and the farmer gets a red-yellow-green map.

The design and the reasoning behind it are in [docs/software-plan.md](docs/software-plan.md).

## Status

Phase 1 of the plan is built: a benchmark on synthetic fields that every sampling strategy is judged on. Nothing here touches a camera or a real field yet.

## Install

```bash
pip install -e ".[dev]"
```

## Run the tests

```bash
pytest
```

## Run the benchmark

Each command prints a table and, with `--check`, fails if the numbers differ from the stored reference results.

Nine strategies with 15 tests per field, twelve scenarios (about 6 minutes):

```bash
soilsight sim benchmark methods --check
```

Accuracy against the number of tests, zone answers and mixed cores (about 15 minutes):

```bash
soilsight sim benchmark budget --check
```

How often a kit-vs-lab split-sample test would pass (instant):

```bash
soilsight sim benchmark validation --check
```

A quick look at two scenarios:

```bash
soilsight sim benchmark methods --scenarios S0,S7 --reps 10
```

## Layout

| Path | What it holds |
|---|---|
| `soilsight/sim/fields.py` | Synthetic fields, the scenarios, and a simulated kit reading |
| `soilsight/sim/baselines.py` | Reference strategies, including the original plan's loop |
| `soilsight/sim/metrics.py` | How a predicted map is scored against the truth |
| `soilsight/sim/benchmark.py` | The three experiments and the check against reference results |
| `soilsight/sim/expected/` | Reference results (the numbers quoted in the plan) |
| `tests/` | Unit tests and a fast regression test |

The fields are made up. The benchmark compares strategies under controlled conditions; it does not show how well any of them works on real soil.
