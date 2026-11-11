# SoilSight

Software for SoilSight: a drone maps a drained rice field, the software picks where to test the soil, a strip kit tests each spot, and the farmer gets a red-yellow-green map.

The design and the reasoning behind it are in [docs/software-plan.md](docs/software-plan.md).

## Status

Phases 1 and 2 of the plan are built:

- **Benchmark:** synthetic fields that every sampling strategy is judged on.
- **Brain:** the map model and the loop that picks where to test next.

On the benchmark the Brain beats the original plan's loop and gives honest uncertainty, but it is level with a plain even spread of tests on accuracy; its gain is less unsafe ground called safe. Details are in the plan under "Phase 2 result".

Nothing here touches a camera or a real field yet.

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

Nine strategies with 15 tests per field, twelve scenarios (a few minutes):

```bash
soilsight sim benchmark methods --check
```

Accuracy against the number of tests, zone answers and mixed cores (a few minutes):

```bash
soilsight sim benchmark budget --check
```

The Brain against the two reference strategies, with the phase 2 gate (about 6 minutes):

```bash
soilsight sim benchmark brain --check
```

How often a kit-vs-lab split-sample test would pass (instant):

```bash
soilsight sim benchmark validation --check
```

A quick look at two scenarios:

```bash
soilsight sim benchmark methods --scenarios S0,S7 --reps 10
```

## Using the Brain

Values are ln(reading / threshold), so a spot is unsafe when its value is above 0.

```python
from soilsight.brain import Brain, Protocol

brain = Brain(xy, protocol=Protocol(budget=15, cores=3, kit_sd=0.22), fixed_xy=[inlet, outlet])
for spot in brain.initial_spots():
    brain.add_result(spot, reading)
while (nxt := brain.next_spot()) is not None:
    brain.add_result(nxt.xy, reading)      # nxt.gain is what this test is expected to fix

cells = brain.map()                         # mean, sd and P(unsafe) per cell
zones = brain.zones(zone_ids)               # green / yellow / red per management zone
```

## Layout

| Path | What it holds |
|---|---|
| `soilsight/brain.py` | One object per field visit: where to test, record results, map and zone answers |
| `soilsight/model/` | Map model, zone answers, kit-to-lab calibration, soil-to-grain threshold |
| `soilsight/sampling/` | Initial spread and the next-spot rule |
| `soilsight/sim/fields.py` | Synthetic fields, the scenarios, and a simulated kit reading |
| `soilsight/sim/baselines.py` | Reference strategies, including the original plan's loop |
| `soilsight/sim/metrics.py` | How a predicted map is scored against the truth |
| `soilsight/sim/benchmark.py` | The four experiments, the gate, and the check against reference results |
| `soilsight/sim/expected/` | Reference results (the numbers quoted in the plan) |
| `tests/` | Unit tests and a fast regression test |

The fields are made up. The benchmark compares strategies under controlled conditions; it does not show how well any of them works on real soil.
