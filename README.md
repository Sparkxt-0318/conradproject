# SoilSight

Software for SoilSight: a drone maps a drained rice field, the software picks where to test the soil, a strip kit tests each spot, and the farmer gets a red-yellow-green map.

The design and the reasoning behind it are in [docs/software-plan.md](docs/software-plan.md).

## Status

Phases 1 to 5 of the plan are built:

- **Benchmark:** synthetic fields that every sampling strategy is judged on.
- **Brain:** the map model and the loop that picks where to test next.
- **Scout processing:** one command turns a folder of drone photos into map layers and a quality report.
- **Capture:** the software for the Raspberry Pi on the drone, which takes the photos and records where each was taken.
- **Field app and report:** a phone web app that runs a whole visit with no signal, and the Field Health Report made from its log.

On the benchmark the Brain beats the original plan's loop and gives honest uncertainty, but it is level with a plain even spread of tests on accuracy; its gain is less unsafe ground called safe. Details are in the plan under "Phase 2 result".

Scout processing has run on public sample photo sets with real stitching. Capture has flown surveys in ArduPilot's simulator with a stand-in camera. The field app has run whole visits in an emulated phone with the network cut, not yet on a real phone. Nothing here has touched the project's own camera, a real flight controller or a real rice field yet; [docs/bench-test.md](docs/bench-test.md) is the procedure for that.

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

## From photos to map layers

Stitching is done by OpenDroneMap. Start its server once on the processing laptop (needs Docker):

```bash
docker run -d -p 3000:3000 opendronemap/nodeodm
```

Put the photos in a field folder:

```
my_field/
  photos/               the drone photos
  field.toml            optional: camera, flying height, grid size
  capture.csv           optional: photo positions from the capture computer
  annotations.geojson   optional: boundary, inlet, outlet, ditches, grey card
```

Judge the flight in seconds, before stitching:

```bash
soilsight scout check my_field
```

Stitch, build the layers and write the report:

```bash
soilsight scout run my_field
```

If the orthophoto and surface model already exist (for example exported from WebODM), put them in `my_field/odm/` as `odm_orthophoto.tif` and `dsm.tif` and skip the stitching:

```bash
soilsight scout run my_field --skip-stitch
```

To try this without photos or Docker, the repository includes real stitched outputs for a small sample field:

```bash
soilsight scout run data/sample_field --skip-stitch
```

The results land in the same folder:

| File | What it is |
|---|---|
| `layers/layers.tif` | One band per layer on a 1 m grid, plus an in-field mask |
| `layers/layers.json` | What each band means and how far to trust it |
| `report/quality.html` | Every check from flight to layers, with previews; open it in a browser |
| `report/quality.json` | The same checks for programs |

`annotations.geojson` is drawn in QGIS, QField or geojson.io, in longitude and latitude. Each feature has a `role` property:

| Role | Shape | Effect |
|---|---|---|
| `boundary` | Polygon | Layers are made inside it. Without one, everything the orthophoto covers is used |
| `inlet`, `outlet` | Point | Distance layers |
| `ditch` | LineString | Distance to the nearest ditch |
| `grey_card` | Polygon | Colour layers are scaled so the card reads its reflectance (`reflectance` property, default 0.18) |

`field.toml` settings, all optional:

```toml
name = "North paddy"
sensor = "pi_gs_6mm"       # pi_gs_6mm, pi_cm3_binned or pi_cm3_full
flight_height_m = 40       # above ground; needed when the photos do not record it
grid_m = 1.0

# or, for any other camera, describe it instead of naming a sensor:
# [camera]
# width_px = 4896
# height_px = 3672
# pixel_um = 1.26
# focal_mm = 4.45
# shutter = "rolling"      # add readout_ms if known
```

To hand the layers to the Brain:

```python
from soilsight.field import FieldDir
from soilsight.layers import load_layers

xy, feats, cells = load_layers(FieldDir("my_field")).brain_inputs()
```

## Capturing photos on the drone

Flight settings for a camera, and a survey mission for a field boundary:

```bash
soilsight capture plan --sensor pi_gs_6mm --height 40 --boundary my_field/annotations.geojson --mission my_field/survey.waypoints
```

On the Raspberry Pi during the flight (it reads the flight controller on the Pi's serial port):

```bash
soilsight capture run my_field --sensor pi_gs_6mm --height 40
```

That fills `my_field/photos/` and `my_field/capture.csv`, which `soilsight scout check` and `soilsight scout run` then use.

To rehearse without hardware, fly the same survey in ArduPilot's simulator. Build the simulator once from an ArduPilot checkout (`./waf configure --board sitl && ./waf copter`), then:

```bash
soilsight capture simulate sim_field --sensor pi_gs_6mm --sitl ardupilot/build/sitl/bin/arducopter --defaults ardupilot/Tools/autotest/default_params/copter.parm
```

To test the real camera on a desk:

```bash
soilsight capture bench ~/bench1 --sensor pi_gs_6mm --count 50
```

## The field visit: phone app and report

Make the file the phone needs from a scouted field (set `threshold` in `field.toml`, or type it in at the start of the visit):

```bash
soilsight app package my_field
```

Try the app on the laptop with that field loaded:

```bash
soilsight app serve --package my_field/visit/field.soilsight.json
```

In the app: walk to each numbered spot (it shows distance and direction), tap "Sample taken here", mix the cores, and enter each reading when it is tested. After the first spread of spots has results, the app proposes the next spot itself. When the tests are used up, finish the visit and export the log.

Back on the laptop, check the log and make the Field Health Report:

```bash
soilsight app report my_field soilsight-log-my_field-2026-11-11.json
```

That writes `my_field/report/field_health.html` (open it in a browser; it prints to A4) and `my_field/visit/results.geojson` for QGIS. A log that was edited after the fact is refused.

Getting the app onto a phone: the app is the folder `soilsight/app/static/`, plain files with no server behind them. Put that folder on any https web host and open it once on the phone with a connection; after that it starts with no signal. Then load the field package file through the app's "Load a field" button. A phone only gives GPS, and only keeps the app for offline use, on an https address, so `soilsight app serve` over the local network is for looking, not for field use.

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
| `soilsight/scout.py` | The photos-to-layers pipeline |
| `soilsight/field.py` | Where everything lives inside a field folder |
| `soilsight/app/` | Field package, measurement log, visit replay, and the phone app in `static/` (with the Brain in JavaScript) |
| `soilsight/report/health.py` | The Field Health Report |
| `soilsight/capture/` | The capture computer: camera drivers, triggers, clock matching, planning, simulator rehearsal, bench test |
| `soilsight/ingest/` | Camera profiles, photo positions, flight checks |
| `soilsight/photogrammetry/` | OpenDroneMap options, upload, outputs and their checks |
| `soilsight/layers/` | Field grid; colour, terrain and distance layers |
| `soilsight/report/` | The map quality report |
| `soilsight/sim/fields.py` | Synthetic fields, the scenarios, and a simulated kit reading |
| `soilsight/sim/baselines.py` | Reference strategies, including the original plan's loop |
| `soilsight/sim/metrics.py` | How a predicted map is scored against the truth |
| `soilsight/sim/benchmark.py` | The four experiments, the gate, and the check against reference results |
| `soilsight/sim/expected/` | Reference results (the numbers quoted in the plan) |
| `data/sample_field/` | Real stitched outputs at reduced resolution, for trying the pipeline and as a regression test |
| `tests/` | Unit tests and a fast regression test; scout tests use small synthetic photos and rasters; `tests/js/` runs the phone's code in Node and in an emulated phone |

The benchmark's fields are made up. It compares strategies under controlled conditions; it does not show how well any of them works on real soil.
