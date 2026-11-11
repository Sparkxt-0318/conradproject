# SoilSight software and AI plan

Status: approved. Phases 1 (benchmark), 2 (Brain), 3 (Scout processing), 4 (Capture) and 5 (Field app and report) are built; phases 6 and 7 need real fields and are not started. Phase 2 met two of its three gate items. Phase 3 met its gate on public sample photos, not yet on bare soil or the project's own camera. Phase 4 met its gate in ArduPilot's simulator; nothing in it has run on the real camera or flight controller. Phase 5 met its gate in an emulated phone, not yet on a real one. See the result sections.

## Context

SoilSight flies a drone over a drained rice field, picks a handful of spots, tests each with a strip kit for plant-available cadmium, pH and nitrate, and gives the farmer a red-yellow-green map. This plan covers the software and AI ("Brain" and "Scout" processing) and its link to a real camera. The "tested starter code" named in the PDF (`capture.py`, `make_layers.py`, `first_spots.py`, `next_spot.py`) was not available, so this is a from-scratch design.

Before designing, I stress-tested the PDF's adaptive loop in simulation, and had the literature, competitors and camera hardware researched. The loop as written does not survive the test, and the camera plan has two hardware errors. The plan below keeps your product idea and changes how the software delivers it.

## What the stress test found

Synthetic 3 ha fields (2 m cells), 12 scenarios × 40 repeats, 9 method variants, 15 tests per field, kit error 22%. Synthetic means my assumptions (within-field spread, 40 m patch size, 20% short-range noise), not proof; real fields must replace them.

The code is in `soilsight/sim/` (numpy only) and reproduces every number in this table. With `--check`, each command fails if its results differ from the stored reference:

```bash
soilsight sim benchmark methods --check
```

```bash
soilsight sim benchmark budget --check
```

```bash
soilsight sim benchmark validation --check
```

| Finding | Number |
|---|---|
| A 15-test map barely beats calling the whole field safe | 74–79% correct vs 70% for "all safe"; about half of the unsafe area is missed |
| The PDF's loop (cLHS start, most-uncertain next, GP on 3 principal components, fitted settings) loses to an even spread of spots plus plain kriging | 74.2% vs 78.6% correct; map error 0.33 vs 0.29 |
| Its uncertainty is overconfident | "90%" ranges held the truth only 68–75% of the time (83–90% once model settings were constrained) |
| So the stop rule fires early and wrongly | In 10–28% of runs (most scenarios) it stopped at about 7 tests with accuracy no better than "all safe"; 42–70% of runs when hotspots were rare |
| The "90% of field at 90% confidence" target is out of reach | In the base case only 30–50% of the field is confidently classed at 15 tests, and 20–30% is still unsure at 60 tests |
| Drone layers fed through principal components gave no reliable gain | Even when layers explained 60% of cadmium variation: 78.7% vs 80.4% without them |
| Picking spots near the safe/unsafe boundary does help when hotspots are rare | Balanced accuracy 0.66 vs 0.57 at 15 tests; 0.79 vs 0.71 at 60 |
| Zone-level answers work at 15 tests | Six ~0.5 ha zones classed correctly 86–98% of the time |
| Mixing 3 cores into each test helps a little (corrected) | Across all 12 scenarios accuracy is unchanged (−0.4 to +0.8 points) and the missed unsafe area falls by 2 to 8 points. An earlier version of this row said "worth about 10 extra tests" from one scenario and 24 fields (78.6% vs 75.8%); the larger run does not support that |
| A real cell-level map needs 40–60 tests | About 85% correct at 60 |
| The "80% of results within 25% of lab" target fails more often than not at 22% kit error | True share within 25% is 75%; chance of passing with 20 split samples is 41%. Kit error must be ≤18% |

## What others do, and what the literature says

- **No commercial product closes the loop** (model picks spot → test → map updates) for soil. Rogo, Radicle, GeoPard, FieldView and Veris all use fixed grids or zones. That part of SoilSight is new as a product.
- **Closest prior work:** Marchant et al. 2013 (later samples placed to minimise misclassifying contamination), Huang et al. 2020 (adaptive wins when fewer than 15 extra samples), Huuskonen & Oksanen 2018 (drone soil-colour zones with automatic sample spots).
- **Handheld sensors failed on accuracy, not software:** Stenon's nitrogen R² was 0.30 per reading in independent tests; AgroCares showed a "limited relationship" to lab values. Honest uncertainty is the differentiator.
- **No paper shows an RGB camera predicting plant-available cadmium.** It sees organic matter, iron and moisture proxies at best.
- **cLHS is unstable at 5 points**; k-means coverage sampling did better (Wadoux et al. 2019).
- **Model settings cannot be learned from 5–15 points**; borrow them from imagery and other fields (Kerry & Oliver 2008).
- **The safe/unsafe line needs pH**: extractable cadmium alone predicts grain cadmium poorly (Simmons 2008, Brus 2009).
- **Nadir-only photos bend the 3D surface** ("doming", James & Robson 2014). Paddy relief of 10–20 cm is at the noise level without ground targets.

## Changes to the PDF's design

| PDF | Recommended | Why |
|---|---|---|
| Stop at "90% of field, 90% confident" | Fixed test budget; report zones plus a three-class map where yellow means "not sure, test more" | Target unreachable; stop rule misfires |
| cLHS + inlet/outlet start | Even spatial spread (k-means) + inlet/outlet for the first 60% of tests | Beat cLHS in simulation and literature |
| Next spot = most uncertain | Next spot = the one expected to remove the most misclassified area (after Marchant et al. 2013), kept 10 m from earlier spots | Matches a classification goal |
| GP on 3 principal components + position, settings fitted per field | Position-based GP that averages over many candidate settings drawn from priors, with known kit error; drone layers are off by default and can enter only as a simple trend | Overconfidence; with 15 tests one field cannot tell useful layers from junk |
| One core per test | Three cores mixed per test (walk mode) | Cuts the missed unsafe area by 2 to 8 points; accuracy unchanged |
| Kit reading vs GB 15618 | Risk from cadmium + pH through a published soil-to-grain formula, marked provisional | The standard is for total cadmium |
| Gradient boosting after 10 fields; U-Net | Shared settings and kit-to-lab correction across fields; hand-drawn or click-to-segment ditches | Too little data for either |
| Pi Camera Module 3 full resolution on Pi Zero 2 W | See below | Rolling-shutter smear of 34 cm (29 pixels) at 5 m/s |

## Sensor recommendation

- **Primary: Raspberry Pi Global Shutter Camera + 6 mm lens.** No smear, rigid lens, 2.3 cm per pixel at 40 m, which is ample for a 1 m grid. Costs twice as many photos (about 740 per 3.7 ha, one per second).
- **Fallback: Camera Module 3 in half-resolution mode** (2.4 cm per pixel, 17 ms readout, 4 pixels of smear, correctable in OpenDroneMap with an explicit readout value).
- **Best extra purchase: one RTK receiver (UM980 class).** Plain GPS returns you to within 2–5 m of a chosen spot; the method needs about 1 m.
- **Stretch: borrow a DJI Mavic 3 Multispectral** for one flight to test whether near-infrared adds anything. Do not buy (¥24,999).
- **Skip scanning lidar.** A flat bare paddy gives it nothing to lock onto, and photos plus ground targets reach similar height accuracy.
- **Corrections:** the TFmini Plus reads 4 m on dark soil, so it serves the 2.5 m sampling hover only; forward overlap should be 80% (1.2 s, not 1.5 s).

## Architecture

One Python package, sensor-agnostic, developed against simulation before hardware exists.

```
soilsight/
  sim/            synthetic fields + benchmark (the stress test, kept as a regression test)  [built]
  model/          GP on ln(Cd) with per-test noise, averaged over candidate settings; optional
                  trend; kit-to-lab correction; grain-risk formula; three-class and zone
                  summaries  [built]
  sampling/       initial spread, next spot by expected drop in misclassified area (which is
                  also the "value of one more test")  [built]
  brain.py        one object per field visit tying model and sampling together  [built]
  capture/        Pi-side: camera drivers, triggers, MAVLink clock matching, capture.csv,
                  flight planning, simulator rehearsal, camera bench test  [built; the Pi
                  camera driver has not run on hardware]
  ingest/         sensor profiles (TOML), photo positions from capture.csv or EXIF,
                  the flight controller's log or EXIF, ODM geo.txt, flight checks  [built]
  photogrammetry/ NodeODM via pyodm → orthophoto + surface model, with checks  [built]
  layers/         colour, terrain and distance layers on one grid, each with a quality
                  flag  [built]
  report/         map quality report and Field Health Report (HTML + JSON)  [built]
  scout.py        the photos-to-layers pipeline  [built]
  field.py        where everything lives in a field folder  [built]
  app/            field package, hash-chained measurement log, visit replay, and the phone
                  web app in static/ with the Brain in JavaScript  [built; no server-side
                  code, so no FastAPI]
  cli.py
tests/  data/sample_field/  docs/
```

Each field is one folder: GeoTIFF layers, the field package for the phone, an append-only, hash-chained measurement log (the "tamper-evident record"), and the tests as GeoJSON points (GeoJSON instead of a GeoPackage: no extra dependency, and QGIS opens it the same).

## Build order (each phase ends on a measured gate)

1. **Benchmark first (done).** The stress test lives in `sim/` with tests. Gate met: the ported code matches the original script run for run, and the stored reference results hold the table above.
2. **Brain (done, gate partly met).** `model/` and `sampling/`. Gate: beats both the PDF loop and the even-spread baseline on the benchmark; 90% ranges cover 85–95%. It beats the PDF loop and its ranges are honest on average, but it does not beat even spread on accuracy. See "Phase 2 result".
3. **Scout processing (done, gate met on sample photos).** `ingest/`, `photogrammetry/`, `layers/` on a public drone dataset. Gate: one command turns a photo folder into layers and a quality report. See "Phase 3 result".
4. **Capture (done in simulation; hardware steps open).** `capture/` against ArduPilot's simulator with a fake camera, then the real camera on a pole or cart, then the drone. Gate: every photo geotagged, overlap ≥80%, smear under 1 pixel. Met in the simulator; the hardware steps are written up in `docs/bench-test.md`. See "Phase 4 result".
5. **Field app and report (done; gate met in an emulated phone).** Gate: a full loop run on a phone with no signal. See "Phase 5 result".
6. **Truth field.** One field tested densely (about 50 kit tests, 10 lab splits). Replay every method on it offline. The simulation does not support a "same accuracy with X% fewer tests" claim, so the number to measure here is the one it does support: how much less unsafe area is missed, and how often a green zone is really unsafe.
7. **Across fields.** Shared settings and kit-to-lab correction once three fields exist.

## Phase 2 result

The Brain was run against the two reference strategies on all 12 scenarios × 40 fields, 15 single-core tests each, on the same fields and with the same test noise. Its settings were chosen beforehand on a separate set of fields.

```bash
soilsight sim benchmark brain --check
```

These are the numbers for the Brain as corrected during phase 5 (see "Correction" below).

| Average over 12 scenarios | PDF loop | Even spread + kriging | Brain |
|---|---|---|---|
| Field classed correctly | 75.8% | 79.3% | 78.8% |
| Balanced accuracy | 0.665 | 0.682 | 0.693 |
| Unsafe area missed | 51.3% | 52.3% | 48.0% |
| Map error (ln units) | 0.349 | 0.301 | 0.309 |
| Truth inside the "90%" ranges | 73.9% | 82.1% | 90.9% |

| Gate item | Result |
|---|---|
| Beats the PDF loop on accuracy | Pass: +3.0 points ± 0.4 (standard error, paired by field) |
| Beats even spread on accuracy | **Fail: −0.5 points ± 0.4.** It is level, not ahead. It misses less unsafe area (−4.3 points ± 1.6) and has better balanced accuracy (+1.1 points ± 0.6) |
| 90% ranges cover 85–95% | Pass: 90.9% on average, and inside 85–95% in eleven of twelve scenarios (the twelfth is 96%, slightly cautious) |

What this means:

- **No strategy beats an even spread on plain accuracy at 15 tests.** With 22% kit error and metre-scale scatter, 15 tests carry too little information for clever placement to matter much. The Brain's gain is where it looks: the safe/unsafe boundary, so less unsafe ground is called safe.
- **The Brain's uncertainty can be taken at face value.** Its 90% ranges hold the truth 91% of the time; the reference strategies' hold it 74% and 82%.
- **The Brain's zone answers are cautious.** It calls 27% of zones "not sure", and 2.7% of the zones it calls green are really unsafe.
- **The comparison is slightly unfair to the Brain.** The reference strategies are told the field's true metre-scale scatter; the Brain has to estimate it.
- **Drone layers stay off.** On the tuning fields they added nothing, even where they explained 60% of the cadmium pattern. With 15 tests the model's weight on "layers matter" averaged 0.42 for junk layers and 0.67 for real ones, against a starting value of 0.50. Whether layers help has to be learned across fields (phase 7).
- **The soil-to-grain formula has no coefficients built in.** None has been checked against this kit's extraction; they must come from a paper using the same extraction or from paired soil and grain samples.

**Correction (made in phase 5).** The Brain as first built in phase 2 was overconfident when tests were few: after one reading a little above the threshold it called the whole field unsafe at 95%. It kept the 24 best-fitting of its 512 candidate settings, and with few tests all candidates fit about equally, so "the 24 best" were simply the 24 that assume the least variation. It now takes a proportional draw across all candidates, and no longer centres readings on their own average. The change was checked on the tuning fields first (ranges holding the truth 91% of the time against 86%, accuracy unchanged) and then run once on the benchmark. Against the first version's benchmark numbers: accuracy 78.8% (was 78.7%), unsafe area missed 48.0% (was 47.6%), map error 0.309 (was 0.320), ranges holding the truth 90.9% (was 86.1%, with four scenarios below 85%), zones answered "not sure" 27% (was 22%). The gate verdict did not change.

## Phase 3 result

One command, `soilsight scout run <field>`, takes a folder of photos through OpenDroneMap stitching to the map layers and a quality report. It was run for real, not only against test doubles:

| Run | Photos | Stitching | Outcome |
|---|---|---|---|
| Brighton Beach sample set (DJI Phantom 3, 40 m) | 18 | 6 min | 18 of 18 photos used, alignment error 0.80 px, 2 cm orthophoto; with a boundary, inlet, outlet and ditch drawn on it, all 11 layers |
| Aukerman sample set (Sony compact on a fixed-wing, about 97 m) | 77 | 44 min | 76 of 77 photos used, alignment error 1.12 px, 8.4 ha mapped, under 5 GB of memory |
| Brighton again, positions from a `capture.csv` and a rolling-shutter profile, low-quality settings | 18 | 2 min | The stitcher's log shows it read our position file and ran rolling-shutter correction with the profile's readout value |

Timings are from a 4-core, 16 GB machine. A reduced copy of the first run's outputs is in `data/sample_field/` as a regression test.

What the software does:

- **Flight check before stitching** (`soilsight scout check`): photo count, positions, overlap along and across flight lines, motion blur and rolling-shutter smear, from photo positions and the camera profile. It takes seconds, so a bad flight can be re-flown the same day.
- **Stitching:** options are chosen from the camera profile; outputs are checked against OpenDroneMap's own statistics (photos used, alignment error, ground control points).
- **Eleven layers on a 1 m grid:** five soil-colour indices, three terrain layers, three distance layers. Each carries a note on how far to trust it.
- **Quality report:** one HTML page and a JSON file listing every check.

On synthetic rasters with known answers the layers recover the grey-card gains to within 2%, the 0.5% field tilt, and the position of a 15 cm hollow to within 2 m.

Problems the real runs exposed, all fixed and tested:

- The slope layer had a false cliff along the field edge.
- Ground speed was dragged down by skipped photos and turns (3.4 m/s reported for a 10 m/s flight).
- The terrain warning's wording was wrong for a scene with metres of relief.
- A camera whose photos lack a 35 mm-equivalent focal length got no profile; `field.toml` can now describe any camera.
- Photos that do not record flying height left overlap unchecked; the height is now worked out from the stitched pixel size.
- Low overlap found that way failed the whole report even though the photos had stitched; it is now a warning.

A test also caught that a partial stitching run could leave old outputs mixed with new ones; outputs are now replaced as a set.

Differences from the plan as first written:

- Sensor profiles and field settings are TOML, not YAML (no extra dependency).
- Photo positions go to OpenDroneMap in its `geo.txt`; photos are never rewritten, so no EXIF or XMP writing was needed.
- Reading positions from a flight log is deferred to phase 4, when the simulator can produce logs to test against.
- Low spots use "how far below the surroundings within 10 m", not a water-flow model, because flow models amplify centimetre noise on a nearly flat field.

Not shown by these runs:

- **Bare soil.** Both sample scenes are parks. The colour indices have not seen real soil.
- **Ground control points.** That path was tested against a stand-in server only.
- **Whether rolling-shutter correction helps.** The stitcher accepted the option, but its effect depends on the readout time, and the Raspberry Pi values in the profiles were computed from the driver, not measured.
- **Terrain on a paddy.** Without ground control points the report warns that hollows and rises of a few centimetres may be stitching artefacts; this warning will be the normal case until control points are used.

## Phase 4 result

The capture software was rehearsed in ArduPilot's simulator: the real ArduCopter 4.7.2 code flew a survey of a 60 × 40 m field at 40 m, generated by `soilsight capture plan`, while the capture software recorded it over a simulated telemetry port with a stand-in camera. The phase 3 flight check then judged each flight from its files alone.

| Simulated flight | Photos | With a position | Forward / side overlap | Smear | Position error |
|---|---|---|---|---|---|
| Capture computer spaces the photos, planned exactly at 80% | 79 | 79 | 79.9% / 70% | none | 11 cm typical, under 14 cm for 19 in 20 |
| Flight controller triggers the camera, planned exactly at 80% | 72 | 72 | 79.7% / 70% | none | 10 cm typical, under 14 cm for 19 in 20 |
| Capture computer spaces the photos, planned 3% closer (now the default) | 81 | 81 | 80.6% / 70% | none | 10 cm typical, under 14 cm for 19 in 20 |

Position error is the recorded position against where the simulator says the drone truly was at the instant of exposure. Smear is "none" because the profile is the global-shutter camera.

Gate: every photo geotagged, overlap ≥80%, smear under 1 pixel. **Met in simulation** with the default settings (third row). The first two flights landed a hair under 80% because a trigger always fires slightly after its planned point; planning 3% closer fixed that.

In the second flight the flight controller's own log held 72 trigger lines for 72 photos, agreeing with `capture.csv` to 4 cm typically (8 cm at most). With `capture.csv` removed, the pipeline took the positions from that log instead and passed the same checks.

What the software does:

- **Clock matching.** The flight controller stamps positions with its own clock; photos are stamped with the Pi's. The offset is learned from the messages (the smallest "arrival minus stamp" seen recently), and each photo's position is interpolated between the position samples on either side of its exposure. A photo's line is written only once a sample from after the exposure has arrived.
- **Three trigger modes:** by distance travelled (default), by time, or on the flight controller's command.
- **Safe files.** Each line of `capture.csv` is forced to disk as it is written; a second flight into the same folder continues the numbering; a photo that could not be placed is listed with an empty position, not dropped.
- **Planning.** `soilsight capture plan` gives photo spacing, line spacing and the fastest allowed speed for a camera and height, and writes a survey mission file for a boundary.
- **Flight log.** Photo positions can come from the flight controller's log when there is no capture computer.

Findings:

- **Camera Module 3 must fly slowly.** For smear under one pixel at 40 m: 1.4 m/s in binned mode, 0.2 m/s at full resolution. The Global Shutter Camera can fly 4.9 m/s, limited by an assumed one photo a second.
- **The camera will not point straight down.** The simulated drone leans 12° at 5 m/s, and a fixed mount leans with it. A steady lean stitches fine and the flight check now reports it, but there are no truly vertical photos without a gimbal.
- **One conversion was 0.5% off.** Degrees to metres used one constant for all latitudes; at 28°N that is 20 cm over a 40 m field. It now depends on latitude, checked against a map projection.
- **Speed estimate.** The flight check's ground speed was skewed by a hover before the first line; it is now estimated per flight line.

Not shown by simulation (procedure in `docs/bench-test.md`):

- **The Pi camera driver has never run.** It was written from the picamera2 manual.
- **Photo rate and focus stability.** The plan assumes one photo a second and a lens that holds focus; `soilsight capture bench` measures both.
- **Real GPS and serial delay.** Simulated positions were within about 10 cm of the truth. A real GPS without RTK is off by a metre or more, and a fixed link delay shifts every recorded position along the flight line by speed × delay.
- **Clock rates.** The clock matching assumes the Pi and the flight controller tick at the same rate, true to parts per million on hardware. The simulator therefore has to run in real time.

## Phase 5 result

The field app is a web page that runs a whole visit on the phone with no network: it guides to each spot, records samples and readings, refits the map after every result, proposes the next spot, and exports the visit's log. The Field Health Report is then made on the laptop from that log.

Gate: a full loop run on a phone with no signal. **Met in emulation.** An automated test loads the app in Chromium emulating a Pixel 7, cuts the network, reloads the page (it starts from the phone's own copy), and then acts as a person would: it moves the phone's GPS position to each spot, taps "Sample taken here", types the readings, and follows the spots the app proposes until the test budget is used. It reloads the app once mid-visit to check nothing is lost.

| Check | Result |
|---|---|
| Visit completed with the network off | Yes: 10 of 10 tests in the automated test (6 spread, 4 proposed by the app); 12 and 15 in two longer runs |
| The phone's log verifies on the laptop | Yes, every entry |
| Phone's zone answers against the reference Brain replaying the log | Same labels; chance-unsafe values agree to better than 0.000001 |
| The phone's Brain against the Python Brain, step by step through a loop | Same candidate weights, same map to 8 decimal places, same next spot at every step |
| A log edited after the fact | Refused, naming the first altered entry |

How it works:

- **The Brain runs on the phone.** The model and the next-spot rule are written a second time in JavaScript (`soilsight/app/static/brain.js`). Python stays the reference: a test runs both on the same field through a whole loop and fails if they differ. The field package carries the model's candidate settings, so both start from the same numbers.
- **One file per field.** `soilsight app package` writes everything the phone needs (grid, photo of the field, zones, protocol, first spots) into one JSON file of about 100 KB.
- **The log is the visit.** Every sample, result and proposed spot is an entry chained to the one before by its hash. The app's screen is rebuilt from the log, a correction is a new entry, and the report is made only from a log that verifies.
- **The report** gives one answer per zone (probably safe, not sure, probably unsafe) with the chance behind it, a three-class map, which samples to send to an accredited lab (about one in ten, plus the highest reading), every test, and the log's fingerprint. It states that it is a screening result and that it does not calculate a lime dose.

Differences from the plan as first written:

- **No server.** The plan had a FastAPI backend. With the Brain on the phone there is nothing for a server to do, so the app is plain files that any web host can serve.
- **Samples are collected first, tested after.** Because a kit test takes about ten minutes at the dock, the app separates "sample taken" from "result entered", and proposes the next spot once every collected sample has its result.
- **The threshold has no default.** The app asks for it at the start of the visit unless `field.toml` sets it, because the right value comes from the kit-to-lab comparison, which does not exist yet.

Not shown by emulation:

- **A real phone.** Only Chromium emulating an Android phone was used. Safari on an iPhone, a small screen in sunlight, and real GPS (off by 2 to 5 m without RTK, and slow to settle) are untested.
- **Getting the app onto the phone.** A phone gives a page GPS and keeps it for offline use only on an https address. The app's folder has to be put on an https web host once; this was not done. Access to such a host from China should be checked before relying on it.
- **Storage limits.** The log and field package live in the browser's storage, which a phone may clear if it runs short of space. Export the log at the end of every visit.
- **The readings.** All test visits used invented cadmium patterns and simulated kit readings.

## Verification

- `pytest`: GP checked against closed-form cases; design and geotag unit tests.
- `soilsight sim benchmark <experiment> --check`: regenerates the stress-test tables and fails if they change. `soilsight sim benchmark brain` also prints the phase 2 gate.
- `soilsight scout run data/sample_field --skip-stitch`: real stitched outputs → layers → quality report. With photos and a NodeODM server, `soilsight scout run <field>` does the whole chain.
- `soilsight capture simulate <field> --sitl ... --defaults ...`: the real ArduCopter code flies a survey in simulation while the capture software records it; `soilsight scout check <field>` then judges the flight.
- `soilsight capture bench`: photo rate and sharpness of the real camera, on the Pi. Procedure in `docs/bench-test.md`.
- `pytest tests/test_app.py`: runs the phone's Brain in Node against the Python Brain, and a whole visit in an emulated phone with the network cut (needs Node.js and Playwright; skipped without them).

## Decisions (agreed)

1. **Output framing:** zone answers plus a three-class map where yellow means "not sure, test more". The "90% of the field at 90% confidence" claim is dropped.
2. **Sensor:** Raspberry Pi Global Shutter Camera as primary, plus one RTK receiver.
3. **Tests per field:** 15 mixed-core tests for zone answers; one truth field at about 50. Three cores per test was confirmed after the corrected figure (accuracy unchanged, 2 to 8 points less unsafe area missed).
4. **App:** phone web app first, WeChat mini-program later.
5. **Starter code:** start fresh.

## Not verified

- The Conrad site could not be fetched from this session; judging weights are taken from your PDF's appendix.
- Prices in yuan, the Rangke patent's claims, and Camera Module 3 lens shake on drones.
- All accuracy figures above are from simulation.
- The Brain's priors (field spread, patch size, metre-scale scatter) are placeholders centred near the simulation's base case. Real ones must come from measured fields.
