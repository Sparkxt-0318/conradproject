# SoilSight software and AI plan

Status: approved. Phase 1 (the benchmark) and phase 2 (the Brain) are built; phases 3 to 7 are not started. Phase 2 met two of its three gate items; see "Phase 2 result".

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
  capture/        Pi-side: picamera2 trigger loop, MAVLink time sync, sidecar CSV
  ingest/         sensor profiles (YAML), geotag from live MAVLink or flight log,
                  EXIF/XMP, ODM geo.txt and gcp_list.txt
  photogrammetry/ NodeODM via pyodm → orthophoto + surface model + quality report
  layers/         colour, terrain and distance layers on one grid, each with a quality flag
  report/         Field Health Report (HTML → PDF)
  app/            phone web app (FastAPI + offline map): next spot, navigate, enter result
  cli.py
tests/  data/sample_field/  docs/
```

Each field is one folder: GeoTIFF layers, a GeoPackage of spots and results, and an append-only, hash-chained measurement log (the "tamper-evident record").

## Build order (each phase ends on a measured gate)

1. **Benchmark first (done).** The stress test lives in `sim/` with tests. Gate met: the ported code matches the original script run for run, and the stored reference results hold the table above.
2. **Brain (done, gate partly met).** `model/` and `sampling/`. Gate: beats both the PDF loop and the even-spread baseline on the benchmark; 90% ranges cover 85–95%. It beats the PDF loop and its ranges are honest on average, but it does not beat even spread on accuracy. See "Phase 2 result".
3. **Scout processing.** `ingest/`, `photogrammetry/`, `layers/` on a public drone dataset. Gate: one command turns a photo folder into layers and a quality report.
4. **Capture.** `capture/` against ArduPilot's simulator with a fake camera, then the real camera on a pole or cart, then the drone. Gate: every photo geotagged, overlap ≥80%, smear under 1 pixel.
5. **Field app and report.** Gate: a full loop run on a phone with no signal.
6. **Truth field.** One field tested densely (about 50 kit tests, 10 lab splits). Replay every method on it offline. The simulation does not support a "same accuracy with X% fewer tests" claim, so the number to measure here is the one it does support: how much less unsafe area is missed, and how often a green zone is really unsafe.
7. **Across fields.** Shared settings and kit-to-lab correction once three fields exist.

## Phase 2 result

The Brain was run against the two reference strategies on all 12 scenarios × 40 fields, 15 single-core tests each, on the same fields and with the same test noise. Its settings were chosen beforehand on a separate set of fields.

```bash
soilsight sim benchmark brain --check
```

| Average over 12 scenarios | PDF loop | Even spread + kriging | Brain |
|---|---|---|---|
| Field classed correctly | 75.8% | 79.3% | 78.7% |
| Balanced accuracy | 0.665 | 0.682 | 0.694 |
| Unsafe area missed | 51.3% | 52.3% | 47.6% |
| Map error (ln units) | 0.349 | 0.301 | 0.320 |
| Truth inside the "90%" ranges | 73.9% | 82.1% | 86.1% |

| Gate item | Result |
|---|---|
| Beats the PDF loop on accuracy | Pass: +2.9 points ± 0.4 (standard error, paired by field) |
| Beats even spread on accuracy | **Fail: −0.6 points ± 0.4.** It is level, not ahead. It misses less unsafe area (−4.8 points ± 1.6) and has better balanced accuracy (+1.3 points ± 0.6) |
| 90% ranges cover 85–95% | Pass on average (86.1%); below 85% in four scenarios (81–84%) |

What this means:

- **No strategy beats an even spread on plain accuracy at 15 tests.** With 22% kit error and metre-scale scatter, 15 tests carry too little information for clever placement to matter much. The Brain's gain is where it looks: the safe/unsafe boundary, so less unsafe ground is called safe.
- **The Brain's zone answers are cautious.** It calls 22% of zones "not sure", and 2.7% of the zones it calls green are really unsafe.
- **The comparison is slightly unfair to the Brain.** The reference strategies are told the field's true metre-scale scatter; the Brain has to estimate it.
- **Drone layers stay off.** On the tuning fields they added nothing, even where they explained 60% of the cadmium pattern. With 15 tests the model's weight on "layers matter" averaged 0.42 for junk layers and 0.67 for real ones, against a starting value of 0.50. Whether layers help has to be learned across fields (phase 7).
- **The soil-to-grain formula has no coefficients built in.** None has been checked against this kit's extraction; they must come from a paper using the same extraction or from paired soil and grain samples.

## Verification

- `pytest`: GP checked against closed-form cases; design and geotag unit tests.
- `soilsight sim benchmark <experiment> --check`: regenerates the stress-test tables and fails if they change. `soilsight sim benchmark brain` also prints the phase 2 gate.
- `soilsight pipeline run data/sample_field`: photos → layers → spots → report, end to end.
- Simulator flight: trigger messages in, geotagged files out, timing error under 50 ms.
- Camera bench test: measured readout, focus stability under vibration, capture rate on the chosen Pi.

## Decisions (agreed)

1. **Output framing:** zone answers plus a three-class map where yellow means "not sure, test more". The "90% of the field at 90% confidence" claim is dropped.
2. **Sensor:** Raspberry Pi Global Shutter Camera as primary, plus one RTK receiver.
3. **Tests per field:** 15 mixed-core tests for zone answers; one truth field at about 50. (Agreed when mixed cores looked worth about 10 extra tests; the corrected figure is smaller, so this is worth a second look.)
4. **App:** phone web app first, WeChat mini-program later.
5. **Starter code:** start fresh.

## Not verified

- The Conrad site could not be fetched from this session; judging weights are taken from your PDF's appendix.
- Prices in yuan, the Rangke patent's claims, and Camera Module 3 lens shake on drones.
- All accuracy figures above are from simulation.
- The Brain's priors (field spread, patch size, metre-scale scatter) are placeholders centred near the simulation's base case. Real ones must come from measured fields.
