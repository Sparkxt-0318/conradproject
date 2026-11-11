# Bench test: camera and capture computer

The capture software has flown surveys in ArduPilot's simulator with a stand-in camera. None of it has run on the real camera or a real flight controller. This is the order to prove it on hardware, cheapest step first. Each step says what passing looks like.

The Raspberry Pi camera driver (`PiCamera` in `soilsight/capture/camera.py`) was written from the picamera2 manual and has never run. Expect to fix something in step 2.

## 1. Install on the Pi

Raspberry Pi OS already has picamera2. Install soilsight so it can see it:

```bash
python3 -m venv --system-site-packages ~/soilsight-env
```

```bash
~/soilsight-env/bin/pip install -e .
```

## 2. Camera alone: photo rate

Point the camera at a detailed, well-lit scene (a printed page of text works) and take 50 photos as fast as it will go:

```bash
soilsight capture bench ~/bench1 --sensor pi_gs_6mm --count 50
```

**Pass:** "safe photo interval" is 1.00 s or less, which is what the flight plan assumes. If it is longer, plan with a slower speed or a greater height until the photo interval in `soilsight capture plan` is at least that long.

Look at a few photos. They should be evenly exposed and not blown out; if they are dark, raise `--gain` before lengthening `--exposure` past 0.001.

## 3. Camera alone: sharpness under vibration

Run the same test at one photo a second, once with the camera still and once while tapping the mount throughout:

```bash
soilsight capture bench ~/bench2 --sensor pi_gs_6mm --count 30 --interval 1
```

**Pass:** "soft photos 0" in both runs. Soft photos only while tapping mean the lens or the mount moves under vibration: stiffen or damp the mount. This matters most for Camera Module 3, whose lens floats on a coil.

## 4. Rolling-shutter cameras only: measure the readout time

Skip this for the Global Shutter Camera. For Camera Module 3, the readout times in `soilsight/ingest/sensors.toml` (68 ms full, 17 ms binned) were computed from the driver, not measured. Measure with OpenDroneMap's RSCalibration tool (https://opendronemap.github.io/RSCalibration/) and put the measured value in a `[camera]` table in `field.toml`.

## 5. With the flight controller, on the ground

Wire the Pi's serial port to a spare telemetry port on the flight controller (TELEM2 below) and set, in the flight controller's parameters:

| Parameter | Value | Meaning |
|---|---|---|
| `SERIAL2_PROTOCOL` | 2 | MAVLink 2 |
| `SERIAL2_BAUD` | 115 | 115200 baud |

Outdoors with a GPS fix, and with the minimum height set to zero so it shoots on the ground, carry the drone or push it on a cart along a straight line of about 30 m:

```bash
soilsight capture run ~/walk1 --sensor pi_gs_6mm --spacing 2 --min-height 0
```

Stop with Ctrl-C, then:

```bash
soilsight scout check ~/walk1
```

**Pass:** every photo has a position ("Photos with a position" is OK), `capture.csv` has one line per photo, and the photo spacing reported is close to 2 m.

## 6. First flight

Make the mission from the field boundary (drawn in QGIS or geojson.io and saved as `annotations.geojson` with a `boundary` feature):

```bash
soilsight capture plan --sensor pi_gs_6mm --height 40 --boundary my_field/annotations.geojson --mission my_field/survey.waypoints
```

Load `survey.waypoints` in Mission Planner or QGroundControl. On the Pi, before take-off:

```bash
soilsight capture run my_field --sensor pi_gs_6mm --height 40
```

After landing, stop it and judge the flight:

```bash
soilsight scout check my_field
```

**Pass (the phase 4 gate, on hardware):** every photo has a position, forward overlap is 80% or more, and rolling-shutter smear is under 1 pixel.

Then stitch and build the layers as in the README:

```bash
soilsight scout run my_field
```

## What simulation could not tell us

- Whether picamera2 behaves as the driver expects.
- The real photo rate and whether the lens holds focus in flight.
- How much the real drone pitches at survey speed (the simulated one leans 12° at 5 m/s).
- GPS error. Simulated positions were within 10 cm of the truth; a real GPS without RTK is off by a metre or more.
- The delay on a real serial link. A fixed delay shifts every photo's recorded position along the flight line by speed × delay: 5 cm per 10 ms at 5 m/s.
