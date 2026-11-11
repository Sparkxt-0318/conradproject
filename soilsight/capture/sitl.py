"""Fly a survey in ArduPilot's simulator while the capture software records it.

This is the rehearsal for a real flight. The simulated flight controller is the real
ArduCopter code; it flies the mission made by soilsight.capture.plan and talks MAVLink on
two ports, as a real one would: a ground-station port (used here to load the mission and
arm) and a telemetry port, which is where the capture computer is wired on the drone.

The simulator must run in real time (speed-up 1): the capture software's clock matching
assumes both clocks tick at the same rate, as they do on real hardware.

Building the simulator (once), from an ArduPilot checkout:

    ./waf configure --board sitl && ./waf copter

which gives build/sitl/bin/arducopter; its default parameters are in
Tools/autotest/default_params/copter.parm.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soilsight.capture.camera import FakeCamera
from soilsight.capture.link import MavLink
from soilsight.capture.plan import FlightPlan, survey_mission
from soilsight.capture.session import Session
from soilsight.capture.telemetry import State, distance_m
from soilsight.capture.trigger import AutopilotTrigger, DistanceTrigger
from soilsight.field import FieldDir

GCS_PORT, TELEMETRY_PORT = 5760, 5762
SIMSTATE_ID = 164


@dataclass(frozen=True)
class SimFlight:
    photos: int
    unplaced: int              # photos with no position
    extrapolated: int          # photos placed by carrying on from the last sample
    skipped_no_fix: int
    flight_s: float            # armed time
    error_median_m: float      # recorded position against the simulator's true position
    error_p95_m: float
    log_triggers: int | None   # camera triggers in the flight controller's own log, if read
    messages: list             # the flight controller's status texts


def _wait_port(connect: str, timeout_s: float):
    from pymavlink import mavutil
    end = time.monotonic() + timeout_s
    while True:
        try:
            conn = mavutil.mavlink_connection(connect, source_system=255)
            if conn.wait_heartbeat(timeout=5) is not None:
                return conn
            conn.close()
        except OSError:
            pass
        if time.monotonic() > end:
            raise TimeoutError(f"the simulator did not answer on {connect}")
        time.sleep(0.5)


def _upload_mission(gcs, items, tries: int = 8) -> None:
    """Load the mission, retrying while the flight controller is still starting up (just after
    boot it has no room for a mission yet)."""
    for attempt in range(tries):
        try:
            return _upload_once(gcs, items)
        except RuntimeError:
            if attempt == tries - 1:
                raise
            time.sleep(3.0)


def _upload_once(gcs, items, timeout_s: float = 30.0) -> None:
    while gcs.recv_match(type="MISSION_ACK", blocking=False) is not None:
        pass                                    # drop answers left over from an earlier try
    m = gcs.mav
    m.mission_count_send(gcs.target_system, gcs.target_component, len(items), 0)
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        msg = gcs.recv_match(type=["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"], blocking=True, timeout=2)
        if msg is None:
            continue
        if msg.get_type() == "MISSION_ACK":
            if msg.type != 0:
                raise RuntimeError(f"the flight controller rejected the mission (code {msg.type})")
            return
        it = items[msg.seq]
        p = it["params"]
        m.mission_item_int_send(gcs.target_system, gcs.target_component, msg.seq, it["frame"], it["command"],
                                0, 1, p[0], p[1], p[2], p[3], int(round(it["lat"] * 1e7)), int(round(it["lon"] * 1e7)),
                                it["alt"], 0)
    raise TimeoutError("mission upload did not finish")


def simulate_flight(field: FieldDir, sitl_binary, defaults, boundary, plan: FlightPlan, profile=None,
                    autopilot_trigger: bool = False, timeout_s: float = 900.0, log=print) -> SimFlight:
    """Fly a survey of boundary ((n, 2) lon/lat) in the simulator and capture it into field.

    Writes photos/ and capture.csv as a real flight would, and copies the flight controller's
    log to flight.bin. Returns what happened, including how far each recorded position was
    from where the simulator says the drone truly was.
    """
    from pymavlink import mavutil
    items = survey_mission(boundary, plan, autopilot_trigger=autopilot_trigger)
    home_lon, home_lat = items[0]["lon"], items[0]["lat"]
    work = Path(tempfile.mkdtemp(prefix="soilsight_sitl_"))
    extra = work / "soilsight.parm"
    extra.write_text("\n".join([
        "AUTO_OPTIONS 3",                      # may arm and take off in AUTO with no pilot
        f"WPNAV_SPEED {plan.speed_ms * 100:.0f}",
        "CAM1_TYPE 5" if autopilot_trigger else "CAM1_TYPE 0",   # 5: camera is a MAVLink device
        "SERIAL2_PROTOCOL 2", "SERIAL2_BAUD 115",                # the capture computer's telemetry port
        "SIM_SPEEDUP 1",
    ]) + "\n")
    proc = subprocess.Popen([str(sitl_binary), "-w", "--model", "+", "--speedup", "1",
                             "--defaults", f"{defaults},{extra}", "--home", f"{home_lat:.7f},{home_lon:.7f},50,0", "-I0"],
                            cwd=work, stdout=open(work / "sitl.out", "w"), stderr=subprocess.STDOUT)
    gcs = link = session = None
    texts, truth = [], []
    try:
        gcs = _wait_port(f"tcp:127.0.0.1:{GCS_PORT}", 60)
        log("simulator is up; loading the mission")
        gcs.mav.command_long_send(gcs.target_system, gcs.target_component, 511, 0, SIMSTATE_ID, 50000, 0, 0, 0, 0, 0)
        gcs.mav.command_long_send(gcs.target_system, gcs.target_component, 511, 0, 33, 200000, 0, 0, 0, 0, 0)
        _upload_mission(gcs, items)

        link = MavLink(f"tcp:127.0.0.1:{TELEMETRY_PORT}")
        camera = FakeCamera(profile)
        trigger = AutopilotTrigger() if autopilot_trigger else DistanceTrigger(plan.photo_spacing_m, min_agl_m=plan.height_m * 0.9)
        session = Session(field, camera, trigger)

        auto = gcs.mode_mapping()["AUTO"]
        armed_at = disarmed_at = None
        last_try = 0.0
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            now = time.monotonic()
            for msg in iter(lambda: gcs.recv_match(blocking=False), None):
                kind = msg.get_type()
                if kind == "SIMSTATE":
                    truth.append((time.monotonic(), msg.lat / 1e7, msg.lng / 1e7))
                elif kind == "STATUSTEXT":
                    texts.append(msg.text)
                elif kind == "HEARTBEAT" and msg.get_srcComponent() == 1:
                    is_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    if is_armed and armed_at is None:
                        armed_at = now
                        log("armed; flying the survey")
                    if armed_at is not None and not is_armed and now - armed_at > 20:
                        disarmed_at = now
            if armed_at is None and now - last_try > 3.0:        # keep asking until the pre-arm checks pass
                gcs.mav.set_mode_send(gcs.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, auto)
                gcs.mav.command_long_send(gcs.target_system, gcs.target_component,
                                          mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
                last_try = now
            for msg_type, fields, recv in link.poll():
                session.handle(msg_type, fields, recv)
            shot = session.step()
            if shot is not None and len(session.shots) % 10 == 0:
                log(f"  {len(session.shots)} photos placed")
            if disarmed_at is not None:
                break
            time.sleep(0.005)
        else:
            raise TimeoutError(f"the simulated flight did not finish in {timeout_s:.0f} s; last messages: {texts[-5:]}")
        session.close()
        flight_s = disarmed_at - armed_at
        log(f"landed after {flight_s:.0f} s with {len(session.shots)} photos")
    finally:
        if session is not None:
            session.close()
        for c in (link, gcs):
            if c is not None:
                c.close()
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()

    # recorded position against the simulator's truth at each exposure
    t = np.array([r[0] for r in truth])
    errors = []
    for shot in session.shots:
        if shot.state is None or not len(t) or not t[0] <= shot.exposure.mid_mono <= t[-1]:
            continue
        lat = np.interp(shot.exposure.mid_mono, t, [r[1] for r in truth])
        lon = np.interp(shot.exposure.mid_mono, t, [r[2] for r in truth])
        errors.append(distance_m(shot.state, State(lat, lon, 0, 0, 0)))
    logs = sorted((work / "logs").glob("*.BIN"))
    triggers = None
    if logs:
        shutil.copy(logs[-1], field.flight_log)
        from soilsight.ingest.flightlog import read_camera_log
        triggers = len(read_camera_log(field.flight_log))
    shutil.copy(work / "sitl.out", field.root / "sitl.out")
    shutil.rmtree(work, ignore_errors=True)
    return SimFlight(photos=len(session.shots), unplaced=session.unplaced, extrapolated=session.extrapolated,
                     skipped_no_fix=session.skipped_no_fix, flight_s=float(flight_s),
                     error_median_m=float(np.median(errors)) if errors else float("nan"),
                     error_p95_m=float(np.percentile(errors, 95)) if errors else float("nan"),
                     log_triggers=triggers, messages=texts)
