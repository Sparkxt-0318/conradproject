"""The MAVLink connection to the flight controller, and the capture loop that runs on it."""
from __future__ import annotations

import time

from soilsight.capture.session import Session

# message id -> how many a second the capture computer asks the flight controller for
STREAMS = {33: 10.0,   # GLOBAL_POSITION_INT: position and velocity
           30: 10.0,   # ATTITUDE
           24: 2.0,    # GPS_RAW_INT: fix quality and accuracy
           2: 1.0}     # SYSTEM_TIME: the date and time
WANTED = {"GLOBAL_POSITION_INT", "ATTITUDE", "GPS_RAW_INT", "SYSTEM_TIME", "COMMAND_LONG"}
MAV_CMD_SET_MESSAGE_INTERVAL = 511


class MavLink:
    """connect is a pymavlink address: a serial port on the Pi ("/dev/serial0"), or a network
    address for the simulator ("tcp:127.0.0.1:5760", "udp:127.0.0.1:14550")."""

    def __init__(self, connect: str, baud: int = 115200, wait_s: float = 30.0, clock=time.monotonic):
        from pymavlink import mavutil
        self._mavutil, self.clock = mavutil, clock
        # The capture computer announces itself as a camera component of the same vehicle.
        self.conn = mavutil.mavlink_connection(connect, baud=baud, source_system=1,
                                               source_component=mavutil.mavlink.MAV_COMP_ID_CAMERA)
        if self.conn.wait_heartbeat(timeout=wait_s) is None:
            raise TimeoutError(f"no flight controller heard on {connect} within {wait_s:.0f} s")
        self._last_heartbeat = 0.0
        self.request_streams()

    def request_streams(self) -> None:
        for msg_id, hz in STREAMS.items():
            self.conn.mav.command_long_send(self.conn.target_system, self.conn.target_component,
                                            MAV_CMD_SET_MESSAGE_INTERVAL, 0, msg_id, 1e6 / hz, 0, 0, 0, 0, 0)

    def poll(self) -> list:
        """Messages that have arrived, as (type, fields, arrival time). Never waits."""
        now = self.clock()
        if now - self._last_heartbeat >= 1.0:
            m = self._mavutil.mavlink
            self.conn.mav.heartbeat_send(m.MAV_TYPE_CAMERA, m.MAV_AUTOPILOT_INVALID, 0, 0, m.MAV_STATE_ACTIVE)
            self._last_heartbeat = now
        out = []
        while True:
            msg = self.conn.recv_match(blocking=False)
            if msg is None:
                return out
            if msg.get_type() in WANTED:
                out.append((msg.get_type(), msg.to_dict(), self.clock()))

    def close(self) -> None:
        self.conn.close()


def run_capture(session: Session, link, stop=lambda: False, period_s: float = 0.01, sleep=time.sleep, log=None) -> int:
    """Run the capture loop until stop() is true. Returns the number of photos taken."""
    taken = 0
    session.camera.start()
    try:
        while not stop():
            for msg_type, fields, recv in link.poll():
                session.handle(msg_type, fields, recv)
            shot = session.step()
            if shot is not None:
                taken += 1
                if log:
                    log(f"photo {taken}: {shot.file}")
            sleep(period_s)
    finally:
        session.camera.stop()
        session.close()
    return taken
