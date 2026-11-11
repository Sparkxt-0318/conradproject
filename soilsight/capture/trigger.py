"""When to take the next photo."""
from __future__ import annotations

from soilsight.capture.telemetry import State, distance_m


class DistanceTrigger:
    """A photo every spacing_m of ground travelled, once the drone is above min_agl_m
    (0 or less: at any height, for tests on the ground).

    Spacing by distance keeps the overlap right whatever the speed or wind.
    """

    def __init__(self, spacing_m: float, min_agl_m: float = 10.0):
        if spacing_m <= 0:
            raise ValueError("spacing_m must be positive")
        self.spacing_m, self.min_agl_m = spacing_m, min_agl_m
        self._last = None

    def due(self, state: State, now: float) -> bool:
        if self.min_agl_m > 0 and state.agl_m < self.min_agl_m:
            return False
        return self._last is None or distance_m(self._last, state) >= self.spacing_m

    def fired(self, state: State, now: float) -> None:
        self._last = state

    def command(self, command: int, params: tuple) -> None:
        pass


class IntervalTrigger:
    """A photo every interval_s seconds, once the drone is above min_agl_m."""

    def __init__(self, interval_s: float, min_agl_m: float = 10.0):
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.interval_s, self.min_agl_m = interval_s, min_agl_m
        self._last = None

    def due(self, state: State, now: float) -> bool:
        high_enough = self.min_agl_m <= 0 or state.agl_m >= self.min_agl_m
        return high_enough and (self._last is None or now - self._last >= self.interval_s)

    def fired(self, state: State, now: float) -> None:
        self._last = now

    def command(self, command: int, params: tuple) -> None:
        pass


MAV_CMD_DO_DIGICAM_CONTROL = 203
MAV_CMD_IMAGE_START_CAPTURE = 2000


class AutopilotTrigger:
    """A photo whenever the flight controller asks for one.

    With ArduPilot's camera type set to MAVLink (CAM1_TYPE = 5) and a trigger distance in the
    mission, the controller sends a camera command at each photo point. The controller then
    also writes each trigger to its own flight log.
    """

    def __init__(self):
        self._pending = 0

    def command(self, command: int, params: tuple) -> None:
        if command == MAV_CMD_DO_DIGICAM_CONTROL and len(params) >= 5 and params[4] == 1:
            self._pending += 1
        elif command == MAV_CMD_IMAGE_START_CAPTURE:
            self._pending += 1

    def due(self, state: State, now: float) -> bool:
        return self._pending > 0

    def fired(self, state: State, now: float) -> None:
        self._pending = max(self._pending - 1, 0)
