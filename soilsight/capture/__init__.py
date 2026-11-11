"""The capture computer on the drone: take photos and record where each was taken."""

from soilsight.capture.camera import Exposure, FakeCamera, PiCamera
from soilsight.capture.plan import FlightPlan, flight_plan, survey_lines, survey_mission, write_waypoints
from soilsight.capture.session import Session
from soilsight.capture.telemetry import State, Telemetry, distance_m
from soilsight.capture.trigger import AutopilotTrigger, DistanceTrigger, IntervalTrigger

__all__ = ["AutopilotTrigger", "DistanceTrigger", "Exposure", "FakeCamera", "FlightPlan", "IntervalTrigger",
           "PiCamera", "Session", "State", "Telemetry", "distance_m", "flight_plan", "survey_lines",
           "survey_mission", "write_waypoints"]
