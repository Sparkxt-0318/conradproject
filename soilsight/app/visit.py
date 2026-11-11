"""A field visit, rebuilt from its measurement log with the reference (Python) Brain.

The phone app decides in the field; this module replays what it recorded, so the report
is made from the verified log and not from whatever the phone had on screen.

Log entry types and their data:

  visit_started    threshold, unit, package (name and made-time of the field package)
  sample_taken     id, cell (the planned cell), x, y (where it was really taken, metres from
                   the grid corner), lon, lat, gps_accuracy_m, cores, planned (true for the
                   first spread, false for a spot the loop proposed)
  result_entered   id, reading (in the visit's unit), ph, nitrate, noise_sd, lab (true for
                   an accredited-lab value). A later entry for the same id replaces an
                   earlier one; void = true withdraws the result.
  spot_proposed    cell, gain, misclassified_now (what the loop suggested, for the record)
  visit_finished
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from soilsight.app.package import cells_xy
from soilsight.app.record import Entry
from soilsight.brain import Brain
from soilsight.model.classify import zone_answers


@dataclass(frozen=True)
class Test:
    id: str
    x: float
    y: float
    cell: int
    reading: float
    value: float              # ln(reading / threshold)
    noise_sd: float
    cores: int
    ph: float | None = None
    nitrate: float | None = None
    lab: bool = False
    lon: float | None = None
    lat: float | None = None
    time: str = ""


@dataclass
class Visit:
    package: dict
    threshold: float | None = None
    unit: str = ""
    samples: dict = field(default_factory=dict)      # id -> the sample_taken data
    tests: list = field(default_factory=list)        # Tests with a current result, in sampling order
    proposals: list = field(default_factory=list)
    finished: bool = False
    head: str = ""                                   # hash of the last log entry
    started: str = ""
    ended: str = ""

    def brain(self) -> Brain:
        b = Brain.from_package(self.package)
        for t in self.tests:
            b.add_result((t.x, t.y), t.value, noise_sd=t.noise_sd, cores=t.cores)
        return b

    def map(self):
        return self.brain().map()

    def zones(self) -> list:
        z = self.package["zones"]
        xy = cells_xy(self.package)
        c = self.package["classes"]
        post = self.brain().posterior()
        out = []
        for i, cells in enumerate(z["samples"]):      # the same cells the phone averages over
            m = post.area_mean(xy[cells])
            from soilsight.model.classify import CLASS_NAMES, ZoneAnswer, classify
            p = float(m.p_unsafe)
            out.append(ZoneAnswer(i, float(m.mean), float(m.sd), p, CLASS_NAMES[int(classify(p, c["lo"], c["hi"]))]))
        return out

    def next_spot(self):
        return self.brain().next_spot()


def replay(package: dict, entries: list) -> Visit:
    """Rebuild a visit from verified log entries (see soilsight.app.record.read_log)."""
    v = Visit(package, threshold=package.get("threshold"), unit=package.get("unit", ""))
    results = {}
    for e in entries:
        d = e.data
        if e.type == "visit_started":
            v.threshold = d.get("threshold", v.threshold)
            v.unit = d.get("unit", v.unit)
            v.started = e.time
        elif e.type == "sample_taken":
            v.samples[d["id"]] = dict(d, time=e.time)
        elif e.type == "result_entered":
            if d.get("void"):
                results.pop(d["id"], None)
            else:
                results[d["id"]] = d
        elif e.type == "spot_proposed":
            v.proposals.append(d)
        elif e.type == "visit_finished":
            v.finished, v.ended = True, e.time
        v.head = e.hash
    if results and not v.threshold:
        raise ValueError("the log has results but no threshold to judge them against")
    p = package["protocol"]
    for sid, s in v.samples.items():
        r = results.get(sid)
        if r is None:
            continue
        v.tests.append(Test(id=sid, x=float(s["x"]), y=float(s["y"]), cell=int(s["cell"]), reading=float(r["reading"]),
                            value=float(np.log(r["reading"] / v.threshold)),
                            noise_sd=float(r.get("noise_sd", p["kit_sd"])), cores=int(s.get("cores", p["cores"])),
                            ph=r.get("ph"), nitrate=r.get("nitrate"), lab=bool(r.get("lab")),
                            lon=s.get("lon"), lat=s.get("lat"), time=s.get("time", "")))
    return v
