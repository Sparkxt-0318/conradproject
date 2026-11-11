"""One line of a quality report."""
from __future__ import annotations

from dataclasses import asdict, dataclass

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"


@dataclass(frozen=True)
class Check:
    stage: str      # flight, stitching or layers
    name: str
    status: str     # ok, warn, fail, or info (reported, not judged)
    value: str      # the measured value, formatted for reading
    detail: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def graded(value: float, ok: float, warn: float, higher_is_better: bool = True) -> str:
    """ok / warn / fail from two thresholds."""
    if value != value:   # NaN
        return WARN
    if higher_is_better:
        return OK if value >= ok else WARN if value >= warn else FAIL
    return OK if value <= ok else WARN if value <= warn else FAIL


def worst(checks) -> str:
    order = {INFO: 0, OK: 0, WARN: 1, FAIL: 2}
    return max((c.status for c in checks), key=order.__getitem__, default=OK).replace(INFO, OK)
