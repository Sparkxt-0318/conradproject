"""The measurement log: every sample and result of a visit, in order, tamper-evident.

Each entry's body is a line of JSON holding the entry's number, time, type, data and the
hash of the entry before it. The entry's own hash is the SHA-256 of that body. Changing,
removing or reordering any entry breaks every hash after it, so a log that verifies is the
log as it was written.

The hash is taken over the body exactly as stored, never over a re-encoding of it, so the
phone app (JavaScript) and this module agree whatever either language does with numbers.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

FORMAT = "soilsight-log/1"
GENESIS = "0" * 64


class RecordError(ValueError):
    """The log is damaged or was altered."""


@dataclass(frozen=True)
class Entry:
    seq: int
    time: str          # ISO 8601, as the phone recorded it
    type: str
    data: dict
    hash: str


def make_entry(seq: int, time: str, type: str, data: dict, prev: str) -> dict:
    """One stored entry: {"body": <JSON text>, "hash": <hex>}."""
    body = json.dumps(dict(seq=seq, time=time, type=type, data=data, prev=prev), separators=(",", ":"), ensure_ascii=False)
    return dict(body=body, hash=hashlib.sha256(body.encode()).hexdigest())


def verify(stored: list) -> list:
    """Check the chain and return the entries. Raises RecordError naming the first bad entry."""
    out, prev = [], GENESIS
    for i, item in enumerate(stored):
        body = item.get("body", "")
        if hashlib.sha256(body.encode()).hexdigest() != item.get("hash"):
            raise RecordError(f"entry {i + 1} was altered: its contents do not match its hash")
        try:
            b = json.loads(body)
        except json.JSONDecodeError as err:
            raise RecordError(f"entry {i + 1} is not readable") from err
        if b.get("prev") != prev:
            raise RecordError(f"entry {i + 1} does not follow entry {i}: an entry was removed, added or reordered")
        if b.get("seq") != i + 1:
            raise RecordError(f"entry {i + 1} is numbered {b.get('seq')}")
        out.append(Entry(b["seq"], b["time"], b["type"], b["data"], item["hash"]))
        prev = item["hash"]
    return out


def read_log(path) -> list:
    """Read a log file exported by the app and verify it."""
    doc = json.loads(Path(path).read_text())
    if doc.get("format") != FORMAT:
        raise RecordError(f"{path} is not a {FORMAT} file")
    return verify(doc.get("entries", []))


def write_log(stored: list, path, field: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(format=FORMAT, field=field, entries=stored), indent=1, ensure_ascii=False) + "\n")


class LogWriter:
    """Builds a log in Python, for tests and for importing results typed in on a laptop."""

    def __init__(self):
        self.stored: list = []

    def add(self, type: str, data: dict, time: str) -> dict:
        prev = self.stored[-1]["hash"] if self.stored else GENESIS
        item = make_entry(len(self.stored) + 1, time, type, data, prev)
        self.stored.append(item)
        return item
