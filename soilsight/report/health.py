"""The Field Health Report: what a farmer or cooperative receives after a visit.

It is made on the laptop from the verified measurement log, with the reference Brain, so it
shows what the tests support and nothing the phone merely had on screen. One HTML page that
prints to A4; no network or script needed to read it.
"""
from __future__ import annotations

import base64
import html
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from soilsight.app.package import cells_xy
from soilsight.model.classify import classify
from soilsight.sampling.acquire import misclassified_share

CLASS_RGB = {"green": (46, 139, 87), "yellow": (214, 158, 20), "red": (192, 57, 43)}
WORD = {"green": "Probably safe", "yellow": "Not sure", "red": "Probably unsafe"}
ADVICE = {
    "green": "No action from this screening. Check again after the next season.",
    "yellow": "Not settled. Add tests here, or send a sample to an accredited lab.",
    "red": "Send a sample to an accredited lab to confirm before planting rice here.",
}


def lab_candidates(visit, cell_p: np.ndarray) -> list:
    """Which samples to send to an accredited lab: about one in ten, starting with those taken
    where the map is least settled, and always the highest reading."""
    tests = [t for t in visit.tests if not t.lab]
    if not tests:
        return []
    n = max(1, math.ceil(len(tests) / 10))
    by_doubt = sorted(tests, key=lambda t: abs(cell_p[t.cell] - 0.5))[:n]
    top = max(tests, key=lambda t: t.reading)
    picked = list(by_doubt) + ([top] if top not in by_doubt else [])
    return [(t, "highest reading" if t is top and t not in by_doubt else "taken where the map is least settled") for t in picked]


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:          # older Pillow: a small fixed font
        return ImageFont.load_default()


def map_png(package: dict, cell_p: np.ndarray, visit, min_side: int = 640) -> bytes:
    """The three-class map with zone borders, zone letters and the samples."""
    g = package["grid"]
    nx, ny = g["nx"], g["ny"]
    rows, cols = np.asarray(g["rows"]), np.asarray(g["cols"])
    scale = max(2, int(math.ceil(min_side / min(nx, ny))))
    c = package["classes"]
    rgb = np.full((ny, nx, 3), 236, "uint8")
    label = classify(cell_p, c["lo"], c["hi"])
    palette = np.array([CLASS_RGB["green"], CLASS_RGB["yellow"], CLASS_RGB["red"]], "uint8")
    rgb[rows, cols] = palette[label]
    field = Image.fromarray(rgb).resize((nx * scale, ny * scale), Image.NEAREST)
    pad = max(14, scale * 2)                      # room for tests on the field's edge
    im = Image.new("RGB", (nx * scale + 2 * pad, ny * scale + 2 * pad), (236, 236, 236))
    im.paste(field, (pad, pad))
    draw = ImageDraw.Draw(im)

    zid = np.full((ny, nx), -1)
    zid[rows, cols] = package["zones"]["ids"]
    for r in range(ny):
        for col in range(nx):
            here = zid[r, col]
            if col + 1 < nx and zid[r, col + 1] != here:
                draw.line([(pad + (col + 1) * scale, pad + r * scale), (pad + (col + 1) * scale, pad + (r + 1) * scale)], fill=(20, 20, 20), width=2)
            if r + 1 < ny and zid[r + 1, col] != here:
                draw.line([(pad + col * scale, pad + (r + 1) * scale), (pad + (col + 1) * scale, pad + (r + 1) * scale)], fill=(20, 20, 20), width=2)
    big, small = _font(max(18, scale * 3)), _font(max(12, scale * 2))
    for z, name in enumerate(package["zones"]["names"]):       # a letter in each zone's north-west corner, clear of the tests
        rr, cc = np.nonzero(zid == z)
        if len(rr):
            k = np.argmin((rr - rr.min()) ** 2 + (cc - cc.min()) ** 2)
            draw.text((pad + (cc[k] + 2.2) * scale, pad + (rr[k] + 2.2) * scale), name, fill=(15, 15, 15), font=big, anchor="mm",
                      stroke_width=3, stroke_fill=(255, 255, 255))
    cell = g["cell_m"]
    for t in visit.tests:
        x, y = pad + t.x / cell * scale, pad + (ny - t.y / cell) * scale
        rad = max(9, scale * 1.6)
        draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=(142, 31, 20) if t.value > 0 else (29, 107, 65), outline=(255, 255, 255), width=2)
        draw.text((x, y), t.id[1:].lstrip("0"), fill=(255, 255, 255), font=small, anchor="mm")
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def write_health_report(path, visit, checks=None) -> dict:
    """Write the report for a replayed visit (soilsight.app.visit.replay). checks are the
    scout quality checks, if the field was scouted, so their warnings travel with the answer.
    Returns the report's content as a dict (also saved next to it as JSON)."""
    package = visit.package
    if not visit.tests:
        raise ValueError("the visit has no test results to report")
    brain = visit.brain()
    pred = brain.map()
    zones = visit.zones()
    xy = cells_xy(package)
    c = package["classes"]
    ids = np.asarray(package["zones"]["ids"])
    cell_area = package["grid"]["cell_m"] ** 2
    cls = classify(pred.p_unsafe, c["lo"], c["hi"])
    share = {name: float((cls == i).mean()) for i, name in enumerate(("green", "yellow", "red"))}
    wrong = misclassified_share(brain.posterior(), xy[np.asarray(package["plan"]["eval"])])
    lab = lab_candidates(visit, pred.p_unsafe)
    counts = {k: sum(z.label == k for z in zones) for k in ("green", "yellow", "red")}
    names = package["zones"]["names"]

    doc = dict(
        field=package["name"], made=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        threshold=visit.threshold, unit=visit.unit, tests=len(visit.tests), budget=package["protocol"]["budget"],
        cores=package["protocol"]["cores"], kit_sd=package["protocol"]["kit_sd"], finished=visit.finished,
        zones=[dict(zone=names[z.zone], label=z.label, p_unsafe=round(z.p_unsafe, 4), area_ha=round(float((ids == z.zone).sum() * cell_area / 1e4), 3),
                    tests=sum(int(ids[t.cell]) == z.zone for t in visit.tests)) for z in zones],
        field_share=share, expected_wrong_share=round(wrong, 4),
        lab=[dict(id=t.id, reading=t.reading, why=why) for t, why in lab],
        record=dict(entries=None, head=visit.head, started=visit.started, ended=visit.ended),
    )

    e = html.escape
    def pct(p):
        return f"{p * 100:.0f}%"

    verdict_bits = [f"{counts[k]} {WORD[k].lower()}" for k in ("green", "yellow", "red") if counts[k]]
    worst = "red" if counts["red"] else "yellow" if counts["yellow"] else "green"
    parts = [
        f"<h1>{e(package['name'])}: Field Health Report</h1>",
        f'<p class="sub">Soil cadmium screening · {len(visit.tests)} tests of {package["protocol"]["cores"]} mixed cores each · '
        f'{e((visit.ended or visit.started or "")[:10])}</p>',
        f'<p class="verdict {worst}"><strong>{len(zones)} zones: {", ".join(verdict_bits)}.</strong> '
        "This is a screening result, not an official classification. A zone marked probably unsafe needs an accredited-lab "
        "test before any decision that depends on it.</p>",
        '<div class="two"><figure><img alt="Map of the field in three classes" src="data:image/png;base64,'
        + base64.b64encode(map_png(package, pred.p_unsafe, visit)).decode() + '">'
        '<figcaption>Green: probably safe. Yellow: not sure. Red: probably unsafe. Circles are the tests, dark red where the '
        "reading was above the threshold. Letters are the zones. North is up.</figcaption></figure>",
    ]
    if package.get("image"):
        parts.append('<figure><img alt="Photo of the field" src="data:image/jpeg;base64,' + package["image"]["jpeg"]
                     + '"><figcaption>The field as the drone saw it.</figcaption></figure>')
    parts.append("</div>")

    parts.append("<h2>Zone by zone</h2><table><thead><tr><th>Zone</th><th>Answer</th><th>Chance unsafe</th><th>Area</th>"
                 "<th>Tests</th><th>What to do</th></tr></thead><tbody>")
    for z, row in zip(zones, doc["zones"]):
        parts.append(f'<tr><td><strong>{e(row["zone"])}</strong></td><td><span class="tag {z.label}">{WORD[z.label]}</span></td>'
                     f'<td class="num">{pct(z.p_unsafe)}</td><td class="num">{row["area_ha"]:.2f} ha</td><td class="num">{row["tests"]}</td>'
                     f"<td>{ADVICE[z.label]}</td></tr>")
    parts.append("</tbody></table>")
    parts.append(f'<p class="note">"Chance unsafe" is the chance that the zone\'s average is above the threshold, given these tests. '
                 f"Across the whole field, {pct(share['green'])} of the area is classed probably safe, {pct(share['yellow'])} not sure and "
                 f"{pct(share['red'])} probably unsafe; about {pct(wrong)} of the area is expected to be on the wrong side of the line. "
                 "This report does not calculate a lime dose.</p>")

    if lab:
        parts.append("<h2>Send to an accredited lab</h2><table><thead><tr><th>Sample</th><th>Kit reading</th><th>Why this one</th></tr></thead><tbody>")
        for t, why in lab:
            parts.append(f'<tr><td><strong>{e(t.id)}</strong></td><td class="num">{t.reading:.3g} {e(visit.unit)}</td><td>{why}</td></tr>')
        parts.append("</tbody></table><p class=\"note\">About one sample in ten goes to the lab. Enter the lab value in the app as a "
                     "lab result for the same sample and remake this report: a lab value counts for more than a kit reading.</p>")

    parts.append("<h2>The tests</h2><table><thead><tr><th>Sample</th><th>Zone</th><th>Reading</th><th>Against threshold</th><th>pH</th>"
                 "<th>Nitrate</th><th>Source</th></tr></thead><tbody>")
    for t in visit.tests:
        ratio = t.reading / visit.threshold
        parts.append(f'<tr><td><strong>{e(t.id)}</strong></td><td>{e(names[int(ids[t.cell])])}</td><td class="num">{t.reading:.3g} {e(visit.unit)}</td>'
                     f'<td><span class="tag {"red" if ratio > 1 else "green"}">{ratio:.2f} × threshold</span></td>'
                     f'<td class="num">{"" if t.ph is None else f"{t.ph:g}"}</td><td class="num">{"" if t.nitrate is None else f"{t.nitrate:g}"}</td>'
                     f'<td>{"accredited lab" if t.lab else "kit"}</td></tr>')
    parts.append("</tbody></table>")

    flagged = [ch for ch in (checks or []) if ch.get("status") in ("warn", "fail")]
    parts.append("<h2>How this was worked out</h2><ul>"
                 f"<li>Unsafe means a reading above <strong>{visit.threshold:g} {e(visit.unit)}</strong>, the threshold entered at the start of the visit.</li>"
                 f"<li>Each test mixed {package['protocol']['cores']} cores. Kit error was taken as {package['protocol']['kit_sd'] * 100:.0f}% "
                 "(one standard deviation); accredited-lab values as 5%.</li>"
                 "<li>The map is a statistical estimate between the tests. Its assumptions about how patchy a field is are placeholders "
                 "until they are set from fields measured in detail.</li>"
                 "<li>pH and nitrate are recorded for the grower but do not change the map.</li>"
                 + (f"<li>The drone map of this field had {len(flagged)} point(s) to check: "
                    + "; ".join(e(f"{ch['name']} ({ch['value']})") for ch in flagged[:6]) + ".</li>" if flagged else "")
                 + "</ul>")
    parts.append("<h2>Record</h2>"
                 f"<p class=\"note\">Made from the visit's measurement log, which was checked entry by entry before use"
                 + ("" if visit.finished else "; <strong>the visit was not marked finished</strong>")
                 + f". Visit started {e(visit.started)}{', ended ' + e(visit.ended) if visit.ended else ''}. "
                 f"Fingerprint of the last entry:</p><p class=\"hash\">{e(visit.head)}</p>")
    parts.append(f'<p class="foot">Made {e(doc["made"])} by soilsight.</p>')

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_PAGE.replace("{title}", e(package["name"])).replace("{body}", "\n".join(parts)))
    path.with_suffix(".json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def write_results_geojson(path, visit) -> int:
    """The tests as map points, for QGIS. Returns how many were written (those with a position)."""
    feats = []
    for t in visit.tests:
        if t.lon is None or t.lat is None:
            continue
        feats.append(dict(type="Feature", geometry=dict(type="Point", coordinates=[t.lon, t.lat]),
                          properties=dict(id=t.id, reading=t.reading, unit=visit.unit, above_threshold=bool(t.value > 0),
                                          ph=t.ph, nitrate=t.nitrate, lab=t.lab, cores=t.cores, time=t.time)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(type="FeatureCollection", features=feats), indent=1) + "\n")
    return len(feats)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}: Field Health Report</title>
<style>
:root { --bg:#fff; --fg:#17201b; --muted:#56625b; --line:#d8dbd5; --soft:#f4f6f3;
        --green:#1f7a48; --yellow:#8a6100; --red:#b3261e; --green-bg:#e3f3ea; --yellow-bg:#fbf0d3; --red-bg:#f9e2df; }
@media screen and (prefers-color-scheme: dark) { :root { --bg:#131814; --fg:#e8ece8; --muted:#9daaa2; --line:#2c3731; --soft:#1b221e;
        --green:#5fcf93; --yellow:#f0be4a; --red:#ff8478; --green-bg:#15301f; --yellow-bg:#3a2f10; --red-bg:#3d1b17; } }
body { background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; max-width:58rem; margin:0 auto; padding:1.5rem 1rem 3rem; }
h1 { font-size:1.7rem; margin:0; } h2 { font-size:1.15rem; margin:2rem 0 .5rem; }
.sub { color:var(--muted); margin:.2rem 0 1rem; }
.verdict { padding:.8rem 1rem; border-left:6px solid var(--line); background:var(--soft); }
.verdict.green { border-color:var(--green); } .verdict.yellow { border-color:var(--yellow); } .verdict.red { border-color:var(--red); }
.two { display:grid; grid-template-columns:repeat(auto-fit, minmax(17rem, 1fr)); gap:1rem; align-items:start; }
figure { margin:0; } figure img { width:100%; height:auto; border:1px solid var(--line); display:block; }
figcaption, .note { color:var(--muted); font-size:.88rem; margin-top:.4rem; }
table { border-collapse:collapse; width:100%; } th, td { text-align:left; padding:.45rem .5rem; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--muted); font-size:.82rem; font-weight:600; } td.num { font-variant-numeric:tabular-nums; white-space:nowrap; }
.tag { display:inline-block; font-size:.82rem; font-weight:650; padding:.1rem .55rem; border-radius:999px; white-space:nowrap; }
.tag.green { background:var(--green-bg); color:var(--green); } .tag.yellow { background:var(--yellow-bg); color:var(--yellow); } .tag.red { background:var(--red-bg); color:var(--red); }
ul { padding-left:1.2rem; } li { margin:.25rem 0; }
.hash { font:.8rem ui-monospace, monospace; word-break:break-all; color:var(--muted); }
.foot { color:var(--muted); font-size:.8rem; margin-top:2rem; }
@media (max-width:40rem) { table, thead, tbody, tr, td { display:block; } thead { display:none; } td { border:0; padding:.1rem 0; } tr { border-bottom:1px solid var(--line); padding:.5rem 0; } }
@media print { body { max-width:none; padding:0; font-size:11pt; } h2 { break-after:avoid; } tr, figure { break-inside:avoid; } @page { size:A4; margin:16mm; } }
</style></head><body>
{body}
</body></html>
"""
