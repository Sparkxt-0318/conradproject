"""The quality report: every check from flight to layers, as JSON for programs and one
self-contained HTML page for people."""
from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from soilsight.field import FieldDir
from soilsight.report.checks import FAIL, INFO, OK, WARN, worst

_STOPS = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)  # viridis
_STAGES = (("flight", "Flight"), ("stitching", "Stitching"), ("layers", "Layers"))
_WORD = {OK: "OK", WARN: "Check", FAIL: "Problem", INFO: ""}


def layer_png(a: np.ndarray, min_side: int = 240) -> bytes:
    """A layer as a small colour image: dark purple is low, yellow is high, grey is no data."""
    valid = np.isfinite(a)
    rgb = np.full(a.shape + (3,), 200, "uint8")
    if valid.any():
        lo, hi = np.nanpercentile(a, [2, 98])
        t = np.clip((a[valid] - lo) / (hi - lo if hi > lo else 1), 0, 1) * (len(_STOPS) - 1)
        i = np.minimum(t.astype(int), len(_STOPS) - 2)
        rgb[valid] = (_STOPS[i] + (_STOPS[i + 1] - _STOPS[i]) * (t - i)[:, None]).astype("uint8")
    im = Image.fromarray(rgb)
    scale = max(1, int(np.ceil(min_side / min(a.shape))))
    im = im.resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _img(png: bytes, alt: str) -> str:
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{base64.b64encode(png).decode()}">'


def write_quality(fd: FieldDir, checks, stack=None, stopped: str = "") -> dict:
    """Write report/quality.json and report/quality.html. stopped says why the pipeline
    ended early, if it did. Returns the JSON content."""
    fd.report_dir.mkdir(parents=True, exist_ok=True)
    overall = FAIL if stopped else worst(checks)
    cfg = fd.config()
    doc = dict(field=cfg.name or fd.root.name, made=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               overall=overall, stopped=stopped, checks=[c.as_dict() for c in checks],
               layers=[dict(name=n, **stack.quality.get(n, {})) for n in stack.names] if stack else [])
    fd.quality_json.write_text(json.dumps(doc, indent=1) + "\n")

    e = html.escape
    headline = {OK: "Ready to use", WARN: "Usable, with points to check", FAIL: "Not usable yet"}[overall]
    parts = [f"<h1>{e(doc['field'])}: map quality</h1>",
             f'<p class="verdict {overall}"><strong>{headline}.</strong> '
             + (e(stopped) if stopped else f"{sum(c.status == WARN for c in checks)} point(s) to check, "
                                           f"{sum(c.status == FAIL for c in checks)} problem(s).") + "</p>"]
    for key, title in _STAGES:
        rows = [c for c in checks if c.stage == key]
        if not rows:
            continue
        parts.append(f"<h2>{title}</h2><table><thead><tr><th>Check</th><th>Result</th><th>Value</th>"
                     "<th>Notes</th></tr></thead><tbody>")
        for c in rows:
            parts.append(f'<tr><td>{e(c.name)}</td><td><span class="tag {c.status}">{_WORD[c.status]}</span></td>'
                         f"<td>{e(c.value)}</td><td>{e(c.detail)}</td></tr>")
        parts.append("</tbody></table>")
    if stack is not None:
        from soilsight.layers.build import DESCRIPTIONS
        parts.append("<h2>Layer previews</h2><p>Dark purple is low, yellow is high, grey has no data. "
                     "Each preview is stretched to its own range.</p><div class=\"grid\">")
        for n in stack.names:
            q = stack.quality.get(n, {})
            parts.append(f"<figure>{_img(layer_png(stack.layer(n)), n)}<figcaption><strong>{e(n)}</strong> "
                         f'<span class="tag {q.get("status", OK)}">{_WORD[q.get("status", OK)]}</span><br>'
                         f"{e(DESCRIPTIONS[n])}<br><em>{e(q.get('note', ''))}</em></figcaption></figure>")
        parts.append("</div>")
    parts.append(f"<p class=\"foot\">Made {e(doc['made'])} by soilsight.</p>")
    fd.quality_html.write_text(_PAGE.replace("{title}", e(doc["field"])).replace("{body}", "\n".join(parts)))
    return doc


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}: map quality</title>
<style>
:root { --bg:#fff; --fg:#1b1f23; --muted:#5b6570; --line:#d9dee3; --ok:#1a7f4b; --warn:#9a6700; --fail:#b42318; --soft:#f4f6f8; }
@media (prefers-color-scheme: dark) { :root { --bg:#14181c; --fg:#e6e9ec; --muted:#9aa4ae; --line:#2e363e; --ok:#4ac27f; --warn:#e0a82e; --fail:#f2766b; --soft:#1c2228; } }
body { background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; max-width:60rem; margin:0 auto; padding:1.5rem 1rem 3rem; }
h1 { font-size:1.6rem; margin:0 0 .5rem; } h2 { font-size:1.15rem; margin:2rem 0 .5rem; }
table { border-collapse:collapse; width:100%; } th, td { text-align:left; padding:.4rem .6rem; border-bottom:1px solid var(--line); vertical-align:top; }
th { font-weight:600; color:var(--muted); font-size:.85rem; }
.verdict { padding:.7rem 1rem; border-left:4px solid var(--line); background:var(--soft); }
.verdict.ok { border-color:var(--ok); } .verdict.warn { border-color:var(--warn); } .verdict.fail { border-color:var(--fail); }
.tag { font-size:.8rem; font-weight:600; white-space:nowrap; } .tag.ok { color:var(--ok); } .tag.warn { color:var(--warn); } .tag.fail { color:var(--fail); }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(15rem, 1fr)); gap:1rem; }
figure { margin:0; } figure img { width:100%; height:auto; image-rendering:pixelated; border:1px solid var(--line); }
figcaption { font-size:.85rem; color:var(--muted); } figcaption strong { color:var(--fg); }
.foot { color:var(--muted); font-size:.8rem; margin-top:2rem; }
@media (max-width:40rem) { table, thead, tbody, tr, td { display:block; } thead { display:none; } td { border:0; padding:.1rem 0; } tr { border-bottom:1px solid var(--line); padding:.5rem 0; } }
</style></head><body>
{body}
</body></html>
"""
