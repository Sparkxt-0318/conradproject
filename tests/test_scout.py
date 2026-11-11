import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scout_helpers import lawnmower, write_annotations, write_photo, write_rasters, write_stats
from soilsight.cli import main
from soilsight.field import FieldConfig, FieldDir
from soilsight.ingest import get_profile
from soilsight.photogrammetry.odm import odm_options, stitch, stitching_checks, used_ground_control
from soilsight.report.checks import Check, graded, worst
from soilsight.report.quality import layer_png, write_quality
from soilsight.scout import check_flight, run


# ---- field folder

def test_field_config_defaults_and_file(field_dir):
    assert field_dir.config() == FieldConfig(name="field")
    field_dir.root.mkdir(parents=True)
    field_dir.config_path.write_text('name = "North paddy"\nsensor = "pi_gs_6mm"\nflight_height_m = 40\ngrid_m = 2.0\n')
    cfg = field_dir.config()
    assert (cfg.name, cfg.sensor, cfg.flight_height_m, cfg.grid_m, cfg.smooth_m) == ("North paddy", "pi_gs_6mm", 40, 2.0, 3.0)
    field_dir.config_path.write_text("gird_m = 2.0\n")
    with pytest.raises(ValueError, match="unknown setting.*gird_m"):
        field_dir.config()


def test_field_config_can_describe_any_camera(flown_field):
    fd, _ = flown_field
    fd.config_path.write_text('flight_height_m = 50\n[camera]\nname = "WX220"\nwidth_px = 4896\nheight_px = 3672\n'
                              'pixel_um = 1.26\nfocal_mm = 4.45\nshutter = "rolling"\n')
    _, profile, checks = check_flight(fd)
    assert profile.name == "WX220" and profile.gsd_m(50) * 100 == pytest.approx(1.42, abs=0.01)
    assert next(c for c in checks if c.name == "Rolling-shutter smear").value == "unknown"


def test_unknown_camera_is_reported(flown_field):
    fd, _ = flown_field
    fd.config_path.unlink()                      # the test photos carry no focal length
    _, profile, checks = check_flight(fd)
    assert profile is None and checks[0].name == "Camera" and "[camera]" in checks[0].detail


# ---- checks

def test_graded_and_worst():
    assert [graded(v, 0.75, 0.6) for v in (0.8, 0.7, 0.5, float("nan"))] == ["ok", "warn", "fail", "warn"]
    assert [graded(v, 1, 2, higher_is_better=False) for v in (0.5, 1.5, 3)] == ["ok", "warn", "fail"]
    mk = lambda s: Check("flight", "x", s, "")
    assert worst([mk("ok"), mk("info")]) == "ok" and worst([mk("ok"), mk("warn")]) == "warn"
    assert worst([mk("warn"), mk("fail")]) == "fail" and worst([]) == "ok"


# ---- stitching options

def test_odm_options_follow_the_camera():
    gs = odm_options(get_profile("pi_gs_6mm"), gsd_cm=2.3)
    assert gs["dsm"] is True and gs["orthophoto-resolution"] == 2.3 and "rolling-shutter" not in gs
    cm3 = odm_options(get_profile("pi_cm3_full"), gsd_cm=1.18)
    assert cm3["rolling-shutter"] is True and cm3["rolling-shutter-readout"] == 68.0
    assert cm3["orthophoto-resolution"] == 2.0                    # never finer than 2 cm
    fast = odm_options(None, fast=True)
    assert fast["feature-quality"] == "medium" and fast["pc-quality"] == "low" and "rolling-shutter" not in fast


# ---- a stand-in for the NodeODM server

class FakeNode:
    """Pretends to stitch: hands back rasters prepared in `outputs`."""

    def __init__(self, outputs: Path, leave_out=()):
        self.outputs, self.leave_out, self.calls = outputs, leave_out, []

    def create_task(self, files, options):
        self.calls.append((files, options))
        return self

    def wait_for_completion(self, status_callback=None):
        for pct in (5, 50, 100):
            status_callback(type("Info", (), {"progress": pct})())

    def download_assets(self, dest):
        layout = {"odm_orthophoto.tif": "odm_orthophoto/odm_orthophoto.tif", "dsm.tif": "odm_dem/dsm.tif",
                  "stats.json": "odm_report/stats.json"}
        for name, rel in layout.items():
            if name not in self.leave_out:
                (Path(dest) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(self.outputs / name, Path(dest) / rel)

    def info(self):
        return type("Info", (), {"uuid": "abc", "processing_time": 120000, "images_count": 3})()


@pytest.fixture()
def flown_field(field_dir, stitched_field):
    """A field folder with 100 tiny photos of a good grid flight and annotations, not yet stitched."""
    field_dir.photos.mkdir(parents=True)
    for name, lat, lon, when in lawnmower():
        write_photo(field_dir.photos / name, lat, lon, 80, when, exposure=1 / 1000, xmp_height=40)
    field_dir.config_path.write_text('sensor = "pi_gs_6mm"\n')
    shutil.copy(stitched_field.annotations, field_dir.annotations)
    return field_dir, FakeNode(stitched_field.odm)


def test_stitch_sends_photos_and_positions_and_keeps_outputs(flown_field):
    fd, node = flown_field
    fd.work.mkdir()
    fd.geo_txt.write_text("EPSG:4326\n")
    (fd.root / "notes.txt").write_text("not for the stitcher")
    logs = []
    photos = sorted(fd.photos.iterdir())[:3]
    info = stitch(fd, photos, {"dsm": True}, log=logs.append, node=node)
    files, options = node.calls[0]
    assert files == [str(p) for p in photos] + [str(fd.geo_txt)]       # any other .txt would be read as control points
    assert options == {"dsm": True} and info["processing_time_s"] == 120
    assert fd.orthophoto.exists() and fd.dsm.exists() and fd.odm_stats.exists()
    assert any("50%" in line for line in logs)


def test_stitch_without_an_orthophoto_is_an_error(flown_field, stitched_field):
    fd, _ = flown_field
    with pytest.raises(RuntimeError, match="odm_orthophoto"):
        stitch(fd, [], {}, node=FakeNode(stitched_field.odm, leave_out=("odm_orthophoto.tif",)))
    assert not fd.odm.exists()                                       # a failed run leaves nothing behind
    stitch(fd, [], {}, node=FakeNode(stitched_field.odm))
    assert fd.odm_stats.exists()
    stitch(fd, [], {}, node=FakeNode(stitched_field.odm, leave_out=("stats.json",)))     # stats are optional
    assert fd.orthophoto.exists() and not fd.odm_stats.exists()      # and old stats do not linger


def test_stitching_checks(tmp_path):
    fd = FieldDir(tmp_path / "f")
    assert [c.status for c in stitching_checks(fd)] == ["fail", "fail"]              # nothing there
    write_rasters(fd)
    st = {c.name: c for c in stitching_checks(fd)}
    assert st["Stitching statistics"].status == "warn" and st["Map projection"].status == "ok"
    assert st["Orthophoto pixel size"].value == "10.0 cm"
    write_stats(fd, used=34, total=40)
    st = {c.name: c for c in stitching_checks(fd)}
    assert st["Photos used"].status == "warn" and st["Photos used"].value == "34 of 40"
    assert st["Alignment error"].status == "ok" and st["Ground control points"].status == "warn"
    assert st["Ground pixel size, measured"].value == "2.0 cm"
    assert not used_ground_control(fd)
    write_stats(fd, gcp=True)
    st = {c.name: c for c in stitching_checks(fd)}
    assert st["Ground control points"].status == "ok" and "0.03 m" in st["Ground control points"].value
    assert st["Photos used"].status == "ok" and used_ground_control(fd)


# ---- report

def test_quality_report_files(built, tmp_path):
    stack, checks = built
    fd = FieldDir(tmp_path / "r")
    doc = write_quality(fd, checks, stack)
    assert doc["overall"] == "warn" and doc == json.loads(fd.quality_json.read_text())
    assert [layer["name"] for layer in doc["layers"]] == list(stack.names)
    page = fd.quality_html.read_text()
    assert "Usable, with points to check" in page and "Terrain relief" in page and "low_spot" in page
    assert page.count("data:image/png;base64,") == 11 and "<script" not in page      # no orthophoto in this folder
    assert "prefers-color-scheme: dark" in page


def test_quality_report_when_stopped(tmp_path):
    fd = FieldDir(tmp_path / "r")
    doc = write_quality(fd, [Check("flight", "Photo count", "fail", "2", "too few")], stopped="The flight checks failed.")
    assert doc["overall"] == "fail" and doc["layers"] == []
    page = fd.quality_html.read_text()
    assert "Not usable yet" in page and "The flight checks failed." in page and "Layer previews" not in page


def test_report_escapes_text(tmp_path):
    fd = FieldDir(tmp_path / "r")
    write_quality(fd, [Check("flight", "Camera", "info", "<b>x</b>", "a & b")])
    page = fd.quality_html.read_text()
    assert "&lt;b&gt;x&lt;/b&gt;" in page and "a &amp; b" in page and "<b>x</b>" not in page


def test_layer_png():
    a = np.arange(12.0).reshape(3, 4)
    a[0, 0] = np.nan
    png = layer_png(a)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert layer_png(np.full((3, 4), np.nan))[:4] == b"\x89PNG"            # nothing to show is still an image


# ---- the pipeline

def test_check_flight_reads_the_folder(flown_field):
    fd, _ = flown_field
    shots, profile, checks = check_flight(fd)
    assert len(shots) == 100 and profile.name == "pi_gs_6mm"
    st = {c.name: c.status for c in checks}
    assert st["Camera"] == "info" and st["Forward overlap"] == "ok" and "fail" not in st.values()


def test_run_end_to_end(flown_field):
    fd, node = flown_field
    logs = []
    result = run(fd, log=logs.append, node=node)
    assert result.overall == "warn" and result.stopped == ""        # warn: no ground control points
    assert len(result.stack.names) == 11
    assert fd.layers_tif.exists() and fd.layers_json.exists() and fd.quality_html.exists()
    page = fd.quality_html.read_text()
    assert "<h2>Orthophoto</h2>" in page and page.count("data:image/png;base64,") == 12
    files, options = node.calls[0]
    assert len(files) == 100 and options["dsm"] is True and "rolling-shutter" not in options
    assert options["orthophoto-resolution"] == pytest.approx(2.3, abs=0.01)
    stages = {c.stage for c in result.checks}
    assert stages == {"flight", "stitching", "layers"}
    assert any("report:" in line for line in logs)


def test_overlap_is_judged_after_stitching_when_height_was_unknown(field_dir, stitched_field):
    field_dir.photos.mkdir(parents=True)
    for name, lat, lon, when in lawnmower():
        write_photo(field_dir.photos / name, lat, lon, 80, when, exposure=1 / 1000)      # no height recorded
    field_dir.config_path.write_text('sensor = "pi_gs_6mm"\n')
    shutil.copy(stitched_field.annotations, field_dir.annotations)
    before = {c.name: c for c in check_flight(field_dir)[2]}
    assert before["Overlap, blur and smear"].value == "not checked"
    result = run(field_dir, log=lambda m: None, node=FakeNode(stitched_field.odm))
    after = {c.name: c for c in result.checks}
    assert "Overlap, blur and smear" not in after
    assert after["Flying height"].value == "35 m"            # 2.0 cm pixels with this camera
    assert after["Forward overlap"].status == "ok" and after["Forward overlap"].value == "77%"
    assert [c.stage for c in result.checks if c.name in ("Flying height", "Side overlap")] == ["flight", "flight"]


def test_low_overlap_found_after_stitching_is_a_warning_not_a_failure(field_dir, stitched_field):
    field_dir.photos.mkdir(parents=True)
    for name, lat, lon, when in lawnmower(spacing=14.0, per_line=8):
        write_photo(field_dir.photos / name, lat, lon, 80, when)
    field_dir.config_path.write_text('sensor = "pi_gs_6mm"\n')
    result = run(field_dir, log=lambda m: None, node=FakeNode(stitched_field.odm))
    forward = next(c for c in result.checks if c.name == "Forward overlap")
    assert forward.status == "warn" and "did stitch" in forward.detail and result.overall == "warn"


def test_run_writes_positions_from_the_capture_computer(flown_field):
    fd, node = flown_field
    rows = ["file,lat,lon,alt_m,agl_m,time_utc"] + [f"{n},{lat},{lon},80,40,{t.isoformat()}Z" for n, lat, lon, t in lawnmower()]
    fd.sidecar.write_text("\n".join(rows) + "\n")
    run(fd, log=lambda m: None, node=node)
    assert fd.geo_txt.read_text().splitlines()[0] == "EPSG:4326" and len(fd.geo_txt.read_text().splitlines()) == 101
    assert str(fd.geo_txt) in node.calls[0][0]


def test_run_stops_before_stitching_a_bad_flight(flown_field):
    fd, node = flown_field
    write_photo(fd.photos / "zzz_no_position.jpg")
    result = run(fd, log=lambda m: None, node=node)
    assert result.overall == "fail" and "not stitched" in result.stopped and node.calls == []
    assert result.stack is None and fd.quality_html.exists() and not fd.layers_tif.exists()
    forced = run(fd, log=lambda m: None, node=node, force=True)
    assert len(node.calls) == 1 and forced.stack is not None
    assert forced.overall == "fail"                                 # the failed check stays in the report


def test_run_with_outputs_already_stitched(stitched_field, tmp_path):
    fd = FieldDir(tmp_path / "copy")
    shutil.copytree(stitched_field.root, fd.root)
    result = run(fd, skip_stitch=True, log=lambda m: None)
    assert result.stack is not None and result.overall == "warn"
    assert next(c for c in result.checks if c.name == "Photos").value == "not supplied"


def test_sample_field_from_real_stitched_outputs(tmp_path):
    """data/sample_field holds what OpenDroneMap really produced (reduced in resolution)."""
    sample = Path(__file__).resolve().parents[1] / "data" / "sample_field"
    fd = FieldDir(tmp_path / "sample")
    shutil.copytree(sample, fd.root)
    result = run(fd, skip_stitch=True, log=lambda m: None)
    by = {c.name: c for c in result.checks}
    assert result.overall == "warn" and len(result.stack.names) == 11
    assert by["Photos used"].value == "18 of 18" and by["Alignment error"].value == "0.80 px"
    assert by["Ground pixel size, measured"].value == "1.6 cm" and by["Map projection"].value == "EPSG:32615"
    assert by["Field area"].value == "0.35 ha" and by["Field covered by the orthophoto"].status == "ok"
    assert "trees or buildings" in by["Terrain relief"].detail
    stack = result.stack
    assert np.isfinite(stack.data[:, stack.grid.mask]).all()
    assert 80 < np.nanmax(stack.layer("dist_inlet")) < 90                  # the drawn boundary is 70 x 50 m
    # Grass is greener than the paved paths, and the paths are brighter than the grass.
    green, bright = stack.layer("excess_green")[stack.grid.mask], stack.layer("brightness")[stack.grid.mask]
    assert np.corrcoef(green, bright)[0, 1] < -0.5
    assert fd.quality_html.read_text().count("data:image/png;base64,") == 12


def test_run_needs_photos_or_outputs(field_dir):
    field_dir.root.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no photos folder"):
        run(field_dir, log=lambda m: None)
    result = run(field_dir, skip_stitch=True, log=lambda m: None)
    assert result.overall == "fail" and "no layers were made" in result.stopped


# ---- command line

def test_cli_scout_check(flown_field, capsys):
    fd, _ = flown_field
    assert main(["scout", "check", str(fd.root)]) == 0
    out = capsys.readouterr().out
    assert "Forward overlap" in out and "80%" in out
    write_photo(fd.photos / "zzz_no_position.jpg")
    assert main(["scout", "check", str(fd.root)]) == 1


def test_cli_scout_run_on_stitched_outputs(stitched_field, tmp_path, capsys):
    fd = FieldDir(tmp_path / "copy")
    shutil.copytree(stitched_field.root, fd.root)
    assert main(["scout", "run", str(fd.root), "--skip-stitch"]) == 0
    captured = capsys.readouterr()
    assert "overall: warn" in captured.out and "Terrain relief" in captured.out
    assert "report:" in captured.err and fd.quality_html.exists()
