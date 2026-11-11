import json

import numpy as np
import pytest
import rasterio

from scout_helpers import (CARD, CRS, FIELD_H, FIELD_W, HOLLOW, ILLUMINATION, RED_PATCH, SOIL, SOUTH, STRAW_PATCH,
                           WEST, to_lonlat, write_annotations, write_rasters, write_stats)
from soilsight.brain import Brain, Protocol
from soilsight.field import FieldDir
from soilsight.layers import build_layers, grid_from_boundary, grid_from_raster, load_annotations, load_layers, save_layers
from soilsight.layers.colour import colour_indices, grey_card_gains, srgb_to_linear
from soilsight.layers.distance import distance_to_lines, distance_to_point
from soilsight.layers.grid import block_stat, read_on_grid, smooth


def _local(grid):
    x, y = grid.centres()
    return x - WEST, y - SOUTH


def _inside(grid, box, pad=2.0):
    x, y = _local(grid)
    return (x > box[0] + pad) & (x < box[2] - pad) & (y > box[1] + pad) & (y < box[3] - pad)


# ---- grid helpers

def test_grid_from_boundary(stitched_field):
    ann = load_annotations(stitched_field.annotations, CRS)
    g = grid_from_boundary(ann.boundary, CRS, 1.0)
    assert g.mask.sum() == pytest.approx(FIELD_W * FIELD_H, rel=0.02)
    x, y = _local(g)
    assert x[g.mask].min() == pytest.approx(0.5, abs=1.0) and y[g.mask].max() == pytest.approx(FIELD_H - 0.5, abs=1.0)
    assert g.origin[0] <= WEST and g.origin[1] <= SOUTH
    coarse = grid_from_boundary(ann.boundary, CRS, 2.0)
    assert coarse.mask.sum() == pytest.approx(FIELD_W * FIELD_H / 4, rel=0.05)


def test_grid_from_raster_pulls_in_from_the_edge(stitched_field):
    g = grid_from_raster(stitched_field.orthophoto, 1.0, margin_m=2.0)
    x, y = _local(g)
    assert g.mask.sum() == pytest.approx((FIELD_W + 10 - 4) * (FIELD_H + 10 - 4), rel=0.03)
    assert x[g.mask].min() >= -5 + 1.5


def test_read_on_grid_averages_and_marks_missing(stitched_field):
    ann = load_annotations(stitched_field.annotations, CRS)
    g = grid_from_boundary(ann.boundary, CRS, 1.0)
    z = read_on_grid(stitched_field.dsm, g, (1,))[0]
    x, _ = _local(g)
    far_west = (x < 5) & g.mask
    assert z[far_west].mean() == pytest.approx(30 + 0.005 * x[far_west].mean(), abs=0.005)
    big = grid_from_boundary({"type": "Polygon", "coordinates": [[(WEST - 50, SOUTH), (WEST - 40, SOUTH),
                              (WEST - 40, SOUTH + 10), (WEST - 50, SOUTH + 10), (WEST - 50, SOUTH)]]}, CRS, 1.0)
    assert np.isnan(read_on_grid(stitched_field.dsm, big, (1,))).all()        # off the raster entirely


def test_block_stat():
    fine = np.arange(16, dtype=float).reshape(4, 4)
    assert block_stat(fine, 2).tolist() == [[2.5, 4.5], [10.5, 12.5]]
    assert block_stat(fine, 2, "sd")[0, 0] == pytest.approx(np.std([0, 1, 4, 5]))
    fine[0, 0] = np.nan
    assert block_stat(fine, 2)[0, 0] == pytest.approx(np.mean([1, 4, 5]))     # ignores the gap
    fine[0, 1] = fine[1, 0] = np.nan
    assert np.isnan(block_stat(fine, 2)[0, 0])                                # under half present
    with pytest.raises(ValueError, match="unknown stat"):
        block_stat(fine, 2, "max")


def test_smooth_ignores_gaps():
    a = np.ones((9, 9))
    a[4, 4] = np.nan
    out = smooth(a, 1.5)
    assert np.isnan(out[4, 4]) and np.allclose(out[np.isfinite(out)], 1.0)
    spike = np.zeros((9, 9))
    spike[4, 4] = 1
    assert 0 < smooth(spike, 1.5)[4, 4] < 1 and smooth(spike, 0)[4, 4] == 1


# ---- colour

def test_srgb_to_linear():
    assert srgb_to_linear(np.array([0, 255])).tolist() == [0.0, 1.0]
    assert srgb_to_linear(np.array([128]))[0] == pytest.approx(0.2158, abs=1e-3)


def test_colour_indices_on_known_colours():
    grey = colour_indices(np.full((3, 1), 0.3))
    assert grey["colour_index"][0] == grey["saturation"][0] == 0 and grey["excess_green"][0] == pytest.approx(0)
    assert grey["brightness"][0] == pytest.approx(0.3)
    red = colour_indices(np.array([[0.4], [0.2], [0.1]]))
    assert red["colour_index"][0] == pytest.approx(1 / 3) and red["saturation"][0] == pytest.approx(0.6)
    assert red["redness"][0] == pytest.approx(np.log10(0.16 / (0.1 * 0.008)))
    assert red["excess_green"][0] < 0 < colour_indices(np.array([[0.2], [0.4], [0.1]]))["excess_green"][0]


def test_grey_card_recovers_the_illumination(stitched_field):
    ann = load_annotations(stitched_field.annotations, CRS)
    gains = grey_card_gains(stitched_field.orthophoto, ann.grey_card, 0.18)
    assert gains == pytest.approx(1 / ILLUMINATION, rel=0.02)
    off = {"type": "Polygon", "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]}
    with pytest.raises(ValueError):
        grey_card_gains(stitched_field.orthophoto, off)


def test_colour_layers_show_the_patches(built):
    stack, _ = built
    g = stack.grid
    x, y = _local(g)
    plain = (x > 30) & (x < 38) & (y > 28) & (y < 38)
    red, straw = _inside(g, RED_PATCH), _inside(g, STRAW_PATCH)
    assert np.nanmean(stack.layer("brightness")[plain]) == pytest.approx(np.sqrt((SOIL ** 2).mean()), rel=0.03)
    assert np.nanmean(stack.layer("redness")[red]) > np.nanmean(stack.layer("redness")[plain]) + 0.3
    assert np.nanmean(stack.layer("colour_index")[red]) > np.nanmean(stack.layer("colour_index")[plain]) + 0.1
    assert np.nanmean(stack.layer("excess_green")[straw]) > np.nanmean(stack.layer("excess_green")[plain]) + 0.3
    assert np.nanmean(stack.layer("saturation")[straw]) == pytest.approx(np.nanmean(stack.layer("saturation")[plain]), abs=0.02)


# ---- terrain

def test_terrain_layers_find_the_tilt_and_the_hollow(built):
    stack, _ = built
    g = stack.grid
    x, y = _local(g)
    elev, low, slope = stack.layer("elevation"), stack.layer("low_spot"), stack.layer("slope")
    ok = np.isfinite(elev)
    assert abs(np.corrcoef(elev[ok], x[ok])[0, 1]) < 0.05                 # the tilt is gone
    assert np.nanmedian(slope) == pytest.approx(0.5, abs=0.05)            # 0.5 percent
    i = np.nanargmax(low)
    assert np.hypot(x.ravel()[i] - HOLLOW[0], y.ravel()[i] - HOLLOW[1]) < 2
    assert np.nanmax(low) > 0.05 and np.nanmin(elev) < -0.08
    edge = stack.grid.mask & ~np.roll(stack.grid.mask, 1, axis=1)          # westernmost column of the field
    assert np.nanmax(slope[edge]) < 1.0                                    # no false cliff at the boundary


def test_terrain_without_ground_control_is_flagged(built, tmp_path):
    stack, checks = built
    relief = next(c for c in checks if c.name == "Terrain relief")
    assert relief.status == "warn" and "no ground control points" in relief.detail
    assert stack.quality["low_spot"]["status"] == "warn" and stack.quality["dist_inlet"]["status"] == "ok"

    fd = FieldDir(tmp_path / "gcp")
    write_rasters(fd)
    write_stats(fd, gcp=True)
    write_annotations(fd)
    relief = next(c for c in build_layers(fd)[1] if c.name == "Terrain relief")
    assert relief.status == "warn" and "close to what stitching can resolve" in relief.detail     # only 5 cm of relief


def test_large_relief_points_at_trees_and_buildings(tmp_path):
    fd = FieldDir(tmp_path / "hilly")
    write_rasters(fd)
    with rasterio.open(fd.dsm, "r+") as dst:                 # put a 6 m block (a building) in the field
        z = dst.read(1)
        z[100:160, 100:200] += 6.0
        dst.write(z, 1)
    write_annotations(fd)
    relief = next(c for c in build_layers(fd)[1] if c.name == "Terrain relief")
    assert relief.status == "warn" and "trees or buildings" in relief.detail and "bow the surface by this much" not in relief.detail


# ---- distance

def test_distances():
    x, y = np.meshgrid(np.arange(5.0), np.arange(4.0))
    assert distance_to_point(x, y, (0, 0))[3, 4] == 5
    line = [np.array([[0.0, 0.0], [4.0, 0.0]])]
    assert distance_to_lines(x, y, line)[2].tolist() == [2, 2, 2, 2, 2]
    assert distance_to_lines(np.array([6.0]), np.array([0.0]), line)[0] == 2          # past the end
    two = line + [np.array([[0.0, 3.0], [4.0, 3.0]])]
    assert distance_to_lines(x, y, two)[2].tolist() == [1, 1, 1, 1, 1]                # nearest of two ditches


def test_distance_layers(built):
    stack, _ = built
    x, y = _local(stack.grid)
    m = stack.grid.mask
    assert np.allclose(stack.layer("dist_inlet")[m], np.hypot(x, y)[m], atol=0.05)
    assert np.allclose(stack.layer("dist_outlet")[m], np.hypot(x - FIELD_W, y - FIELD_H)[m], atol=0.05)
    assert np.allclose(stack.layer("dist_ditch")[m], y[m], atol=0.05)


# ---- annotations

def test_annotations_roles_and_problems(tmp_path):
    fd = FieldDir(tmp_path / "f")
    bad = [{"type": "Feature", "properties": {"role": "pond"}, "geometry": {"type": "Point", "coordinates": [113, 28]}},
           {"type": "Feature", "properties": {"role": "inlet"},
            "geometry": {"type": "LineString", "coordinates": [[113, 28], [113.1, 28]]}}]
    write_annotations(fd, card=False, extra=bad)
    ann = load_annotations(fd.annotations, CRS)
    assert ann.boundary["type"] == "Polygon" and len(ann.ditches) == 1 and ann.grey_card is None
    assert ann.inlet == pytest.approx((WEST, SOUTH), abs=0.01)
    assert len(ann.problems) == 2 and "unknown role 'pond'" in ann.problems[0] and "must be a Point" in ann.problems[1]
    empty = load_annotations(tmp_path / "none.geojson", CRS)
    assert empty.boundary is None and empty.ditches == ()


# ---- the whole stack

def test_build_makes_all_eleven_layers(built):
    stack, checks = built
    assert stack.names == ("brightness", "redness", "colour_index", "saturation", "excess_green",
                           "elevation", "low_spot", "slope", "dist_inlet", "dist_outlet", "dist_ditch")
    assert stack.data.shape == (11,) + stack.grid.shape
    assert np.isfinite(stack.data[:, stack.grid.mask]).all() and np.isnan(stack.data[:, ~stack.grid.mask]).all()
    st = {c.name: c.status for c in checks}
    assert st["Field boundary"] == st["Colour calibration"] == st["Distance layers"] == "ok"
    assert st["Field covered by the orthophoto"] == "ok"


def test_build_without_annotations_still_works_and_says_what_is_missing(tmp_path):
    fd = FieldDir(tmp_path / "bare")
    write_rasters(fd)
    stack, checks = build_layers(fd)
    assert len(stack.names) == 8 and "dist_inlet" not in stack.names
    st = {c.name: c for c in checks}
    assert st["Field boundary"].status == "warn" and st["Colour calibration"].status == "warn"
    assert st["Distance layers"].value == "0 of 3" and "inlet, outlet, ditch" in st["Distance layers"].detail
    assert stack.quality["brightness"]["status"] == "warn"


def test_boundary_off_the_orthophoto_is_reported(tmp_path):
    fd = FieldDir(tmp_path / "off")
    write_rasters(fd, margin=0.0)
    far = {"type": "Feature", "properties": {"role": "boundary"}, "geometry": {
        "type": "Polygon", "coordinates": [[list(to_lonlat(*p)) for p in ((30, 0), (120, 0), (120, 40), (30, 40), (30, 0))]]}}
    write_annotations(fd, boundary=False, inlet=False, outlet=False, ditch=False, card=False, extra=[far])
    cover = next(c for c in build_layers(fd)[1] if c.name == "Field covered by the orthophoto")
    assert cover.status == "fail" and cover.value == "33%"


def test_save_and_load(built, tmp_path):
    stack, _ = built
    fd = FieldDir(tmp_path / "saved")
    save_layers(stack, fd)
    back = load_layers(fd)
    assert back.names == stack.names and np.array_equal(back.grid.mask, stack.grid.mask)
    assert np.allclose(back.data, stack.data, equal_nan=True)
    assert (back.grid.west, back.grid.north, back.grid.cell) == (stack.grid.west, stack.grid.north, stack.grid.cell)
    meta = json.loads(fd.layers_json.read_text())
    assert meta["cells_in_field"] == 2400 and meta["layers"][5]["group"] == "terrain"
    with rasterio.open(fd.layers_tif) as src:
        assert src.descriptions[0] == "brightness" and src.descriptions[-1] == "in_field" and src.crs.to_epsg() == 32650


def test_layers_feed_the_brain(built):
    stack, _ = built
    xy, feats, cells = stack.brain_inputs(["brightness", "low_spot", "dist_inlet"])
    assert xy.shape == (2400, 2) and feats.shape == (2400, 3) and cells.shape == (2400, 2)
    assert np.allclose(feats.mean(0), 0, atol=1e-9) and np.allclose(feats.std(0), 1)
    assert 0 <= xy[:, 0].min() < 2 and xy[:, 0].max() <= FIELD_W + 1 and xy[:, 1].max() <= FIELD_H + 1
    brain = Brain(xy, feats, protocol=Protocol(budget=8))
    for spot in brain.initial_spots():
        brain.add_result(spot, -0.3)
    nxt = brain.next_spot()
    assert nxt is not None and brain.map().mean.shape == (2400,)
    xy_only, none, _ = stack.brain_inputs([])
    assert xy_only.shape == (2400, 2) and none.shape == (2400, 0)
