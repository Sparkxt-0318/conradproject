import numpy as np
import pytest

from soilsight.model.gp import SoilGP
from soilsight.sampling import misclassified_share, next_spot, spread_spots


def test_spread_spots_cover_the_field(small_grid):
    idx = spread_spots(small_grid.xy, 8, np.random.default_rng(50))
    assert len(idx) == len(set(idx.tolist())) == 8
    p = small_grid.xy[idx]
    d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1)) + np.eye(8) * 1e9
    assert d.min() > 12
    for quadrant_x in (p[:, 0] < 40, p[:, 0] >= 40):
        for quadrant_y in (p[:, 1] < 30, p[:, 1] >= 30):
            assert (quadrant_x & quadrant_y).any()


def test_spread_spots_keep_the_fixed_ones(small_grid):
    inlet, outlet = small_grid.xy[0], small_grid.xy[-1]
    idx = spread_spots(small_grid.xy, 6, np.random.default_rng(51), fixed_xy=[inlet, outlet])
    assert idx[0] == 0 and idx[1] == small_grid.n - 1 and len(idx) == 6
    with pytest.raises(ValueError, match="do not fit"):
        spread_spots(small_grid.xy, 1, np.random.default_rng(0), fixed_xy=[inlet, outlet])


@pytest.fixture(scope="module")
def half_tested(small_grid):
    """Six tests on the left third only, all clearly safe; the right side is unknown."""
    rng = np.random.default_rng(52)
    xy = np.c_[rng.uniform(0, 25, 6), rng.uniform(0, 58, 6)]
    return xy, SoilGP().fit(xy, np.full(6, -1.2), 0.1)


def test_misclassified_share_falls_as_the_boundary_is_pinned_down(small_grid):
    def truth(xy):
        return (xy[:, 0] - 40) / 40              # safe on the left, unsafe on the right

    cells = small_grid.xy[::7]
    few_xy = np.array([[10.0, 30.0], [40.0, 30.0], [70.0, 30.0]])
    many_xy = small_grid.xy[np.random.default_rng(53).choice(small_grid.n, 20, replace=False)]
    few = SoilGP().fit(few_xy, truth(few_xy), 0.05)
    many = SoilGP().fit(np.vstack([few_xy, many_xy]), truth(np.vstack([few_xy, many_xy])), 0.05)
    a, b = misclassified_share(few, cells), misclassified_share(many, cells)
    assert 0 < b < a < 0.5


def test_next_spot_goes_where_the_map_is_unsure(small_grid, half_tested):
    xy, post = half_tested
    cand = small_grid.xy[::23]
    pick = next_spot(post, cand, small_grid.xy[::7], kit_sd=0.22)
    assert cand[pick.index][0] > 40                       # the untested side
    assert 0 < pick.gain <= pick.misclassified_now < 0.5
    assert pick.gains.shape == (len(cand),)
    tested_side = pick.gains[cand[:, 0] < 15]
    assert tested_side.max() < 0.5 * pick.gain            # re-testing known ground is worth little


def test_next_spot_keeps_its_distance(small_grid, half_tested):
    xy, post = half_tested
    cand = small_grid.xy[::23]
    taken = np.vstack([xy, [[60.0, 30.0]]])
    pick = next_spot(post, cand, small_grid.xy[::7], kit_sd=0.22, taken_xy=taken, min_sep_m=20)
    d = np.sqrt(((cand[:, None] - taken[None]) ** 2).sum(-1)).min(1)
    assert d[pick.index] >= 20
    assert np.isneginf(pick.gains[d < 20]).all() and np.isfinite(pick.gains[d >= 20]).all()


def test_a_better_test_is_worth_more(small_grid, half_tested):
    _, post = half_tested
    cand, cells = small_grid.xy[::23], small_grid.xy[::7]
    precise = next_spot(post, cand, cells, kit_sd=0.05).gain
    rough = next_spot(post, cand, cells, kit_sd=0.8).gain
    mixed = next_spot(post, cand, cells, kit_sd=0.05, cores=5).gain
    assert precise > rough
    assert mixed >= precise


def test_next_spot_works_with_layers(small_field):
    g = small_field.grid
    rng = np.random.default_rng(54)
    idx = rng.choice(g.n, 8, replace=False)
    post = SoilGP().fit(g.xy[idx], small_field.y[idx], 0.22, small_field.pcs[idx])
    cand, cells = np.arange(0, g.n, 23), np.arange(0, g.n, 7)
    pick = next_spot(post, g.xy[cand], g.xy[cells], kit_sd=0.22, cand_feats=small_field.pcs[cand],
                     cells_feats=small_field.pcs[cells])
    assert np.isfinite(pick.gain) and pick.gain > -1e-9
