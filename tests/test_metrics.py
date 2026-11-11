import numpy as np
import pytest

from soilsight.sim.fields import Field, Grid
from soilsight.sim.metrics import norm_cdf, score


@pytest.fixture()
def half_field():
    """Left half unsafe (+1), right half safe (-1), no noise."""
    g = Grid(30, 20, 2.0)
    y = np.where(g.xy[:, 0] < 30, 1.0, -1.0)
    return Field(grid=g, y=y, yb=g.block_mean(y), pcs=np.zeros((g.n, 3)), i_in=0, i_out=g.n - 1,
                 sd=1.0, nugget=0.0)


def test_norm_cdf():
    assert norm_cdf(np.array([0.0]))[0] == pytest.approx(0.5)
    assert norm_cdf(np.array([1.6449]))[0] == pytest.approx(0.95, abs=1e-4)
    assert norm_cdf(np.array([-1.6449]))[0] == pytest.approx(0.05, abs=1e-4)


def test_perfect_map(half_field):
    m = score(half_field.yb.copy(), np.full(half_field.grid.n, 0.01), half_field)
    assert m["acc"] == 1 and m["bal"] == 1 and m["missed"] == 0
    assert m["rmse"] == 0 and m["cov"] == 1
    assert m["conf"] == 1 and m["cacc"] == 1 and m["yellow"] == pytest.approx(0)
    assert m["fgreen"] == 0 and m["caught"] == 1
    assert m["zacc"] == 1 and m["zrmse"] == pytest.approx(0)


def test_calling_everything_safe(half_field):
    n = half_field.grid.n
    m = score(np.full(n, -1.0), np.full(n, 0.01), half_field)
    assert m["acc"] == 0.5 and m["bal"] == 0.5
    assert m["missed"] == 1          # every unsafe cell was called safe
    assert m["fgreen"] == 0.5        # half of the green area is really unsafe
    assert m["caught"] == 0
    assert m["cov"] == 0.5           # the unsafe half lies outside the narrow ranges


def test_an_unsure_map_is_all_yellow(half_field):
    n = half_field.grid.n
    m = score(np.full(n, 0.1), np.full(n, 5.0), half_field)
    assert m["yellow"] == pytest.approx(1) and m["conf"] == 0
    assert np.isnan(m["cacc"])
    assert m["caught"] == 1          # nothing is green, so nothing unsafe is green
    assert m["cov"] == 1


def test_zone_scores(half_field):
    # The middle column of zones straddles the boundary; predict zone truth shifted by 0.5.
    n = half_field.grid.n
    m = score(half_field.yb - 0.5, np.full(n, 0.01), half_field)
    assert m["zrmse"] == pytest.approx(0.5)
    assert 0 < m["zacc"] <= 1
