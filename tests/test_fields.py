import numpy as np
import pytest

from soilsight.sim.fields import DEFAULT_GRID, SCENARIOS, Grid, grf, make_field, observe


def test_grid_layout(small_grid):
    g = small_grid
    assert g.n == 1200 and g.xy.shape == (1200, 2)
    assert g.xy[0].tolist() == [0.0, 0.0] and g.xy[-1].tolist() == [78.0, 58.0]
    for i in (0, 41, 777, 1199):
        assert g.cell_at(g.xy[i]) == i
    assert g.cell_at([-50.0, -50.0]) == 0            # clipped to the field
    assert g.cell_at([1e4, 1e4]) == g.n - 1


def test_default_grid_is_three_hectares():
    g = DEFAULT_GRID
    assert g.n * g.cell ** 2 == 30000


def test_zones_split_the_field_in_six():
    zone = DEFAULT_GRID.zone
    assert sorted(np.unique(zone)) == [0, 1, 2, 3, 4, 5]
    share = np.bincount(zone) / zone.size
    assert share.min() > 0.15 and share.max() < 0.18
    # zones 0-2 are the row nearest the inlet edge, left to right
    assert zone[0] == 0 and zone[DEFAULT_GRID.nx - 1] == 2 and zone[-1] == 5


def test_block_mean(small_grid):
    v = np.random.default_rng(1).standard_normal(small_grid.n)
    b = small_grid.block_mean(v, 5)
    assert b.mean() == pytest.approx(v.mean())
    first_block = b.reshape(small_grid.ny, small_grid.nx)[:5, :5]
    assert np.all(first_block == first_block[0, 0])
    assert first_block[0, 0] == pytest.approx(v.reshape(small_grid.ny, small_grid.nx)[:5, :5].mean())


def test_grf_is_standardised_and_patch_size_matters(small_grid):
    def neighbour_corr(f):
        m = f.reshape(small_grid.ny, small_grid.nx)
        return np.corrcoef(m[:, :-1].ravel(), m[:, 1:].ravel())[0, 1]

    rough = grf(np.random.default_rng(2), 4, small_grid)
    smooth = grf(np.random.default_rng(2), 40, small_grid)
    for f in (rough, smooth):
        assert f.mean() == pytest.approx(0, abs=1e-9) and f.std() == pytest.approx(1)
    assert neighbour_corr(smooth) > neighbour_corr(rough)
    assert neighbour_corr(smooth) > 0.99


@pytest.mark.parametrize("prevalence", [0.1, 0.3, 0.5])
def test_field_has_requested_unsafe_share_and_spread(small_grid, prevalence):
    f = make_field(np.random.default_rng(3), small_grid, prevalence=prevalence, sd=0.6)
    assert (f.y > 0).mean() == pytest.approx(prevalence, abs=2 / small_grid.n)
    assert f.y.std() == pytest.approx(0.6)
    assert f.sd == 0.6 and f.nugget == 0.2


def test_field_is_reproducible(small_grid):
    a = make_field(np.random.default_rng(4), small_grid)
    b = make_field(np.random.default_rng(4), small_grid)
    c = make_field(np.random.default_rng(5), small_grid)
    assert np.array_equal(a.y, b.y) and np.array_equal(a.pcs, b.pcs)
    assert not np.array_equal(a.y, c.y)


def test_principal_components_are_unit_variance_and_uncorrelated(small_field):
    p = small_field.pcs
    assert p.shape == (small_field.grid.n, 3)
    assert np.allclose(p.T @ p / len(p), np.eye(3), atol=1e-9)


def test_inlet_and_outlet_are_opposite_corners(small_field):
    g = small_field.grid
    assert g.xy[small_field.i_in].tolist() == [0.0, 0.0]
    assert g.xy[small_field.i_out].tolist() == g.xy.max(0).tolist()


@pytest.mark.parametrize("sid", list(SCENARIOS))
def test_every_scenario_builds(small_grid, sid):
    sc = SCENARIOS[sid]
    f = make_field(np.random.default_rng(6), small_grid, **sc.field)
    assert np.isfinite(f.y).all() and np.isfinite(f.pcs).all()
    assert (f.y > 0).mean() == pytest.approx(sc.field.get("prevalence", 0.3), abs=2 / small_grid.n)
    assert sc.kit_sd > 0


def test_plume_field_is_worst_at_the_inlet():
    f = make_field(np.random.default_rng(7), DEFAULT_GRID, cov_r2=0.6, plume_only=True)
    d_in = np.hypot(*f.grid.xy.T)
    assert f.yb[d_in < 30].mean() > f.yb[d_in > 120].mean() + 0.3


def test_observe_without_error_returns_truth(small_field):
    rng = np.random.default_rng(8)
    for i in (0, 500, 1199):
        assert observe(small_field, i, rng, kit_sd=0.0, loc_err=0.0) == small_field.y[i]


def test_observe_adds_kit_noise(small_field):
    rng = np.random.default_rng(9)
    r = np.array([observe(small_field, 600, rng, kit_sd=0.22, loc_err=0.0) for _ in range(4000)])
    assert r.mean() == pytest.approx(small_field.y[600], abs=0.02)
    assert r.std() == pytest.approx(0.22, rel=0.06)


def test_mixing_cores_reduces_scatter(small_grid):
    f = make_field(np.random.default_rng(10), small_grid, nugget=0.5)
    rng = np.random.default_rng(11)
    one = [observe(f, 600, rng, kit_sd=0.0, loc_err=4.0, cores=1) for _ in range(600)]
    five = [observe(f, 600, rng, kit_sd=0.0, cores=5) for _ in range(600)]
    assert np.var(five) < 0.6 * np.var(one)
