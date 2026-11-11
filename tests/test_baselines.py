import numpy as np
import pytest

from soilsight.sim.baselines import (BUDGET_STRATEGIES, MEAN_PRIOR_VAR, METHODS, N_CANDIDATES, Strategy,
                                     clhs, fit_predict, hp_table, kmeans_pick, run_strategy)


def test_hp_table_is_fixed_and_in_range():
    a, b = hp_table(["sf", "ls", "sb"]), hp_table(["sf", "ls", "sb"])
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert a["sf"].shape == (N_CANDIDATES,)
    assert 0.1 <= a["sf"].min() and a["sf"].max() <= 1.5
    assert 5 <= a["ls"].min() and a["ls"].max() <= 300


def _direct_gp(fld, idx, y, hp, model):
    """Textbook GP algebra for one hyperparameter setting, written independently."""
    xy, f = fld.grid.xy, fld.pcs

    def kern(a, b, fa, fb):
        d2 = ((a[:, None] - b[None]) ** 2).sum(-1)
        k = hp["sf"] ** 2 * np.exp(-d2 / (2 * hp["ls"] ** 2))
        if model == "PC":
            k = k * np.exp(-((fa[:, None] - fb[None]) ** 2).sum(-1) / (2 * hp["lf"] ** 2))
        if model == "LIN":
            k = k + hp["sb"] ** 2 * fa @ fb.T / 3
        return k + MEAN_PRIOR_VAR

    kxx = kern(xy[idx], xy[idx], f[idx], f[idx]) + (hp["sn"] ** 2 + 1e-8) * np.eye(len(idx))
    ksx = kern(xy, xy[idx], f, f[idx])
    mu = ksx @ np.linalg.solve(kxx, y - y.mean()) + y.mean()
    prior = hp["sf"] ** 2 + MEAN_PRIOR_VAR + (hp["sb"] ** 2 * (f ** 2).sum(1) / 3 if model == "LIN" else 0)
    var = prior - np.einsum("ij,ji->i", ksx, np.linalg.solve(kxx, ksx.T))
    return mu, np.sqrt(np.maximum(var, 1e-10))


@pytest.mark.parametrize("model", ["OK", "PC", "LIN"])
@pytest.mark.parametrize("mode", ["ML", "MAP"])
def test_fit_predict_matches_textbook_gp(small_field, hp, model, mode):
    rng = np.random.default_rng(20)
    idx = rng.choice(small_field.grid.n, 9, replace=False)
    y = small_field.y[idx] + 0.2 * rng.standard_normal(9)
    fit = fit_predict(hp, small_field.grid, idx, y, small_field.pcs, model, mode, noise_sd=0.25)
    mu, sd = _direct_gp(small_field, idx, y, fit.hp, model)
    assert np.allclose(fit.mu, mu, atol=1e-8)
    assert np.allclose(fit.sd, sd, atol=1e-6)


def test_map_mode_uses_the_given_noise_and_ml_does_not(small_field, hp):
    idx = np.arange(0, small_field.grid.n, 150)
    y = small_field.y[idx]
    assert fit_predict(hp, small_field.grid, idx, y, small_field.pcs, "OK", "MAP", 0.31).hp["sn"] == 0.31
    assert fit_predict(hp, small_field.grid, idx, y, small_field.pcs, "OK", "ML", 0.31).hp["sn"] != 0.31


def test_uncertainty_is_lowest_at_sampled_cells(small_field, hp):
    g = small_field.grid
    idx = [g.cell_at([10, 10]), g.cell_at([20, 14]), g.cell_at([12, 24])]   # one corner only
    fit = fit_predict(hp, g, idx, small_field.y[idx], small_field.pcs, "OK", "MAP", 0.1)
    far = g.cell_at([78, 58])
    assert fit.sd[idx].max() < fit.sd[far]
    assert fit.sd.argmax() not in idx


def test_kmeans_pick_spreads_spots(small_grid):
    idx = kmeans_pick(small_grid.xy / 50.0, 6, np.random.default_rng(21))
    assert len(idx) == len(set(idx)) == 6
    p = small_grid.xy[idx]
    d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1)) + np.eye(6) * 1e9
    assert d.min() > 15                        # no two spots crowd together on an 80 x 60 m field
    assert (p[:, 0] < 40).any() and (p[:, 0] > 40).any()


def test_clhs_returns_distinct_cells(small_field):
    idx = clhs(small_field.pcs, 4, np.random.default_rng(22))
    assert len(idx) == len(set(idx)) == 4
    assert all(0 <= i < small_field.grid.n for i in idx)


def test_static_strategy_uses_the_whole_budget(small_field, hp):
    r = run_strategy(hp, small_field, METHODS["spatial + OK MAP"], np.random.default_rng(23), 0.22, 12)
    assert len(r["idx"]) == len(set(r["idx"])) == 12
    assert np.isnan(r["stop_n"]) and np.isnan(r["stop_acc"])   # no stop rule without a loop
    assert 0 <= r["acc"] <= 1


def test_pdf_loop_starts_at_inlet_and_outlet_then_adds_one_at_a_time(small_field, hp):
    r = run_strategy(hp, small_field, METHODS["PDF loop: cLHS + maxvar, GP-PC ML"],
                     np.random.default_rng(24), 0.22, 9)
    assert len(r["idx"]) == len(set(r["idx"])) == 9
    assert r["idx"][3:5] == [small_field.i_in, small_field.i_out]


def test_hybrid_keeps_new_spots_apart(small_field, hp):
    strat = BUDGET_STRATEGIES["hybrid OK"]
    r = run_strategy(hp, small_field, strat, np.random.default_rng(25), 0.22, 10)
    xy = small_field.grid.xy[r["idx"]]
    assert len(xy) == 10
    for j in range(6, 10):      # 60% of 10 come from the design, the rest from the loop
        assert np.sqrt(((xy[:j] - xy[j]) ** 2).sum(1)).min() >= strat.min_sep_m


def test_run_strategy_is_reproducible(small_field, hp):
    s = METHODS["cLHS + straddle, LIN MAP"]
    a = run_strategy(hp, small_field, s, np.random.default_rng(26), 0.22, 8)
    b = run_strategy(hp, small_field, s, np.random.default_rng(26), 0.22, 8)
    assert a["idx"] == b["idx"] and a["acc"] == b["acc"]


def test_unknown_names_are_rejected(small_field, hp):
    with pytest.raises(ValueError, match="design"):
        run_strategy(hp, small_field, Strategy("grid", None, "OK", "MAP"), np.random.default_rng(0), 0.22, 6)
    with pytest.raises(ValueError, match="acquisition"):
        run_strategy(hp, small_field, Strategy("spatial", "ucb", "OK", "MAP", n_init=4),
                     np.random.default_rng(0), 0.22, 6)
