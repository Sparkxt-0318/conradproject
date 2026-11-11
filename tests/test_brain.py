import numpy as np
import pytest

from soilsight.brain import Brain, Protocol, _thin
from soilsight.sim.brain_runner import run_brain
from soilsight.sim.fields import observe


def test_thin_gives_an_even_lattice(small_grid):
    idx = _thin(small_grid.xy, 10.0)
    assert len(idx) == len(set(idx.tolist()))
    p = small_grid.xy[idx]
    assert len(idx) == 9 * 7                                    # 80 x 60 m at 10 m spacing
    assert np.unique(p[:, 0]).tolist() == [0, 10, 20, 30, 40, 50, 60, 70, 78]   # last column is the field edge
    d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1)) + np.eye(len(p)) * 1e9
    assert d.min() >= 8


def _visit(field, budget=10, layers=True, seed=7, **protocol):
    g = field.grid
    rng = np.random.default_rng(60)
    brain = Brain(g.xy, field.pcs if layers else None, fixed_xy=g.xy[[field.i_in, field.i_out]],
                  protocol=Protocol(budget=budget, cores=1, **protocol), seed=seed)
    spots = []

    def test(xy):
        spots.append(np.array(xy))
        brain.add_result(xy, observe(field, brain.nearest_cell(xy), rng, 0.22))

    first = brain.initial_spots()
    for xy in first:
        test(xy)
    gains = []
    while (nxt := brain.next_spot()) is not None:
        gains.append(nxt)
        test(nxt.xy)
    return brain, np.array(spots), first, gains


def test_a_full_visit(small_field):
    brain, spots, first, picks = _visit(small_field)
    g = small_field.grid
    assert len(first) == 6 and len(picks) == 4 and brain.n_tests == 10
    assert first[0].tolist() == g.xy[small_field.i_in].tolist()
    assert first[1].tolist() == g.xy[small_field.i_out].tolist()
    for j in range(6, 10):                       # loop spots keep their distance
        assert np.sqrt(((spots[:j] - spots[j]) ** 2).sum(1)).min() >= 10
    for p in picks:
        assert 0 <= p.gain <= p.misclassified_now < 0.5
    assert brain.next_spot() is None             # budget spent

    cells = brain.map()
    assert cells.mean.shape == cells.sd.shape == cells.p_unsafe.shape == (g.n,)
    assert (cells.sd > 0).all() and ((cells.p_unsafe >= 0) & (cells.p_unsafe <= 1)).all()

    zones = brain.zones(g.zone)
    assert [z.zone for z in zones] == [0, 1, 2, 3, 4, 5]
    assert all(z.label in ("green", "yellow", "red") and 0 <= z.p_unsafe <= 1 and z.sd > 0 for z in zones)


def test_a_visit_without_drone_layers(small_field):
    brain, spots, _, _ = _visit(small_field, budget=8, layers=False)
    assert brain.n_tests == 8 and np.isfinite(brain.map().mean).all()


def test_same_seed_same_visit(small_field):
    a = _visit(small_field, budget=8)[1]
    b = _visit(small_field, budget=8)[1]
    c = _visit(small_field, budget=8, seed=8)[1]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_no_answers_before_the_first_test(small_field):
    brain = Brain(small_field.grid.xy)
    with pytest.raises(ValueError, match="no test results"):
        brain.map()


def test_a_lab_result_overrules_kit_results(small_field):
    g = small_field.grid
    brain = Brain(g.xy, protocol=Protocol(kit_sd=0.3, cores=1))
    spot = g.xy[600]
    for xy in g.xy[[100, 300, 900, 1100]]:
        brain.add_result(xy, -0.4)
    brain.add_result(spot, -0.4)
    before = brain.map().mean[600]
    brain.add_result(spot, 0.8, noise_sd=0.03, cores=5)        # accredited lab, mixed sample
    after = brain.map()
    assert before < -0.2 and after.mean[600] > 0.4
    assert after.p_unsafe[600] > 0.8


def test_protocol_sets_the_initial_share(small_field):
    g = small_field.grid
    assert len(Brain(g.xy, protocol=Protocol(budget=20, init_frac=0.5)).initial_spots()) == 10
    assert len(Brain(g.xy, protocol=Protocol(budget=10, init_frac=1.0)).initial_spots()) == 10


def test_run_brain_scores_a_field(small_field):
    r = run_brain(small_field, np.random.default_rng(61), 0.22, budget=9)
    assert len(r["idx"]) == 9
    for k in ("acc", "bal", "missed", "cov", "yellow", "zone_acc", "zone_brier", "zone_yellow", "zone_fgreen"):
        assert 0 <= r[k] <= 1
    assert np.isnan(r["stop_n"])
