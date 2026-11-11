import numpy as np
import pytest
from scipy.special import ndtr

from soilsight.model.gp import SQRT3, Priors, SoilGP
from soilsight.sim.fields import make_field


@pytest.fixture(scope="module")
def data(small_field):
    rng = np.random.default_rng(40)
    idx = rng.choice(small_field.grid.n, 10, replace=False)
    y = small_field.y[idx] + 0.2 * rng.standard_normal(10)
    return idx, small_field.grid.xy[idx], y


def _textbook(post, c, xy, feats, obs_y, noise_sd, cores):
    """GP algebra for one component, written independently of the model code."""
    def kern(a, b, fa, fb):
        d = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1))
        if post.kernel == "rbf":
            k = np.exp(-0.5 * (d / c.length_m) ** 2)
        else:
            k = (1 + SQRT3 * d / c.length_m) * np.exp(-SQRT3 * d / c.length_m)
        k = c.signal_sd ** 2 * k + post.mean_var
        if fa is not None:
            k = k + c.trend_sd ** 2 * fa @ fb.T / fa.shape[1]
        return k

    fo = post.obs_feats
    kxx = kern(post.obs_xy, post.obs_xy, fo, fo) + np.diag(
        np.full(len(obs_y), noise_sd ** 2 + c.nugget_sd ** 2 / cores + 1e-8))
    ks = kern(xy, post.obs_xy, feats, fo)
    mu = ks @ np.linalg.solve(kxx, obs_y - post.y_mean) + post.y_mean
    var = np.diag(kern(xy, xy, feats, feats)) - np.einsum("ij,ji->i", ks, np.linalg.solve(kxx, ks.T))
    return mu, np.sqrt(var)


@pytest.mark.parametrize("kernel", ["matern32", "rbf"])
@pytest.mark.parametrize("layers", [False, True])
@pytest.mark.parametrize("cores", [1, 3])
def test_single_component_matches_textbook_gp(small_field, data, kernel, layers, cores):
    idx, xy, y = data
    feats = small_field.pcs if layers else None
    post = SoilGP(kernel=kernel, top_k=1).fit(xy, y, 0.22, None if feats is None else feats[idx], cores=cores)
    assert len(post.components) == 1 and post.components[0].weight == pytest.approx(1)
    where = np.arange(0, small_field.grid.n, 37)
    pred = post.predict(small_field.grid.xy[where], None if feats is None else feats[where])
    mu, sd = _textbook(post, post.components[0], small_field.grid.xy[where],
                       None if feats is None else feats[where], y, 0.22, cores)
    assert np.allclose(pred.mean, mu, atol=1e-8)
    assert np.allclose(pred.sd, sd, atol=1e-6)
    assert np.allclose(pred.p_unsafe, ndtr(mu / sd), atol=1e-8)


def test_prediction_averages_the_surviving_candidates(small_field, data):
    idx, xy, y = data
    post = SoilGP(top_k=24).fit(xy, y, 0.22)
    w = post.weights
    assert 1 < len(w) <= 24 and w.sum() == pytest.approx(1) and (w > 0).all()
    cells = small_field.grid.xy[::50]
    mu, sd = post.component_moments(cells)
    pred = post.predict(cells)
    assert np.allclose(pred.mean, w @ mu)
    assert np.allclose(pred.sd ** 2, w @ (sd ** 2 + mu ** 2) - pred.mean ** 2)
    assert np.allclose(pred.p_unsafe, w @ ndtr(mu / sd))
    assert ((pred.p_unsafe > 0) & (pred.p_unsafe < 1)).all()


def test_averaging_candidates_widens_uncertainty(small_field, data):
    idx, xy, y = data
    cells = small_field.grid.xy[::50]
    many = SoilGP(top_k=24).fit(xy, y, 0.22)
    one = many.thin(1)
    assert one.weights.tolist() == [1.0]
    spread = many.weights @ many.component_moments(cells)[1] ** 2     # average within-candidate variance
    assert (many.predict(cells).sd ** 2 >= spread - 1e-12).all()     # disagreement only adds


def test_uncertainty_grows_away_from_the_tests(small_grid):
    xy = np.array([[10.0, 10.0], [14.0, 12.0], [12.0, 16.0]])
    post = SoilGP().fit(xy, [0.3, 0.1, 0.2], 0.1)
    near = post.predict(np.array([[12.0, 12.0]])).sd[0]
    far = post.predict(np.array([[78.0, 58.0]])).sd[0]
    assert far > 1.5 * near


def test_a_precise_test_outweighs_a_noisy_one():
    xy = np.array([[20.0, 20.0], [20.0, 20.0]])
    here = np.array([[20.0, 20.0]])
    # Many cores per test, so the disagreement cannot be blamed on metre-scale scatter.
    lab_high = SoilGP().fit(xy, [0.5, -0.5], [0.02, 0.4], cores=25).predict(here).mean[0]
    lab_low = SoilGP().fit(xy, [0.5, -0.5], [0.4, 0.02], cores=25).predict(here).mean[0]
    assert lab_high > 0.3 and lab_low < -0.3


def test_one_core_is_not_trusted_as_the_whole_spot():
    # Two single-core tests at one spot that disagree: even the precise one is only one core.
    xy = np.array([[20.0, 20.0], [20.0, 20.0]])
    one_core = SoilGP().fit(xy, [0.5, -0.5], [0.02, 0.4], cores=1).predict(xy[:1]).mean[0]
    many_cores = SoilGP().fit(xy, [0.5, -0.5], [0.02, 0.4], cores=25).predict(xy[:1]).mean[0]
    assert 0 < one_core < many_cores


def test_mixing_cores_tightens_the_map_at_the_test():
    xy = np.array([[20.0, 20.0], [50.0, 30.0], [35.0, 50.0]])
    y = [0.2, -0.1, 0.3]
    sd1 = SoilGP().fit(xy, y, 0.02, cores=1).predict(xy[:1]).sd[0]
    sd9 = SoilGP().fit(xy, y, 0.02, cores=9).predict(xy[:1]).sd[0]
    assert sd9 < sd1


def test_layers_are_used_only_when_given(small_field, data):
    idx, xy, y = data
    plain = SoilGP().fit(xy, y, 0.22)
    assert all(c.trend_sd == 0 for c in plain.components)
    plain.predict(small_field.grid.xy[:5])                       # no layers needed
    with_layers = SoilGP().fit(xy, y, 0.22, small_field.pcs[idx])
    with pytest.raises(ValueError, match="drone layers"):
        with_layers.predict(small_field.grid.xy[:5])


def test_layers_help_when_they_carry_the_signal(small_grid):
    rng = np.random.default_rng(41)
    feats = rng.standard_normal((small_grid.n, 2))               # no spatial pattern at all
    truth = 0.7 * feats[:, 0]
    idx = rng.choice(small_grid.n, 30, replace=False)
    y = truth[idx] + 0.05 * rng.standard_normal(30)
    with_layers = SoilGP().fit(small_grid.xy[idx], y, 0.05, feats[idx]).predict(small_grid.xy, feats)
    without = SoilGP().fit(small_grid.xy[idx], y, 0.05).predict(small_grid.xy)
    def rmse(p): return np.sqrt(((p.mean - truth) ** 2).mean())
    assert rmse(with_layers) < 0.5 * rmse(without)


def test_weight_on_layers_follows_the_evidence(small_grid):
    # Averaged over fields, junk layers lose weight and informative ones gain it. On a single
    # field with this few tests the answer is noisy, which is why layers stay optional.
    def weight_on_layers(informative):
        out = []
        for s in range(12):
            rng = np.random.default_rng(100 + s)
            f = make_field(rng, small_grid, cov_r2=0.6 if informative else 0.0)
            idx = rng.choice(small_grid.n, 30, replace=False)
            y = f.y[idx] + 0.22 * rng.standard_normal(30)
            feats = f.pcs if informative else rng.standard_normal((small_grid.n, 3))
            post = SoilGP().fit(small_grid.xy[idx], y, 0.22, feats[idx])
            out.append(sum(c.weight for c in post.components if c.trend_sd > 0))
        return np.mean(out)

    junk, real = weight_on_layers(False), weight_on_layers(True)
    assert junk < 0.4 < real         # the prior weight is 0.5


def test_area_mean(small_field, data):
    idx, xy, y = data
    post = SoilGP().fit(xy, y, 0.22)
    g = small_field.grid
    one = g.xy[[500]]
    assert float(post.area_mean(one).mean) == pytest.approx(post.predict(one).mean[0])
    assert float(post.area_mean(one).sd) == pytest.approx(post.predict(one).sd[0], rel=1e-6)
    block = g.xy[g.zone == 0][::4]
    cells = post.predict(block)
    area = post.area_mean(block)
    assert float(area.mean) == pytest.approx(cells.mean.mean())
    assert float(area.sd) < cells.sd.mean()          # an average is better known than a single cell
    assert 0 < float(area.p_unsafe) < 1


def test_resample_represents_the_whole_spread():
    from soilsight.model.gp import resample
    idx, w = resample(np.array([0.1, 0.1, 0.6, 0.1, 0.1]), 4)
    assert idx.tolist() == [1, 2, 3] and w.tolist() == [0.25, 0.5, 0.25]
    idx, w = resample(np.ones(512), 24)                       # all equal: an even spread, not the first 24
    assert len(idx) == 24 and np.allclose(w, 1 / 24) and idx.max() > 480 and np.all(np.diff(idx) > 15)
    idx, w = resample(np.array([1e-9, 1.0, 1e-9]), 24)        # one dominant: just that one
    assert idx.tolist() == [1] and w.tolist() == [1.0]


def test_one_test_does_not_decide_the_whole_field():
    """A single reading a little above the threshold leaves the far side of the field unsure."""
    post = SoilGP().fit(np.array([[10.0, 10.0]]), [0.47], 0.22, cores=3)
    near, far = post.predict(np.array([[10.0, 10.0], [100.0, 80.0]])).p_unsafe
    assert near > 0.9 and 0.6 < far < 0.8
    # The two shortcuts this model avoids, each of which calls the far side "unsafe" outright:
    old = SoilGP(select="heaviest", center_on_tests=True).fit(np.array([[10.0, 10.0]]), [0.47], 0.22, cores=3)
    assert old.predict(np.array([[100.0, 80.0]])).p_unsafe[0] > 0.9


def test_field_average_is_pulled_towards_the_threshold_only_weakly():
    xy = np.array([[10.0, 10.0], [60.0, 20.0], [30.0, 50.0], [70.0, 55.0]])
    high = SoilGP().fit(xy, [1.5, 1.4, 1.6, 1.5], 0.1).predict(np.array([[40.0, 30.0]]))
    assert high.mean[0] == pytest.approx(1.5, abs=0.15) and high.p_unsafe[0] > 0.95


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError, match="kernel"):
        SoilGP(kernel="cubic")
    with pytest.raises(ValueError, match="select"):
        SoilGP(select="best")
    with pytest.raises(ValueError, match="at least one test"):
        SoilGP().fit(np.empty((0, 2)), [], 0.2)


def test_priors_shape_the_candidates():
    gp = SoilGP(Priors(length_m=(100.0, 0.1), trend_prob=0.0), seed=1)
    assert 60 < gp.length_m.min() and gp.length_m.max() < 160
    assert (gp.trend_sd == 0).all()
    assert np.array_equal(SoilGP(seed=3).length_m, SoilGP(seed=3).length_m)
