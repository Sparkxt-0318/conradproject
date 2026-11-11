import numpy as np
import pytest

from soilsight.model import GrainTransfer, classify, fit_calibration, fixed_threshold_index


# ---- kit-to-lab calibration

def test_no_pairs_means_trust_the_kit_but_say_so():
    cal = fit_calibration()
    assert cal.intercept == pytest.approx(0) and cal.slope == pytest.approx(1) and cal.n_pairs == 0
    mean, sd = cal.correct([-1.0, 0.0, 1.0])
    assert mean.tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert (sd > 0.22).all()                      # wider than the kit error alone
    assert sd[0] > sd[1] and sd[2] > sd[1]        # least sure far from the threshold


def test_many_pairs_recover_the_true_line():
    rng = np.random.default_rng(70)
    kit = rng.uniform(-1.5, 1.5, 200)
    lab = 0.3 + 0.8 * kit + 0.1 * rng.standard_normal(200)
    cal = fit_calibration(kit, lab)
    assert cal.intercept == pytest.approx(0.3, abs=0.03)
    assert cal.slope == pytest.approx(0.8, abs=0.03)
    assert cal.resid_sd == pytest.approx(0.1, abs=0.02)
    mean, sd = cal.correct(1.0)
    assert float(mean) == pytest.approx(1.1, abs=0.04)
    assert float(sd) == pytest.approx(0.1, abs=0.02)


def test_few_pairs_move_only_part_of_the_way():
    kit = np.array([-0.5, 0.0, 0.5])
    lab = kit + 0.6                                # the kit reads low by 0.6
    cal = fit_calibration(kit, lab)
    assert 0.1 < cal.intercept < 0.6
    more = fit_calibration(np.tile(kit, 10), np.tile(lab, 10))
    assert more.intercept > cal.intercept
    assert float(more.correct(0.0)[1]) < float(cal.correct(0.0)[1])


def test_calibration_rejects_mismatched_lists():
    with pytest.raises(ValueError, match="same length"):
        fit_calibration([0.1, 0.2], [0.1])


# ---- soil-to-grain transfer (coefficients here are invented for the test)

def test_threshold_is_where_grain_hits_the_limit():
    t = GrainTransfer(a=1.0, b=0.6, c=-0.3)
    for ph in (4.5, 5.5, 6.5):
        assert float(t.grain_cd(t.soil_threshold(ph), ph)) == pytest.approx(0.2)
        assert float(t.risk_index(t.soil_threshold(ph), ph)) == pytest.approx(0.0, abs=1e-12)


def test_acid_soil_lowers_the_threshold():
    t = GrainTransfer(a=1.0, b=0.6, c=-0.3)
    assert t.soil_threshold(4.5) < t.soil_threshold(5.5) < t.soil_threshold(6.5)
    cd = float(t.soil_threshold(5.5))
    assert t.risk_index(cd, 4.5) > 0 > t.risk_index(cd, 6.5)       # same soil Cd, different risk


def test_transfer_needs_a_positive_cadmium_effect():
    with pytest.raises(ValueError, match="positive"):
        GrainTransfer(a=1.0, b=0.0, c=-0.3)


def test_fixed_threshold_index():
    assert fixed_threshold_index([0.03, 0.06], 0.03).tolist() == pytest.approx([0.0, np.log(2)])


# ---- three classes

def test_classify():
    assert classify([0.05, 0.2, 0.5, 0.8, 0.95]).tolist() == [0, 1, 1, 1, 2]
    assert classify([0.3], lo=0.4, hi=0.6).tolist() == [0]
