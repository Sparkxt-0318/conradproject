"""Correcting kit readings to what an accredited lab would report.

Both are in ln units (ln of reading / threshold works, as does ln of mg/kg). The
correction is a straight line, lab = intercept + slope * kit, learned from split
samples that went to both. With few pairs it stays close to "the kit is right"
(intercept 0, slope 1) and reports a wide uncertainty; pairs from every field are
pooled, so it sharpens over time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KitCalibration:
    intercept: float
    slope: float
    resid_sd: float            # scatter of one kit reading around the line
    cov: np.ndarray            # uncertainty of (intercept, slope)
    n_pairs: int

    def correct(self, kit_ln):
        """Lab-equivalent value and its sd for kit readings. The sd is what to pass to the
        map model as the test's noise."""
        kit_ln = np.asarray(kit_ln, float)
        x = np.stack([np.ones_like(kit_ln), kit_ln], -1)
        mean = self.intercept + self.slope * kit_ln
        var = self.resid_sd ** 2 + np.einsum("...i,ij,...j->...", x, self.cov, x)
        return mean, np.sqrt(var)


def fit_calibration(kit_ln=(), lab_ln=(), *, prior_resid_sd: float = 0.22, prior_pairs: float = 4.0,
                    prior_intercept_sd: float = 0.3, prior_slope_sd: float = 0.3) -> KitCalibration:
    """Learn the correction from paired readings (may be empty).

    prior_resid_sd  kit error assumed before any pair is seen
    prior_pairs     how many pairs that assumption is worth
    """
    kit_ln = np.asarray(kit_ln, float)
    lab_ln = np.asarray(lab_ln, float)
    if kit_ln.shape != lab_ln.shape or kit_ln.ndim != 1:
        raise ValueError("kit_ln and lab_ln must be two lists of the same length")
    n = len(kit_ln)
    x = np.c_[np.ones(n), kit_ln]
    prior_mean = np.array([0.0, 1.0])
    prior_prec = np.diag([prior_intercept_sd ** -2, prior_slope_sd ** -2])

    resid_var = prior_resid_sd ** 2
    for _ in range(20):   # the line and the scatter around it depend on each other
        cov = np.linalg.inv(x.T @ x / resid_var + prior_prec)
        coef = cov @ (x.T @ lab_ln / resid_var + prior_prec @ prior_mean)
        sse = float(((lab_ln - x @ coef) ** 2).sum())
        new_var = (prior_pairs * prior_resid_sd ** 2 + sse) / (prior_pairs + n)
        if abs(new_var - resid_var) < 1e-12:
            break
        resid_var = new_var
    return KitCalibration(float(coef[0]), float(coef[1]), float(np.sqrt(resid_var)), cov, n)
