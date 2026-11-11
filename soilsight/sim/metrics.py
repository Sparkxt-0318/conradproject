"""Scoring a predicted map against a synthetic field's truth."""
from __future__ import annotations

from math import erf

import numpy as np

from soilsight.sim.fields import Field

_verf = np.vectorize(erf)


def norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + _verf(x / np.sqrt(2)))


def score(mu: np.ndarray, sd: np.ndarray, fld: Field) -> dict:
    """Compare a prediction (mean and sd per cell) with the 10 m block truth.

    acc      share of cells classed correctly as safe / unsafe
    bal      balanced accuracy (mean of the two per-class accuracies)
    missed   share of the unsafe area that was called safe
    rmse     map error in ln units
    cov      share of cells whose truth lies inside the 90% range (should be 0.90)
    conf     share of cells classed with at least 90% confidence
    cacc     accuracy among those confident cells
    yellow   share of cells in the "not sure" class (P(unsafe) between 0.2 and 0.8)
    fgreen   share of the green area that is really unsafe
    caught   share of the unsafe area that is not green
    zacc     share of the six zones whose mean is on the right side of the threshold
    zrmse    error of the zone means in ln units
    """
    yb, zone = fld.yb, fld.grid.zone
    pred, tr = mu > 0, yb > 0
    missed = ((~pred) & tr).sum() / max(tr.sum(), 1)
    false_alarm = (pred & ~tr).sum() / max((~tr).sum(), 1)
    p = norm_cdf(mu / sd)
    conf = (p > 0.9) | (p < 0.1)
    green, red = p < 0.2, p > 0.8
    zones = np.unique(zone)
    zt = np.array([fld.y[zone == z].mean() for z in zones])
    zp = np.array([mu[zone == z].mean() for z in zones])
    return dict(
        acc=(pred == tr).mean(),
        bal=1 - (missed + false_alarm) / 2,
        missed=missed,
        rmse=np.sqrt(((mu - yb) ** 2).mean()),
        cov=(np.abs(yb - mu) < 1.645 * sd).mean(),
        conf=conf.mean(),
        cacc=(pred == tr)[conf].mean() if conf.any() else np.nan,
        yellow=1 - green.mean() - red.mean(),
        fgreen=(tr & green).sum() / max(green.sum(), 1),
        caught=(tr & ~green).sum() / max(tr.sum(), 1),
        zacc=((zt > 0) == (zp > 0)).mean(),
        zrmse=np.sqrt(((zt - zp) ** 2).mean()),
    )


def score_zones(answers, fld: Field) -> dict:
    """Score the green / yellow / red zone answers against the true zone means.

    zone_acc      share of zones whose P(unsafe) is on the right side of one half
    zone_yellow   share of zones answered "not sure"
    zone_fgreen   share of green zones that are really unsafe
    zone_fred     share of red zones that are really safe
    zone_brier    mean squared error of P(unsafe) (0 is perfect, 0.25 is a coin toss)
    """
    zone = fld.grid.zone
    truth = np.array([fld.y[zone == a.zone].mean() > 0 for a in answers])
    p = np.array([a.p_unsafe for a in answers])
    green = np.array([a.label == "green" for a in answers])
    red = np.array([a.label == "red" for a in answers])
    return dict(
        zone_acc=((p > 0.5) == truth).mean(),
        zone_yellow=1 - green.mean() - red.mean(),
        zone_fgreen=(green & truth).sum() / max(green.sum(), 1),
        zone_fred=(red & ~truth).sum() / max(red.sum(), 1),
        zone_brier=((p - truth) ** 2).mean(),
    )
