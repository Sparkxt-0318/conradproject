"""From soil cadmium and pH to rice-grain risk.

The soil standard (GB 15618) is written for total cadmium, but the kit reads
plant-available cadmium, and how much of that reaches the grain depends on pH.
Published soil-to-grain models have the form

    log10(grain Cd) = a + b * log10(soil Cd) + c * pH

This module turns such a model into a pH-dependent soil threshold, so each test can
be expressed as ln(reading / threshold) for the map model.

No coefficients are built in. They differ between studies, soils and extraction
methods, and none has been checked against this kit. Take a, b and c from a paper
that used the same extraction, or fit them to paired soil and grain samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GrainTransfer:
    a: float
    b: float                     # effect of soil Cd; positive
    c: float                     # effect of pH; negative (acid soil passes more cadmium on)
    grain_limit: float = 0.2     # mg/kg in rice; check the food standard in force

    def __post_init__(self):
        if self.b <= 0:
            raise ValueError("b must be positive: more cadmium in soil means more in grain")

    def grain_cd(self, soil_cd, ph):
        """Predicted grain cadmium, mg/kg."""
        return 10 ** (self.a + self.b * np.log10(soil_cd) + self.c * np.asarray(ph, float))

    def soil_threshold(self, ph):
        """Soil reading at which predicted grain cadmium reaches the limit, at this pH."""
        return 10 ** ((np.log10(self.grain_limit) - self.a - self.c * np.asarray(ph, float)) / self.b)

    def risk_index(self, soil_cd, ph):
        """ln(reading / threshold): above 0 means the grain is predicted to exceed the limit."""
        return np.log(np.asarray(soil_cd, float) / self.soil_threshold(ph))


def fixed_threshold_index(soil_cd, threshold: float):
    """ln(reading / threshold) for a single fixed threshold, when no transfer model is used."""
    return np.log(np.asarray(soil_cd, float) / threshold)
