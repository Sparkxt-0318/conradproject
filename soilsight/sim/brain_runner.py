"""Runs the Brain on a synthetic field, the same way the baselines are run."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from soilsight.brain import Brain, Protocol
from soilsight.sim.fields import Field, observe
from soilsight.sim.metrics import score, score_zones


@dataclass(frozen=True)
class BrainSetup:
    """A Brain configuration to benchmark: cores per test, whether it sees the drone
    layers, and any other Brain or Protocol option."""

    cores: int = 1
    use_layers: bool = False
    options: dict = field(default_factory=dict)


def run_brain(fld: Field, rng: np.random.Generator, kit_sd: float, budget: int = 15, cores: int = 1,
              use_layers: bool = False, fixed_spots: bool = True, **brain_options) -> dict:
    """Let the Brain sample one field and score its final map and zone answers.

    The Brain is told the kit error and, with fixed_spots, the field's inlet and outlet;
    nothing else.
    """
    grid = fld.grid
    protocol = Protocol(budget=budget, cores=cores, kit_sd=kit_sd,
                        **{k: brain_options.pop(k) for k in list(brain_options) if k in Protocol.__dataclass_fields__})
    brain = Brain(grid.xy, fld.pcs if use_layers else None,
                  fixed_xy=grid.xy[[fld.i_in, fld.i_out]] if fixed_spots else None,
                  protocol=protocol, seed=int(rng.integers(1 << 31)), **brain_options)
    idx = []

    def test(xy):
        i = brain.nearest_cell(xy)
        idx.append(i)
        brain.add_result(xy, observe(fld, i, rng, kit_sd, cores=cores))

    for xy in brain.initial_spots():
        test(xy)
    while (spot := brain.next_spot()) is not None:
        test(spot.xy)
    pred = brain.map()
    return (score(pred.mean, pred.sd, fld) | score_zones(brain.zones(grid.zone), fld)
            | dict(stop_n=np.nan, stop_acc=np.nan, idx=idx))
