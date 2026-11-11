import numpy as np
import pytest

from soilsight.sim.baselines import hp_table
from soilsight.sim.fields import Grid, make_field


@pytest.fixture(scope="session")
def small_grid():
    return Grid(40, 30, 2.0)   # 80 m x 60 m, fast enough for unit tests


@pytest.fixture(scope="session")
def small_field(small_grid):
    return make_field(np.random.default_rng(0), small_grid)


@pytest.fixture(scope="session")
def hp():
    return hp_table(["sf", "ls", "lf", "sn", "sb"])
