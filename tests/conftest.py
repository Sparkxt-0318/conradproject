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


@pytest.fixture()
def field_dir(tmp_path):
    from soilsight.field import FieldDir
    return FieldDir(tmp_path / "field")


@pytest.fixture(scope="session")
def stitched_field(tmp_path_factory):
    """A field folder holding a synthetic orthophoto, surface model, stats and annotations."""
    from scout_helpers import write_annotations, write_rasters, write_stats
    from soilsight.field import FieldDir
    fd = FieldDir(tmp_path_factory.mktemp("stitched") / "field")
    write_rasters(fd)
    write_stats(fd)
    write_annotations(fd)
    return fd


@pytest.fixture(scope="session")
def built(stitched_field):
    from soilsight.layers import build_layers
    return build_layers(stitched_field)
