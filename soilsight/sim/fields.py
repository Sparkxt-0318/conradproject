"""Synthetic rice fields.

Every value is ln(Cd / threshold): a cell is unsafe when its value is above 0.
The fields are made up. They exist to compare sampling strategies under
controlled conditions, not to predict how a real paddy behaves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class Grid:
    """A rectangular field cut into square cells. Cell i sits at xy[i] (metres)."""

    nx: int = 100
    ny: int = 75
    cell: float = 2.0

    @property
    def n(self) -> int:
        return self.nx * self.ny

    @cached_property
    def xy(self) -> np.ndarray:
        gx, gy = np.meshgrid(np.arange(self.nx) * self.cell, np.arange(self.ny) * self.cell)
        return np.c_[gx.ravel(), gy.ravel()]

    @cached_property
    def zone(self) -> np.ndarray:
        """Management zone of each cell: 3 columns x 2 rows, numbered 0-5."""
        x, y = self.xy.T
        col = np.floor(3 * x / (self.nx * self.cell)).astype(int)
        row = np.floor(2 * y / (self.ny * self.cell)).astype(int)
        return col + 3 * row

    def cell_at(self, p) -> int:
        """Index of the cell nearest to point p, clipped to the field."""
        cx = int(np.clip(round(p[0] / self.cell), 0, self.nx - 1))
        cy = int(np.clip(round(p[1] / self.cell), 0, self.ny - 1))
        return cy * self.nx + cx

    def block_mean(self, v: np.ndarray, b: int = 5) -> np.ndarray:
        """Mean of v over b x b cell blocks, repeated back to every cell."""
        V = v.reshape(self.ny, self.nx)
        out = np.empty_like(V)
        for i in range(0, self.ny, b):
            for j in range(0, self.nx, b):
                out[i:i + b, j:j + b] = V[i:i + b, j:j + b].mean()
        return out.ravel()


DEFAULT_GRID = Grid()  # 200 m x 150 m (3 ha) on a 2 m grid


@dataclass(frozen=True)
class Field:
    """One synthetic field: the hidden truth plus what the drone would see."""

    grid: Grid
    y: np.ndarray        # truth per cell
    yb: np.ndarray       # truth averaged over 10 m blocks (the scale a farmer can act on)
    pcs: np.ndarray      # first 3 principal components of the 11 drone layers, unit variance
    i_in: int            # cell at the irrigation inlet
    i_out: int           # cell at the outlet
    sd: float            # within-field spread of ln(Cd)
    nugget: float        # share of variance that is metre-scale noise


@dataclass(frozen=True)
class Scenario:
    label: str
    field: dict = field(default_factory=dict)   # keyword arguments for make_field
    kit_sd: float = 0.22                        # kit error, sd of ln(reading / truth)


SCENARIOS = {
    "S0": Scenario("base"),
    "S1": Scenario("drone layers useless", dict(cov_r2=0.0)),
    "S2": Scenario("drone layers strong", dict(cov_r2=0.6)),
    "S3": Scenario("short patches, 12 m", dict(range_m=12)),
    "S4": Scenario("long patches, 100 m", dict(range_m=100)),
    "S5": Scenario("kit error 10%", kit_sd=0.1),
    "S6": Scenario("kit error 40%", kit_sd=0.4),
    "S7": Scenario("unsafe area 10%", dict(prevalence=0.1)),
    "S8": Scenario("unsafe area 50%", dict(prevalence=0.5)),
    "S9": Scenario("strong contrast in field", dict(sd=0.8)),
    "S10": Scenario("short-range noise 50%", dict(nugget=0.5)),
    "S11": Scenario("inlet plume dominant", dict(cov_r2=0.6, plume_only=True)),
}


def _std(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / v.std()


def grf(rng: np.random.Generator, ell: float, grid: Grid = DEFAULT_GRID) -> np.ndarray:
    """Smooth random field with patch size about ell metres, zero mean, unit variance."""
    s = ell / grid.cell / np.sqrt(2)
    pad = 2  # generate on a larger grid so the crop is not periodic
    w = rng.standard_normal((grid.ny * pad, grid.nx * pad))
    fy = np.fft.fftfreq(grid.ny * pad)[:, None]
    fx = np.fft.fftfreq(grid.nx * pad)[None, :]
    f = np.fft.ifft2(np.fft.fft2(w) * np.exp(-2 * np.pi ** 2 * s ** 2 * (fx ** 2 + fy ** 2)))
    f = f.real[:grid.ny, :grid.nx].ravel()
    return (f - f.mean()) / f.std()


def make_field(rng: np.random.Generator, grid: Grid = DEFAULT_GRID, *, cov_r2: float = 0.3,
               nugget: float = 0.2, range_m: float = 40, sd: float = 0.4,
               prevalence: float = 0.3, plume_only: bool = False) -> Field:
    """Draw one field.

    cov_r2      share of Cd variance explained by things the drone can see
    nugget      share that is metre-scale noise no map can resolve
    range_m     patch size of the remaining spatial pattern
    sd          within-field spread of ln(Cd); 0.4 is roughly a 40% CV
    prevalence  share of the field above the threshold
    plume_only  the visible part is only a plume spreading from the inlet
    """
    n, xy = grid.n, grid.xy
    a = grf(rng, 30, grid)
    b = grf(rng, 15, grid)
    elev = 0.5 * _std(xy[:, 0]) * rng.choice([-1, 1]) + grf(rng, 60, grid)
    g = np.c_[np.ones(n), xy]
    low = -(elev - g @ np.linalg.lstsq(g, elev, rcond=None)[0])   # low spots, tilt removed
    gyy, gxx = np.gradient(elev.reshape(grid.ny, grid.nx))
    slope = np.hypot(gxx, gyy).ravel()
    d_in = np.hypot(*xy.T)                     # inlet at one corner
    d_out = np.hypot(*(xy - xy.max(0)).T)      # outlet at the opposite corner

    def nz():
        return 0.15 * rng.standard_normal(n)

    # The 11 layers of the engineering plan: 5 colour, 3 terrain, 3 distance.
    layers = np.c_[a + 0.3 * b + nz(), -0.8 * a + 0.5 * b + nz(), 0.6 * a + 0.6 * b + nz(),
                   b + nz(), grf(rng, 10, grid) + nz(), -low, low + 0.3 * grf(rng, 8, grid),
                   slope, xy[:, 1], d_in, d_out]
    layers = (layers - layers.mean(0)) / layers.std(0)

    plume = _std(np.exp(-d_in / 40.0))
    if plume_only:
        visible = plume
    else:
        w = rng.dirichlet([1, 1, 1])
        visible = _std(w[0] * rng.choice([-1, 1]) * a + w[1] * _std(low) + w[2] * plume)
    spatial = _std(0.8 * grf(rng, range_m, grid) + 0.6 * grf(rng, range_m / 4, grid))
    z = _std(np.sqrt(cov_r2) * visible + np.sqrt(max(1 - cov_r2 - nugget, 0)) * spatial
             + np.sqrt(nugget) * rng.standard_normal(n))
    y = sd * (z - np.quantile(z, 1 - prevalence))

    u, _, _ = np.linalg.svd(layers, full_matrices=False)
    return Field(grid=grid, y=y, yb=grid.block_mean(y), pcs=u[:, :3] * np.sqrt(n),
                 i_in=0, i_out=n - 1, sd=sd, nugget=nugget)


def observe(fld: Field, i: int, rng: np.random.Generator, kit_sd: float,
            loc_err: float = 2.0, cores: int = 1) -> float:
    """One kit reading aimed at cell i.

    GPS error moves the real spot by loc_err metres (sd), then kit noise is added.
    With cores > 1, that many cores from within a few metres are mixed into one test.
    """
    grid = fld.grid
    spread = loc_err if cores == 1 else 4.0
    v = [fld.y[grid.cell_at(grid.xy[i] + rng.standard_normal(2) * spread)] for _ in range(cores)]
    val = v[0] if cores == 1 else np.log(np.mean(np.exp(v)))
    return val + kit_sd * rng.standard_normal()
