from __future__ import annotations
import typing as tp

from .solver import Solver


class Kernel(Solver):
    """Compute kernel and / or misfit."""
    # length of a time step
    dt: float

    # period range
    period_range: tp.List[float]

    # sum of misfit values of all kernels
    misfit_value: float
