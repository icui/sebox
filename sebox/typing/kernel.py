from __future__ import annotations
import typing as tp

from sebox import Workspace


class Kernel(Workspace):
    """Compute kernel and / or misfit."""
    # root kernel workspace
    root_kernel: Kernel

    # current iteration
    iteration: tp.Optional[int]

    # length of a time step
    dt: float

    # simulation duration in minutes
    duration: float

    # period range
    period_range: tp.List[float]

    # path to current model
    path_model: str

    # inherit from an existing kernel workspace
    inherit_kernel: tp.Optional[Kernel]

    # sum of misfit values of all kernels
    misfit_value: float
