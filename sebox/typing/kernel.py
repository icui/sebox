from __future__ import annotations
import typing as tp


if tp.TYPE_CHECKING:
    from sebox import Task, Workspace


    class KernelModule(tp.Protocol):
        """Required functions in a solver module."""
        # compute kernels
        kernel: Task[Kernel]

        # compute misfit
        misfit: Task[Kernel]


    class Kernel(Workspace):
        """Compute kernel and / or misfit."""
        # current iteration
        iteration: int

        # length of a time step
        dt: float

        # simulation duration in minutes
        duration: float

        # period range
        period_range: tp.List[float]

        # path to current model
        path_model: str
