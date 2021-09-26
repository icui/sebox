from __future__ import annotations
import typing as tp


if tp.TYPE_CHECKING:
    from sebox import Task, Workspace


    class KernelModule(tp.Protocol):
        """Required functions in a solver module."""
        kernel: Task[Kernel]


    class Misfit(Workspace):
        """Compute misfit only."""


    class Kernel(Workspace):
        """Compute misfit and kernel."""
        # compute misfit only (no adjoint simulation)
        misfit_only: bool
