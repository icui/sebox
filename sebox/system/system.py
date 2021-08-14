from __future__ import annotations
import typing as tp


class System(tp.Protocol):
    """Required functions in a system module."""
    # run a MPI task
    mpiexec: tp.Callable[[str, int, int, int], str]

    # resubmit current job
    requeue: tp.Callable[[], None]
