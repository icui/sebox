from __future__ import annotations
import typing as tp


class System(tp.Protocol):
    """Required functions in a system module."""
    # run a MPI task
    mpiexec: tp.Callable[[str, int, int, int], str]

    # resubmit current job
    requeue: tp.Callable[[], None]

    # number of CPUs per node
    cpus_per_node: int

    # number of GPUs per node
    gpus_per_node: int
