from sebox import Task
from .solver import *
from .kernel import *


class Solver(tp.Protocol):
    """Required functions in a solver module."""
    # generate mesh
    mesh: Task[Forward]

    # forward simulation
    forward: Task[Forward]

    # adjoint simulation
    adjoint: Task[Adjoint]

    # sum and smooth kernels
    postprocess: Task[Sum]


class Kernel(tp.Protocol):
    """Required functions in a solver module."""
    # compute kernels
    kernel: Task[Kernel]

    # compute misfit
    misfit: Task[Kernel]


class System(tp.Protocol):
    """Required functions in a kernel module."""
    # run a MPI task
    mpiexec: tp.Callable[[str, int, int, int], str]

    # resubmit current job
    requeue: tp.Callable[[], None]

    # number of CPUs per node
    cpus_per_node: int

    # number of GPUs per node
    gpus_per_node: int
