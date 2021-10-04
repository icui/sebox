from sebox import Task
from .solver import *
from .kernel import *
from .optimizer import *
from .search import *


class Solver(tp.Protocol):
    """Required functions in a solver module."""
    # forward simulation
    main: Task[Forward]

    # generate mesh
    mesh: Task[Forward]

    # adjoint simulation
    adjoint: Task[Adjoint]

    # sum and smooth kernels
    postprocess: Task[Postprocess]


class Kernel(tp.Protocol):
    """Required functions in a solver module."""
    main: Task[Kernel]
    
    # compute misfit
    misfit: Task[Kernel]


class Optimizer(tp.Protocol):
    """Required functions in a optimizer module."""
    # create iterations
    main: Task[Optimizer]

    # add an iteration
    iterate: Task[Optimizer]

    # compute search direction
    direction: Task[Optimizer]


class Search(tp.Protocol):
    """Required functions in a search module."""
    # create search steps
    main: Task[Search]

    # add a search step
    step: Task[Search]


class System(tp.Protocol):
    """Required functions in a kernel module."""
    # run a MPI task
    mpiexec: tp.Callable[[str, int, int, int], str]

    # resubmit current job
    requeue: tp.Callable[[], None]

    # submit new job
    submit: tp.Callable[[str, str], None]

    # number of CPUs per node
    cpus_per_node: int

    # number of GPUs per node
    gpus_per_node: int
