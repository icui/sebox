import typing as tp

from sebox import Task
from sebox.solver import Mesh, Forward, Adjoint
from sebox.trace import Trace


class Solver(tp.Protocol):
    """Required functions in a solver module."""
    # generate mesh
    mesh: Task[Mesh]

    # forward simulation
    forward: Task[Forward]

    # adjoint simulation
    adjoint: Task[Adjoint]


class Trace(tp.Protocol):
    """Required functions in a trace module."""
    # extract from archive format to MPI format
    scatter: Task[Trace]

    # bundle MPI format to archive format
    gather: Task[Trace]
