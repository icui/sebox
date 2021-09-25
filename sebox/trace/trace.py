from __future__ import annotations
import typing as tp


if tp.TYPE_CHECKING:
    from sebox import Workspace


    class Trace(Workspace):
        """A workspace to convert trace from / to MPI format."""
        # path to trace file
        path_trace: str

        # path to MPI trace file
        path_mpi: str


def gather(ws: Workspace, name: str, *,
    path_trace: tp.Optional[str] = None,
    path_mpi: tp.Optional[str] = None):
    """Convert MPI trace to bundled trace."""
    trace = tp.cast(Trace, ws.add(name, { 'task': ('sebox.trace', 'gather') }))

    if path_trace is not None:
        trace.path_trace = path_trace
    
    if path_mpi is not None:
        trace.path_mpi = path_mpi
    
    return trace


def scatter(ws: Workspace, name: str, *,
    path_trace: tp.Optional[str] = None,
    path_mpi: tp.Optional[str] = None):
    """Convert bundled trace to MPI trace."""
    trace = tp.cast(Trace, ws.add(name, { 'task': ('sebox.trace', 'scatter') }))

    if path_trace is not None:
        trace.path_trace = path_trace
    
    if path_mpi is not None:
        trace.path_mpi = path_mpi
    
    return trace
