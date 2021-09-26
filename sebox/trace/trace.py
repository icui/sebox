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


def gather(ws: Workspace, name: tp.Optional[str] = None,
    path_trace: tp.Optional[str] = None,
    path_mpi: tp.Optional[str] = None):
    """Convert MPI trace to bundled trace."""
    data = tp.cast(dict, { 'task': ('sebox.trace', 'gather') })
    
    if path_trace is not None:
        data['path_trace'] = path_trace
    
    if path_mpi is not None:
        data['path_mpi'] = path_mpi

    if name is None:
        return ws.add(data)

    return ws.add(name, data)


def scatter(ws: Workspace, name: tp.Optional[str] = None,
    path_trace: tp.Optional[str] = None,
    path_mpi: tp.Optional[str] = None):
    """Convert bundled trace to MPI trace."""
    data = tp.cast(dict, { 'task': ('sebox.trace', 'scatter') })
    
    if path_trace is not None:
        data['path_trace'] = path_trace
    
    if path_mpi is not None:
        data['path_mpi'] = path_mpi

    if name is None:
        return ws.add(data)
    
    return ws.add(name, data)

