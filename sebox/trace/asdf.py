from __future__ import annotations
import typing as tp

from sebox import root

if tp.TYPE_CHECKING:
    from sebox.trace import Trace


def gather(ws: Trace):
    """Convert MPI trace to ASDF trace."""
    from pyasdf import ASDFDataSet


def _scatter(stas: tp.List[str]):
    print(stas)


async def scatter(ws: Trace):
    """Convert ASDF trace to MPI trace."""
    from pyasdf import ASDFDataSet

    src = ws.rel(ws.path_trace)

    with ASDFDataSet(src, mode='r', mpi=False) as ds:
        stas = ds.waveforms.list()
        await ws.mpiexec(_scatter, root.task_nprocs, arg=src, arg_mpi=stas)
