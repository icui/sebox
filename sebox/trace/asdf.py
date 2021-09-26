import typing as tp

from sebox import root

if tp.TYPE_CHECKING:
    from sebox.trace import Trace


def gather(ws: Trace):
    """Convert MPI trace to ASDF trace."""
    from pyasdf import ASDFDataSet


def scatter(ws: Trace):
    """Convert ASDF trace to MPI trace."""
    import numpy as np
    from pyasdf import ASDFDataSet

    nprocs = root.task_nprocs

    with ASDFDataSet(ws.rel(ws.path_trace), mode='r', mpi=False) as ds:
        stas = ds.waveforms.list()
        stas_mpi: tp.List[tp.List[str]] = []
        chunk = max(1, int(np.round(len(stas) / nprocs)))

        # assign processors with stations
        for i in range(nprocs - 1):
            stas_mpi.append(stas[i * chunk: (i + 1) * chunk])
        
        stas_mpi.append(stas[(nprocs - 1) * chunk:])
