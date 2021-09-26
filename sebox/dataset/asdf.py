from __future__ import annotations
import typing as tp

from sebox import root

if tp.TYPE_CHECKING:
    from sebox.typing import Convert


def _scatter(path: tp.Tuple[str, str], stas: tp.List[str]):
    from pyasdf import ASDFDataSet
    from sebox import Directory
    from sebox.core.mpi import comm

    rank, size = comm()
    print(path)

    with ASDFDataSet(path[0], mode='r', mpi=False) as ds:
        data = {}

        for sta in stas:
            wav = ds.waveforms[sta]
            data[sta] = wav[wav.get_waveform_tags()[0]]
        
        zeros = '0' * (len(str(size - 1)) - len(str(rank)))
        Directory(path[1]).dump(data, f'p{zeros}{rank}.pickle', mkdir=False)


def gather(ws: Convert):
    """Convert MPI trace to ASDF trace."""
    from pyasdf import ASDFDataSet


async def scatter(ws: Convert):
    """Convert ASDF trace to MPI trace."""
    from pyasdf import ASDFDataSet

    src = ws.rel(ws.path_bundle)
    dst = ws.rel(ws.path_mpi)
    path = (src, dst)
    ws.mkdir(ws.path_mpi)

    with ASDFDataSet(src, mode='r', mpi=False) as ds:
        stas = ds.waveforms.list()
        await ws.mpiexec(_scatter, root.task_nprocs, arg=path, arg_mpi=stas)
