from __future__ import annotations
import typing as tp

from sebox import root

if tp.TYPE_CHECKING:
    from pyasdf import ASDFDataSet
    from obspy import Trace

    from sebox.typing import Convert


def get_stream(ds: ASDFDataSet, sta: str) -> tp.List[Trace]:
    wav = ds.waveforms[sta]
    return tp.cast(tp.Any, wav[wav.get_waveform_tags()[0]])


async def gather(ws: Convert):
    """Convert MPI trace to ASDF trace."""
    from pyasdf import ASDFDataSet
    from sebox import Directory

    d = Directory(ws.path_mpi)

    with ASDFDataSet(ws.rel(ws.path_bundle), mode='w', mpi=False) as ds:
        for pid in d.ls():
            for stream in d.load(pid).values():
                ds.add_waveforms(stream, ws.tag_bundle or 'sebox')


def _scatter(path: tp.Tuple[str, str, dict], stas: tp.List[str]):
    from pyasdf import ASDFDataSet

    from sebox import Directory
    from sebox.core.mpi import pid

    with ASDFDataSet(path[0], mode='r', mpi=False) as ds:
        data = {}

        for sta in stas:
            data[sta] = get_stream(ds, sta)
        
        Directory(path[1]).dump(data, f'{pid}.pickle', mkdir=False)


async def scatter(ws: Convert):
    """Convert ASDF trace to MPI trace."""
    from pyasdf import ASDFDataSet

    src = ws.rel(ws.path_bundle)
    dst = ws.rel(ws.path_mpi)
    ws.mkdir(ws.path_mpi)

    with ASDFDataSet(src, mode='r', mpi=False) as ds:
        stas = ds.waveforms.list()
        await ws.mpiexec(_scatter, root.task_nprocs, arg=(src, dst), arg_mpi=stas)
