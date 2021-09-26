from __future__ import annotations
from os import path
import typing as tp

from sebox import root

if tp.TYPE_CHECKING:
    from pyasdf import ASDFDataSet
    from obspy import Trace

    from sebox.typing import Convert

    class Stats(tp.TypedDict):
        # number of timesteps
        nt: int

        # length of a timestep
        dt: int

        # trace components
        cmps: tp.List[str]


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


def _scatter_npy(arg: tp.Tuple[str, str, Stats], stas: tp.List[str]):
    import numpy as np
    from pyasdf import ASDFDataSet

    from sebox import Directory
    from sebox.core.mpi import pid

    src, dst, stats = arg

    with ASDFDataSet(src, mode='r', mpi=False) as ds:
        data = np.zeros([len(stas), len(stats['cmps']), stats['nt']])

        for i, sta in enumerate(stas):
            stream = get_stream(ds, sta)

            for j, cmp in enumerate(stats['cmps']):
                trace = stream[j]
                
                assert trace.stats.npts == stats['nt']
                assert trace.stats.delta == stats['dt']
                assert trace.stats.component == cmp

                data[i, j, :] = trace.data
        
        Directory(dst).dump(data, f'{pid}.npy', mkdir=False)


async def scatter(ws: Convert):
    """Convert ASDF trace to MPI trace."""
    from pyasdf import ASDFDataSet

    src = ws.rel(ws.path_bundle)
    dst = ws.rel(ws.path_mpi)
    ws.mkdir(ws.path_mpi)

    with ASDFDataSet(src, mode='r', mpi=False) as ds:
        stas = ds.waveforms.list()

        if stats := ws.stats:
            # save data as numpy array with a collective stats file
            if 'nt' not in stats or 'dt' not in stats or 'cmps' not in stats:
                # read nt and dt from the first trace
                stream = get_stream(ds, stas[0])
                stats['nt'] = stream[0].stats.npts
                stats['dt'] = stream[0].stats.delta
                stats['cmps'] = []

                for trace in stream:
                    stats['cmps'].append(trace.stats.component)

                ws.dump(stats, path.join(ws.path_mpi, 'stats.pickle'))
                await ws.mpiexec(_scatter_npy, root.task_nprocs, arg=(src, dst, stats), arg_mpi=stas)
        
        else:
            # save data as traces
            await ws.mpiexec(_scatter, root.task_nprocs, arg=(src, dst), arg_mpi=stas)
