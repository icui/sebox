from __future__ import annotations
from os import path
import typing as tp

from sebox import root
from .asdf import get_stream

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


async def gather(ws: Convert):
    """Convert MPI trace to ASDF trace."""
    from pyasdf import ASDFDataSet


def _scatter(arg: tp.Tuple[str, str, Stats], stas: tp.List[str]):
    import numpy as np
    from pyasdf import ASDFDataSet

    from sebox import Directory
    from sebox.core.mpi import pid

    src, dst, stats = arg

    with ASDFDataSet(src, mode='r', mpi=False) as ds:
        ncmps = len(stats['cmps'])
        data = np.zeros([len(stas), ncmps, stats['nt']])

        for i, sta in enumerate(stas):
            stream = get_stream(ds, sta)

            for j, trace in enumerate(stream):
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
        stats: Stats = tp.cast(tp.Any, ws.stats or {})

        if 'nt' not in stats or 'dt' not in stats or 'cmps' not in stats:
            # read nt and dt from the first trace
            stream = get_stream(ds, stas[0])
            stats['nt'] = stream[0].stats.npts
            stats['dt'] = stream[0].stats.delta
            stats['cmps'] = []

            for trace in stream:
                stats['cmps'].append(trace.stats.component)

        # make sure nt, dt and component number are consistent across all traces
        for sta in stas:
            stream = get_stream(ds, sta)
            
            for i, cmp in enumerate(stats['cmps']):
                assert cmp == stream[i].stats.component
            
            for trace in stream:
                assert stats['nt'] == trace.stats.npts
                assert stats['dt'] == trace.stats.delta

        ws.dump(stats, path.join(ws.path_mpi, 'stats.pickle'))
        await ws.mpiexec(_scatter, root.task_nprocs, arg=(src, dst, stats), arg_mpi=stas)
