from __future__ import annotations
from os import path
import typing as tp

from sebox import root

if tp.TYPE_CHECKING:
    from pyasdf import ASDFDataSet
    from obspy import Trace

    from sebox import Workspace

    class Stats(tp.TypedDict):
        # number of timesteps
        nt: int

        # length of a timestep
        dt: int

        # trace components
        cmps: tp.List[str]
    

    class Convert(Workspace):
        """A workspace to convert dataset from / to MPI format."""
        # path to bundled data file
        path_bundle: str

        # path to MPI data file
        path_mpi: str

        # tag of bundled data file
        tag_bundle: tp.Optional[str]

        # tag of MPI data file
        tag_mpi: tp.Optional[str]

        # collective data
        stats: tp.Optional[dict]


def get_stream(ds: ASDFDataSet, sta: str) -> tp.List[Trace]:
    wav = ds.waveforms[sta]
    return tp.cast(tp.List['Trace'], wav[wav.get_waveform_tags()[0]])


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
    nt = stats['nt']

    with ASDFDataSet(src, mode='r', mpi=False) as ds:
        data = np.zeros([len(stas), len(stats['cmps']), nt])

        for i, sta in enumerate(stas):
            stream = get_stream(ds, sta)

            for j, cmp in enumerate(stats['cmps']):
                trace = stream[j]
                
                assert trace.stats.npts >= nt
                assert trace.stats.delta == stats['dt']
                assert trace.stats.component == cmp

                data[i, j, :] = trace.data[:nt]
        
        Directory(dst).dump(data, f'{pid}.npy', mkdir=False)


async def scatter(ws: Convert):
    """Convert ASDF trace to MPI trace."""
    from pyasdf import ASDFDataSet

    src = ws.rel(ws.path_bundle)
    dst = ws.rel(ws.path_mpi)
    ws.mkdir(ws.path_mpi)

    with ASDFDataSet(src, mode='r', mpi=False) as ds:
        stas = ds.waveforms.list()

        if (stats := ws.stats) is not None:
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
