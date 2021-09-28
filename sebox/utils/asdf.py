from __future__ import annotations
from os import path
import typing as tp

from sebox import root, Workspace
from sebox.utils.catalog import getstations, getcomponents

if tp.TYPE_CHECKING:
    from pyasdf import ASDFDataSet
    from obspy import Trace

    class Stats(tp.TypedDict, total=False):
        # length of a trace data
        n: int

        # trace components
        cmps: tp.List[str]

        # stations used
        stas: tp.List[str]

        # station channel
        cha: tp.Optional[str]

        # length of a timestep
        dt: float
        
        # data type
        dtype: tp.Any
    

    class Convert(Workspace):
        """A workspace to convert dataset from / to MPI format."""
        # path to bundled data file
        path_bundle: str

        # path to MPI data file
        path_mpi: str

        # tag of output file
        output_tag: tp.Optional[str]

        # collective data
        stats: tp.Optional[Stats]

        # use auxiliary data instead of waveform data
        aux: tp.Optional[bool]


def gettrace(ds: ASDFDataSet, sta: str, cmp: str) -> Trace:
    """Get trace based on station and component."""
    wav = ds.waveforms[sta]
    return tp.cast('Trace', wav[wav.get_waveform_tags()[0]].select(component=cmp)[0])


async def gather(ws: Convert):
    """Convert MPI trace to ASDF trace."""
    from pyasdf import ASDFDataSet
    from sebox import Directory

    d = Directory(ws.path_mpi)

    with ASDFDataSet(ws.path_bundle, mode='w', mpi=False) as ds:
        for pid in d.ls():
            for stream in d.load(pid).values():
                ds.add_waveforms(stream, ws.output_tag or 'sebox')


def _scatter(arg: tp.Tuple[str, str, bool, Stats], stas: tp.List[str]):
    import numpy as np
    from pyasdf import ASDFDataSet

    from sebox import Directory
    from sebox.mpi import pid

    src, dst, aux, stats = arg
    cmps = stats['cmps']
    cha = stats['cha'] if aux else None

    with ASDFDataSet(src, mode='r', mpi=False) as ds:
        data = np.zeros([len(stas), len(cmps), stats['n']], dtype=stats.get('dtype') or float)
        
        if cha is not None:
            aux = ds.auxiliary_data[ds.auxiliary_data.list()[0]]
            tags = aux.list()
        
        else:
            aux = None
            tags = ds.waveforms.list()

        for i, sta in enumerate(stas):
            for j, cmp in enumerate(cmps):
                if cha is not None:
                    tag = sta.replace('.', '_') + f'_{cha}{cmp}'
                    if tag in tags:
                        data[i, j, :] = np.array(tp.cast(tp.Any, aux)[tag].data)
                
                else:
                    if sta in tags:
                        data[i, j, :] = gettrace(ds, sta, cmp).data
        
        Directory(dst).dump(data, f'{pid}.npy', mkdir=False)


async def scatter(ws: Convert):
    """Convert ASDF trace to MPI trace."""
    from pyasdf import ASDFDataSet

    ws.mkdir(ws.rel(ws.path_mpi))

    with ASDFDataSet(ws.path_bundle, mode='r', mpi=False) as ds:
        stats = tp.cast('Stats', ws.stats or {})

        # fill stattions and components
        if 'stas' not in stats:
            stats['stas'] = getstations()
        
        if 'cmps' not in stats:
            stats['cmps'] = getcomponents()
        
        # fill data info
        if ws.aux:
            aux = ds.auxiliary_data[ds.auxiliary_data.list()[0]]

            if 'cha' not in stats:
                stats['cha'] = aux.list()[0].split('_')[-1][:-1]
            
            data = aux[aux.list()[0]].data

            if 'n' not in stats:
                stats['n'] = len(data)
            
            if 'dtype' not in stats:
                stats['dtype'] = data.dtype
        
        else:
            trace = gettrace(ds, stats['stas'][0], stats['cmps'][0])
            
            if 'n' not in stats:
                stats['n'] = trace.stats.npts
            
            if 'dt' not in stats:
                stats['dt'] = trace.stats.delta

        # save stats
        ws.dump(stats, ws.rel(ws.path_mpi, 'stats.pickle'))

        await ws.mpiexec(_scatter, root.task_nprocs,
            arg=(ws.path_bundle, ws.path_mpi, ws.aux, stats), arg_mpi=stats['stas'])
