from __future__ import annotations
import typing as tp

from sebox import root
from .mesh import setup as setup_mesh
from .shared import setpars, xmeshfem, xspecfem, getsize

if tp.TYPE_CHECKING:
    from .typing import Par_file, Specfem

    class Stats(tp.TypedDict, total=False):
        # total number of timesteps
        nt: int

        # length of a timestep
        dt: float

        # trace components
        cmps: tp.Tuple[str, str, str]


def setup(node: Specfem):
    """Create forward node."""
    if not node.path_mesh and not node.path_model:
        raise AttributeError('path_mesh or path_model is required')
    
    setup_mesh(node)

    # update Par_file
    pars: Par_file = { 'SIMULATION_TYPE': 1 }

    if node.save_forward is not None:
        pars['SAVE_FORWARD'] = node.save_forward
    
    if node.monochromatic_source is not None:
        pars['USE_MONOCHROMATIC_CMT_SOURCE'] = node.monochromatic_source
    
    if node.duration is not None:
        pars['RECORD_LENGTH_IN_MINUTES'] = node.duration
    
    if node.transient_duration is not None:
        if node.duration is None:
            raise ValueError('solver duration must be set if transient_duration exists')

        pars['STEADY_STATE_KERNEL'] = True
        pars['STEADY_STATE_LENGTH_IN_MINUTES'] = node.duration - node.transient_duration
    
    else:
        pars['STEADY_STATE_KERNEL'] = False
    
    if node.use_asdf:
        pars['OUTPUT_SEISMOS_ASDF'] = True
        pars['OUTPUT_SEISMOS_3D_ARRAY'] = False
    
    else:
        pars['OUTPUT_SEISMOS_ASDF'] = False
        pars['OUTPUT_SEISMOS_3D_ARRAY'] = True
    
    setpars(node, pars)


def align(node: Specfem):
    """Convert output seismograms with processing format."""
    from sebox.utils.catalog import getstations

    lines = node.readlines('OUTPUT_FILES/seismogram_stats.txt')
    stats: Stats = {
        'dt': float(lines[0].split('=')[-1]),
        'nt': int(lines[1].split('=')[-1]),
        'cmps': ('N', 'E', 'Z')
    }
    nodes = {}

    for p in range(getsize(node)):
        if node.has(fname := f'OUTPUT_FILES/array_stations_node_{p:05d}.txt'):
            lines = node.readlines(fname)
            nodes[p] = []

            for line in lines:
                if '#' in line:
                    sta = line.split('#')[1].rstrip().split(' ')[-1]
                    nodes[p].append(sta)

    node.dump(stats, 'traces/stats.pickle')
    node.mkdir('stations')
    node.add_mpi(_align, arg=(stats, nodes), arg_mpi=getstations())


def _align(arg: tp.Tuple[Stats, tp.Dict[int, tp.List[str]]], stas: tp.List[str]):
    import numpy as np
    from scipy.io import FortranFile
    from sebox import root
    
    stats, nodes = arg
    data = np.full([len(stas), len(stats['cmps']), stats['nt']], np.nan)

    for p, pstas in nodes.items():
        if any(sta in stas for sta in pstas):
            d = FortranFile(root.mpi.path(f'OUTPUT_FILES/array_seismograms_node_{p:05d}.bin')).read_reals(dtype='float32')
            d = d.reshape([stats['nt'], len(pstas), 3]) # type: ignore
            
            for k, sta in enumerate(pstas):
                if sta in stas:
                    for j in range(3):
                        data[stas.index(sta), j] = d[:, k, j]
    
    root.mpi.mpidump(data, 'traces')
    root.mpi.mpidump(stas, 'stations')


def forward(node: Specfem):
    """Forward simulation."""
    node.add(setup)
    xmeshfem(node)
    xspecfem(node)

    if node.use_asdf:
        node.add(('sebox.utils.asdf', 'scatter'),
            stats={'cmps': ['N', 'E', 'Z']}, save_stations=node.path('stations'),
            path_input=node.path('OUTPUT_FILES/synthetic.h5'), path_output=node.path('traces'))
    
    else:
        node.add('solver.align')
