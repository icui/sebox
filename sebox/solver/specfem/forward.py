from __future__ import annotations
import typing as tp

from sebox import root
from .mesh import setup as setup_mesh
from .shared import setpars, xmeshfem, xspecfem

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
    
    setpars(node, pars)


def scatter(node: Specfem):
    """Convert output seismograms with processing format."""
    from sebox.utils.catalog import getstations
    from obspy import read

    stas = getstations()
    tr = read(node.path(f'OUTPUT_FILES/{stas[0]}.MXE.sem.sac'))[0]
    stats: Stats = {
        'nt': tr.stats.npts,
        'dt': tr.stats.delta,
        'cmps': ('N', 'E', 'Z')
    }

    node.dump(stats, 'traces/stats.pickle')
    node.add_mpi(_scatter, arg=stats, arg_mpi=stas)


def _scatter(stats: Stats, stas: tp.List[str]):
    import numpy as np
    from obspy import read

    data = np.zeros([len(stas), len(stats['cmps']), stats['nt']], dtype='float32')

    for i, sta in enumerate(stas):
        for j, cmp in enumerate(stats['cmps']):
            tr = read(root.mpi.path(f'OUTPUT_FILES/{sta}.MX{cmp}.sem.sac'))[0]
            data[i, j] = tr.data
    
    root.mpi.mpidump(data, 'traces')



def forward(node: Specfem):
    """Forward simulation."""
    node.add(setup)
    xmeshfem(node)
    xspecfem(node)
    node.add(('sebox.utils.asdf', 'scatter'),
        path_input=node.path('OUTPUT_FILES/synthetic.h5'), path_output=node.path('traces'),
        stats={'cmps': ['N', 'E', 'Z']})
