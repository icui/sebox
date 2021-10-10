from __future__ import annotations
import typing as tp

from .mesh import setup as setup_mesh
from .shared import setpars, xmeshfem, xspecfem, getsize

if tp.TYPE_CHECKING:
    from .typing import Par_file, Specfem


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

    stamap = {}
    
    for i in range(getsize(node)):
        if node.has(fname := f'OUTPUT_FILES/array_stations_node_{i:05d}.txt'):
            for line in node.readlines(fname):
                if '.' in line:
                    stamap[line.lstrip()] = i

    print(stamap)
    print(len(stamap.keys()))
    node.mkdir('traces')
    node.add_mpi(_scatter, arg=stamap, arg_mpi=getstations())


def _scatter(stamap: tp.Dict[str, int], stas: tp.List[str]):
    import numpy as np
    from scipy.io import FortranFile
    from sebox import root

    nt = None
    nsta = len(stas)
    data = None
    cache = {}

    for i, sta in enumerate(stas):
        for j in range(3):
            p = stamap[sta]

            if p not in cache:
                d = FortranFile(f'OUTPUT_FILES/array_seismograms_node_{p:05d}.bin').read_reals(dtype='float32')
                
                if nt is None:
                    nt = int(len(d) / 3 / nsta)
                    data = np.zeros((3, nsta, nt))
                
                cache[p] = d.reshape((nt, nsta, 3)) # type: ignore
            
            data[i, j, :] = cache[p][:, i, j] # type: ignore
    
    root.mpi.mpidump(data, 'traces')



def forward(node: Specfem):
    """Forward simulation."""
    node.add(setup)
    xmeshfem(node)
    xspecfem(node)
    node.add(('sebox.utils.asdf', 'scatter'),
        path_input=node.path('OUTPUT_FILES/synthetic.h5'), path_output=node.path('traces'),
        stats={'cmps': ['N', 'E', 'Z']})
