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

    smap = {}
    nt = None
    
    for i in range(getsize(node)):
        if node.has(fname := f'OUTPUT_FILES/array_stations_node_{i:05d}.txt'):
            lines = node.readlines(fname)

            n = int(lines[0])

            if nt is None:
                nt = int(lines[1])
                dt = float(lines[2])

            for line in lines:
                if '#' in line:
                    idx, sta = line.split('#')[1].lstrip().split(' ')
                    smap[sta] = i, int(idx) - 1, n

    node.mkdir('traces')
    node.add_mpi(_scatter, arg=(smap, nt), arg_mpi=getstations())


def _scatter(arg: tp.Tuple[tp.Dict[str, tp.Tuple[int, int, int]], int], stas: tp.List[str]):
    import numpy as np
    from scipy.io import FortranFile
    from sebox import root

    smap, nt = arg
    data = np.zeros([len(stas), 3, nt], dtype='float32')
    cache = {}

    for i, sta in enumerate(stas):
        for j in range(3):
            p, k, n = smap[sta]

            if p not in cache:
                d = FortranFile(f'OUTPUT_FILES/array_seismograms_node_{p:05d}.bin').read_reals(dtype='float32')
                cache[p] = d.reshape([nt, n, 3]).transpose() # type: ignore
            
            data[i, j] = cache[p][j, k]
    
    root.mpi.mpidump(data, 'traces')



def forward(node: Specfem):
    """Forward simulation."""
    node.add(setup)
    xmeshfem(node)
    xspecfem(node)
    node.add(('sebox.utils.asdf', 'scatter'),
        path_input=node.path('OUTPUT_FILES/synthetic.h5'), path_output=node.path('traces'),
        stats={'cmps': ['N', 'E', 'Z']})
