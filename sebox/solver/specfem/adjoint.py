from __future__ import annotations
import typing as tp

from sebox import Directory
from .forward import xspecfem
from .shared import setpars

if tp.TYPE_CHECKING:
    from .typing import Specfem


def setup(node: Specfem):
    """Create adjoint node."""
    # specfem directory for forward simulation
    if not node.path_forward:
        raise ValueError('forward solver not specified')
    
    if not node.path_misfit:
        raise ValueError('adjoint source not specified')

    d = Directory(node.path_forward)

    # create directories
    node.mkdir('SEM')
    node.mkdir('DATA')
    node.mkdir('OUTPUT_FILES')
    
    # link files
    node.ln(node.rel(d, 'bin'))
    node.ln(node.rel(d, 'DATABASES_MPI'))
    node.ln(node.rel(d, 'DATA/*'), 'DATA')
    node.ln(node.rel(node.path_misfit), 'SEM/adjoint.h5')
    node.ln('DATA/STATIONS', 'DATA/STATIONS_ADJOINT')

    #  update Par_file
    node.rm('DATA/Par_file')
    node.cp(node.rel(d, 'DATA/Par_file'), 'DATA')
    setpars(node, { 'SIMULATION_TYPE': 3, 'SAVE_FORWARD': False })


def adjoint(node: Specfem):
    """Forward simulation."""
    node.add(setup)
    xspecfem(node)
