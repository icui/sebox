from __future__ import annotations
import typing as tp

from sebox import Directory
from .specfem import xspecfem
from .utils import setpars

if tp.TYPE_CHECKING:
    from sebox.solver import Adjoint


def setup(ws: Adjoint):
    """Create adjoint workspace."""
    # specfem directory for forward simulation
    if not ws.path_forward:
        raise ValueError('forward solver not specified')
    
    if not ws.path_adjoint:
        raise ValueError('adjoint source not specified')

    d = Directory(ws.path_forward)

    # create directories
    ws.mkdir('SEM')
    ws.mkdir('DATA')
    ws.mkdir('OUTPUT_FILES')
    
    # link files
    ws.ln(d.abs('bin'))
    ws.ln(d.abs('DATABASES_MPI'))
    ws.ln(d.abs('DATA/*'), 'DATA')
    ws.ln(ws.path_misfit, 'SEM/adjoint.h5')
    ws.ln('DATA/STATIONS', 'DATA/STATIONS_ADJOINT')

    #  update Par_file
    ws.cp(d.abs('DATA/Par_file'), 'DATA')
    setpars(ws, { 'SIMULATION_TYPE': 3, 'SAVE_FORWARD': False })


def finalize(ws: Adjoint):
    """Move generated kernels."""
    ws.mv('OUTPUT_FILES/kernels.bp', 'kernels.bp')


def adjoint(ws: Adjoint):
    """Forward simulation."""
    ws.add(setup)
    ws.add(xspecfem)
    ws.add(finalize)
