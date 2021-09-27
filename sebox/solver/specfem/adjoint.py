from __future__ import annotations
import typing as tp

from sebox import Directory
from .specfem import xspecfem, setpars

if tp.TYPE_CHECKING:
    from sebox.typing import Adjoint


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
    ws.ln(d.path('bin'))
    ws.ln(d.path('DATABASES_MPI'))
    ws.ln(d.path('DATA/*'), 'DATA')
    ws.ln(ws.rel(ws.path_misfit), 'SEM/adjoint.h5')
    ws.ln('DATA/STATIONS', 'DATA/STATIONS_ADJOINT')

    #  update Par_file
    ws.cp(d.path('DATA/Par_file'), 'DATA')
    setpars(ws, { 'SIMULATION_TYPE': 3, 'SAVE_FORWARD': False })


def finalize(ws: Adjoint):
    """Move generated kernels."""
    ws.mv('OUTPUT_FILES/kernels.bp', 'kernels.bp')


def adjoint(ws: Adjoint):
    """Forward simulation."""
    ws.add(setup)
    xspecfem(ws)
    ws.add(finalize)
