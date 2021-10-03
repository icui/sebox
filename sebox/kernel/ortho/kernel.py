from __future__ import annotations
import typing as tp

from sebox.utils.catalog import getdir, getstations
from .ft import ft
from .diff import diff, gather

if tp.TYPE_CHECKING:
    from .typing import Kernel


def compute_kernel(ws: Kernel):
    # forward simulation
    ws.add('solver', 'forward',
        path_event= ws.path('SUPERSOURCE'),
        path_stations= getdir().path('SUPERSTATION'),
        path_mesh= ws.path('../mesh'),
        monochromatic_source= True,
        save_forward= True)
    
    # compute misfit
    ws.add(_compute_misfit)

    # adjoint simulation
    if not ws.misfit_only:
        ws.add('solver.adjoint', 'adjoint',
            path_forward = ws.path('forward'),
            path_misfit = ws.path('adjoint.h5'))


def _compute_misfit(ws: Kernel):
    stas = getstations()

    # process traces
    ws.add_mpi(ft, arg=ws, arg_mpi=stas, cwd='enc_syn')
    
    # compute misfit
    ws.add_mpi(diff, arg=ws, arg_mpi=stas, cwd='enc_mf')

    # convert adjoint sources to ASDF format
    ws.add(gather)
