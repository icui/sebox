from __future__ import annotations
import typing as tp

from sebox.utils.catalog import getdir, getstations
from .ft import ft, mf, mfadj
from .main import dirs

if tp.TYPE_CHECKING:
    from .typing import Ortho


def forward(node: Ortho):
    """Forward simulation."""
    node.concurrent = True

    for cwd in dirs(node):
        node.add('solver', f'{cwd}/forward', cwd,
            path_event=node.path(f'{cwd}/SUPERSOURCE'),
            path_stations=node.path(f'{cwd}/SUPERSTATION'),
            path_mesh= node.path('mesh'),
            monochromatic_source= True,
            save_forward=not node.misfit_only)


def misfit(node: Ortho):
    """Compute misfit and adjoint source."""
    node.concurrent = True

    for cwd in dirs(node):
        node.add(_misfit, cwd)


def adjoint(node: Ortho):
    """Adjoint simulation."""
    if node.misfit_only:
        return

    node.concurrent = True

    for cwd in dirs(node):
        node.add('solver.adjoint', f'{cwd}/adjoint', cwd,
            path_forward = node.path(f'{cwd}/forward'),
            path_misfit = node.path(f'{cwd}/adjoint.h5'))


def _misfit(node: Ortho):
    stas = getstations()
    enc = node.load('encoding.pickle')

    # process traces
    node.add_mpi(ft, arg=enc, arg_mpi=stas, cwd='enc_syn')
    
    # compute misfit
    node.add_mpi(mf if node.misfit_only else mfadj, arg=enc, arg_mpi=stas, cwd='enc_mf')
