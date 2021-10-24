from __future__ import annotations
import typing as tp

from sebox.utils.catalog import getstations
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
            path_mesh=node.path('mesh'),
            monochromatic_source=True,
            save_forward=not node.misfit_only)
        
        if node.test_encoding:
            node.add('solver', f'{cwd}/observed', cwd,
                path_event=node.path(f'{cwd}/SUPERSOURCE'),
                path_stations=node.path(f'{cwd}/SUPERSTATION'),
                path_mesh=node.path_mesh2,
                monochromatic_source=True,
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
    node.mkdir('enc_syn')
    node.add_mpi(ft, arg=(enc, 'forward/traces', 'enc_syn', True), arg_mpi=stas)

    if node.test_encoding:
        node.mkdir('enc_obs')
        node.add_mpi(ft, arg=(enc, 'observed/traces', 'enc_obs', True), arg_mpi=stas)
    
    # compute misfit
    node.add_mpi(mf if node.misfit_only else mfadj, arg=enc, arg_mpi=stas, cwd='enc_mf')
