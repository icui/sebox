from __future__ import annotations
import typing as tp

from sebox.utils.catalog import getdir, getstations
from .ft import ft, diff, gather

if tp.TYPE_CHECKING:
    from .typing import Ortho


def forward(node: Ortho):
    """Forward simulation."""
    for cwd in _cwd(node):
        node.add('solver', f'k{cwd}/forward', cwd,
            path_event= node.path(f'{cwd}/SUPERSOURCE'),
            path_stations= getdir().path('SUPERSTATION'),
            path_mesh= node.path('mesh'),
            monochromatic_source= True,
            save_forward= True)


def misfit(node: Ortho):
    """Compute misfit and adjoint source."""
    for cwd in _cwd(node):
        node.add(_misfit, cwd)


def adjoint(node: Ortho):
    """Adjoint simulation."""
    if node.misfit_only:
        return

    for cwd in _cwd(node):
        node.add('solver.adjoint', f'{cwd}/adjoint', cwd,
            path_forward = node.path(f'{cwd}/forward'),
            path_misfit = node.path(f'{cwd}/adjoint.h5'))


def _cwd(node: Ortho):
    return [f'kl_{iker:02d}' for iker in range(node.nkernels or 1)]


def _misfit(node: Ortho):
    stas = getstations()

    # load source encoding parameters
    node.update(node.load('encoding.pickle'))
    node.fslots = node.load('fslots.pickle')

    # process traces
    node.add_mpi(ft, arg=node, arg_mpi=stas, cwd='enc_syn')
    
    # compute misfit
    node.add_mpi(diff, arg=node, arg_mpi=stas, cwd='enc_mf')

    # convert adjoint sources to ASDF format
    node.add(gather)
