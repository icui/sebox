from __future__ import annotations
import typing as tp

from sebox.utils.catalog import getdir, getstations
from .ft import ft, diff, gather

if tp.TYPE_CHECKING:
    from .typing import Ortho


def forward(node: Ortho):
    """Forward simulation."""
    for iker in range(node.nkernels or 1):
        # add steps to run forward and adjoint simulation
        node.add('solver', f'kl_{iker:02d}/forward',
            path_event= node.path(f'kl_{iker:02d}/SUPERSOURCE'),
            path_stations= getdir().path('SUPERSTATION'),
            path_mesh= node.path('mesh'),
            monochromatic_source= True,
            save_forward= True)


def misfit(node: Ortho):
    """Compute misfit and adjoint source."""
    for iker in range(node.nkernels or 1):
        node.add(_misfit, f'kl_{iker:02d}')


def adjoint(node: Ortho):
    """Adjoint simulation."""
    if not node.misfit_only:
        for iker in range(node.nkernels or 1):
            node.add('solver.adjoint', f'kl_{iker:02d}/adjoint',
                path_forward = node.path(f'kl_{iker:02d}/forward'),
                path_misfit = node.path(f'kl_{iker:02d}/adjoint.h5'))


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
