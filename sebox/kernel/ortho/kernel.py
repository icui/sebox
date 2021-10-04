from __future__ import annotations
import typing as tp

from sebox.utils.catalog import getdir, getstations
from .ft import ft, diff, gather

if tp.TYPE_CHECKING:
    from .typing import Ortho


def kernel(node: Ortho):
    """Compute kernels."""
    for iker in range(node.nkernels or 1):
        # add steps to run forward and adjoint simulation
        node.add(_compute_kernel, f'kl_{iker:02d}', iker=iker)


def _compute_kernel(node: Ortho):
    # load source encoding parameters
    node.update(node.load('encoding.pickle'))
    node.fslots = node.load('fslots.pickle')

    # forward simulation
    node.add('solver', 'forward',
        path_event= node.path('SUPERSOURCE'),
        path_stations= getdir().path('SUPERSTATION'),
        path_mesh= node.path('../mesh'),
        monochromatic_source= True,
        save_forward= True)
    
    # compute misfit
    node.add(_compute_misfit)

    # adjoint simulation
    if not node.misfit_only:
        node.add('solver.adjoint', 'adjoint',
            path_forward = node.path('forward'),
            path_misfit = node.path('adjoint.h5'))


def _compute_misfit(node: Ortho):
    stas = getstations()

    # process traces
    node.add_mpi(ft, arg=node, arg_mpi=stas, cwd='enc_syn')
    
    # compute misfit
    node.add_mpi(diff, arg=node, arg_mpi=stas, cwd='enc_mf')

    # convert adjoint sources to ASDF format
    node.add(gather)
