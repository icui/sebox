from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from .typing import Ortho


def main(node: Ortho):
    # prepare catalog (executed only once for a catalog)
    node.add('kernel.catalog', 'catalog')

    # mesher and preprocessing
    node.add('kernel.preprocess')

    # forward simulation
    node.add('kernel.forward')

    # misfit calculation
    node.add('kernel.misfit')

    # adjoint simulation
    node.add('kernel.adjoint')

    # sum and smooth kernels
    node.add('kernel.postprocess')


def dirs(node: Ortho):
    """Get all kernel directories."""
    return [f'kl_{iker:02d}' for iker in range(node.nkernels or 1)]
