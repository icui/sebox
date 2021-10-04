from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from .typing import Ortho


def main(node: Ortho):
    # prepare catalog (executed only once for a catalog)
    node.add('kernel.catalog', 'catalog', concurrent=True)

    # mesher and preprocessing
    node.add('kernel.preprocess', concurrent=True)

    # forward simulation
    node.add('kernel.forward', concurrent=True)

    # misfit calculation
    node.add('kernel.misfit', concurrent=True)

    # adjoint simulation
    node.add('kernel.adjoint', concurrent=True)

    # sum and smooth kernels
    node.add('kernel.postprocess', concurrent=True)
