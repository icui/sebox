from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from .typing import Ortho


def main(node: Ortho):
    """Compute kernels."""
    _main(node, False)


def misfit(node: Ortho):
    """Compute misfit."""
    _main(node, True)


def _main(node: Ortho, misfit_only: bool):
    # prepare catalog (executed only once for a catalog)
    node.add('kernel.catalog', 'catalog', concurrent=True)

    # mesher and preprocessing
    node.add('kernel.preprocess', concurrent=True, misfit_only=misfit_only)

    # kernel computation
    node.add('kernel.kernel', concurrent=True, misfit_only=misfit_only)

    # sum and smooth kernels
    node.add('kernel.postprocess', misfit_only=misfit_only)
