from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from .typing import Ortho


def postprocess(node: Ortho):
    """Sum misfit and process kernels."""
    # sum misfit
    kernel = node.parent
    kernel.misfit_value = sum([kl.misfit_kl for kl in kernel[2]])

    if not node.misfit_only:        
        # link kernels
        node.ln('postprocess/kernels.bp')
        node.ln('postprocess/precond.bp')

        # process kernels
        node.add('solver.postprocess', 'postprocess',
            path_kernels=[kl.path('adjoint') for kl in node.parent[1]][:node.nkernels],
            path_mesh= node.path('mesh'))
