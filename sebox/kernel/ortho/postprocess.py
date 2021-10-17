from __future__ import annotations
import typing as tp

from .main import dirs

if tp.TYPE_CHECKING:
    from .typing import Ortho


def postprocess(node: Ortho):
    """Sum misfit and process kernels."""
    node.concurrent = True
    
    # sum misfit
    node.add(_sum_misfit)

    if not node.misfit_only:        
        # link kernels
        node.ln('postprocess/kernels.bp')
        node.ln('postprocess/precond.bp')

        # process kernels
        node.add('solver.postprocess', 'postprocess', 'sum_smooth_precond',
            path_kernels=[node.path(kl, 'adjoint') for kl in dirs(node)],
            path_mesh= node.path('mesh'))

def _sum_misfit(node: Ortho):
    mf = None

    for cwd in dirs(node):
        if mf is None:
            mf = node.load(f'{cwd}/phase_mf.npy')

        else:
            mf += node.load(f'{cwd}/phase_mf.npy')

        if node.amplitude_factor > 0:
            mf += node.load(f'{cwd}/amp_mf.npy')

    node.dump(mf, 'misfit.npy')
