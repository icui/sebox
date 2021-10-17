from __future__ import annotations
import typing as tp

from .main import dirs

if tp.TYPE_CHECKING:
    from .typing import Ortho


def postprocess(node: Ortho):
    """Sum misfit and process kernels."""
    node.concurrent = True
    
    # sum misfit
    mf = 0.0
    for cwd in dirs(node):
        mf += node.load(f'{cwd}/phase_mf.npy').sum()

        if node.amplitude_factor > 0:
            mf += node.load(f'{cwd}/amp_mf.npy').sum()

    node.parent.misfit_value = mf

    if not node.misfit_only:        
        # link kernels
        node.ln('postprocess/kernels.bp')
        node.ln('postprocess/precond.bp')

        # process kernels
        node.add('solver.postprocess', 'postprocess',
            path_kernels=[node.path(kl, 'adjoint') for kl in dirs(node)],
            path_mesh= node.path('mesh'))
