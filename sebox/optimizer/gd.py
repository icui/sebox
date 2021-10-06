from __future__ import annotations
import typing as tp

from sebox.utils.adios import xgd

if tp.TYPE_CHECKING:
    from sebox.typing import Optimizer

def main(node: Optimizer):
    if len(node) == 0:
        node.add('optimizer.iterate', 'iter_00', iteration=0)


def iterate(node: Optimizer):
    """Add an iteration."""
    node.ln(node.rel(node.path_model), 'model_init.bp')

    # generate or link mesh
    node.add('solver.mesh', 'mesh')

    # compute kernels
    kl = node.add('kernel', 'kernel', path_mesh=node.path('mesh'))

    # compute direction
    node.add('optimizer.direction')

    # line search
    node.add('search', inherit_kernel=kl)

    # add new iteration
    node.add('optimizer.check')


def direction(node: Optimizer):
    """Compute direction."""
    xgd(node)


def check(node: Optimizer):
    """Add a new iteration if necessary."""
    optim = tp.cast('Optimizer', node.parent.parent)
    i = len(optim)

    if i < optim.niters and node.parent is optim[-1]:
        optim.add(iterate, f'iter_{i:02d}',
            iteration=i,
            path_model=optim.path(f'iter_{len(optim)-1:02d}/model_new.bp'),
            path_mesh=optim.path(f'iter_{len(optim)-1:02d}/mesh_new'))


def check_misfit():
    """Check misfit values."""
    from sebox import root

    root.restore()

    for i, optim in enumerate(root):
        if len(optim) < 2 or optim[1].misfit_value is None:
            continue

        print(f'Iteration {i}')
    
        steps = [0.0]
        vals = [optim[1].misfit_value]

        if len(optim):
            for step in optim[-2]:
                steps.append(step.step) # type: ignore
                vals.append(step[1].misfit_value)
        
        for step, val in zip(steps, vals):
            if val is None:
                break
            
            if optim.done and val == min(vals):
                print(f' {step:.3e}: {val:.3e} *')
            
            else:
                print(f' {step:.3e}: {val:.3e}')
        
        print()
