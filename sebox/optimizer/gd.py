from __future__ import annotations
import typing as tp

from sebox.utils.adios import xgd

if tp.TYPE_CHECKING:
    from sebox.typing import Optimizer

def main(ws: Optimizer):
    if len(ws) == 0:
        ws.add(iterate, 'iter_00', iteration=0)


def iterate(ws: Optimizer):
    """Add an iteration."""
    # link model
    ws.ln(ws.rel(ws.path_model), 'model_init.bp')
    ws.path_model = ws.path('model_init.bp')

    # generate or link mesh
    ws.add('solver.mesh', 'mesh', path_mesh=ws.path_mesh)
    ws.path_mesh = ws.path('mesh')

    # compute kernels
    kl = ws.add('kernel', 'kernel')

    # compute direction
    ws.add('optimizer.direction')

    # line search
    ws.search = tp.cast(tp.Any, ws.add('search', inherit_kernel=kl))

    # add new iteration
    ws.add(_add)


def direction(ws: Optimizer):
    """Compute direction."""
    xgd(ws)


def check_misfit():
    """Check misfit values."""
    from sebox import root

    root.restore()

    for i, optim in enumerate(root._ws):
        if len(optim) < 2 or optim[1].misfit_value is None:
            continue

        print(f'Iteration {i}')
    
        steps = [0.0]
        vals = [optim[1].misfit_value]

        if optim.search:
            for step in optim.search._ws:
                steps.append(step.step)
                vals.append(step[1].misfit_value)
        
        for step, val in zip(steps, vals):
            print(f'  step {step}: {val:.2e}')
        
        print()


def _add(ws: Optimizer):
    optim = tp.cast('Optimizer', ws.parent.parent)

    if len(optim) < optim.niters:
        optim.add(iterate, f'iter_{len(optim):02d}',
            path_model=optim.path(f'iter_{len(optim)-1:02d}/model_new.bp'),
            path_mesh=optim.path(f'iter_{len(optim)-1:02d}/mesh_new'))
