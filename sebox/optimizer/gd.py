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
    ws.ln(ws.rel(ws.path_model), 'model_init.bp')

    # compute kernels
    ws.add('kernel', 'kernel', path_model=ws.path('model_init.bp'))

    # compute direction
    ws.add('optimizer.direction')

    # line search
    ws.add('search')

    # add new iteration
    ws.add(_add)


def direction(ws: Optimizer):
    """Compute direction."""
    xgd(ws)


def _add(ws: Optimizer):
    optim = tp.cast('Optimizer', ws.parent.parent)

    if len(optim) < optim.niters:
        optim.add(iterate, f'iter_{len(optim):02d}',
            path_model=optim.path(f'iter_{len(optim)-1:02d}/model_new.bp'))
