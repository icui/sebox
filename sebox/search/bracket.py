from __future__ import annotations
import typing as tp

from sebox.utils.adios import xupdate

if tp.TYPE_CHECKING:
    from sebox.typing import Search


def main(ws: Search):
    """Perform line search"""
    ws.add(step, 'step_00', step=ws.step_init)


def step(ws: Search):
    # update model
    xupdate(ws, ws.step)

    # compute misfit
    ws.add('kernel.misfit', path_model=ws.path('model_gll.bp'), path_mesh=None)

    # check bracket
    ws.add(_check)


def _check(ws: Search):
    search = tp.cast('Search', ws.parent.parent)
    x = [0.0]
    f = [ws.inherit_kernel.misfit_value]

    for step in search._ws:
        x.append(step.step) # type: ignore
        f.append(step[1].misfit_value) # type: ignore
    
    print(x, f)
    exit()

    
