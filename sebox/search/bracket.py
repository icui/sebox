from __future__ import annotations
import typing as tp

from sebox.utils.adios import xupdate

if tp.TYPE_CHECKING:
    from sebox.typing import Search


def main(ws: Search):
    """Perform line search"""
    ws.add(step, 'step_00', step=ws.step_init)


def step(ws: Search):
    """Perform a search step."""
    # update model
    ws.mkdir()
    xupdate(ws, ws.step)

    # compute misfit
    ws.add('kernel.misfit', path_model=ws.path('model_gll.bp'), path_mesh=None)

    # check bracket
    ws.add(_check, cwd='..', name='check_bracket')


def _check(ws: Search):
    import numpy as np

    search = tp.cast('Search', ws.parent.parent)
    steps = [0.0]
    vals = [ws.inherit_kernel.misfit_value]

    for step in search._ws:
        steps.append(step.step) # type: ignore
        vals.append(step[1].misfit_value) # type: ignore
    
    x = np.array(steps)
    f = np.array(vals)
    f = f[abs(x).argsort()]
    x = x[abs(x).argsort()]

    alpha = None

    if check_bracket(f):
        if good_enough(x, f):
            step_optim = x[f.argmin()]

            for j, s in enumerate(steps):
                if np.isclose(step_optim, s):
                    ws.ln(f'step_{j-1:02d}/model_gll.bp', 'model_new.bp')
                    return
            
        alpha = polyfit(x,f)
        
    elif len(steps) - 1 < ws.nsteps:
        if all(f <= f[0]):
            alpha = 1.618034 * x[-1]
        
        else:
            alpha = x[1] / 1.618034
    
    if alpha:
        search.add(step_optim, f'step_{len(steps)-1:02d}')
    
    else:
        print('line search failed')
        ws.ln('model_init.bp', 'model_new.bp')


def check_bracket(f):
    """Check history has bracket shape."""
    imin, fmin = f.argmin(), f.min()

    return (fmin < f[0]) and any(f[imin:] > fmin)

def good_enough(x, f):
    """Check current result is good."""
    import numpy as np

    if not check_bracket(f):
        return 0

    x0 = polyfit(x, f)

    return any(np.abs(np.log10(x[1:]/x0)) < np.log10(1.2))

def polyfit(x, f):
    import numpy as np

    i = np.argmin(f)
    p = np.polyfit(x[i-1:i+2], f[i-1:i+2], 2)
    
    if p[0] > 0:
        return -p[1]/(2*p[0])
    
    raise RuntimeError('polyfit failed')
