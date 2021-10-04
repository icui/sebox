from __future__ import annotations
import typing as tp

from sebox.utils.adios import xupdate

if tp.TYPE_CHECKING:
    from sebox.typing import Search


def main(node: Search):
    """Perform line search"""
    node.add(step, 'step_00', step=node.step_init)


def step(node: Search):
    """Perform a search step."""
    # update model
    node.mkdir()
    xupdate(node, node.step)

    # compute misfit
    node.add('kernel.misfit', path_model=node.path('model_gll.bp'), path_mesh=None)

    # check bracket
    node.add(_check, cwd='..', name='check_bracket')


def _check(node: Search):
    import numpy as np

    search = tp.cast('Search', node.parent.parent)
    steps = [0.0]
    vals = [node.inherit_kernel.misfit_value]

    for st in search:
        steps.append(st.step) # type: ignore
        vals.append(st[1].misfit_value) # type: ignore
    
    x = np.array(steps)
    f = np.array(vals)
    f = f[abs(x).argsort()]
    x = x[abs(x).argsort()]

    alpha = None

    if check_bracket(f):
        if good_enough(x, f):
            st = x[f.argmin()]

            for j, s in enumerate(steps):
                if np.isclose(st, s):
                    node.ln(f'step_{j-1:02d}/model_gll.bp', 'model_new.bp')
                    node.ln(f'step_{j-1:02d}/mesh', 'mesh_new')
                    return
            
        alpha = polyfit(x,f)
        
    elif len(steps) - 1 < node.nsteps:
        if all(f <= f[0]):
            alpha = 1.618034 * x[-1]
        
        else:
            alpha = x[1] / 1.618034
    
    if alpha:
        search.add(step, f'step_{len(steps)-1:02d}', step=alpha)
    
    else:
        print('line search failed')
        node.ln('model_init.bp', 'model_new.bp')


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
