from __future__ import annotations
import typing as tp

from sebox.utils.adios import xupdate

if tp.TYPE_CHECKING:
    from sebox.typing import Search


def main(node: Search):
    """Perform line search"""
    node.dump(node.step_init, 'step_00/step.pickle')
    node.add('search.step', 'step_00')


def step(node: Search):
    """Perform a search step."""
    # update model
    node.add('search.update')

    # compute misfit
    node.add('kernel', path_mesh=None, path_model=node.path('model_gll.bp'), misfit_only=True)

    # check bracket
    node.add('search.check')


def update(node: Search):
    """Update model and mesh."""
    kl = node.inherit_kernel
    step = node.load('step.pickle')
    xupdate(node, step, node.rel(kl.path_model), node.rel(kl.path_mesh))


def check(node: Search):
    """Check bracket and add new step if necessary."""
    import numpy as np

    search = node.parent.parent
    x, f, steps = hist(search)

    # new step length
    alpha = None

    if _check_bracket(f):
        # search history has bracket shape
        if _good_enough(x, f):
            # find the step that minimizes misfit value
            st = x[f.argmin()]

            for j, s in enumerate(steps):
                if np.isclose(st, s):
                    # index is j-1 because the first step is 0.0
                    search.rm('mesh_new')
                    search.ln(f'step_{j-1:02d}/mesh', 'mesh_new')
                    search.ln(f'step_{j-1:02d}/model_gll.bp', 'model_new.bp')
                    search.dump(j - 1, 'step_final.pickle')
                    return
            
        alpha = _polyfit(x,f)
        
    elif len(steps) - 1 < search.nsteps:
        # history is monochromatically increasing or decreasing
        if all(f <= f[0]):
            alpha = 1.618034 * x[-1]
        
        else:
            alpha = x[1] / 1.618034
    
    if alpha:
        # add a new search step
        search.add(step, f'step_{len(steps)-1:02d}', step=alpha)
    
    else:
        # use initial model as new model
        print('line search failed')
        search.rm('mesh_new')
        search.ln('mesh', 'mesh_new')
        search.ln('model_init.bp', 'model_new.bp')


def hist(node: Search):
    """Get search history."""
    import numpy as np

    # seach step lengths and misfit values
    steps = [0.0]
    vals = [node.inherit_kernel.load('misfit.npy').sum()]

    for i in range(node.nsteps):
        cwd = f'step_{i:02d}'
        
        if node.has(f'{cwd}/step.pickle') and node.has(f'{cwd}/misfit.npy'):
            steps.append(node.load(f'{cwd}/step.pickle'))
            vals.append(node.load(f'{cwd}/misfit.npy').sum())
        
        else:
            break
    
    # sort by step length
    x = np.array(steps)
    f = np.array(vals)
    f = f[abs(x).argsort()]
    x = x[abs(x).argsort()]

    return x, f, steps


def _check_bracket(f):
    """Check history has bracket shape."""
    imin, fmin = f.argmin(), f.min()

    return (fmin < f[0]) and any(f[imin:] > fmin)


def _good_enough(x, f):
    """Check current result is good."""
    import numpy as np

    if not _check_bracket(f):
        return 0

    x0 = _polyfit(x, f)

    return any(np.abs(np.log10(x[1:] / x0)) < np.log10(1.2))

def _polyfit(x, f):
    import numpy as np

    i = np.argmin(f)
    p = np.polyfit(x[i-1:i+2], f[i-1:i+2], 2)
    
    if p[0] > 0:
        return -p[1]/(2*p[0])
    
    raise RuntimeError('polyfit failed')
