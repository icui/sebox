from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir
from .catalog import merge, scatter_obs, scatter_diff
from .encoding import encode_obs, encode_diff
from .preprocess import prepare_frequencies, encode_events, link_observed

if tp.TYPE_CHECKING:
    from .typing import Kernel


def kernel(ws: Kernel):
    """Compute kernels."""
    _compute(ws, False)


def misfit(ws: Kernel):
    """Compute misfit."""
    _compute(ws, True)


def _compute(ws: Kernel, misfit_only: bool):
    # prepare catalog (executed only once for a catalog)
    cdir = getdir()
    cat = ws.add('catalog', concurrent=True)

    # merge stations into a single file
    if not cdir.has('SUPERSTATION'):
        cat.add(merge)
    
    #FIXME create_catalog (measurements.npy, weightings.npy, noise.npy, ft_obs, ft_diff)

    # convert observed traces into MPI format
    if not cdir.has(f'ft_obs_p{root.task_nprocs}'):
        cat.add(scatter_obs, concurrent=True)

    # convert differences between observed and synthetic data into MPI format
    if not cdir.has(f'ft_diff_p{root.task_nprocs}'):
        cat.add(scatter_diff, concurrent=True)

    # mesher and preprocessing
    pre = ws.add('preprocess', concurrent=True, target=ws)

    # # run mesher
    # pre.add('mesh', ('module:solver', 'mesh'))

    for iker in range(ws.nkernels or 1):
        kl = pre.add(f'kl_{iker:02d}', iker=iker)

        # determine frequency range
        kl.add(prepare_frequencies, target=kl)

        if ws.path_encoded:
            # link encoded observed data
            kl.add(link_observed, target=kl)
        
        else:
            # encode events
            kl.add(encode_events, target=kl)

            # encode observed data
            enc = kl.add(concurrent=True)
            enc.add('enc_obs', encode_obs)
            enc.add('enc_diff', encode_diff)
