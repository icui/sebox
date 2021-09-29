from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir
from .catalog import merge, scatter_obs, scatter_diff, scatter_baz
from .encoding import encode_obs, encode_diff
from .preprocess import prepare_frequencies, encode_events, link_observed
from .ft import ft

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
    ws.add('catalog', _catalog, concurrent=True)

    # mesher and preprocessing
    ws.pre = ws.add(_preprocess, concurrent=True, kls={})

    # kernel computation
    ws.add(_main, concurrent=True)


def _catalog(ws: Kernel):
    # prepare catalog (executed only once for a catalog)
    cdir = getdir()

    # merge stations into a single file
    if not cdir.has('SUPERSTATION'):
        ws.add(merge)
    
    #FIXME create_catalog (measurements.npy, weightings.npy, noise.npy, ft_obs, ft_diff)

    # convert observed traces into MPI format
    if not cdir.has(f'ft_obs_p{root.task_nprocs}'):
        ws.add(scatter_obs, concurrent=True)

    # convert differences between observed and synthetic data into MPI format
    if not cdir.has(f'ft_diff_p{root.task_nprocs}'):
        ws.add(scatter_diff, concurrent=True)
    
    # compute back-azimuth
    if not cdir.has(f'baz_p{root.task_nprocs}'):
        ws.add(scatter_baz, concurrent=True)


def _preprocess(ws: Kernel):
    # run mesher
    ws.add('mesh', ('module:solver', 'mesh'))

    for iker in range(ws.nkernels or 1):
        # create workspace for individual kernels
        kl = ws.add(f'kl_{iker:02d}', iker=iker)
        tp.cast(tp.Any, ws.kls)[iker] = kl

        # determine frequency range
        kl.add(prepare_frequencies, target=kl)

        if ws.path_encoded:
            # link encoded observed data
            kl.add(link_observed, target=kl)
        
        else:
            # encode events
            kl.add(encode_events, target=kl)

            # encode observed data
            enc = kl.add(_encode, concurrent=True)


def _main(ws: Kernel):
    cdir = getdir()

    for iker in range(ws.nkernels or 1):
        kl = ws.add(f'kl_{iker:02d}', inherit=tp.cast(tp.Any, ws.pre).kls[iker])

        # forward simulation
        kl.add('forward', ('module:solver', 'forward'),
            path_event= kl.path('SUPERSOURCE'),
            path_stations= cdir.path('SUPERSTATION'),
            path_mesh= ws.path('mesh'),
            monochromatic_source= True,
            save_forward= True)
        
        # process traces
        kl.add(ft, path_input=kl.path('forward/traces'), path_output=kl.path('ft_syn'), ft_event=None)


def _encode(ws: Kernel):
    ws.add('enc_obs', encode_obs)
    ws.add('enc_diff', encode_diff)
