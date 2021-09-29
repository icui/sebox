from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir
from .catalog import merge, scatter_obs, scatter_diff, scatter_baz
from .preprocess import prepare_encoding
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
    ws.add(_preprocess, concurrent=True)

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
    ws.encoding = {}
    
    for iker in range(ws.nkernels or 1):
        # create workspace for individual kernels
        ws.encoding[iker] = tp.cast('Kernel', ws.add(f'kl_{iker:02d}', prepare_encoding, iker=iker))

    # run mesher
    ws.add('mesh', ('module:solver', 'mesh'))


def _main(ws: Kernel):
    for iker in range(ws.nkernels or 1):
        # add steps to run forward and adjoint simulation
        ws.add(f'kl_{iker:02d}', _compute_kernel, inherit=tp.cast('Kernel', ws.parent).encoding[iker])


def _compute_kernel(ws: Kernel):
    cdir = getdir()

    # forward simulation
    ws.add('forward', ('module:solver', 'forward'),
        path_event= ws.path('SUPERSOURCE'),
        path_stations= cdir.path('SUPERSTATION'),
        path_mesh= ws.path('../mesh'),
        monochromatic_source= True,
        save_forward= True)
    
    # process traces
    ws.add(ft, ft_event=None,
        path_input=ws.path('forward/traces'),
        path_output=ws.path('ft_syn'))
