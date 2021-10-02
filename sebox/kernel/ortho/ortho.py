from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir, getstations
from .catalog import merge, scatter_obs, scatter_diff, scatter_baz
from .preprocess import prepare_encoding
from .ft import ft
from .diff import diff, gather

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
    ws.add(_main, concurrent=True, misfit_only=misfit_only)

    # sum and smooth kernels
    ws.add('postprocess', ('module:solver', 'postprocess'),
        path_kernels=[ws.path(f'kl_{iker:02d}/adjoint/kernels.bp') for iker in range(ws.nkernels or 1)],
        path_mesh= ws.path('mesh'))


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
        ws.add_mpi(scatter_baz, arg=ws, arg_mpi=getstations())


def _preprocess(ws: Kernel):
    ws.parent.encoding = {}
    
    for iker in range(ws.nkernels or 1):
        # create workspace for individual kernels
        ws.parent.encoding[iker] = tp.cast('Kernel', ws.add(f'kl_{iker:02d}', prepare_encoding, iker=iker))

    # run mesher
    ws.add('mesh', ('module:solver', 'mesh'))


def _main(ws: Kernel):
    for iker in range(ws.nkernels or 1):
        # add steps to run forward and adjoint simulation
        ws.add(f'kl_{iker:02d}', _compute_kernel, inherit=tp.cast('Kernel', ws.parent).encoding[iker])


def _compute_kernel(ws: Kernel):
    # forward simulation
    ws.add('forward', ('module:solver', 'forward'),
        path_event= ws.path('SUPERSOURCE'),
        path_stations= getdir().path('SUPERSTATION'),
        path_mesh= ws.path('../mesh'),
        monochromatic_source= True,
        save_forward= True)
    
    # compute misfit
    ws.add(_compute_misfit)

    # forward simulation
    ws.add('adjoint', ('module:solver', 'adjoint'),
        path_forward = ws.path('forward'),
        path_misfit = ws.path('adjoint.h5'))


def _compute_misfit(ws: Kernel):
    stas = getstations()

    ws.mkdir('enc_syn')
    ws.mkdir('enc_mf')
    ws.mkdir('adstf')

    # process traces
    ws.add_mpi(ft, arg=ws, arg_mpi=stas)
    
    # compute misfit
    ws.add_mpi(diff, arg=ws, arg_mpi=stas)

    # convert adjoint sources to ASDF format
    ws.add(gather)
