from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir, getstations
from .catalog import merge, scatter_obs, scatter_diff, scatter_baz
from .preprocess import prepare_encoding
from .kernel import compute_kernel

if tp.TYPE_CHECKING:
    from .typing import Kernel


def main(ws: Kernel):
    """Compute kernels."""
    ws.encoding = {}
    _compute(ws, False)


def misfit(ws: Kernel):
    """Compute misfit."""
    ws.encoding = tp.cast(Kernel, ws.inherit_kernel).encoding
    _compute(ws, True)


def _compute(ws: Kernel, misfit_only: bool):
    # prepare catalog (executed only once for a catalog)
    ws.add(_catalog, 'catalog', concurrent=True)

    # mesher and preprocessing
    ws.add(_preprocess, concurrent=True)

    # kernel computation
    ws.add(_main, concurrent=True, misfit_only=misfit_only)

    # sum and smooth kernels
    ws.add(_postprocess)


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
    for iker in range(ws.nkernels or 1):
        # create workspace for individual kernels
        kl = tp.cast('Kernel', ws.add(prepare_encoding, f'kl_{iker:02d}', iker=iker))

        if not ws.inherit_kernel:
            ws.encoding[iker] = kl

    #FIXME # run mesher
    # ws.add('mesh', ('module:solver', 'mesh'))


def _main(ws: Kernel):
    for iker in range(ws.nkernels or 1):
        # add steps to run forward and adjoint simulation
        ws.add(compute_kernel, f'kl_{iker:02d}', inherit=ws.encoding[iker])


def _postprocess(ws: Kernel):
    if not ws.misfit_only:
        ws.add('solver.postprocess', 'postprocess',
            path_kernels=[kl.path('adjoint') for kl in ws.encoding.values()],
            path_mesh= ws.path('mesh'))
        
        ws.add(_link_kernels)


def _link_kernels(ws: Kernel):
    ws.ln('postprocess/kernels.bp')
    ws.ln('postprocess/precond.bp')
