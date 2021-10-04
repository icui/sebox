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
    ws.misfit_only = False
    _compute(ws)


def misfit(ws: Kernel):
    """Compute misfit."""
    ws.misfit_only = True
    _compute(ws)


def _compute(ws: Kernel):
    ws.encoding = {}
    ws.solvers = {}

    # prepare catalog (executed only once for a catalog)
    ws.add(_catalog, 'catalog', concurrent=True)

    # mesher and preprocessing
    ws.add(_preprocess, concurrent=True)

    # kernel computation
    ws.add(_main, concurrent=True)

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
        ws.encoding[iker] = tp.cast('Kernel', ws.add(prepare_encoding, f'kl_{iker:02d}', iker=iker))

    # run mesher
    ws.add('solver.mesh', 'mesh')


def _main(ws: Kernel):
    for iker in range(ws.nkernels or 1):
        # add steps to run forward and adjoint simulation
        ws.solvers[iker] = tp.cast('Kernel', ws.add(compute_kernel, f'kl_{iker:02d}', inherit=ws.encoding[iker]))


def _postprocess(ws: Kernel):
    # sum misfit
    ws.add(_sum_misfit)

    if not ws.misfit_only:
        # process kernels
        ws.add('solver.postprocess', 'postprocess',
            path_kernels=[kl.path('adjoint') for kl in ws.encoding.values()],
            path_mesh= ws.path('mesh'))
        
        ws.add(_link_kernels)


def _link_kernels(ws: Kernel):
    ws.ln('postprocess/kernels.bp')
    ws.ln('postprocess/precond.bp')


def _sum_misfit(ws: Kernel):
    print('#@', ws.parent.parent)
    ws.parent.parent.misfit_value = sum([kl.misfit_kl for kl in ws.solvers.values()])
