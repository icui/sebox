from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir, getstations
from .catalog import merge, scatter_obs, scatter_diff, scatter_baz
from .preprocess import prepare_encoding
from .ft import ft, diff, gather

if tp.TYPE_CHECKING:
    from .typing import Kernel


def main(node: Kernel):
    """Compute kernels."""
    node.misfit_only = False
    _main(node)


def misfit(node: Kernel):
    """Compute misfit."""
    node.misfit_only = True
    _main(node)


def _main(node: Kernel):
    node.root_kernel = node

    # prepare catalog (executed only once for a catalog)
    node.add(_catalog, 'catalog', concurrent=True)

    # mesher and preprocessing
    node.add(_preprocess, concurrent=True)

    # kernel computation
    node.add(_kernel, concurrent=True)

    # sum and smooth kernels
    node.add(_postprocess)


def _catalog(node: Kernel):
    # prepare catalog (executed only once for a catalog)
    cdir = getdir()

    # merge stations into a single file
    if not cdir.has('SUPERSTATION'):
        node.add(merge)
    
    #FIXME create_catalog (measurements.npy, weightings.npy, noise.npy, ft_obs, ft_diff)

    # convert observed traces into MPI format
    if not cdir.has(f'ft_obs_p{root.task_nprocs}'):
        node.add(scatter_obs, concurrent=True)

    # convert differences between observed and synthetic data into MPI format
    if not cdir.has(f'ft_diff_p{root.task_nprocs}'):
        node.add(scatter_diff, concurrent=True)
    
    # compute back-azimuth
    if not cdir.has(f'baz_p{root.task_nprocs}'):
        node.add_mpi(scatter_baz, arg=node, arg_mpi=getstations())


def _preprocess(node: Kernel):
    for iker in range(node.nkernels or 1):
        # create node for individual kernels
        node.add(prepare_encoding, f'kl_{iker:02d}', iker=iker)

    # run mesher
    node.add('solver.mesh', 'mesh')


def _kernel(node: Kernel):
    for iker in range(node.nkernels or 1):
        # add steps to run forward and adjoint simulation
        node.add(_compute_kernel, f'kl_{iker:02d}', inherit=node.parent[1][iker])


def _postprocess(node: Kernel):
    # sum misfit
    node.add(_sum_misfit)

    if not node.misfit_only:
        # process kernels
        node.add('solver.postprocess', 'postprocess',
            path_kernels=[kl.path('adjoint') for kl in node.parent[1]][:node.nkernels],
            path_mesh= node.path('mesh'))
        
        node.add(_link_kernels)


def _compute_kernel(node: Kernel):
    # forward simulation
    node.add('solver', 'forward',
        path_event= node.path('SUPERSOURCE'),
        path_stations= getdir().path('SUPERSTATION'),
        path_mesh= node.path('../mesh'),
        monochromatic_source= True,
        save_forward= True)
    
    # compute misfit
    node.add(_compute_misfit)

    # adjoint simulation
    if not node.misfit_only:
        node.add('solver.adjoint', 'adjoint',
            path_forward = node.path('forward'),
            path_misfit = node.path('adjoint.h5'))


def _compute_misfit(node: Kernel):
    stas = getstations()

    # process traces
    node.add_mpi(ft, arg=node, arg_mpi=stas, cwd='enc_syn')
    
    # compute misfit
    node.add_mpi(diff, arg=node, arg_mpi=stas, cwd='enc_mf')

    # convert adjoint sources to ASDF format
    node.add(gather)


def _link_kernels(node: Kernel):
    node.ln('postprocess/kernels.bp')
    node.ln('postprocess/precond.bp')


def _sum_misfit(node: Kernel):
    kernel = node.parent.parent
    kernel.misfit_value = sum([kl.misfit_kl for kl in kernel[2]])
