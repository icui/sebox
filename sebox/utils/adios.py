import typing as tp

from sebox import Node
from sebox.utils.catalog import getdir


def _check(out: str):
    if 'ERROR' in out or 'MPI_ABORT' in out:
        raise RuntimeError('adios execution failed')


def _adios(node: Node, cmd: str):
    from sebox.solver.specfem.shared import getsize
    node.add_mpi(node.rel(tp.cast(str, node.path_adios), 'bin', cmd), getsize, check_output=_check)


def xsum(node: Node, mask: bool):
    """Sum and mask kernels."""
    _adios(node, f'xsum_kernels path.txt kernels_raw.bp')

    if mask:
        _adios(node, f'xsrc_mask kernels_raw.bp {getdir().path("source_mask")} kernels_masked.bp')


def xmerge(node: Node, precond: float):
    """Merge smoothed kernels and create preconditioner."""
    _adios(node, 'xmerge_kernels smooth kernels.bp')
    _adios(node, 'xcompute_vp_vs_hess kernels.bp DATABASES_MPI/solver_data.bp hess.bp')
    _adios(node, f'xprepare_vp_vs_precond hess.bp precond.bp {precond}')


def xprecond(node: Node):
    """Merge smoothed kernels and create preconditioner."""
    _adios(node, 'xprecond_kernels kernels.bp precond.bp kernels_precond.bp')


def xgd(node: Node):
    """Compute gradient descent direction."""
    _adios(node, 'xsteepDescent kernels.bp precond.bp direction.bp')


def xcg(node: Node):
    """Compute conjugate gradient direction."""
    dir0 = f'iter_{tp.cast(int, node.iteration)-1:02d}'
    _adios(node, f'xcg_direction ../{dir0}/kernels.bp ../{dir0}/precond.bp kernels.bp precond.bp ' +
        f'../{dir0}/direction.bp mesh/DATABASES_MPI/solver_data.bp direction.bp')


def xmm(node: Node):
    """Compute conjugate gradient direction."""
    dir0 = f'iter_{tp.cast(int, node.iteration)-1:02d}'
    _adios(node, f'xmm_direction {node.momentum} kernels.bp precond.bp ' +
        f'../{dir0}/direction.bp mesh/DATABASES_MPI/solver_data.bp direction.bp')


# def xlbfgs(node: Node):
#     """Compute L-BFGS direction."""
#     iter_min = max(node.iteration_start, node.iteration-node.mem) # type: ignore
#     lines = [str(node.iteration - iter_min)]

#     for i in range(iter_min, node.iteration): # type: ignore
#         lines.append(f'../iter_{i:02d}/kernels.bp')
#         lines.append(f'../iter_{i:02d}/direction.bp')
    
#     lines.append('kernels.bp')
#     node.write('\n'.join(lines), 'lbfgs.txt')

#     _adios(node, f'xlbfgs lbfgs.txt mesh/DATABASES_MPI/solver_data.bp direction.bp')


def xlbfgs(node: Node):
    """Compute L-BFGS direction."""
    iter_min = max(node.iteration_start, node.iteration-node.mem) # type: ignore
    lines = [str(node.iteration - iter_min)]

    for i in range(iter_min, node.iteration): # type: ignore
        lines.append(f'../iter_{i:02d}/postprocess/kernels.bp')
        step = node.load(f'../iter_{i:02d}/step_final.pickle')
        lines.append(f'../iter_{i:02d}/step_{step:02d}/dkernels.bp')
    
    lines.append('kernels.bp')
    lines.append('precond.bp')
    node.write('\n'.join(lines), 'lbfgs.txt')

    _adios(node, f'xlbfgs lbfgs.txt mesh/DATABASES_MPI/solver_data.bp direction.bp')


def xupdate(node: Node, step: float, path_model: str, path_mesh: str):
    """Update model."""
    _adios(node, f'xupdate_model {step} {path_model} {path_mesh}/DATABASES_MPI/solver_data.bp ../direction.bp .')
