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
    _adios(node, f'xprepare_vp_vs_precond hess.bp precond.bp {1/precond}')


def xgd(node: Node):
    """Compute gradient descent direction."""
    _adios(node, 'xsteepDescent kernels.bp precond.bp direction.bp')


def xupdate(node: Node, step: float, path_model: str, path_mesh: str):
    """Update model."""
    _adios(node, f'xupdate_model {step} {path_model} {path_mesh}/DATABASES_MPI/solver_data.bp ../direction.bp .')
