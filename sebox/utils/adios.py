import typing as tp

from sebox import Workspace
from sebox.utils.catalog import getdir
from sebox.solver.specfem.shared import getsize


def _check(out: str):
    if 'ERROR' in out:
        raise RuntimeError('adios execution failed')


def _adios(ws: Workspace, cmd: str):
    ws.add_mpi(ws.rel(tp.cast(str, ws.path_adios), 'bin', cmd), getsize, check_output=_check)


def xsum(ws: Workspace, mask: bool):
    """Sum and mask kernels."""
    _adios(ws, f'xsum_kernels path.txt kernels_raw.bp')

    if mask:
        _adios(ws, f'xsrc_mask kernels_raw.bp {getdir().path("source_mask")} kernels_masked.bp')


def xmerge(ws: Workspace, precond: float):
    """Merge smoothed kernels and create preconditioner."""
    _adios(ws, 'xmerge_kernels smooth kernels.bp')
    _adios(ws, 'xcompute_vp_vs_hess kernels.bp DATABASES_MPI/solver_data.bp hess.bp')
    _adios(ws, f'xprepare_vp_vs_precond hess.bp precond.bp {1/precond}')


def xgd(ws: Workspace):
    _adios(ws, 'xsteepDescent kernel/kernels.bp kernel/precond.bp direction.bp')
