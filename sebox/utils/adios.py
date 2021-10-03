import typing as tp

from sebox import Workspace
from sebox.utils.catalog import getdir
from sebox.solver.specfem.specfem import getsize


def _check(out: str):
    if 'ERROR' in out:
        raise RuntimeError('adios execution failed')


def _adios(ws: Workspace, cmd: str):
    ws.add_mpi(ws.rel(tp.cast(str, ws.path_adios), 'bin', cmd), getsize, check_output=_check)


def xsum(ws: Workspace):
    _adios(ws, f'xsum_kernels path.txt kernels_raw.bp')


def xmask(ws: Workspace):
    _adios(ws, f'xsrc_mask kernels_raw.bp {getdir().path("source_mask")} kernels_masked.bp')


def xmerge(ws: Workspace):
    _adios(ws, f'xmerge_kernels smooth kernels.bp')
