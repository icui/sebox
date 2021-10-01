import typing as tp

from sebox import Workspace
from sebox.solver.specfem.specfem import getsize


async def _adios(ws: Workspace, cmd: str):
    name = await ws.mpiexec(ws.rel(tp.cast(str, ws.path_adios), 'bin', cmd + ' > '), getsize(ws))

    if 'ERROR' in ws.read(name):
        raise RuntimeError('adios execution failed')


async def xsum(ws: Workspace):
    await _adios(ws, f'xsum_kernels path.txt kernels.bp')


async def xmerge(ws: Workspace):
    await _adios(ws, f'xmerge_kernels smooth kernels_smooth.bp')
