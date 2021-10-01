from __future__ import annotations
import typing as tp

from sebox.utils.adios import xsum, xmerge
from .specfem import getsize

if tp.TYPE_CHECKING:
    from sebox.typing import Sum


def setup(ws: Sum):
    """Create mesher workspace."""
    # link directories
    ws.mkdir('smooth')
    ws.ln(ws.rel(ws.path_mesh, 'bin'))
    ws.ln(ws.rel(ws.path_mesh, 'DATA'))
    ws.ln(ws.rel(ws.path_mesh, 'OUTPUT_FILES'))
    ws.ln(ws.rel(ws.path_mesh, 'DATABASES_MPI'))

    # create text file with kernel paths
    ws.write(f'{len(ws.path_kernels)}\n', 'path.txt')

    for kl in ws.path_kernels:
        ws.write('1.0\n' + ws.rel(kl) + '\n', 'path.txt', 'a')


def postprocess(ws: Sum):
    """Generate mesh."""
    ws.add(setup)
    ws.add(xsum)
    ws.add(_smooth, concurrent=True)
    ws.add(xmerge)


def _smooth(ws: Sum):
    from functools import partial

    for kl in ws.kernel_names:
        ws.add(partial(_xsmooth, kl, False), prober=partial(probe_smoother, kl))
    
    for kl in ws.hessian_names:
        ws.add(partial(_xsmooth, kl, True), prober=partial(probe_smoother, kl))


async def _xsmooth(kl: str, hess: bool, ws: Sum):
    rad = ws.smooth_hessian if hess else ws.smooth_kernels

    if isinstance(rad, list):
        rad = max(rad[1], rad[0] * rad[2] ** (ws.iteration or 0))

    if rad:
        await ws.mpiexec(f'bin/xsmooth_laplacian_sem_adios  {rad} {rad*(ws.smooth_vertical or 1)} {kl} kernels.bp DATABASES_MPI/ smooth/kernels_smooth_{kl}_crust_mantle.bp > OUTPUT_FILES/smooth_{kl}.txt', getsize(ws))


def probe_smoother(kl: str, ws: Sum):
    """Prober of smoother progress."""
    if ws.has(out := f'OUTPUT_FILES/smooth_{kl}.txt'):
        n = 0

        lines = ws.readlines(out)
        niter = '0'

        for line in lines:
            if 'Initial residual:' in line:
                n += 1
            
            elif 'Iterations' in line:
                niter = line.split()[1]
        
        n = max(1, n)

        return f'{n}/2 iter{niter}'
