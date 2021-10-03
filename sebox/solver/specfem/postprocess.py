from __future__ import annotations
import typing as tp
from functools import partial

from sebox.utils.adios import xsum, xmerge
from .shared import getsize

if tp.TYPE_CHECKING:
    from sebox.typing import Postprocess


def setup(ws: Postprocess):
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
        ws.write('1.0\n' + ws.rel(kl, 'OUTPUT_FILES/kernels.bp') + '\n', 'path.txt', 'a')


def postprocess(ws: Postprocess):
    """Sum and smooth kernels."""
    ws.add(setup)
    xsum(ws, ws.source_mask)
    ws.add(_smooth, concurrent=True)
    xmerge(ws, ws.precondition)


def _smooth(ws: Postprocess):
    klen = ws.smooth_kernels
    hlen = ws.smooth_hessian

    if isinstance(klen, list):
        klen = max(klen[1], klen[0] * klen[2] ** (ws.iteration or 0))

    if isinstance(hlen, list):
        hlen = max(hlen[1], hlen[0] * hlen[2] ** (ws.iteration or 0))

    if klen:
        for kl in ws.kernel_names:
            _xsmooth(ws, kl, klen)
    
    if hlen:
        for kl in ws.hessian_names:
            _xsmooth(ws, kl, hlen)


def _xsmooth(ws: Postprocess, kl: str, length: float):
    args = [
        f'{length} {length*(ws.smooth_vertical or 1)}', kl,
        'kernels_masked.bp' if ws.source_mask else 'kernels_raw.bp',
        'DATABASES_MPI/',
        f'smooth/kernels_smooth_{kl}_crust_mantle.bp',
        f'> OUTPUT_FILES/smooth_{kl}.txt'
    ]
    ws.add_mpi('bin/xsmooth_laplacian_sem_adios ' + ' '.join(args),
        getsize, name=f'smooth_{kl}', data={'prober': partial(_probe, kl)})


def _probe(kl: str, ws: Postprocess):
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
