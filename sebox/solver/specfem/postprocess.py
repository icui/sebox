from __future__ import annotations
import typing as tp
from functools import partial

from sebox.utils.adios import xsum, xmerge
from .shared import getsize

if tp.TYPE_CHECKING:
    from sebox.typing import Postprocess


def setup(node: Postprocess):
    """Create mesher node."""
    # link directories
    node.mkdir('smooth')
    node.ln(node.rel(node.path_mesh, 'bin'))
    node.ln(node.rel(node.path_mesh, 'DATA'))
    node.ln(node.rel(node.path_mesh, 'OUTPUT_FILES'))
    node.ln(node.rel(node.path_mesh, 'DATABASES_MPI'))

    # create text file with kernel paths
    node.write(f'{len(node.path_kernels)}\node', 'path.txt')

    for kl in node.path_kernels:
        node.write('1.0\node' + node.rel(kl, 'OUTPUT_FILES/kernels.bp') + '\node', 'path.txt', 'a')


def postprocess(node: Postprocess):
    """Sum and smooth kernels."""
    node.add(setup)
    xsum(node, node.source_mask)
    node.add(_smooth, concurrent=True)
    xmerge(node, node.precondition)


def _smooth(node: Postprocess):
    klen = node.smooth_kernels
    hlen = node.smooth_hessian

    if isinstance(klen, list):
        klen = max(klen[1], klen[0] * klen[2] ** (node.iteration or 0))

    if isinstance(hlen, list):
        hlen = max(hlen[1], hlen[0] * hlen[2] ** (node.iteration or 0))

    if klen:
        for kl in node.kernel_names:
            _xsmooth(node, kl, klen)
    
    if hlen:
        for kl in node.hessian_names:
            _xsmooth(node, kl, hlen)


def _xsmooth(node: Postprocess, kl: str, length: float):
    args = [
        f'{length} {length*(node.smooth_vertical or 1)}', kl,
        'kernels_masked.bp' if node.source_mask else 'kernels_raw.bp',
        'DATABASES_MPI/',
        f'smooth/kernels_smooth_{kl}_crust_mantle.bp',
        f'> OUTPUT_FILES/smooth_{kl}.txt'
    ]
    node.add_mpi('bin/xsmooth_laplacian_sem_adios ' + ' '.join(args),
        getsize, name=f'smooth_{kl}', data={'prober': partial(probe_smoother, kl)})


def probe_smoother(kl: str, node: Postprocess):
    """Prober of smoother progress."""
    if node.has(out := f'OUTPUT_FILES/smooth_{kl}.txt'):
        i = 0

        lines = node.readlines(out)
        niter = '0'

        for line in lines:
            if 'Initial residual:' in line:
                i += 1
            
            elif 'Iterations' in line:
                niter = line.split()[1]

        return f'{max(1, i)}/2 iter{niter}'
