from __future__ import annotations
import typing as tp
from functools import partial

from sebox.utils.adios import xsum, xmerge, xprecond
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
    node.write(f'{len(node.path_kernels)}\n', 'path.txt')

    for kl in node.path_kernels:
        node.write('1.0\n' + node.rel(kl, 'OUTPUT_FILES/kernels.bp') + '\n', 'path.txt', 'a')


def postprocess(node: Postprocess):
    """Sum and smooth kernels."""
    node.add(setup)
    xsum(node, node.source_mask)
    node.add(_smooth, concurrent=True)
    xmerge(node, node.precondition)

    if node.save_vtu:
        xprecond(node)
        node.add(_vtu, concurrent=True)


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
    node.add_mpi('bin/xsmooth_laplacian_sem_adios ' + ' '.join([
        f'{length} {length*(node.smooth_vertical or 1)}', kl,
        'kernels_masked.bp' if node.source_mask else 'kernels_raw.bp',
        'DATABASES_MPI/',
        f'smooth/kernels_smooth_{kl}_crust_mantle.bp',
        f'{node.smooth_with_prem or 0}',
        f'> OUTPUT_FILES/smooth_{kl}.txt'
    ]), getsize, name=f'smooth_{kl}', data={'prober': partial(probe_smoother, kl)})


def _vtu(node: Postprocess):
    node.writelines(list(str(i) for i in range(getsize(node))), 'addressing.txt')
    for kl in tp.cast(list, node.save_vtu):    
        node.add_mpi('bin/xcombine_vol_data_vtu_adios ' + ' '.join([
            'addressing.txt',
            kl,
            'kernels_precond.bp',
            'DATABASES_MPI/solver_data.bp',
            '. 0 1'
        ]), 1, name=f'save_vtu_{kl}')


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
