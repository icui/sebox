from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from .typing import Kernel

from sebox.utils.catalog import getstations


async def diff(ws: Kernel):
    ws.mkdir('misfit')
    ws.mkdir('adjoint')
    await ws.mpiexec(_diff, arg=ws, arg_mpi=getstations())
    exit()


async def _diff(ws: Kernel, stas: tp.List[str]):
    import numpy as np
    from sebox.mpi import pid
    from mpi4py.MPI import COMM_WORLD as comm

    if 'II.OBN' in stas:
        print(pid, stas.index('II.OBN'))

    # read data
    syn = ws.load(f'enc_syn/{pid}.npy')
    obs = ws.load(f'enc_obs/{pid}.npy')
    ref = ws.load(f'enc_diff/{pid}.npy')
    weight = ws.load(f'enc_weight/{pid}.npy')

    # compute diff
    phase_diff = np.angle(syn / obs) * weight
    amp_diff = np.abs(syn) / np.abs(obs) * weight

    # unwrap or clip phases
    shape = phase_diff.shape
    nsta = len(getstations())
    phase_gather = np.zeros([nsta, shape[1], shape[2]], dtype=phase_diff.dtype)
    comm.allgather(phase_diff, phase_gather)

    if 'II.OBN' in stas:
        print('@', phase_diff.shape)
        print('@', phase_gather.shape)

