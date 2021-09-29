from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from .typing import Kernel

from sebox.utils.catalog import getstations


async def diff(ws: Kernel):
    ws.mkdir('misfit')
    ws.mkdir('adjoint')
    await ws.mpiexec(_diff, arg=ws, arg_mpi=getstations())


async def _diff(ws: Kernel, stas: tp.List[str]):
    from sebox.mpi import pid

    obs = ws.load(f'../enc_obs/{pid}.pickle')
    diff = ws.load(f'../enc_diff/{pid}.pickle')
    syn = ws.load(f'enc_syn/{pid}.pickle')
