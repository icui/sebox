from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir, getevents, getstations

if tp.TYPE_CHECKING:
    from .typing import Kernel


async def encode_obs(ws: Kernel):
    """Encode observed data."""
    stas = getstations()
    await ws.mpiexec(_encode_obs, root.task_nprocs, arg=ws, arg_mpi=stas)


def encode_diff(ws: Kernel):
    """Encode diff data."""


def _encode_obs(ws: Kernel, stas: tp.List[str]):
    import numpy as np
    from sebox.mpi import pid

    # cdir = getdir()
    encoded = np.zeros([len(stas), ws.imax - ws.imin])
    print(encoded.shape)

    # for event in getevents():
    #     data = cdir.load(f'ft_obs_p{root.task_nprocs}/{event}/{pid}.pickle')
