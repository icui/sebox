from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir, getevents, getstations, merge_stations

if tp.TYPE_CHECKING:
    from .typing import Kernel


async def merge(_):
    cdir = getdir()
    merge_stations(cdir.subdir('stations'), cdir, True)


async def scatter_obs(ws: Kernel):
    cdir = getdir()

    for src in cdir.ls('ft_obs'):
        event = src.split('.')[0]
        dst = f'ft_obs_p{root.task_nprocs}/{event}'
        
        if not cdir.has(dst):
            ws.add(event, ('sebox.utils.asdf', 'scatter'),
                aux=True, path_bundle=cdir.path(f'ft_obs/{src}'), path_mpi=cdir.path(dst))


async def scatter_diff(_):
    cdir = getdir()
