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
        dst = f'ft_obs_p{root.task_nprocs}/{src}'
        
        if not cdir.has(dst):
            ws.add(('sebox.utils.asdf', 'scatter'), aux=True)


async def scatter_diff(_):
    cdir = getdir()
