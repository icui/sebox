from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir, getevents, getstations, merge_stations

if tp.TYPE_CHECKING:
    from .typing import Kernel


def merge(_):
    cdir = getdir()
    merge_stations(cdir.subdir('stations'), cdir, True)


def scatter_obs(ws: Kernel):
    """Convert ASDF observed data to MPI format."""
    _scatter(ws, 'obs')


def scatter_diff(ws: Kernel):
    """Convert ASDF diff data to MPI format."""
    _scatter(ws, 'diff')


def _scatter(ws: Kernel, tag: tp.Literal['obs', 'diff']):
    cdir = getdir()

    for src in cdir.ls(f'ft_{tag}'):
        event = src.split('.')[0]
        dst = f'ft_{tag}_p{root.task_nprocs}/{event}'
        
        if not cdir.has(dst):
            ws.add(event, ('sebox.utils.asdf', 'scatter'), aux=True, dtype=complex,
                path_bundle=cdir.path(f'ft_{tag}/{src}'), path_mpi=cdir.path(dst))
