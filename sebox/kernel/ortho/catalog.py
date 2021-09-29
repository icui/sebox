from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir, getevents, getstations, merge_stations, locate_event, locate_station

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


async def scatter_baz(ws: Kernel):
    """Compute back azimuth."""
    await ws.mpiexec(_compute_baz, arg=ws, arg_mpi=getstations())


def _compute_baz(ws: Kernel, stas: tp.List[str]):
    import numpy as np
    from obspy.geodetics import gps2dist_azimuth
    from sebox.mpi import pid

    root.restore(ws)
    baz = {}

    for event in getevents():
        elat, elon = locate_event(event)
        baz[event] = np.zeros(len(stas))

        for i, sta in enumerate(stas):
            slat, slon = locate_station(sta)
            baz[event][i] = gps2dist_azimuth(elat, elon, slat, slon)[2]

    getdir().dump(baz, f'baz_p{root.task_nprocs}/{pid}.pickle')


def _scatter(ws: Kernel, tag: tp.Literal['obs', 'diff']):
    cdir = getdir()

    for src in cdir.ls(f'ft_{tag}'):
        event = src.split('.')[0]
        dst = f'ft_{tag}_p{root.task_nprocs}/{event}'
        
        if not cdir.has(dst):
            ws.add(event, ('sebox.utils.asdf', 'scatter'), aux=True, dtype=complex,
                path_bundle=cdir.path(f'ft_{tag}/{src}'), path_mpi=cdir.path(dst))
