from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir, getevents, getstations, merge_stations, locate_event, locate_station

if tp.TYPE_CHECKING:
    from .typing import Kernel


def merge(_):
    cdir = getdir()
    merge_stations(cdir.subdir('stations'), cdir, True)


def scatter_obs(node: Kernel):
    """Convert ASDF observed data to MPI format."""
    _scatter(node, 'obs')


def scatter_diff(node: Kernel):
    """Convert ASDF diff data to MPI format."""
    _scatter(node, 'diff')


def scatter_baz(node: Kernel, stas: tp.List[str]):
    import numpy as np
    from math import radians
    from obspy.geodetics import gps2dist_azimuth
    from sebox.mpi import pid

    root.restore(node)
    baz = {}

    for event in getevents():
        elat, elon = locate_event(event)
        baz[event] = np.zeros(len(stas))

        for i, sta in enumerate(stas):
            slat, slon = locate_station(sta)
            baz[event][i] = radians(gps2dist_azimuth(elat, elon, slat, slon)[2])

    getdir().dump(baz, f'baz_p{root.task_nprocs}/{pid}.pickle')


def _scatter(node: Kernel, tag: tp.Literal['obs', 'diff']):
    cdir = getdir()

    for src in cdir.ls(f'ft_{tag}'):
        event = src.split('.')[0]
        dst = f'ft_{tag}_p{root.task_nprocs}/{event}'
        
        if not cdir.has(dst):
            node.add(('sebox.utils.asdf', 'scatter'), event, aux=True, dtype=complex,
                path_input=cdir.path(f'ft_{tag}/{src}'), path_output=cdir.path(dst))
