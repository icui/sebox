from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir, getevents, getstations, merge_stations, locate_event, locate_station

if tp.TYPE_CHECKING:
    from .typing import Ortho


def catalog(node: Ortho):
    """Create a catalog (run only once for each catalog directory)."""
    node.concurrent = True
    
    # prepare catalog (executed only once for a catalog)
    cdir = getdir()

    # merge stations into a single file
    if not cdir.has('SUPERSTATION'):
        node.add(_merge)
    
    #FIXME create_catalog (measurements.npy, weightings.npy, noise.npy, ft_obs, ft_diff)

    # convert observed traces into MPI format
    if not cdir.has(f'ft_obs_p{root.task_nprocs}'):
        node.add(_scatter_obs, concurrent=True)

    # convert differences between observed and synthetic data into MPI format
    if not cdir.has(f'ft_diff_p{root.task_nprocs}'):
        node.add(_scatter_diff, concurrent=True)
    
    # compute back-azimuth
    if not cdir.has(f'baz_p{root.task_nprocs}'):
        node.add_mpi(_scatter_baz, arg_mpi=getstations())


def _merge(_):
    cdir = getdir()
    merge_stations(cdir.subdir('stations'), cdir, True)


def _scatter_obs(node: Ortho):
    """Convert ASDF observed data to MPI format."""
    _scatter(node, 'obs')


def _scatter_diff(node: Ortho):
    """Convert ASDF diff data to MPI format."""
    _scatter(node, 'diff')


def _scatter_baz(stas: tp.List[str]):
    import numpy as np
    from math import radians
    from obspy.geodetics import gps2dist_azimuth

    baz = {}

    for event in getevents():
        elat, elon = locate_event(event)
        baz[event] = np.zeros(len(stas))

        for i, sta in enumerate(stas):
            slat, slon = locate_station(sta)
            baz[event][i] = radians(gps2dist_azimuth(elat, elon, slat, slon)[2])

    getdir().dump(baz, f'baz_p{root.task_nprocs}/{root.mpi.pid}.pickle')


def _scatter(node: Ortho, tag: tp.Literal['obs', 'diff']):
    cdir = getdir()

    for src in cdir.ls(f'ft_{tag}'):
        event = src.split('.')[0]
        dst = f'ft_{tag}_p{root.task_nprocs}/{event}'
        
        if not cdir.has(dst):
            node.add(('sebox.utils.asdf', 'scatter'), event, aux=True, dtype=complex,
                path_input=cdir.path(f'ft_{tag}/{src}'), path_output=cdir.path(dst))
