from __future__ import annotations
import typing as tp
from functools import partial

from sebox import root
from sebox.utils.catalog import getdir, getevents, getstations, index_events, index_stations, locate_event, locate_station

if tp.TYPE_CHECKING:
    from .typing import Ortho


def catalog(node: Ortho):
    """Create a catalog (run only once for each catalog directory)."""
    node.concurrent = True
    
    # prepare catalog (executed only once for a catalog)
    cdir = getdir()

    # get available stations of an event
    if not cdir.has('event_stations.pickle'):
        node.add(index_events, args=()) # type: ignore

    # merge stations into a single file
    if not cdir.has('station_lines.pickle'):
        node.add(index_stations, args=()) # type: ignore
    
    #FIXME create_catalog (measurements.npy, weightings.npy, noise.npy, ft_obs, ft_diff)
    
    # compute back-azimuth
    if not cdir.has(f'baz_p{root.task_nprocs}'):
        node.add_mpi(_scatter_baz, arg_mpi=getstations())
    
    
    # convert observed traces into MPI format
    if not cdir.has(f'ft_obs_p{root.task_nprocs}'):
        solver = node.add()

        if not cdir.has(f'raw_obs_p{root.task_nprocs}'):
            solver.add(partial(_forward, 'obs'), concurrent=True)
            solver.add(partial(_move_forward, 'obs'))

        solver.add(partial(_ft, 'obs'), concurrent=True)

    # # convert observed traces into MPI format
    # if not cdir.has(f'ft_obs_p{root.task_nprocs}'):
    #     node.add(partial(_scatter, 'obs'), concurrent=True)

    # convert differences between observed and synthetic data into MPI format
    if not cdir.has(f'ft_diff_p{root.task_nprocs}'):
        node.add(partial(_scatter, 'diff'), concurrent=True)


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


def _forward(tag: tp.Literal['obs', 'syn'], node: Ortho):
    cdir = getdir()

    for event in cdir.ls('events'):
        node.add('solver', cwd=f'forward_{event}',
            path_model = node.path_model if tag == 'syn' else node.path_model2,
            duration=node.duration - node.transient_duration + 10,
            path_event=cdir.path('events', event),
            path_stations=cdir.path('stations', f'STATION.{event}'),
            monochromatic_source=False,
            save_forward=False)


def _move_forward(tag: tp.Literal['obs', 'syn'], node: Ortho):
    cdir = getdir()

    for event in cdir.ls('events'):
        dst = f'raw_{tag}_p{root.task_nprocs}/{event}'
        cdir.rm(dst)
        node.mv(f'forward_{event}/traces', node.rel(cdir, dst))


def _ft(tag: tp.Literal['obs', 'syn'], node: Ortho):
    from .ft import ft
    from .preprocess import getenc

    cdir = getdir()
    stas = getstations()
    enc = getenc(node, True)
    node.dump(enc, 'encoding.pickle')

    for event in cdir.ls('events'):
        node.add_mpi(ft, arg=({
            **enc, 'fslots': {event: list(range(enc['nfreq']))}
        }, node.rel(cdir, f'raw_{tag}_p{root.task_nprocs}/{event}'),
        node.rel(cdir, f'ft_{tag}_p{root.task_nprocs}/{event}'), False), arg_mpi=stas)


def _scatter(tag: tp.Literal['obs', 'diff'], node: Ortho):
    cdir = getdir()

    for src in cdir.ls(f'ft_{tag}'):
        event = src.split('.')[0]
        dst = f'ft_{tag}_p{root.task_nprocs}/{event}'
        
        if not cdir.has(dst):
            node.add(('sebox.utils.asdf', 'scatter'), event, aux=True, dtype=complex,
                path_input=cdir.path(f'ft_{tag}/{src}'), path_output=cdir.path(dst))
