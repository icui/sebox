import typing as tp

import numpy as np
from nnodes import root, Node
from collections import ChainMap

from .catalog import d


def index_events(node):
    """Create event list and event data array."""
    if not d.has('components.pickle'):
        d.dump(['R', 'T', 'Z'], 'components.pickle')

    if not d.has('events.pickle') or not d.has('event_data.npy'):
        node.add_mpi(_index_events, root.job.cpus_per_node, mpiarg=d.ls('events'))


def _index_events(evts):
    # dict of event data
    evt_dict = {}

    for event in evts:
        lines = d.readlines(f'events/{event}')

        # event time shift, half duration, latitude, longitude, depth and moment tensor
        evt_dict[event] = np.array([float(line.split()[-1]) for line in lines[2:13]])

    # gather results
    evt_dict = root.mpi.comm.gather(evt_dict, root=0)
    
    if root.mpi.rank == 0:
        event_dict = dict(ChainMap(*tp.cast(tp.List, evt_dict)))
        events = sorted(list(event_dict.keys()))

        # merge event data into one array
        event_data = np.zeros([len(events), 11])

        for i, event in enumerate(events):
            event_data[i, :] = event_dict[event]

        # save data
        d.dump(events, 'events.pickle')
        d.dump(event_data, 'event_data.npy')


def index_encoding(node):
    """Save source encoding parameters and DFT matrix."""
    from .catalog import process

    encoding = {
        'dt': process['dt'],
        'nt_ts': int(np.round(process['duration'] * 60 / process['dt'])),
        'nt_se': int(np.round(process['duration_encoding'] * 60 / process['dt']))
    }


def index(node: Node):
    """Index events and stations."""
    from .catalog import catalog

    if not catalog.has('components.pickle'):
        catalog.dump(['R', 'T', 'Z'], 'components.pickle')
    
    events = catalog.ls('events')

    if not catalog.has('events.pickle') or not catalog.has('event_data.npy'):
        # save event data
        node.add_mpi(index_events, arg_mpi=events)
    
    if not catalog.has('stations.pickle') or not catalog.has('station_data.npy') or not catalog.has('SUPERSTATION'):
        # save station data
        node.add_mpi(index_stations, arg_mpi=events)


def index_bands(node: Node):
    from .catalog import catalog

    syn = '/gpfs/alpine/scratch/ccui/geo111/north_syn'
    m = root.load(f'{syn}/measurements.npy')
    stations = root.load(f'{syn}/stations.pickle')
    events = root.load('events.pickle')
    components = root.load('components.pickle')
    band_data = np.zeros([len(events), len(stations), len(components), catalog.nbands], dtype=int)

    for i in range(catalog.nbands):
        band_data[..., i] = m[..., i * 3: i * 3 + 2].sum(axis=-1)

    node.dump(band_data, 'band_data.npy')
    node.dump(stations, 'stations.pickle')
    
    df = 1 / catalog.duration_ft / 60
    kf = int(np.ceil(catalog.duration / catalog.duration_ft))

    imin = int(np.ceil(1 / catalog.period_max / df))
    imax = int(np.floor(1 / catalog.period_min / df)) + 1
    fincr = (imax - imin) // catalog.nbands
    imax = imin + catalog.nbands * fincr

    catalog.dump((imin, imax, fincr, kf), 'bands.pickle')


def index_stations(evts):
    from .catalog import catalog

    band_data = root.load('band_data.npy').sum(axis=-1).sum(axis=-1)
    events = root.load('events.pickle')
    stations = root.load('stations.pickle')

    # dict of station data
    sta_dict = {}

    # content of SUPERSTATION file
    sta_lines = {}

    for event in evts:
        eid = events.index(event)

        for line in catalog.readlines(f'stations/STATIONS.{event}'):
            if len(ll := line.split()) == 6:
                station = ll[1] + '.' + ll[0]

                if station in stations and band_data[eid][stations.index(station)] > 0:
                    lat = float(ll[2])
                    lon = float(ll[3])
                    elevation = float(ll[4])
                    burial = float(ll[5])

                    # station latitude, longitude, elevation and burial depth
                    sta_dict[station] = lat, lon, elevation, burial

                    # format line in SUPERSTATION
                    sta_lines[station] = format_station(ll)
    
    # gather and save results
    sta_dict = root.mpi.comm.gather(sta_dict, root=0)
    sta_lines = root.mpi.comm.gather(sta_lines, root=0)

    if root.mpi.rank == 0:
        station_dict = dict(ChainMap(*sta_dict))
        station_lines = dict(ChainMap(*sta_lines))

        # merge station data into one array
        station_npy = np.zeros([len(stations), 4])

        for i, station in enumerate(stations):
            station_npy[i, :] = station_dict[station]
        
        # save result
        catalog.dump(station_npy, 'station_data.npy')
        catalog.write(''.join(station_lines.values()), 'SUPERSTATION')


def format_station(ll: list):
    """Format a line in STATIONS file."""
    # location of dots for floating point numbers
    dots = 28, 41, 55, 62

    # line with station name
    line = ll[0].ljust(13) + ll[1].ljust(5)

    # add numbers with correct indentation
    for i in range(4):
        num = ll[i + 2]

        if '.' in num:
            nint, _ = num.split('.')
        
        else:
            nint = num

        while len(line) + len(nint) < dots[i]:
            line += ' '
        
        line += num
    
    return line + '\n'
