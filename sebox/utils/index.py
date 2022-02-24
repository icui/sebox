import numpy as np
from nnodes import root, Node
from collections import ChainMap


def create_index(node: Node):
    """Index events and stations."""
    from .catalog import catalog

    if not catalog.has('components.pickle'):
        catalog.dump(['R', 'T', 'Z'], 'components.pickle')
    
    events = catalog.ls('events')

    if not catalog.has('events.pickle') or not catalog.has('event_data.npy'):
        # save event data
        node.add_mpi(index_events, arg_mpi=events)
    
    if not catalog.has('bands.pickle') or not catalog.has('stations.pickle') :
        # save trace bands and station list
        node.add(index_bands, events=events)
    
    if not catalog.has('station_data.npy') or not catalog.has('SUPERSTATION'):
        # save station data
        node.add_mpi(index_stations, arg_mpi=events)


def index_events(events):
    from .catalog import catalog
    
    # dict of event data
    event_dict = {}

    for event in events:
        lines = catalog.readlines(f'events/{event}')
        lat = float(lines[4].split()[-1])
        lon = float(lines[5].split()[-1])
        depth = float(lines[6].split()[-1])
        hdur = float(lines[3].split()[-1])
        event_dict[event] = lat, lon, depth, hdur

    # gather results
    event_dict = root.mpi.comm.gather(event_dict, root=0)
    
    if root.mpi.rank == 0:
        event_dict = dict(ChainMap(*event_dict))
        events = sorted(list(event_dict.keys()))

        # merge event data into one array
        event_npy = np.zeros([len(events), 4])

        for i, event in enumerate(events):
            event_npy[i, :] = event_dict[event]  

        # save data
        catalog.dump(events, 'events.pickle')
        catalog.dump(event_npy, 'event_data.npy')


def index_bands(node: Node):
    from .catalog import catalog

    syn = '/gpfs/alpine/scratch/ccui/geo111/north_syn'
    m = root.load(f'{syn}/measurements.npy')
    print(m.shape, catalog.nbands)
    stations = root.load(f'{syn}/stations.pickle')
    events = root.load('events.pickle')
    events = root.load('components.pickle')
    bands = np.zeros([len(events), len(stations), len(components), catalog.nbands], dtype=int)

    for i in range(catalog.nbands):
        bands[..., i] = m[..., i * 3: i * 3 + 2].sum(axis=-1)

    print(np.count_nonzero(bands), np.count_nonzero(bands.sum(axis=-1)))
    print(np.count_nonzero(m), np.count_nonzero(m.sum(axis=-1)))

def index_stations(events):
    from .catalog import catalog

    # dict of station data
    station_dict = {}

    # content of SUPERSTATION file
    station_lines = {}

    for event in events:
        for line in catalog.readlines(f'stations/STATIONS.{event}'):
            if len(ll := line.split()) == 6:
                station = ll[1] + '.' + ll[0]
                lat = float(ll[2])
                lon = float(ll[3])
                elevation = float(ll[4])
                burial = float(ll[5])

                station_dict[station] = lat, lon, elevation, burial
                station_lines[station] = _format_station(ll)
    
    # gather and save results
    station_dict = root.mpi.comm.gather(station_dict, root=0)
    station_lines = root.mpi.comm.gather(station_lines, root=0)

    if root.mpi.rank == 0:
        station_dict = dict(ChainMap(*station_dict))
        station_lines = dict(ChainMap(*station_lines))
        stations = sorted(list(station_dict.keys()))


        # merge station data into one array
        station_npy = np.zeros([len(stations), 4])

        for i, station in enumerate(stations):
            station_npy[i, :] = station_dict[station]
        
        # save result
        catalog.dump(stations, 'stations.pickle')
        catalog.dump(station_npy, 'station_data.npy')
        catalog.write(''.join(station_lines.values()), 'SUPERSTATION')


def _format_station(ll: list):
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
    
    return line
