import typing as tp
import numpy as np
from nnodes import root, Node


def create_index(node: Node):
    """Index events and stations."""
    from .catalog import catalog

    node.concurrent = True

    if not catalog.has('components.pickle'):
        catalog.dump(['R', 'T', 'Z'], 'components.pickle')
    
    events = catalog.ls('events')

    if not catalog.has('events.pickle') or not catalog.has('event_data.npy'):
        node.add_mpi(index_events, arg_mpi=events)
    
    # if not catalog.has('stations.pickle') or not catalog.has('station_data.npy'):
    #     node.add(index_stations, arg_mpi=events)


def index_events(events: tp.List[str]):
    from .catalog import catalog

    data = {}

    for event in events:
        lines = catalog.readlines(f'events/{event}')
        lat = float(lines[4].split()[-1])
        lon = float(lines[5].split()[-1])
        depth = float(lines[6].split()[-1])
        hdur = float(lines[3].split()[-1])

        data[event] = lat, lon, depth, hdur
    
    # create indices
    events = sorted(data.keys())
    event_data = np.zeros([len(events), 4])

    for i, event in enumerate(events):
        event_data[i, :] = data[event]
    
    # save results
    print(root.mpi.rank)
    # catalog.dump(events, 'events.pickle')
    # catalog.dump(event_data, 'event_data.npy')


# def index_stations(_):
#     data = {}
#     lines = {}
#     event_stations = {}

#     for event in catalog.ls('events'):
#         event_stations[event] = []

#         for line in catalog.readlines(f'stations/STATIONS.{event}'):
#             if len(ll := line.split()) == 6:
#                 station = ll[1] + '.' + ll[0]
#                 lat = float(ll[2])
#                 lon = float(ll[3])
#                 elevation = float(ll[4])
#                 burial = float(ll[5])

#                 data[station] = lat, lon, elevation, burial
#                 event_stations[event].append(station)
#                 _format_station(lines, ll)
    
#     # create indices
#     events = sorted(data.keys())
#     event_data = np.zeros([len(events), 4])

#     for i, event in enumerate(events):
#         event_data[i, :] = data[event]
    
#     catalog.dump(stations, 'stations.pickle')
#     catalog.dump(event_data, 'event_data.pickle')
#     catalog.dump(station_data, 'station_data.pickle')


# def _format_station(lines: dict, ll: tp.List[str]):
#     """Format a line in STATIONS file."""
#     # location of dots for floating point numbers
#     dots = 28, 41, 55, 62

#     # line with station name
#     line = ll[0].ljust(13) + ll[1].ljust(5)

#     # add numbers with correct indentation
#     for i in range(4):
#         num = ll[i + 2]

#         if '.' in num:
#             nint, _ = num.split('.')
        
#         else:
#             nint = num

#         while len(line) + len(nint) < dots[i]:
#             line += ' '
        
#         line += num
    
#     lines[ll[1] + '.' + ll[0]] = line


# def index_components(_):
#     catalog.dump(['R', 'T', 'Z'], 'components.pickle')
