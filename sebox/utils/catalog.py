from __future__ import annotations
import typing as tp
from os import path

from sebox import root, Directory, Node

if tp.TYPE_CHECKING:
    from numpy import ndarray
    from sebox.core.root import Root

    class Catalog(Root):
        """Root directory with catalog configurations."""
        # path to catalog directory
        path_catalog: str

        # maximum relative measurement weighting
        measurement_threshold: tp.Optional[float]

        # maximum relative noise weighting
        noise_threshold: tp.Optional[float]
    
    root = tp.cast(Catalog, root)


def getdir(*paths: str) -> Directory:
    """Get catalog directory."""
    return Directory(path.join(tp.cast(str, root.path_catalog), *paths))


def getevents(group: tp.Optional[int] = None):
    """Get list of events."""
    cache = root.cache

    if 'events' not in cache:
        cache['events'] = getdir().load('events.pickle')

    if group is None:
        return cache['events']

    events = []

    for event in cache['events']:
        if getmeasurements(event=event, group=group).any():
            events.append(event)

    return events


def getstations(event: tp.Optional[str] = None, group: tp.Optional[int] = None):
    """Get list of stations."""
    cache = root.cache

    if 'stations' not in cache:
        cache['stations'] = getdir().load('stations.pickle')

    if event is None and group is None:
        return cache['stations']

    stations = []

    for station in cache['stations']:
        if getmeasurements(event=event, station=station, group=group).any():
            stations.append(station)

    return stations


def getcomponents(event: tp.Optional[str] = None, station: tp.Optional[str] = None, group: tp.Optional[int] = None):
    """Get list of components."""
    cache = root.cache

    if 'components' not in cache:
        cache['components'] = getdir().load('components.pickle')

    if event is None and station is None and group is None:
        return cache['components']

    components = []

    for cmp in cache['components']:
        if getmeasurements(event=event, station=station, component=cmp, group=group).any():
            components.append(cmp)

    return components


def hasstation(event: str, station: str):
    """Check if event contains station."""
    if event not in getevents():
        return False

    if station not in getstations():
        return False

    return getmeasurements(event=event, station=station).any()



def getmeasurements(event: tp.Union[str, int, None] = None, station: tp.Union[str, int, None] = None,
    component: tp.Union[str, int, None] = None, group: tp.Optional[int] = None,
    categorical: bool = True, geographical: bool = False, noise: bool = False, balance: bool = False) -> ndarray:
    """Get array of measuremnets."""
    cache = root.cache

    if not categorical and not geographical and not noise:
        raise TypeError('no weighting type specified')

    if categorical and 'measurements' not in cache:
        cache['measurements'] = getdir().load('measurements.npy')

    if geographical and 'weightings' not in cache:
        cache['weightings'] = getdir().load('weightings.npy')

    if noise and 'noise' not in cache:
        cache['noise'] = getdir().load('noise.npy')
    
    m = cache.get('measurements') if categorical else None
    w = cache.get('weightings') if geographical else None
    n = cache.get('noise') if noise else None

    def toint(target, get):
        return target if isinstance(target, int) else get().index(target)

    if event is not None:
        if m is not None:
            m = m[toint(event, getevents)]

        if w is not None:
            w = w[toint(event, getevents)]
        
        if n is not None:
            n = n[toint(event, getevents)]
    
    if station is not None:
        if m is not None:
            m = m[..., toint(station, getstations), :, :]

        if w is not None:
            w = w[..., toint(station, getstations)]
        
        if n is not None:
            n = n[..., toint(station, getstations), :]
    
    if component is not None:
        if m is not None:
            m = m[..., toint(component, getcomponents), :]

        if n is not None:
            n = n[..., toint(component, getcomponents)]
    
    if group is not None:
        if m is not None:
            m = m[..., group]
    
    if balance:
        import numpy as np

        if m is not None:
            # clip by measurement weighting
            m = m.copy()
            mw = root.measurement_threshold

            if isinstance(mw, (int, float)):
                m[m <= mw] = 0
            
            m[m > 0] = 1
        
        if w is not None:
            # limit the condition number of geographical weighting
            w = np.sqrt(w)
        
        if n is not None:
            # clip by signal to noise ratial
            n = n.copy()
            sw = root.noise_threshold

            if isinstance(sw, (int, float)):
                n[n > sw] = 0.0

            n[n > 0] = 1.0

    # output matrix
    out = None

    for a in m, w, n:
        if a is None:
            continue

        if out is None:
            out = a
        
        else:
            out = (out.transpose() * a.transpose()).transpose()
    
    return tp.cast('ndarray', out)


def _format_station(lines: dict, ll: tp.List[str]):
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
    
    lines[ll[1] + '.' + ll[0]] = line


def create_catalog(node: Node):
    """Create catalog datatase."""
    import numpy as np

    #FIXME create_catalog (measurements.npy, weightings.npy, noise.npy, ft_obs, ft_diff)
    cdir = getdir()
    event_data = {}
    station_locations = {}
    event_stations = {}
    station_lines = {}

    for event in cdir.ls('events'):
        event_stations[event] = []
        lines = cdir.readlines(f'events/{event}')
        elat = float(lines[4].split()[-1])
        elon = float(lines[5].split()[-1])
        depth = float(lines[6].split()[-1])
        hdur = float(lines[3].split()[-1])
        event_data[event] = elat, elon, depth, hdur

        for line in getdir().readlines(f'stations/STATIONS.{event}'):
            if len(ll := line.split()) == 6:
                station = ll[1] + '.' + ll[0]
                slat = float(ll[2])
                slon = float(ll[3])
                station_locations[station] = slat, slon
                event_stations[event].append(station)
                _format_station(station_lines, ll)
    
    events = sorted(event_data.keys())
    stations = sorted(station_locations.keys())
    cmps = sorted(['R', 'T', 'Z'])
    
    # save results
    cdir.dump(events, 'events.pickle')
    cdir.dump(stations, 'stations.pickle')
    cdir.dump(cmps, 'components.pickle')
    cdir.dump(event_data, 'event_data.pickle')
    cdir.dump(station_locations, 'station_locations.pickle')

    #FIXME
    c = np.zeros([len(events), len(stations), len(cmps), 15], dtype=int)

    for i, event in enumerate(events):
        for j, station in enumerate(stations):
            if station in event_stations[event]:
                c[i, j, :, :] = 1

    cdir.dump(c, 'measurements.npy')
    cdir.dump(event_stations, 'event_stations.pickle')
    cdir.dump(station_lines, 'station_lines.pickle')


def index_stations():
    """Create an index of stations."""
    from sebox.utils.catalog import getstations

    lines = {}
    cdir = getdir()
    d = cdir.subdir('stations')
    stations = getstations()

    for src in d.ls():
        for line in d.readlines(src):
            if len(ll := line.split()) == 6:
                station = ll[1] + '.' + ll[0]

                if station in lines or station not in stations:
                    continue

                _format_station(lines, ll)
    
    cdir.dump(lines, 'station_lines.pickle')


def index_events():
    """Create an index of available stations of events."""
    event_stations = {}
    m = getmeasurements().sum(axis=-1).sum(axis=-1)

    for i, event in enumerate(getevents()):
        event_stations[event] = []

        for j, station in enumerate(getstations()):
            if m[i, j] >= 1:
                event_stations[event].append(station)
    
    getdir().dump(event_stations, 'event_stations.pickle')


def merge_stations(dst: Directory, events: tp.List[str], bury: tp.Optional[float] = None):
    """Merge multiple stations into one."""
    stations = set()
    event_stations = getdir().load('event_stations.pickle')

    for event in events:
        for station in event_stations[event]:
            stations.add(station)

    lines = getdir().load('station_lines.pickle')
    sta = ''
    
    for station in stations:
        if bury is not None and bury > 1:
            ll = lines[station].split(' ')
            ll[-1] = f'{float(ll[-1]) + bury*1000:.1f}'
            lines[station] = ' '.join(ll)

        sta += lines[station] + '\n'
    
    dst.write(sta, 'SUPERSTATION')


def extract_stations(d: Directory, dst: Directory):
    """Extract STATIONS from ASDFDataSet."""
    from pyasdf import ASDFDataSet

    for src in d.ls():
        event = src.split('.')[0]
        lines = {}
        fname = f'STATIONS.{event}'

        if dst.has(fname):
            continue

        with ASDFDataSet(src, mode='r', mpi=False) as ds:
            for station in ds.waveforms.list():
                if not hasattr(ds.waveforms[station], 'StationXML'):
                    print('  ' + station)
                    continue

                sta = ds.waveforms[station].StationXML.networks[0].stations[0] # type: ignore

                ll = station.split('.')
                ll.reverse()
                ll.append(f'{sta.latitude:.4f}')
                ll.append(f'{sta.longitude:.4f}')
                ll.append(f'{sta.elevation:.1f}')
                ll.append(f'{sta.channels[0].depth:.1f}')

                _format_station(lines, ll)
        
        dst.writelines(lines.values(), fname)


def locate_event(event: str, depth: bool = False) -> tp.List[float]:
    """Get event latitude and longitide."""
    cache = root.cache

    if 'event_locations' not in cache:
        cache['event_locations'] = getdir().load('event_data.pickle')
    
    loc = cache['event_locations'][event]

    if depth:
        return loc
    
    return loc[:2]


def locate_station(station: str) -> tp.Tuple[float, float]:
    """Get station latitude and longitide."""
    cache = root.cache

    if 'station_locations' not in cache:
        cache['station_locations'] = getdir().load('station_locations.pickle')

    return cache['station_locations'][station]
