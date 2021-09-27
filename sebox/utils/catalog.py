import typing as tp

from sebox import root, Directory

if tp.TYPE_CHECKING:
    from numpy import ndarray


def getdir() -> Directory:
    return Directory(tp.cast(str, root.path_catalog))


def getevents(group: tp.Optional[int] = None):
    """Get list of events."""
    from sebox import root

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
    from sebox import root

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
    from sebox import root

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
    from sebox import root

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
        if m is not None:
            # normalize categorical weighting (1-5) by multiplying 1/3
            # m = np.sqrt(m / 3)

            # clip by measurement weighting
            m = m.copy()
            mw = root.measurement_threshold

            if isinstance(mw, (int, float)):
                m[m < mw] = 0
            
            m[m > 0] = 1
        
        if w is not None:
            # limit the condition number of geographical weighting
            w = np.sqrt(w) # type: ignore
        
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
    
    return tp.cast(tp.Any, out)
