from nnodes import root, Node, Directory


# determine working directory
if root.has('config.toml'):
    cwd = root.load('config.toml')['root'].get('path_catalog') or '.'

else:
    cwd = '.'

# directory object for catalog
d = Directory(_cwd)

# read catalog.toml
if d.has('catalog.toml'):
    _catalog = d.load('catalog.toml')

else:
    _catalog = {}

# cache of catalog items stored as pickle
_cache = {
    # event names
    'events': None,

    # event latitude, longitude, depth and hdur
    'event_data': None,

    # station names
    'stations': None,

    # station latitude, longitude, elevation and burial
    'station_data': None,

    # source encoding parameters
    'encoding': None,

    # geographical weightings
    'weighting': None,

    # source-receiver pairs used for measurement
    'traces': None
}


def __getattr__(name):
    if name in _cache:
        # read items stored as pickle in catalog directory
        if _cache[name] is None:
            if d.has(f'{name}.pickle'):
                _cache[name] = d.load(f'{name}.pickle')

            if d.has(f'{name}.npy'):
                _cache[name] = d.load(f'{name}.npy')
            
    if name in _catalog:
        # items in config.toml
        _catalog[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def create_catalog(node: Node):
    """Create a catalog database."""
    # create pickle file for events and event_data
    node.add(index_events)

    # create pickle file for encoding parameters
    node.add(index_bands)

    # # download raw seismic data
    # node.add(download)

    # # compute and process synthetic data
    # node.add(simulate)

    # # process observed and synthetic seismic data
    # node.add(process)

    # create trace windows
    node.add(window)

    # select traces based on windows
    node.add(select)

    # create pickle file for stations and station_data
    node.add(index_stations)

    # create pickle file for traces
    node.add(index_traces)

    # create pickle file for weighting
    node.add(weight)
