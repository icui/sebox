from nnodes import root, Node, Directory


# determine working directory
if root.has('config.toml'):
    _cwd = root.load('config.toml')['root'].get('path_catalog') or '.'

else:
    _cwd = '.'

# directory object for catalog
_dir = Directory(_cwd)

# cache of config.toml
_config = _dir.load('catalog.toml') if _dir.has('catalog.toml') else {}


# cache of catalog items stored as pickle
_catalog = {
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
    if name == 'd':
        return _dir

    if name in _catalog:
        # items stored as pickle in catalog directory
        if _catalog[name] is None:
            _catalog[name] = _dir.load(f'{name}.pickle')
            
    if name in _config:
        # items in config.toml
        _config[name]

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
