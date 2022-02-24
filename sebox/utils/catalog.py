import typing as tp
import numpy as np

from nnodes import root, Node, Directory

from .index import create_index
from .weight import create_weighting


def create_catalog(node: Node):
    """Create a catalog database."""
    catalog.init()
    node.add(create_index)
    node.add(create_weighting)


class Catalog(Directory):
    """Getter of items in catalog directory."""
    # self.init() has been called
    _inited = False

    # catalog configuration from catalog.toml
    _config: dict

    # list of event names
    _events = None

    # 2D array of event_latitude, event_longitude, event_depth, event_hdur
    _event_data = None

    # list of station names
    _stations = None

    # 2D array of station_latitude, station_longitude, station_elevation, station_burial
    _station_data = None

    @property
    def events(self) -> tp.List[str]:
        self.init()

        if self._events is None:
            self._events = self.load('events.pickle')
        
        return self._events

    @property
    def stations(self) -> tp.List[str]:
        self.init()

        if self._stations is None:
            self._stations = self.load('stations.pickle')
        
        return self._stations
    
    @property
    def event_data(self) -> np.ndarray:
        self.init()

        if self._event_data is None:
            self._event_data = self.load('event_data.pickle')
        
        return self._event_data
    
    @property
    def station_data(self) -> np.ndarray:
        self.init()

        if self._station_data is None:
            self._station_data = self.load('station_data.pickle')
        
        return self._station_data

    def init(self):
        """Initialize catalog directory."""
        if self._inited:
            return
        
        self._cwd = tp.cast(str, root.path_catalog)
        self._config = self.load('catalog.toml') if self.has('catalog.toml') else {}
        self._inited = True


catalog = Catalog(root.load('config.toml')['root'].get('path_catalog') or '.')
