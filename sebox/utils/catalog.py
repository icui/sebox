import typing as tp
import numpy as np

from nnodes import root, Node, Directory

from .index import create_index
from .weight import create_weighting


def create_catalog(node: Node):
    """Create a catalog database."""
    node.add(create_index)
    node.add(create_weighting)


class Catalog(Directory):
    """Getter of items in catalog directory."""
    # catalog configuration from catalog.toml
    _config = {}

    @property
    def events(self) -> tp.List[str]:
        """List of event names."""
        if self._events is None:
            self._events = self.load('events.pickle')
        
        return self._events
    
    @property
    def event_data(self) -> np.ndarray:
        """2D array of event data (event_latitude, event_longitude, event_depth, event_hdur)."""
        if self._event_data is None:
            self._event_data = self.load('event_data.npy')
        
        return self._event_data

    @property
    def stations(self) -> tp.List[str]:
        """List of station names."""
        if self._stations is None:
            self._stations = self.load('stations.pickle')
        
        return self._stations
    
    @property
    def station_data(self) -> np.ndarray:
        """2D array of station data (station_latitude, station_longitude, station_elevation, station_burial)."""
        if self._station_data is None:
            self._station_data = self.load('station_data.npy')
        
        return self._station_data
    
    @property
    def bands(self) -> tp.List[int]:
        """Frequecy band info (min_index, max_index, index_increment, ceil(transient_dur / station_dur))."""
        if self._bands is None:
            self._bands = self.load('bands.pickle')
        
        return self._bands
    
    @property
    def band_data(self) -> np.ndarray:
        """4D array of measurements of bands (event, station, component, band)."""
        if self._band_data is None:
            self._band_data = self.load('band_data.npy')
        
        return self._band_data

    @property
    def nbands(self) -> int:
        """Total number of bands."""
        return self._config.get('nbands') or 1

    @property
    def period(self) -> tp.Tuple[float, float]:
        """Min and max period in seconds."""
        return self._config['period']

    @property
    def duration(self) -> tp.Tuple[float, float]:
        """Transient duration and station duration in minutes."""
        return self._config['duration']


catalog = Catalog(root.load('config.toml')['root'].get('path_catalog') or '.')

if catalog.has('catalog.toml'):
    catalog._config = catalog.load('catalog.toml')
