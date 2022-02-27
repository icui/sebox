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

    def __init__(self):
        cwd = root.load('config.toml')['root'].get('path_catalog') or '.'
        
        super().__init__(cwd)

        if self.has('catalog.toml'):
            self._config = self.load('catalog.toml')

    @property
    def events(self) -> tp.List[str]:
        """List of event names."""
        if self._events is None:
            self._events = self.load('events.pickle')
        
        return self._events
    
    @property
    def event_data(self) -> np.ndarray:
        """2D array of event data (latitude, longitude, depth, hdur)."""
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
        """2D array of station data (latitude, longitude, elevation, burial)."""
        if self._station_data is None:
            self._station_data = self.load('station_data.npy')
        
        return self._station_data
    
    @property
    def bands(self) -> tp.List[int]:
        """Frequecy band info (min_index, max_index, index_increment, transient_ratio."""
        if self._bands is None:
            self._bands = self.load('bands.pickle')
        
        return self._bands
    
    @property
    def imin(self) -> int:
        """Minimum frequency index."""
        return self.bands[0]
    
    @property
    def imax(self) -> int:
        """Maximum frequency index."""
        return self.bands[1]
    
    @property
    def fincr(self) -> int:
        """Number of frequencies per band."""
        return self.bands[2]
    
    @property
    def kf(self) -> int:
        """Ratio between transient duration and stationary duration."""
        return self.bands[3]
    
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


catalog = Catalog()
