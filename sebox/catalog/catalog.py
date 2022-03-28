import typing as tp
import numpy as np

from nnodes import root, Node, Directory

from .download import download
from .process import process
# from .index import index
# from .weight import weight
from .window import window


def create_catalog(node: Node):
    """Create a catalog database."""
    node.add(download)
    node.add(process)
    node.add(window)
    # node.add(index)
    # node.add(weight)


class Catalog(Directory):
    """Getter of items in catalog directory."""
    # event names
    _events = None

    # event latitude, longitude, depth and hdur
    _event_data = None

    # station names
    _stations = None

    # station latitude, longitude, elevation and burial
    _station_data = None

    # band index data
    _bands_pkl = None

    # band measurements
    _measurements = None

    def __init__(self):
        if root.has('config.toml'):
            cwd = root.load('config.toml')['root'].get('path_catalog') or '.'
        
        else:
            cwd = '.'
        
        super().__init__(cwd)

        if self.has('catalog.toml'):
            self._config = self.load('catalog.toml')
        
        else:
            self._config = {}
    
    @property
    def weight(self):
        """Weighting configurations."""
        return self._config['weight']
    
    @property
    def download(self):
        """Download configurations."""
        return self._config['download']
    
    @property
    def process(self):
        """Trace configurations."""
        return self._config['process']
    
    @property
    def window(self):
        """Window configurations."""
        return self._config['window']

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
    def _bands(self) -> tp.List[int]:
        """Frequecy band info (min_index, max_index, index_increment, transient_ratio."""
        if self._bands_pkl is None:
            self._bands_pkl = self.load('bands.pickle')
        
        return self._bands_pkl
    
    @property
    def imin(self) -> int:
        """Minimum frequency index."""
        return self._bands[0]
    
    @property
    def imax(self) -> int:
        """Maximum frequency index."""
        return self._bands[1]
    
    @property
    def fincr(self) -> int:
        """Number of frequencies per band."""
        return self._bands[2]
    
    @property
    def kf(self) -> int:
        """Ratio between transient duration and stationary duration."""
        return self._bands[3]
    
    @property
    def measurements(self) -> np.ndarray:
        """4D array of measurements of bands (event, station, component, band)."""
        if self._measurements is None:
            self._measurements = self.load('band_data.npy')
        
        return self._measurements

    @property
    def dt(self) -> int:
        """Length of a time step."""
        return self.process['dt']

    @property
    def nbands(self) -> int:
        """Total number of bands."""
        return self.process['nbands']

    @property
    def period_min(self) -> float:
        """Min period in minutes."""
        return self.process['period_min']

    @property
    def period_max(self) -> float:
        """Max period in minutes."""
        return self.process['period_max']

    @property
    def duration(self) -> float:
        """Transient duration and stationary duration in minutes."""
        return self.process['duration']

    @property
    def duration_ft(self) -> float:
        """Transient duration and station duration in minutes."""
        return self.process['duration_ft']


catalog = Catalog()