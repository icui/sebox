# def create_catalog(node: Node):
#     """Create a catalog database."""
#     node.add(download)
#     node.add(process)
#     node.add(window)
    # node.add(index)
    # node.add(weight)


    # @property
    # def _bands(self) -> tp.List[int]:
    #     """Frequecy band info (min_index, max_index, index_increment, transient_ratio."""
    #     if self._bands_pkl is None:
    #         self._bands_pkl = self.load('bands.pickle')
        
    #     return self._bands_pkl
    
    # @property
    # def imin(self) -> int:
    #     """Minimum frequency index."""
    #     return self._bands[0]
    
    # @property
    # def imax(self) -> int:
    #     """Maximum frequency index."""
    #     return self._bands[1]
    
    # @property
    # def fincr(self) -> int:
    #     """Number of frequencies per band."""
    #     return self._bands[2]
    
    # @property
    # def kf(self) -> int:
    #     """Ratio between transient duration and stationary duration."""
    #     return self._bands[3]
    
    # @property
    # def measurements(self) -> np.ndarray:
    #     """4D array of measurements of bands (event, station, component, band)."""
    #     if self._measurements is None:
    #         self._measurements = self.load('band_data.npy')
        
    #     return self._measurements

    # @property
    # def dt(self) -> int:
    #     """Length of a time step."""
    #     return self.process['dt']

    # @property
    # def nbands(self) -> int:
    #     """Total number of bands."""
    #     return self.process['nbands']

    # @property
    # def period_min(self) -> float:
    #     """Min period in minutes."""
    #     return self.process['period_min']

    # @property
    # def period_max(self) -> float:
    #     """Max period in minutes."""
    #     return self.process['period_max']

    # @property
    # def duration(self) -> float:
    #     """Duration of wavefield recording in minutes."""
    #     return self.process['duration']

    # @property
    # def duration_ft(self) -> float:
    #     """Duration of stationary wavefield for frequency measurement in minutes."""
    #     return self.process['duration_ft']
