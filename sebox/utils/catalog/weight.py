import typing as tp
import numpy as np

from nnodes import Node


def prepare_weighting(node: Node):
    """Computes weightings."""
    from sebox import catalog

    node.add_mpi(_weight, arg_mpi = list(range(-1, len(catalog.events))))


def _weight(indices: tp.List[int]):
    from sebox import catalog

    for idx in indices:
        if idx < 0:
            # event weighting
            locs = catalog.event_data[:, :2]
        
        else:
            # station weighting
            locs = catalog.station_data[idx][:, :2]
        
        print(locs.shape, np.count_nonzero(locs))


def weight_events(node: Node):
    """Comeputes event geographical weighting."""


def weight_stations(node: Node):
    """Computes station geographical weighting."""


def weight_bands(node: Node):
    """Computes band weighting."""
