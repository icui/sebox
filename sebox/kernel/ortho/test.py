from __future__ import annotations
import typing as tp

from sebox.utils.catalog import merge_stations, getevents, getstations, getdir
from .preprocess import prepare_frequencies
from .ft import ft
from .main import dirs

if tp.TYPE_CHECKING:
    from .typing import Ortho


def test_traces(node: Ortho):
    """Check the orthogonality of traces."""
    node.add(_catalog, path_catalog=node.path('catalog'))

    # node.add('solver', cwd='forward_mono',
    #     path_event=node.path('SUPERSOURCE'),
    #     path_stations=node.path('SUPERSTATION'),
    #     path_mesh=node.path('mesh'),
    #     monochromatic_source=True,
    #     save_forward=False)

    # node.add('solver', cwd='forward_regular',
    #     path_event=getdir().path('event', node.test_event),
    #     path_stations=node.path('SUPERSTATION'),
    #     path_mesh=node.path('mesh'),
    #     monochromatic_source=True,
    #     save_forward=False)
    
    # ft = node.add(_ft, concurrent=True)


def _catalog(node: Ortho):
    node.mkdir('catalog')
    node.ln(getdir().path('*'), 'catalog')
    node.rm('catalog/events')
    node.mkdir('catalog/events')
    prepare_frequencies(node)
    merge_stations(node, getevents())


def _ft(node: Ortho):
    stas = getstations()
    enc = node.load('encoding.pickle')


