from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import merge_stations, getevents, getstations, getdir, getmeasurements, getcomponents
from .preprocess import _prepare_frequencies, _freq
from .ft import ft, ft_obs

if tp.TYPE_CHECKING:
    from .typing import Ortho, Encoding


def test_traces(node: Ortho):
    """Check the orthogonality of traces."""
    node.add(_catalog, nkernels=1, reference_velocity=None)

    node.add('solver', cwd='forward_mono',
        path_event=node.path('kl_00/SUPERSOURCE'),
        path_stations=node.path('SUPERSTATION'),
        path_mesh=node.path('mesh'),
        monochromatic_source=True,
        save_forward=False)

    node.add('solver', cwd='forward_regular',
        path_event=getdir().path('events', node.test_event),
        path_stations=node.path('SUPERSTATION'),
        path_mesh=node.path('mesh'),
        monochromatic_source=False,
        save_forward=False)
    
    node.add(_ft)


def _catalog(node: Ortho):
    node.mkdir('catalog')
    node.ln(getdir().path('*'), 'catalog')
    node.rm('catalog/events')
    node.mkdir('catalog/events')
    merge_stations(node, getevents())
    node.ln(getdir().path('events', node.test_event), 'catalog/events')
    root.cache['events'] = [node.test_event]
    _prepare_frequencies(node)
    root.path_catalog = node.path('catalog')


def _ft(node: Ortho):
    stas = getstations()
    enc = node.load('kl_00/encoding.pickle')
    node.add_mpi(ft, arg=(enc, 'forward_mono', 'enc_mono', True), arg_mpi=stas)
    node.add_mpi(ft, arg=(enc, 'forward_regular', f'enc_regular', False), arg_mpi=stas)
    node.add_mpi(_enc, arg=enc, arg_mpi=stas)


def _enc(enc: Encoding, stas: tp.List[str]):
    import numpy as np

    # kernel configuration
    nt = enc['kf'] * enc['nt_se']
    t = np.linspace(0, (nt - 1) * enc['dt'], nt)
    freq = _freq(enc)

    # data from catalog
    cmps = getcomponents()
    event_data = getdir().load('event_data.pickle')
    encoded = np.full([len(stas), 3, enc['imax'] - enc['imin']], np.nan, dtype=complex)

    # get global station index
    sidx = []
    stations = getstations()

    for sta in stas:
        sidx.append(stations.index(sta))
    
    sidx = np.array(sidx)

    # read event data
    data = root.mpi.mpiload('enc_regular')
    event = list(enc['fslots'].keys())[0]
    slots = enc['fslots'][event]
    hdur = event_data[event][-1]
    tshift = 1.5 * hdur
    
    # source time function of observed data and its frequency component
    stf = np.exp(-((t - tshift) / (hdur / 1.628)) ** 2) / np.sqrt(np.pi * (hdur / 1.628) ** 2)
    sff = ft_obs(enc, stf)

    # phase difference from source time function
    pff = np.exp(2 * np.pi * 1j * freq * (enc['nt_ts'] * enc['dt'] - tshift)) / sff

    # record frequency components
    for idx in slots:
        group = idx // enc['frequency_increment']
        pshift = pff[idx]

        for j, cmp in enumerate(cmps):
            m = getmeasurements(event, None, cmp, group)[sidx]
            i = np.squeeze(np.where(m))
            encoded[i, j, idx] = data[i, j, idx] * pshift

    root.mpi.mpidump(encoded, 'enc_shifted')
