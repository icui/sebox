from __future__ import annotations
import typing as tp
from obspy.geodetics import locations2degrees

from sebox import root
from sebox.utils.catalog import merge_stations, getevents, getstations, getdir, locate_station
from .preprocess import _prepare_frequencies, _freq
from .ft import ft, ft_obs

if tp.TYPE_CHECKING:
    from .typing import Ortho, Encoding


def test_traces(node: Ortho):
    """Check the orthogonality of traces."""
    node.add(_catalog, nkernels=1, reference_velocity=None)

    node.add('solver', cwd='forward_syn',
        path_event=node.path('kl_00/SUPERSOURCE'),
        path_stations=node.path('SUPERSTATION'),
        monochromatic_source=True,
        save_forward=False)

    for event in node.test_events:
        node.add('solver', cwd=f'forward_{event}',
            path_event=getdir().path('events', event),
            path_stations=node.path('SUPERSTATION'),
            monochromatic_source=False,
            save_forward=False)

    node.add(_ft)
    node.add(_read)


def _catalog(node: Ortho):
    node.mkdir('catalog')
    node.ln(getdir().path('*'), 'catalog')
    node.rm('catalog/events')
    node.mkdir('catalog/events')
    merge_stations(node, getevents())

    for event in node.test_events:
        node.ln(getdir().path('events', event), 'catalog/events')
        node.mkdir(f'enc_{event}')

    root.cache['events'] = node.test_events
    _prepare_frequencies(node)
    root.path_catalog = node.path('catalog')

    node.mkdir('enc_syn')
    node.mkdir('enc_obs')
    node.mkdir('enc_stas')


def _ft(node: Ortho):
    stas = getstations()
    enc = node.load('kl_00/encoding.pickle')
    node.add_mpi(ft, arg=(enc, 'forward_syn', 'enc_syn', True), arg_mpi=stas)

    for event in node.test_events:
        node.add_mpi(ft, arg=(enc, f'forward_{event}', f'enc_{event}', False), arg_mpi=stas)

    node.add_mpi(_enc, arg=enc, arg_mpi=stas)


def _enc(enc: Encoding, stas: tp.List[str]):
    import numpy as np

    # kernel configuration
    nt = enc['kf'] * enc['nt_se']
    t = np.linspace(0, (nt - 1) * enc['dt'], nt)
    freq = _freq(enc)

    # data from catalog
    event_data = getdir().load('event_data.pickle')
    encoded = np.full([len(stas), 3, enc['imax'] - enc['imin']], np.nan, dtype=complex)

    # get global station index
    sidx = []
    stations = getstations()

    for sta in stas:
        sidx.append(stations.index(sta))
    
    sidx = np.array(sidx)

    # read event data
    for event, slots in enc['fslots'].items():
        data = root.mpi.mpiload(f'enc_{event}')
        hdur = event_data[event][-1]
        tshift = 1.5 * hdur
        
        # source time function of observed data and its frequency component
        stf = np.exp(-((t - tshift) / (hdur / 1.628)) ** 2) / np.sqrt(np.pi * (hdur / 1.628) ** 2)
        sff = ft_obs(enc, stf)

        # phase difference from source time function (note: pff is different from _enc_obs because ft_obs in catalog is trimed to event starttime)
        pff = np.exp(2 * np.pi * 1j * freq * (enc['nt_ts'] * enc['dt'])) / sff

        # record frequency components
        for idx in slots:
            pshift = pff[idx]
            encoded[:, :, idx] = data[:, :, idx] * pshift

    root.mpi.mpidump(encoded, 'enc_obs')
    root.mpi.mpidump(stas, 'enc_stas')

def _read(node: Ortho):
    import numpy as np

    lines = []

    for p in range(root.task_nprocs):
        syn = node.load(f'enc_syn/p{p:02d}.npy')
        obs = node.load(f'enc_obs/p{p:02d}.npy')
        stas = node.load(f'enc_stas/p{p:02d}.pickle')
        d = abs(np.angle(syn / obs))
        idx = np.unravel_index(d.argmax(), d.shape)
        str_idx = f'{idx}'
        str_idx += ' ' * (15 - len(str_idx))
        sta = stas[idx[0]]
        str_sta = sta +  ' ' * (8 - len(sta))
        lines.append(f'p{p:02d}: {str_sta} {str_idx}{d[idx]:.2f}  {d[idx[:2]].std():.2f}  {d.std():.2f}')
        lat, lon = locate_station(sta)
        print(lines[-1], locations2degrees(lat, lon, 67, 91))

    lines.append('')

    node.writelines(lines, 'mf.txt')
