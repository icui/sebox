from __future__ import annotations
import typing as tp

from sebox.utils.catalog import getstations, getcomponents, locate_event

if tp.TYPE_CHECKING:
    from numpy import ndarray
    from .typing import Kernel


def ft_syn(ws: Kernel, data: ndarray):
    from scipy.fft import fft
    return fft(data[..., ws.nt_ts: ws.nt_ts + ws.nt_se])[..., ws.imin: ws.imax] # type: ignore


def ft_obs(ws: Kernel, data: ndarray):
    import numpy as np
    from scipy.fft import fft

    shape = data.shape

    if (nt := ws.kf * ws.nt_se) > len(data):
        # expand observed data with zeros
        pad = list(shape)
        pad[-1] = nt - len(data)
        data = np.concatenate([data, pad])
    
    else:
        data = data[..., :nt]

    return fft(data)[..., ::ws.kf][..., ws.imin: ws.imax] # type: ignore


async def ft(ws: Kernel):
    # load trace parameters
    await ws.mpiexec(_ft, arg=ws, arg_mpi=getstations())


def _ft(ws: Kernel, stas: tp.List[str]):
    import numpy as np
    from sebox import root
    from sebox.mpi import pid

    root.restore(ws)
    stats = ws.load('forward/traces/stats.pickle')
    data = ws.load(f'forward/traces/{pid}.npy')

    if 'II.OBN' in stas:
        ws.dump(data[stas.index('II.OBN'), 2], '../ii_obn.npy')
    
    # resample if necessary
    if not np.isclose(stats['dt'], ws.dt):
        from scipy.signal import resample
        print('resample:', stats['dt'], '->', ws.dt)
        resample(data, int(round(stats['nt'] * stats['dt'] / ws.dt)), axis=-1)

    # FFT
    if ws.ft_event is None:
        data_nez = ft_syn(ws, data)
        data_rtz = rotate_frequencies(ws, data_nez, stas, stats['cmps'], True)

        if 'II.OBN' in stas:
            ws.dump(data_rtz[stas.index('II.OBN'), 2], '../ii_obn_rot.npy')
        # print(stats['cmps'])

    #     # rotate frequencies
    #     output_rtz = rotate_frequencies(output_nez, self.fslots, params, station, inv)
    #     output = {}

    #     for cmp, data in output_rtz.items():
    #         output[f'MX{cmp}'] = data, params
    
    # else:
    #     for trace in stream:
    #         output[f'MX{trace.stats.component}'] = self._ft_obs(trace.data), params

    # return output

def rotate_frequencies(ws: Kernel, data: ndarray, stas: tp.List[str], cmps: tp.Tuple[str, str, str], direction: bool = True):
    import numpy as np
    from obspy import Stream, Trace
    from pytomo3d.signal.process import rotate_stream
    from sebox.mpi import pid

    data_rot = np.zeros(data.shape)
    invs = ws.load(f'forward/traces/{pid}_inv.pickle')

    if direction:
        mode = 'NE->RT'
        cmps_from = cmps
        cmps_to = getcomponents()
    
    else:
        mode = 'RT->NE'
        cmps_from = getcomponents()
        cmps_to = cmps

    for i, sta in enumerate(stas):
        for event, slots in ws.fslots.items():
            if len(slots) == 0:
                continue

            # frequency domain traces of current event
            traces = []
            n, s = sta.split('.')
            
            for j, cmp in enumerate(cmps_from):
                traces.append(Trace(data[i, j, slots], {
                    'component': cmp, 'channel': f'MX{cmp}', 'delta': ws.dt,
                    'network': n, 'station': s, 'location': 'S3'
                }))

            # rotate frequencies
            lat, lon = locate_event(event)
            stream = rotate_stream(Stream(traces), lat, lon, invs[i], mode=mode)

            for j, cmp in enumerate(cmps_to):
                data_rot[i, j, slots] = stream.select(component=cmp)[0].data # type: ignore

    return data_rot
