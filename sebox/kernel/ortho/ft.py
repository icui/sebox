from __future__ import annotations
import typing as tp

from sebox.utils.catalog import getstations

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
    from sebox.mpi import pid

    stats = ws.load('forward/traces/stats.pickle')
    data = ws.load(f'forward/traces/{pid}.npy')
    
    # resample if necessary
    if not np.isclose(stats['dt'], ws.dt):
        from scipy.signal import resample
        print('resample:', stats['dt'], '->', ws.dt)
        resample(data, int(round(stats['nt'] * stats['dt'] / ws.dt)), axis=-1)

    # FFT
    if ws.event is None:
        data_nez = ft_syn(ws, data)
        print(stats['cmps'])

    #     # rotate frequencies
    #     output_rtz = rotate_frequencies(output_nez, self.fslots, params, station, inv)
    #     output = {}

    #     for cmp, data in output_rtz.items():
    #         output[f'MX{cmp}'] = data, params
    
    # else:
    #     for trace in stream:
    #         output[f'MX{trace.stats.component}'] = self._ft_obs(trace.data), params

    # return output