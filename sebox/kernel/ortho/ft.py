from __future__ import annotations
import typing as tp

from sebox import Directory
from sebox.utils.catalog import getstations

if tp.TYPE_CHECKING:
    from numpy import ndarray
    from .typing import Kernel

    class Stats(tp.TypedDict, total=False):
        """Source encoding data dict."""
        # length of original trace
        npts: int

        # time step of original trace
        delta: float

        # time step
        dt: float

        # total number of time steps
        nt: int

        # transient time step
        nt_ts: int

        # steady state time step
        nt_se: int

        # minimum frequency index
        imin: int

        # maximum frequency index
        imax: int

        # event name
        event: tp.Optional[str]

        # MPI trace directory
        path: str


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
    stats: Stats = ws.load('forward/traces/stats.pickle')
    stats.update({
        'npts': stats['nt'],
        'delta': stats['dt'],
        'dt': ws.dt,
        'nt_ts': ws.nt_ts,
        'nt_se': ws.nt_se,
        'imin': ws.imin,
        'imax': ws.imax,
        'event': ws.event,
        'path': ws.path('forward/traces')
    })

    await ws.mpiexec(_ft, arg=stats, arg_mpi=getstations())


def _ft(stats: Stats, stas: tp.List[str]):
    import numpy as np
    from sebox.mpi import pid

    data = Directory(stats['path']).load(f'{pid}.npy')
    output = {}
    
    # resample if necessary
    if not np.isclose(stats['dt'], stats['delta']):
        from scipy.signal import resample
        print('resample:', stats['delta'], '->', stats['dt'])
        resample(data, int(round(stats['npts'] * stats['delta'] / stats['dt'])), axis=-1)

    # # FFT
    # if event is None:
    #     if (inv := acc.inventory) is None or station is None:
    #         return
        
    #     output_nez = {}

    #     for trace in stream:
    #         output_nez[trace.stats.component] = self._ft_syn(trace.data)

    #     # rotate frequencies
    #     output_rtz = rotate_frequencies(output_nez, self.fslots, params, station, inv)
    #     output = {}

    #     for cmp, data in output_rtz.items():
    #         output[f'MX{cmp}'] = data, params
    
    # else:
    #     for trace in stream:
    #         output[f'MX{trace.stats.component}'] = self._ft_obs(trace.data), params

    # return output