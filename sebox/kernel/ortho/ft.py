from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from numpy import ndarray
    from .typing import Kernel


def ft_syn(ws: Kernel, data: ndarray):
    from scipy.fft import fft
    return fft(data[ws.nt_ts: ws.nt_ts + ws.nt_se])[ws.imin: ws.imax]


def ft_obs(ws: Kernel, data: ndarray):
    from scipy.fft import fft

    if (nt := ws.kf * ws.nt_se) > len(data):
        # expand observed data with zeros
        data = np.concatenate([data, np.zeros(nt - len(data))]) # type: ignore
    
    else:
        data = data[:nt]
    
    return fft(data)[::ws.kf][ws.imin: ws.imax]


# def _ft(ws: Kernel):
#         output = {}

#         # process stream
#         if (stream := acc.stream) is None:
#             return

#         # save the stats of original trace
#         station = acc.station
#         stats = stream[0].stats # type: ignore

#         # Time and frequency parameters
#         params = {
#             'npts': stats.npts,
#             'delta': stats.delta,
#             'nt': len(stream[0].data), # type: ignore
#             'dt': self.dt,
#             'nt_ts': self.nt_ts,
#             'nt_se': self.nt_se,
#             'fidx': self.fidx
#         }
        
#         # resample if necessary
#         if not np.isclose(self.dt, stats.delta):
#             print(f'resample {stats.delta} -> {self.dt}')
#             stream.resample(sampling_rate=1/self.dt)

#         # FFT
#         if event is None:
#             if (inv := acc.inventory) is None or station is None:
#                 return
            
#             output_nez = {}

#             for trace in stream:
#                 output_nez[trace.stats.component] = self._ft_syn(trace.data)

#             # rotate frequencies
#             output_rtz = rotate_frequencies(output_nez, self.fslots, params, station, inv)
#             output = {}

#             for cmp, data in output_rtz.items():
#                 output[f'MX{cmp}'] = data, params
        
#         else:
#             for trace in stream:
#                 output[f'MX{trace.stats.component}'] = self._ft_obs(trace.data), params

#         return output