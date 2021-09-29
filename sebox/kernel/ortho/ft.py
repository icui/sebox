from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir, getstations, getcomponents, locate_event, locate_station

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
            ws.dump(data_rtz[stas.index('II.OBN'), 0], '../ii_obn_r.npy')
            ws.dump(data_rtz[stas.index('II.OBN'), 1], '../ii_obn_t.npy')
            ws.dump(data_rtz[stas.index('II.OBN'), 2], '../ii_obn_z.npy')
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
    from sebox.mpi import pid

    cmps_rt = getcomponents()
    data_rot = np.zeros(data.shape, dtype=complex)
    data_rot[:, 2, :] = data[:, 2, :]
    baz = getdir().load(f'baz_p{root.task_nprocs}/{pid}.pickle')

    n_i = cmps.index('N')
    e_i = cmps.index('E')
    r_i = cmps_rt.index('R')
    t_i = cmps_rt.index('T')

    for event, slots in ws.fslots.items():
        ba = baz[event]

        if len(slots) == 0:
            continue
        
        for slot in slots:
            if direction:
                # rotate from NE to RT
                n = data[:, n_i, slot]
                e = data[:, e_i, slot]
                data_rot[:, r_i, slot] = - e * np.sin(ba) - n * np.cos(ba)
                data_rot[:, t_i, slot] = - e * np.sin(ba) - n * np.cos(ba)
            
            else:
                # rotate from RT to NE
                ba = 2 * np.pi - ba
                r = data[:, r_i, slot]
                t = data[:, t_i, slot]
                data_rot[:, n_i, slot] = - r * np.sin(ba) - t * np.cos(ba)
                data_rot[:, e_i, slot] = - r * np.sin(ba) - t * np.cos(ba)
            
    return data_rot
