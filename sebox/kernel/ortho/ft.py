from __future__ import annotations
import typing as tp

from sebox import root, Directory
from sebox.utils.catalog import getdir, getstations, getcomponents

if tp.TYPE_CHECKING:
    from numpy import ndarray
    from .typing import Kernel

    class FT(Kernel):
        # input time domain trace data
        path_input: str

        # output frequencies
        path_output: str

        # event being processed (none for source-encoded event)
        ft_event: tp.Optional[str]


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


async def ft(ws: FT):
    # load trace parameters
    ws.mkdir(ws.path_output)
    await ws.mpiexec(_ft, arg=ws, arg_mpi=getstations())


def _ft(ws: FT, stas: tp.List[str]):
    import numpy as np
    from sebox import root
    from sebox.mpi import pid

    root.restore(ws)
    d = Directory(ws.path_input)
    stats = d.load('stats.pickle')
    data = d.load(f'{pid}.npy')

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
        data_rtz = rotate_frequencies(ws, data_nez, stats['cmps'], True)
        ws.dump(data_rtz, f'{ws.path_output}/{pid}.npy', mkdir=False)

        ######
        data_nez2 = rotate_frequencies(ws, data_rtz, stats['cmps'], False)

        if 'IU.PET' in stas:
            ws.dump(data_nez[stas.index('IU.PET'), 0], '../iu_pet_n.npy')
            ws.dump(data_nez[stas.index('IU.PET'), 1], '../iu_pet_e.npy')
            ws.dump(data_nez2[stas.index('IU.PET'), 0], '../iu_petn2.npy')
            ws.dump(data_nez2[stas.index('IU.PET'), 1], '../iu_pete2.npy')
            ws.dump(data_nez2[stas.index('IU.PET'), 2], '../iu_petz2.npy')
            ws.dump(data_rtz[stas.index('IU.PET'), 0], '../iu_pet_r.npy')
            ws.dump(data_rtz[stas.index('IU.PET'), 1], '../iu_pet_t.npy')
            ws.dump(data_rtz[stas.index('IU.PET'), 2], '../iu_pet_z.npy')
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

def rotate_frequencies(ws: Kernel, data: ndarray, cmps_ne: tp.Tuple[str, str, str], direction: bool):
    import numpy as np
    from sebox.mpi import pid

    cmps_rt = getcomponents()
    data_rot = np.zeros(data.shape, dtype=complex)
    baz = getdir().load(f'baz_p{root.task_nprocs}/{pid}.pickle')

    if direction:
        # rotate from NE to RT
        data_rot[:, cmps_rt.index('Z'), :] = data[:, cmps_ne.index('Z'), :]
    
    else:
        # rotate from RT to NE
        data_rot[:, cmps_ne.index('Z'), :] = data[:, cmps_rt.index('Z'), :]

    n_i = cmps_ne.index('N')
    e_i = cmps_ne.index('E')
    r_i = cmps_rt.index('R')
    t_i = cmps_rt.index('T')

    for event, slots in ws.fslots.items():
        if len(slots) == 0:
            continue
        
        if direction:
            # rotate from NE to RT
            ba = baz[event]

            for slot in slots:
                n = data[:, n_i, slot]
                e = data[:, e_i, slot]
                data_rot[:, r_i, slot] = - e * np.sin(ba) - n * np.cos(ba)
                data_rot[:, t_i, slot] = - e * np.cos(ba) + n * np.sin(ba)
        
        else:
            # rotate from RT to NE
            ba = 2 * np.pi - baz[event]

            for slot in slots:
                r = data[:, r_i, slot]
                t = data[:, t_i, slot]
                data_rot[:, n_i, slot] = - t * np.sin(ba) - r * np.cos(ba)
                data_rot[:, e_i, slot] = - t * np.cos(ba) + r * np.sin(ba)
            
    return data_rot
