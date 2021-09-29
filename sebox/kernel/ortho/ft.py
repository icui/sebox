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
    ws.mkdir(ws.rel(ws.path_output))
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
        ws.dump(data_rtz, ws.rel(f'{ws.path_output}/{pid}.npy'), mkdir=False)

def rotate_frequencies(ws: Kernel, data: ndarray, cmps_ne: tp.Tuple[str, str, str], direction: bool):
    import numpy as np
    from sebox.mpi import pid

    cmps_rt = getcomponents()
    data_rot = np.zeros(data.shape, dtype=complex)
    baz = getdir().load(f'baz_p{root.task_nprocs}/{pid}.pickle')

    if direction:
        # rotate from NE to RT
        cmps_from = cmps_ne
        cmps_to = cmps_rt
        c1, c2 = cmps_ne.index('N'), cmps_ne.index('E')
        c3, c4 = cmps_rt.index('R'), cmps_rt.index('T')
    
    else:
        # rotate from RT to NE
        cmps_from = cmps_rt
        cmps_to = cmps_ne
        c1, c2 = cmps_rt.index('R'), cmps_rt.index('T')
        c3, c4 = cmps_ne.index('N'), cmps_ne.index('E')

    data_rot[:, cmps_from.index('Z'), :] = data[:, cmps_to.index('Z'), :]

    for event, slots in ws.fslots.items():
        if len(slots) == 0:
            continue
        
        ba = baz[event] if direction else 2 * np.pi - baz[event]

        for slot in slots:
            a = data[:, c1, slot]
            b = data[:, c2, slot]
            data_rot[:, c3, slot] = - b * np.sin(ba) - a * np.cos(ba)
            data_rot[:, c4, slot] = - b * np.cos(ba) + a * np.sin(ba)
            
    return data_rot
