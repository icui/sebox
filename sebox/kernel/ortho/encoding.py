from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir, getevents, getstations, getcomponents

if tp.TYPE_CHECKING:
    from numpy import ndarray
    from .typing import Kernel


async def encode_obs(ws: Kernel):
    """Encode observed data."""
    stas = getstations()
    await ws.mpiexec(_encode_obs, root.task_nprocs, arg=ws, arg_mpi=stas)


def encode_diff(ws: Kernel):
    """Encode diff data."""


def _encode_obs(ws: Kernel, stas: tp.List[str]):
    import numpy as np
    from sebox.mpi import pid
    from .preprocess import getfreq

    # kernel configuration
    nt = ws.kf * ws.nt_se
    t = np.linspace(0, (nt - 1) * ws.dt, nt)
    freq = getfreq(ws)

    # data from catalog
    cdir = getdir()
    event_data = cdir.load('event_data.pickle')
    encoded = np.zeros([len(stas), ws.imax - ws.imin])

    for event in getevents():
        # read event data
        data = cdir.load(f'ft_obs_p{root.task_nprocs}/{event}/{pid}.pickle')
        slots = ws.fslots[event]
        hdur = event_data[event][-1]
        tshift = 1.5 * hdur
        
        # source time function of observed data and its frequency component
        stf = np.exp(-((t - tshift) / (hdur / 1.628)) ** 2) / np.sqrt(np.pi * (hdur / 1.628) ** 2)
        sff = _ft_obs(ws, stf)
        pff = np.exp(2 * np.pi * 1j * freq * (ws.nt_ts * ws.dt - tshift)) / sff

        # record frequency components
        for idx in slots:
            group = idx // ws.frequency_increment
            pshift = pff[idx]

            # phase shift due to the measurement of observed data
            for i, sta in enumerate(stas):
                for cmp in getcomponents(event=event, station=sta, group=group):
                    encoded[i][cmp][idx] = data[i][idx] * pshift
        
        break
    
    ws.dump(encoded, f'{pid}.pickle')


def _ft_syn(ws: Kernel, data: ndarray):
    from scipy.fft import fft
    return fft(data[ws.nt_ts: ws.nt_ts + ws.nt_se])[ws.imin: ws.imax]


def _ft_obs(ws: Kernel, data: ndarray):
    from scipy.fft import fft

    if (nt := ws.kf * ws.nt_se) > len(data):
        # expand observed data with zeros
        data = np.concatenate([data, np.zeros(nt - len(data))]) # type: ignore
    
    else:
        data = data[:nt]
    
    return fft(data)[::ws.kf][ws.imin: ws.imax]
