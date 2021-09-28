from __future__ import annotations
import typing as tp

from sebox import root, Directory
from sebox.utils.catalog import getdir, getevents, getstations, getcomponents, getmeasurements

if tp.TYPE_CHECKING:
    from numpy import ndarray
    from .typing import Kernel


async def encode_obs(ws: Kernel):
    """Encode observed data."""
    stas = getstations()
    ws.cdir = getdir().path()
    ws.mkdir()
    await ws.mpiexec(_encode_obs, root.task_nprocs, arg=ws, arg_mpi=stas)


async def encode_diff(ws: Kernel):
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
    root.restore(ws)
    cdir = getdir()
    event_data = cdir.load('event_data.pickle')
    encoded = np.full([len(stas), 3, ws.imax - ws.imin], np.nan, dtype=complex)
    cmps = getcomponents()

    for event in getevents():
        # read event data
        data = cdir.load(f'ft_obs_p{root.task_nprocs}/{event}/{pid}.npy')

        if len(ws.fslots[event]) == 0:
            continue

        slots = np.array(ws.fslots[event])
        groups = slots // ws.frequency_increment
        hdur = event_data[event][-1]
        tshift = 1.5 * hdur
        
        # source time function of observed data and its frequency component
        stf = np.exp(-((t - tshift) / (hdur / 1.628)) ** 2) / np.sqrt(np.pi * (hdur / 1.628) ** 2)
        sff = _ft_obs(ws, stf)
        pshift = np.exp(2 * np.pi * 1j * freq * (ws.nt_ts * ws.dt - tshift)) / sff

        # record frequency components
        for i, sta in enumerate(stas):
            m = getmeasurements(event=event, station=sta)

            for j in range(3):
                idx = slots[np.squeeze(np.where(m[j, groups]))]
                encoded[i, j, idx] = data[i, j, idx] * pshift[idx]
    
    ws.dump(encoded, f'{pid}.pickle', mkdir=False)

    if 'II.OBN' in stas:
        i = stas.index('II.OBN')
        print('@', )
        print(encoded[i][2])


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
