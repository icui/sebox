from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir, getevents, getcomponents, getstations, getmeasurements

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
    stas = getstations()
    ws.cdir = getdir().path()
    ws.mkdir()
    ws.mkdir('../enc_weight')
    await ws.mpiexec(_encode_diff, root.task_nprocs, arg=ws, arg_mpi=stas)


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
    cmps = getcomponents()
    event_data = cdir.load('event_data.pickle')
    encoded = np.full([len(stas), 3, ws.imax - ws.imin], np.nan, dtype=complex)

    # get global station index
    sidx = []
    stations = getstations()

    for sta in stas:
        sidx.append(stations.index(sta))
    
    sidx = np.array(sidx)

    for event in getevents():
        # read event data
        data = cdir.load(f'ft_obs_p{root.task_nprocs}/{event}/{pid}.npy')
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
            for j, cmp in enumerate(cmps):
                m = getmeasurements(event, None, cmp, group)[sidx]
                i = np.squeeze(np.where(m))
                encoded[i, j, idx] = data[i, j, idx] * pshift
    
    ws.dump(encoded, f'{pid}.pickle', mkdir=False)


def _encode_diff(ws: Kernel, stas: tp.List[str]):
    """Encode diff data."""
    import numpy as np
    from sebox.mpi import pid

    # data from catalog
    root.restore(ws)
    cdir = getdir()
    cmps = getcomponents()
    encoded = np.full([len(stas), 3, ws.imax - ws.imin], np.nan)
    weight = np.full([len(stas), 3, ws.imax - ws.imin], np.nan)

    # get global station index
    sidx = []
    stations = getstations()

    for sta in stas:
        sidx.append(stations.index(sta))
    
    sidx = np.array(sidx)

    for event in getevents():
        # read event data
        data = cdir.load(f'ft_diff_p{root.task_nprocs}/{event}/{pid}.npy')
        slots = ws.fslots[event]

        # record frequency components
        for idx in slots:
            group = idx // ws.frequency_increment

            # phase shift due to the measurement of observed data
            for j, cmp in enumerate(cmps):
                w = getmeasurements(event, None, cmp, group, True, True, True, True)[sidx]
                i = np.squeeze(np.where(w > 0))
                encoded[i, j, idx] = data[i, j, idx]
                weight[i, j, idx] = w[i]
    
    ws.dump(encoded, f'{pid}.pickle', mkdir=False)
    ws.dump(weight, f'../enc_weight/{pid}.pickle', mkdir=False)
    
    if 'II.OBN' in stas:
        i = stas.index('II.OBN')
        print('@', )
        print(encoded[i][2], weight[i][2])

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
