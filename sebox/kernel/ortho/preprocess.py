from __future__ import annotations
import typing as tp
import random

from sebox import root
from sebox.utils.catalog import getdir, getevents, getstations, getcomponents, getmeasurements
from .ft import ft_obs

if tp.TYPE_CHECKING:
    from numpy import ndarray
    from .typing import Kernel


def getseed(ws: Kernel):
    """Random seed based on kernel configuration."""
    if isinstance(ws.nkernels_rng, int):
        rng_iter = ws.nkernels_rng
    
    else:
        rng_iter = ws.nkernels or 1
    
    return (ws.iteration or 0) * rng_iter + (ws.seed or 0) + (ws.iker or 0)


def getfreq(ws: Kernel) -> ndarray:
    """Frequencies used for encoding."""
    from scipy.fft import fftfreq
    return fftfreq(ws.nt_se, ws.dt)[ws.imin: ws.imax]


def prepare_encoding(ws: Kernel):
    """Prepare source encoding data."""
    if ws.path_encoded:
        # link existing encoding workspace
        ws.add(_link_observed)
    
    else:
        # determine frequencies
        ws.add(_prepare_frequencies)

        # create SUPERSOURCE
        ws.add(_encode_events)

        # encode observed traces and diffs
        ws.add(_encode, concurrent=True)


def _link_observed(ws: Kernel):
    pass


def _prepare_frequencies(ws: Kernel):
    import numpy as np
    from scipy.fft import fftfreq

    if ws.fmax:
        return

    if ws.duration <= ws.transient_duration:
        raise ValueError('duration should be larger than transient_duration')

    # number of time steps to reach steady state
    nt_ts = int(round(ws.transient_duration * 60 / ws.dt))

    # number of time steps after steady state
    nt_se = int(round((ws.duration - ws.transient_duration) * 60 / ws.dt))

    # frequency step
    df = 1 / nt_se / ws.dt

    # factor for observed data
    kf = int(np.ceil(nt_ts / nt_se))

    # frequencies to be encoded
    freq = fftfreq(nt_se, ws.dt)
    fincr = ws.frequency_increment
    imin = int(np.ceil(1 / ws.period_range[1] / df))
    imax = int(np.floor(1 / ws.period_range[0] / df))
    nf = imax - imin + 1
    nbands = nf // fincr
    imax = imin + nbands * fincr
    nf = nbands * fincr
    
    # get number of frequency bands actually used (based on referency_velocity and smooth_kernels)
    if ws.reference_velocity is not None and (smooth := ws.smooth_kernels):
        if isinstance(smooth, list):
            smooth = max(smooth[1], smooth[0] * smooth[2] ** (ws.iteration or 0))

        # exclude band where reference_volocity * period < smooth_radius
        for i in range(nbands):
            # compare smooth radius with the highest frequency of current band
            if ws.reference_velocity / freq[(i + 1) * ws.frequency_increment - 1] < smooth:
                # parent.nbands_used = i #FIXME
                ws.nbands_used = nbands
                break

    # save results to parent workspace
    ws.parent.update({
        'df': df,
        'kf': kf,
        'nt_ts': nt_ts,
        'nt_se': nt_se,
        'nbands': nbands,
        'nbands_used': nbands,
        'imin': imin,
        'imax': imax,
        'seed_used': getseed(ws)
    })


def _encode_events(ws: Kernel):
    # load catalog
    cmt = ''
    cdir = getdir()

    # randomize frequency
    freq = getfreq(ws)
    fslots = {}
    events = getevents()
    nbands = ws.nbands_used
    fincr = ws.frequency_increment
    slots = set(range(nbands * fincr))

    # set random seed
    random.seed(ws.seed_used)

    # get available frequency bands for each event (sumed over tations and components)
    event_bands = {}

    for event in events:
        event_bands[event] = getmeasurements(event, balance=True, noise=True).sum(axis=0).sum(axis=0)

    # fill frequency slots
    len_slots = 0

    while len(slots) > 0 and len(slots) != len_slots:
        # stop iteration if no slot is selected
        len_slots = len(slots)
        events = random.sample(events, len(events))

        for event in events:
            if event not in fslots:
                fslots[event] = []

            idx = None

            for i in range(nbands):
                m = event_bands[event][i]

                if m < 1:
                    # no available trace in current band
                    continue
                
                # loop over frequency indices of current band
                for j in range(fincr):
                    k = i * fincr + j

                    if k in slots:
                        # assign slot to current event
                        idx = k
                        slots.remove(k)
                        fslots[event].append(k)
                        break
                
                if idx is not None:
                    break
            
            if len(slots) == 0:
                break

    # encode events
    for event in fslots:
        lines = cdir.readlines(f'events/{event}')

        for idx in fslots[event]:
            f0 = freq[idx]
            lines[2] = 'time shift:           0.0000'
            lines[3] = f'half duration:{" " * (9 - len(str(int(1 / f0))))}{1/f0:.4f}'

            # normalize sources to the same order of magnitude
            if ws.normalize_source:
                mref = 1e25
                mmax = max(abs(float(lines[i].split()[-1])) for i in range(7, 13))
                
                for j in range(7, 13):
                    line = lines[j].split()
                    line[-1] = f'{(float(line[-1]) * mref / mmax):.6e}'
                    lines[j] = '           '.join(line)

            cmt += '\n'.join(lines)

            if cmt[-1] != '\n':
                cmt += '\n'

    # save frequency slots and encoded source
    ws.parent.fslots = fslots
    ws.write(cmt, 'SUPERSOURCE')


def _encode(ws: Kernel):
    ws.add('enc_obs', _enc_obs)
    ws.add('enc_diff', _enc_diff)


async def _enc_obs(ws: Kernel):
    ws.mkdir()
    await ws.mpiexec(_encode_obs, arg=ws, arg_mpi=getstations())


async def _enc_diff(ws: Kernel):
    ws.mkdir()
    ws.mkdir('enc_weight')
    await ws.mpiexec(_encode_diff, arg=ws, arg_mpi=getstations())


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
        sff = ft_obs(ws, stf)
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
    
    ws.dump(encoded, f'{pid}.npy', mkdir=False)


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
    
    ws.dump(encoded, f'{pid}.npy', mkdir=False)
    ws.dump(weight, f'../enc_weight/{pid}.pickle', mkdir=False)
