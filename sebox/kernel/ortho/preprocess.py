from __future__ import annotations
import typing as tp
import random

from sebox import root
from sebox.utils.catalog import getdir, getevents, getstations, getcomponents, getmeasurements
from .ft import ft_obs

if tp.TYPE_CHECKING:
    from numpy import ndarray
    from .typing import Ortho


def preprocess(node: Ortho):
    """Determine encoding parameters and run mesher."""
    for iker in range(node.nkernels or 1):
        # create node for individual kernels
        node.add(_prepare_encoding, f'kl_{iker:02d}', iker=iker)

    # run mesher
    node.add('solver.mesh', 'mesh')


def _prepare_encoding(node: Ortho):
    """Prepare source encoding data."""
    if node.inherit_kernel:
        # link existing encoding node
        node.add(_link_encoded)

    else:
        # determine frequencies
        node.add(_prepare_frequencies)

        # create SUPERSOURCE
        node.add(_encode_events)

        # encode observed traces and diffs
        node.add(_encode, concurrent=True)


def _seed(node: Ortho):
    if isinstance(node.nkernels_rng, int):
        rng_iter = node.nkernels_rng
    
    else:
        rng_iter = node.nkernels or 1
    
    return (node.iteration or 0) * rng_iter + (node.seed or 0) + (node.iker or 0)


def _freq(node: Ortho) -> ndarray:
    from scipy.fft import fftfreq
    return fftfreq(node.nt_se, node.dt)[node.imin: node.imax]


def _link_encoded(node: Ortho):
    kl = node.inherit_kernel[1][node.iker] # type: ignore
    node.cp(node.rel(kl, 'SUPERSOURCE'))
    node.ln(node.rel(kl, 'enc_obs'))
    node.ln(node.rel(kl, 'enc_diff'))
    node.ln(node.rel(kl, 'enc_weight'))


def _prepare_frequencies(node: Ortho):
    import numpy as np
    from scipy.fft import fftfreq

    if node.fmax:
        return

    if node.duration <= node.transient_duration:
        raise ValueError('duration should be larger than transient_duration')

    # number of time steps to reach steady state
    nt_ts = int(round(node.transient_duration * 60 / node.dt))

    # number of time steps after steady state
    nt_se = int(round((node.duration - node.transient_duration) * 60 / node.dt))

    # frequency step
    df = 1 / nt_se / node.dt

    # factor for observed data
    kf = int(np.ceil(nt_ts / nt_se))

    # frequencies to be encoded
    freq = fftfreq(nt_se, node.dt)
    fincr = node.frequency_increment
    imin = int(np.ceil(1 / node.period_range[1] / df))
    imax = int(np.floor(1 / node.period_range[0] / df))
    nf = imax - imin + 1
    nbands = nf // fincr
    imax = imin + nbands * fincr
    nf = nbands * fincr
    
    # get number of frequency bands actually used (based on referency_velocity and smooth_kernels)
    if node.reference_velocity is not None and (rad := node.smooth_kernels):
        if isinstance(rad, list):
            rad = max(rad[1], rad[0] * rad[2] ** (node.iteration or 0))

        # exclude band where reference_volocity * period < smooth_radius
        for i in range(nbands):
            # compare smooth radius with the highest frequency of current band
            if node.reference_velocity / freq[(i + 1) * node.frequency_increment - 1] < rad:
                nbands_used = i
                break

    # save results to parent node
    encoding = {
        'df': df,
        'kf': kf,
        'nt_ts': nt_ts,
        'nt_se': nt_se,
        'nbands': nbands,
        'nbands_used': nbands_used,
        'imin': imin,
        'imax': imax,
        'seed_used': _seed(node)
    }

    node.parent.update(encoding)
    node.dump(encoding, 'encoding.pickle')


def _encode_events(node: Ortho):
    # load catalog
    cmt = ''
    cdir = getdir()

    # randomize frequency
    freq = _freq(node)
    fslots = {}
    events = getevents()
    nbands = node.nbands_used
    fincr = node.frequency_increment
    slots = set(range(nbands * fincr))

    # set random seed
    random.seed(node.seed_used)

    # get available frequency bands for each event (sumed over tations and components)
    event_bands = {}

    for event in events:
        fslots[event] = []
        event_bands[event] = getmeasurements(event, balance=True, noise=True).sum(axis=0).sum(axis=0)

    # fill frequency slots
    len_slots = 0

    while len(slots) > 0 and len(slots) != len_slots:
        # stop iteration if no slot is selected
        len_slots = len(slots)
        events = random.sample(events, len(events))

        for event in events:
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
            if node.normalize_source:
                mref = 1e25
                mmax = max(abs(float(lines[i].split()[-1])) for i in range(7, 13))
                
                for j in range(7, 13):
                    line = lines[j].split()
                    line[-1] = f'{(float(line[-1]) * mref / mmax):.6e}'
                    lines[j] = '           '.join(line)

            cmt += '\node'.join(lines)

            if cmt[-1] != '\node':
                cmt += '\node'

    # save frequency slots and encoded source
    node.dump(fslots, 'fslots.pickle')
    node.write(cmt, 'SUPERSOURCE')


def _encode(node: Ortho):
    stas = getstations()
    node.mkdir('enc_weight')
    node.add_mpi(_enc_obs, arg=node, arg_mpi=stas, cwd='enc_obs')
    node.add_mpi(_enc_diff, arg=node, arg_mpi=stas, cwd='enc_diff')


def _enc_obs(node: Ortho, stas: tp.List[str]):
    import numpy as np
    from sebox.mpi import pid

    # kernel configuration
    nt = node.kf * node.nt_se
    t = np.linspace(0, (nt - 1) * node.dt, nt)
    freq = _freq(node)

    # data from catalog
    root.restore(node)
    cdir = getdir()
    cmps = getcomponents()
    event_data = cdir.load('event_data.pickle')
    encoded = np.full([len(stas), 3, node.imax - node.imin], np.nan, dtype=complex)

    # get global station index
    sidx = []
    stations = getstations()

    for sta in stas:
        sidx.append(stations.index(sta))
    
    sidx = np.array(sidx)

    for event in getevents():
        # read event data
        data = cdir.load(f'ft_obs_p{root.task_nprocs}/{event}/{pid}.npy')
        slots = node.fslots[event]
        hdur = event_data[event][-1]
        tshift = 1.5 * hdur
        
        # source time function of observed data and its frequency component
        stf = np.exp(-((t - tshift) / (hdur / 1.628)) ** 2) / np.sqrt(np.pi * (hdur / 1.628) ** 2)
        sff = ft_obs(node, stf)
        pff = np.exp(2 * np.pi * 1j * freq * (node.nt_ts * node.dt - tshift)) / sff

        # record frequency components
        for idx in slots:
            group = idx // node.frequency_increment
            pshift = pff[idx]

            # phase shift due to the measurement of observed data
            for j, cmp in enumerate(cmps):
                m = getmeasurements(event, None, cmp, group)[sidx]
                i = np.squeeze(np.where(m))
                encoded[i, j, idx] = data[i, j, idx] * pshift
    
    node.dump(encoded, f'enc_obs/{pid}.npy', mkdir=False)


def _enc_diff(node: Ortho, stas: tp.List[str]):
    """Encode diff data."""
    import numpy as np
    from sebox.mpi import pid

    # data from catalog
    root.restore(node)
    cdir = getdir()
    cmps = getcomponents()
    encoded = np.full([len(stas), 3, node.imax - node.imin], np.nan)
    weight = np.full([len(stas), 3, node.imax - node.imin], np.nan)

    # get global station index
    sidx = []
    stations = getstations()

    for sta in stas:
        sidx.append(stations.index(sta))
    
    sidx = np.array(sidx)

    for event in getevents():
        # read event data
        data = cdir.load(f'ft_diff_p{root.task_nprocs}/{event}/{pid}.npy')
        slots = node.fslots[event]

        # record frequency components
        for idx in slots:
            group = idx // node.frequency_increment

            # phase shift due to the measurement of observed data
            for j, cmp in enumerate(cmps):
                w = getmeasurements(event, None, cmp, group, True, True, True, True)[sidx]
                i = np.squeeze(np.where(w > 0))
                encoded[i, j, idx] = data[i, j, idx]
                weight[i, j, idx] = w[i]
    
    node.dump(encoded, f'enc_diff/{pid}.npy', mkdir=False)
    node.dump(weight, f'enc_weight/{pid}.npy', mkdir=False)
