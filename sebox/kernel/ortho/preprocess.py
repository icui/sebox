from __future__ import annotations
import typing as tp
import random

from sebox import root
from sebox.utils.catalog import getdir, getevents, getstations, getcomponents, getmeasurements
from .ft import ft_obs
from .main import dirs

if tp.TYPE_CHECKING:
    from numpy import ndarray
    from .typing import Ortho, Encoding


def preprocess(node: Ortho):
    """Determine encoding parameters and run mesher."""
    for iker, cwd in enumerate(dirs(node)):
        # create node for individual kernels
        node.add(_prepare_encoding, cwd, iker=iker)

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

        # encode observed traces and diffs
        node.add(_encode, concurrent=True)


def _seed(node: Ortho):
    if isinstance(node.nkernels_rng, int):
        rng_iter = node.nkernels_rng
    
    else:
        rng_iter = node.nkernels or 1
    
    return (node.iteration or 0) * rng_iter + (node.seed or 0) + (node.iker or 0)


def _freq(enc: Encoding) -> ndarray:
    from scipy.fft import fftfreq
    return fftfreq(enc['nt_se'], enc['dt'])[enc['imin']: enc['imax']]


def _link_encoded(node: Ortho):
    cwd = node.inherit_kernel.path(f'kl_{node.iker:02d}')
    node.cp(node.rel(cwd, 'SUPERSOURCE'))
    node.ln(node.rel(cwd, 'enc_obs'))
    node.ln(node.rel(cwd, 'enc_diff'))
    node.ln(node.rel(cwd, 'enc_weight'))
    node.ln(node.rel(cwd, 'encoding.pickle'))


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
    nbands_used = nbands
    
    if node.reference_velocity is not None and (rad := node.smooth_kernels):
        if isinstance(rad, list):
            rad = max(rad[1], rad[0] * rad[2] ** (node.iteration or 0))

        # exclude band where reference_volocity * period < smooth_radius
        for i in range(nbands):
            # compare smooth radius with the highest frequency of current band
            if node.reference_velocity / freq[(i + 1) * node.frequency_increment - 1] < rad:
                nbands_used = i
                break
    
    seed_used = _seed(node)

    # save encoding parameters
    enc: Encoding = {
        'dt': node.dt,
        'df': df,
        'kf': kf,
        'nt_ts': nt_ts,
        'nt_se': nt_se,
        'imin': imin,
        'imax': imax,
        'nbands_used': nbands_used,
        'seed_used': seed_used,
        'fslots': {},
        'frequency_increment': node.frequency_increment,
        'double_difference': node.double_difference,
        'phase_factor': node.phase_factor,
        'amplitude_factor': node.amplitude_factor,
        'taper': node.taper,
        'unwrap_phase': node.unwrap_phase,
        'normalize_source': node.normalize_source,
        'normalize_frequency': node.normalize_frequency
    }

    node.write(_encode_events(enc, node.band_interval), 'SUPERSOURCE')
    node.dump(enc, 'encoding.pickle')
    exit()


def _encode_events(enc: Encoding, band_interval: int):
    # load catalog
    cmt = ''
    cdir = getdir()

    # randomize frequency
    freq = _freq(enc)
    events = getevents()
    fincr = enc['frequency_increment']
    fslots = enc['fslots']
    nbands = enc['nbands_used']
    slots = set(range(nbands * fincr))

    # set random seed
    random.seed(enc['seed_used'])

    # get available frequency bands for each event (sumed over tations and components)
    event_bands = {}

    for event in events:
        fslots[event] = []
        event_bands[event] = getmeasurements(event, balance=True, noise=True).sum(axis=0).sum(axis=0)

    len_slots = 0

    while len(slots) > 0 and len(slots) != len_slots:
        # stop iteration if no slot is selected
        len_slots = len(slots)
        events = random.sample(events, len(events))

        for event in events:
            for l in range(0, nbands, band_interval or nbands):
                for i in range(l, min(l + (band_interval or nbands), nbands)):
                    idx = None
                    m = event_bands[event][i]

                    if m < 1:
                        # no available trace in current band
                        continue
                    
                    # loop over frequency indices of current band
                    for j in range(fincr):
                        k = i * fincr + j

                        if k in slots:
                            # slot found in current band
                            idx = k
                            slots.remove(k)
                            fslots[event].append(k)
                            break
                    
                    if idx is not None:
                        # stop if a slot is found
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
            if enc['normalize_source']:
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
    return cmt


def _encode(node: Ortho):
    stas = getstations()
    node.mkdir('enc_weight')
    enc = node.load('encoding.pickle')
    node.add_mpi(_enc_obs, arg=enc, arg_mpi=stas, cwd='enc_obs')
    node.add_mpi(_enc_diff, arg=enc, arg_mpi=stas, cwd='enc_diff')


def _enc_obs(enc: Encoding, stas: tp.List[str]):
    import numpy as np

    # kernel configuration
    nt = enc['kf'] * enc['nt_se']
    t = np.linspace(0, (nt - 1) * enc['dt'], nt)
    freq = _freq(enc)

    # data from catalog
    cdir = getdir()
    cmps = getcomponents()
    event_data = cdir.load('event_data.pickle')
    encoded = np.full([len(stas), 3, enc['imax'] - enc['imin']], np.nan, dtype=complex)

    # get global station index
    sidx = []
    stations = getstations()

    for sta in stas:
        sidx.append(stations.index(sta))
    
    sidx = np.array(sidx)

    for event in getevents():
        # read event data
        data = cdir.load(f'ft_obs_p{root.task_nprocs}/{event}/{root.mpi.pid}.npy')
        slots = enc['fslots'][event]
        hdur = event_data[event][-1]
        tshift = 1.5 * hdur
        
        # source time function of observed data and its frequency component
        stf = np.exp(-((t - tshift) / (hdur / 1.628)) ** 2) / np.sqrt(np.pi * (hdur / 1.628) ** 2)
        sff = ft_obs(enc, stf)

        # phase difference from source time function
        pff = np.exp(2 * np.pi * 1j * freq * (enc['nt_ts'] * enc['dt'] - tshift)) / sff

        # record frequency components
        for idx in slots:
            group = idx // enc['frequency_increment']
            pshift = pff[idx]

            for j, cmp in enumerate(cmps):
                m = getmeasurements(event, None, cmp, group)[sidx]
                i = np.squeeze(np.where(m))
                encoded[i, j, idx] = data[i, j, idx] * pshift
    
    root.mpi.mpidump(encoded)


def _enc_diff(enc: Encoding, stas: tp.List[str]):
    """Encode diff data."""
    import numpy as np

    # data from catalog
    cdir = getdir()
    cmps = getcomponents()
    encoded = np.full([len(stas), 3, enc['imax'] - enc['imin']], np.nan)
    weight = np.full([len(stas), 3, enc['imax'] - enc['imin']], np.nan)

    # get global station index
    sidx = []
    stations = getstations()

    for sta in stas:
        sidx.append(stations.index(sta))
    
    sidx = np.array(sidx)

    for event in getevents():
        # read event data
        data = cdir.load(f'ft_diff_p{root.task_nprocs}/{event}/{root.mpi.pid}.npy')
        slots = enc['fslots'][event]

        # record frequency components
        for idx in slots:
            group = idx // enc['frequency_increment']

            for j, cmp in enumerate(cmps):
                w = getmeasurements(event, None, cmp, group, True, True, True, True)[sidx]
                i = np.squeeze(np.where(w > 0))
                encoded[i, j, idx] = data[i, j, idx]
                weight[i, j, idx] = w[i]
    
    root.mpi.mpidump(encoded)
    root.mpi.mpidump(weight, '../enc_weight')
