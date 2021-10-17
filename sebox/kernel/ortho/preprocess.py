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
    # assign frequencies for all kernels
    if node.inherit_kernel is None:
        node.add(_prepare_frequencies)

    # create or link individual kernel encodings
    enc = node.add(name='encode', concurrent=True)

    for iker, cwd in enumerate(dirs(node)):
        enc.add(_link_encoded if node.inherit_kernel else _encode, cwd, iker=iker, concurrent=True)

    # run mesher while encoding
    enc.add('solver.mesh', 'mesh')


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
        'seed_used': (node.iteration or 0) + (node.seed or 0),
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

    # assign frequency slots to events
    random.seed(enc['seed_used'])
    freq = _freq(enc)
    nfreq = len(freq)
    events = getevents()
    event_bands = {}
    nkl = node.nkernels or 1
    band_interval = max(1, int(round(nbands / (nfreq * nkl / len(events)))))
    fslots = []
    slots = []
    
    for _ in range(nkl):
        fslots.append({})
        slots.append(set(range(nfreq)))

    # get available frequency bands for each event (sumed over stations and components)
    for event in events:
        for s in fslots:
            s[event] = []

        event_bands[event] = getmeasurements(event, balance=True, noise=True).sum(axis=0).sum(axis=0)
    
    def find_slot(e: str, b: int):
        for i in range(b, min(b + band_interval, nbands)):
            # check if current band has trace
            if event_bands[e][i] < 1:
                continue
            
            # loop over frequency indices of current band
            for j in range(fincr):
                k = i * fincr + j

                for iker in range(nkl):
                    if k in slots[iker]:
                        slots[iker].remove(k)
                        fslots[iker][e].append(k)
                        return
    
    def has_slot():
        return any(len(s) for s in slots)

    while has_slot():
        for event in random.sample(events, len(events)):
            # find slots from different frequency sections
            for i in range(0, nbands, band_interval):
                find_slot(event, i)

                if not has_slot():
                    break

            if not has_slot():
                break
    
    for event in events:
            n = []
            for f in fslots:
                for sl in f[event]:
                    n.append(sl // 180)

            print(event, len(n))


    # get encoding parameters for individual kernels
    for iker, cwd in enumerate(dirs(node)):
        # CMTSOLUTION striing
        cmt = ''

        # catalog directory
        cdir = getdir()

        # encode events
        for event in fslots[iker]:
            lines = cdir.readlines(f'events/{event}')

            for idx in fslots[iker][event]:
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
        enc['fslots'] = fslots[iker]
        node.write(cmt, cwd + '/SUPERSOURCE')
        node.dump(enc, cwd + '/encoding.pickle')
    
    node.dump(fslots, 'e.pickle')
    exit()


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
