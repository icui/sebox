from __future__ import annotations
import typing as tp
import random

from sebox.utils.catalog import getdir, getevents, getmeasurements

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


def prepare_frequencies(ws: Kernel):
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

    # save results to parent (kernel) workspace
    ws.nt_ts = nt_ts
    ws.nt_se = nt_se
    ws.df = df
    ws.kf = kf
    ws.nbands = nbands
    ws.imin = imin
    ws.imax = imax
    
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

    # save and print source encoding parameters
    ws.write('\n'.join([
        f'time step length: {ws.dt}',
        f'frequency step length: {ws.df:.2e}',
        f'transient state: {nt_ts} steps, {nt_ts * ws.dt / 60:.2f}min',
        f'steady state duration: {nt_se}steps, {nt_se * ws.dt / 60:.2f}min',
        f'frequency slots: {nbands} x {fincr}',
        f'frequency indices: [{imin}, {imax}] x {kf}',
        f'period range: [{1/freq[imax-1]:.2f}s, {1/freq[imin]:.2f}s]',
        f'random seed: {getseed(ws)}'
        ''
    ]), 'encoding.log')


def link_observed(ws: Kernel):
    pass


def encode_events(ws: Kernel):
    # load catalog
    cmt = ''
    cdir = getdir()

    # randomize frequency
    freq = getfreq(ws)
    fslots = ws.fslots = {}
    events = getevents()
    nbands = ws.nbands_used
    fincr = ws.frequency_increment
    slots = set(range(nbands * fincr))

    # set random seed
    random.seed(getseed(ws))

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

    # save fslots and CMTSOLUTION
    ws.dump(ws.fslots, 'fslots.pickle')
    ws.write(cmt, 'SUPERSOURCE')