from __future__ import annotations
import typing as tp
import random

from sebox.utils.catalog import getdir, getevents, getmeasurements, merge_stations

if tp.TYPE_CHECKING:
    from sebox import typing
    from numpy import ndarray

    class Kernel(typing.Kernel):
        """Source encoded kernel computation."""
        # number of kernel computations per iteration
        nkernels: int

        # number of kernels to randomize frequency per iteration
        nkernels_rng: tp.Optional[int]

        # alter global randomization
        seed: int

        # index of current kernel
        iker: tp.Optional[int]

        # path to encoded observed traces
        path_encoded: tp.Optional[str]

        # time duration to reach steady state for source encoding
        transient_duration: float

        # number of frequencies in a frequency band
        frequency_increment: int

        # index of lowest frequency
        imin: int

        # index of highest frequency
        imax: int

        # frequency interval
        df: float

        # frequency step for observed traces
        kf: int

        # number of frequency bands
        nbands: int

        # number of frequency bands actually used
        nbands_used: int

        # frequency slots assigned to events
        fslots: tp.Dict[str, tp.List[int]]

        # number of time steps in transient state
        nt_ts: int

        # number of time steps in stationary state
        nt_se: int

        # determine frequency bands to by period * reference_velocity = smooth_radius
        reference_velocity: tp.Optional[float]

        # radius for smoothing kernels
        smooth_kernels: tp.Optional[tp.Union[float, tp.List[float]]]


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


def kernel(ws: Kernel):
    """Compute kernels."""
    _compute(ws, False)


def misfit(ws: Kernel):
    """Compute misfit."""
    _compute(ws, True)


def _compute(ws: Kernel, misfit_only: bool):
    # determine frequency range
    ws.add(_prepare_frequencies)

    if ws.path_encoded:
        # link encoded observed data
        ws.add(_link_observed)
    
    else:
        # encode events
        ws.add(_encode_events)


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

    # save and print source encoding parameters
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
            smooth = max(smooth[1], smooth[0] * smooth[2] ** ws.iteration)

        # exclude band where reference_volocity * period < smooth_radius
        for i in range(ws.nbands):
            # compare smooth radius with the highest frequency of current band
            if ws.reference_velocity / freq[(i + 1) * ws.frequency_increment - 1] < smooth:
                ws.nbands_used = i
                break

    # save source encoding parameters to kernel directory
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


def _link_observed(ws: Kernel):
    pass


def _encode_events(ws: Kernel):
    # load catalog
    cmt = ''
    cdir = getdir()

    # merge stations into a single station file
    merge_stations(cdir.subdir('stations'), ws.path('SUPERSTATION'), True)

    # randomize frequency
    freq = getfreq(ws)
    fslots = ws.fslots = {}
    events = getevents()
    nbands = ws.nbands_used
    fincr = ws.frequency_increment
    slots = set(range(nbands * fincr))

    # set random seed
    random.seed(getseed(ws))

    # fill frequency slots
    len_slots = 0

    while len(slots) > 0 and len(slots) != len_slots:
        # stop iteration if no slot is selected
        len_slots = len(slots)
        events = random.sample(events, len(events))

        for event in events:
            if event not in fslots:
                fslots[event] = []

            # selected frequency bands of current event (sumed over tations and components)
            bands = getmeasurements(event, balance=True, noise=True).sum(axis=0).sum(axis=0)
            idx = None

            for i in range(nbands):
                m = bands[i]

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
