from __future__ import annotations
import typing as tp

from sebox.utils.catalog import getdir

if tp.TYPE_CHECKING:
    from sebox import typing

    class Kernel(typing.Kernel):
        """Source encoded kernel computation."""
        # current iteration
        iteration: int

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

        # indices of lowest and highest frequency
        fidx: tp.Tuple[int, int]

        # simulation duration in minutes
        duration: float

        # time duration to reach steady state for source encoding
        transient_duration: float

        # length of a time step
        dt: float

        # period range
        period_range: tp.List[float]

        # number of frequencies in a frequency band
        frequency_increment: int


def rng(ws: Kernel):
    if isinstance(ws.nkernels_rng, int):
        rng_iter = ws.nkernels_rng
    
    else:
        rng_iter = ws.nkernels or 1
    
    return (ws.iteration or 0) * rng_iter + (ws.seed or 0) + (ws.iker or 0)


def _prepare_frequencies(ws: Kernel):
    import numpy as np
    from scipy.fft import fftfreq

    if ws.fidx:
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
    ws.fidx = imin, imax

    ws.write('\n'.join([
        f'time step length: {ws.dt}',
        f'frequency step length: {ws.df:.2e}',
        f'transient state: {nt_ts} steps, {nt_ts * ws.dt / 60:.2f}min',
        f'steady state duration: {nt_se}steps, {nt_se * ws.dt / 60:.2f}min',
        f'frequency slots: {nbands} x {fincr}',
        f'frequency indices: [{imin}, {imax}] x {kf}',
        f'period range: [{1/freq[imax-1]:.2f}s, {1/freq[imin]:.2f}s]',
        f'rng seed: {rng(ws)}'
        ''
    ]), 'encoding.log')


def kernel(ws: Kernel):
    """Compute kernels."""
    # determine frequency range
    ws.add(_prepare_frequencies)

    if ws.path_encoded:
        pass

    else:
        cdir = getdir()
