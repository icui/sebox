from __future__ import annotations
import typing as tp

from sebox import typing


class Kernel(typing.Kernel):
    """Source encoded kernel computation."""
    # number of kernel computations per iteration
    nkernels: tp.Optional[int]

    # number of kernels to randomize frequency per iteration
    nkernels_rng: tp.Optional[int]

    # alter global random seed
    seed: int

    # random seed actually used
    seed_used: int

    # index of current kernel
    iker: int

    # workspace with source encoding paremeters
    encoding: tp.Dict[int, Kernel]

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

    # attempt to unwrap phases
    unwrap_phase: bool
    
    # use double difference measurement
    double_difference: bool

    # global phase weighting
    phase_factor: float

    # global amplitude weighting
    amplitude_factor: float

    # skip computing adjoint kernels
    misfit_only: bool

    # taper adjoint source
    taper: tp.Optional[float]

    # normalize source magnitudes
    normalize_source: bool
