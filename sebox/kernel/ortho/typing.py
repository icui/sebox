from __future__ import annotations
import typing as tp

from sebox.typing import Kernel


class Encoding(tp.TypedDict):
    """Source encoding parameters."""
    # time step
    dt: float
    
    # index of lowest frequency
    imin: int

    # index of highest frequency
    imax: int

    # frequency interval
    df: float

    # frequency step for observed traces
    kf: int

    # number of time steps in transient state
    nt_ts: int

    # number of time steps in stationary state
    nt_se: int

    # random seed actually used
    seed_used: int

    # number of frequency bands actually used
    nbands_used: int

    # number of frequencies actually used
    nfreq: int

    # frequency slots assigned to events
    fslots: tp.Dict[str, tp.List[int]]

    # number of frequencies in a frequency band
    frequency_increment: int
    
    # use double difference measurement
    double_difference: bool

    # global phase weighting
    phase_factor: float

    # global amplitude weighting
    amplitude_factor: float

    # taper adjoint source
    taper: tp.Optional[float]

    # attempt to unwrap phases
    unwrap_phase: bool

    # normalize source magnitudes
    normalize_source: bool

    # divide phase difference by frequency
    normalize_frequency: bool

    # encode all frequencies
    test_encoding: bool


class Ortho(Kernel):
    """Source encoded kernel computation."""
    # number of kernel computations per iteration
    nkernels: tp.Optional[int]

    # alter global random seed
    seed: tp.Optional[int]

    # index of current kernel
    iker: int

    # number of frequency bands
    nbands: int

    # determine frequency bands to by period * reference_velocity = smooth_radius
    reference_velocity: tp.Optional[float]

    # radius for smoothing kernels
    smooth_kernels: tp.Optional[tp.Union[float, tp.List[float]]]

    # number of frequencies in a frequency band
    frequency_increment: int
    
    # use double difference measurement
    double_difference: bool

    # global phase weighting
    phase_factor: float

    # global amplitude weighting
    amplitude_factor: float

    # taper adjoint source
    taper: tp.Optional[float]

    # attempt to unwrap phases
    unwrap_phase: bool

    # normalize source magnitudes
    normalize_source: bool

    # divide phase difference by frequency
    normalize_frequency: bool

    # encode all frequencies
    test_encoding: bool

    # check the trace orthogonality of a specific event
    test_events: tp.List[str]

    # bury the stations (depth in km) if test_events is enabled
    test_bury: float
