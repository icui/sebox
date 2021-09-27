import typing as tp

from sebox import typing

class Kernel(typing.Kernel):
    """Source encoded kernel computation."""
    # number of kernel computations per iteration
    nkernels: tp.Optional[int]

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
