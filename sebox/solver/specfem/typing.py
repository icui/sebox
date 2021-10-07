import typing as tp

from sebox.typing import Solver


class Par_file(tp.TypedDict, total=False):
    """DATA/Par_file in specfem."""
    # 1 for forward simulation, 3 for adjoint simulation
    SIMULATION_TYPE: int

    # save forward wavefield
    SAVE_FORWARD: bool

    # use monochromatic source time function
    USE_MONOCHROMATIC_CMT_SOURCE: bool

    # simulation duration
    RECORD_LENGTH_IN_MINUTES: float

    # model name
    MODEL: str

    # use high order time scheme
    USE_LDDRK: bool

    # number of processors in XI direction
    NPROC_XI: int

    # number of processors in ETA direction
    NPROC_ETA: int

    # number of chunks
    NCHUNKS: int

    # compute steady state kernel for source encoded FWI
    STEADY_STATE_KERNEL: bool

    # steady state duration for source encoded FWI
    STEADY_STATE_LENGTH_IN_MINUTES: float

    # sponge absorbing boundary
    ABSORB_USING_GLOBAL_SPONGE: bool

    # center latitude of sponge
    SPONGE_LATITUDE_IN_DEGREES: float

    # center longitude of sponge
    SPONGE_LONGITUDE_IN_DEGREES: float

    # radius of the sponge
    SPONGE_RADIUS_IN_DEGREES: float


class Specfem(Solver):
    # specfem directory
    path_specfem: str

    # use LDDRK time scheme
    lddrk: bool
