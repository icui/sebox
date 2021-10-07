from __future__ import annotations
import typing as tp

from sebox import Node


class Postprocess(Node):
    """A node to generate mesh."""
    # current iteration
    iteration: tp.Optional[int]
    
    # path to mesher directory
    path_mesh: str

    # path to kernels to be summed
    path_kernels: tp.List[str]

    # radius for smoothing kernels
    smooth_kernels: tp.Optional[tp.Union[float, tp.List[float]]]

    # radius for smoothing hessian
    smooth_hessian: tp.Optional[tp.Union[float, tp.List[float]]]

    # ratio between vertival and horizontal smoothing length
    smooth_vertical: tp.Optional[float]

    # increase smoothing length with PREM velocity (length *= (vp / vp0) ** smooth_depth)
    smooth_with_prem: tp.Optional[float]

    # names of the kernels to be smoothed
    kernel_names: tp.List[str]

    # names of the hessian to be smoothed
    hessian_names: tp.List[str]

    # apply source mask
    source_mask: bool

    # precondition threshold
    precondition: float


class Forward(Node):
    """A node to run forward simulation."""
    # path to CMTSOLUTION
    path_event: str

    # path to STATIONS
    path_stations: str

    # path to generated mesh
    path_mesh: str

    # path to model file (if self.path_mesh is None)
    path_model: str

    # save forward wavefield for adjoint computation
    save_forward: bool

    # use monochromatic source time function
    monochromatic_source: bool

    # simulation duration in minutes
    duration: float

    # time duration to reach steady state for source encoding
    transient_duration: float

    # geographical limit of the catalog (center_lat, center_lon, radius, max_epicentral_dist)
    catalog_boundary: tp.Tuple[float, float, float, float]


class Adjoint(Node):
    """A node to run adjiont simulation."""
    # path to adjoint source
    path_misfit: str

    # path to forward simulation directory
    path_forward: str


class Solver(Forward, Adjoint, Postprocess):
    """All solver configurations."""
