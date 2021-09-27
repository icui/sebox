from __future__ import annotations
import typing as tp

from sebox import Task, Workspace


class SolverModule(tp.Protocol):
    """Required functions in a solver module."""
    # generate mesh
    mesh: Task[Mesh]

    # forward simulation
    forward: Task[Forward]

    # adjoint simulation
    adjoint: Task[Adjoint]


class Mesh(Workspace):
    """A workspace to generate mesh."""
    # path to model file
    path_model: str


class Forward(Workspace):
    """A workspace to run forward simulation."""
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


class Adjoint(Workspace):
    """A workspace to run adjiont simulation."""
    # path to adjoint source
    path_misfit: str

    # path to forward simulation directory
    path_forward: str
