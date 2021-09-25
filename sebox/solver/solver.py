from __future__ import annotations
import typing as tp

from sebox import Task, Workspace


if tp.TYPE_CHECKING:
    class Solver(tp.Protocol):
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


def add_mesh(ws: Workspace, name: str, *,
    path_model: tp.Optional[str] = None) -> Mesh:
    """A a workspace to run mesher."""
    solver = tp.cast(Mesh, ws.add(name, False, ('sebox.solver', 'mesh')))

    if path_model is not None:
        solver.path_model = path_model
    
    return solver


def add_forward(ws: Workspace, name: str, *,
    path_event: tp.Optional[str] = None,
    path_stations: tp.Optional[str] = None,
    path_mesh: tp.Optional[str] = None,
    path_model: tp.Optional[str] = None) -> Forward:
    """A a workspace to run forward simulation."""
    solver = tp.cast(Forward, ws.add(name, False, ('sebox.solver', 'forward')))

    if path_event is not None:
        solver.path_event = path_event
    
    if path_stations is not None:
        solver.path_stations = path_stations
    
    if path_mesh is not None:
        solver.path_mesh = path_mesh
    
    if path_model is not None:
        solver.path_model = path_model
    
    return solver


def add_adjoint(ws: Workspace, name: str, *,
    path_misfit: tp.Optional[str] = None,
    path_forward: tp.Optional[str] = None) -> Adjoint:
    """A a workspace to run adjoint simulation."""
    solver = tp.cast(Adjoint, ws.add(name, False, ('sebox.solver', 'adjoint')))

    if path_misfit is not None:
        solver.path_misfit = path_misfit
    
    if path_forward is not None:
        solver.path_forward = path_forward
    
    return solver
