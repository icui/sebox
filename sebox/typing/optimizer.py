from __future__ import annotations
import typing as tp

from sebox import Workspace
from .search import Search


class Optimizer(Workspace):
    """Gradient optimization."""
    # total number of iterations
    niters: int

    # current iteration
    iteration: int

    # current model
    path_model: str

    # mesh of current model
    path_mesh: tp.Optional[str]

    # workspace for line search
    search: Search
