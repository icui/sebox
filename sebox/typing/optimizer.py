from __future__ import annotations
import typing as tp

from sebox import Workspace
from .kernel import Kernel


class Optimizer(Workspace):
    """Gradient optimization."""
    # total number of iterations
    niters: int

    # current iteration
    iteration: int

    # current model
    path_model: str
