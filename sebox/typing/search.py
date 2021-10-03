from __future__ import annotations
import typing as tp

from sebox import Workspace


class Search(Workspace):
    """Gradient optimization."""
    # maximum number of search steps
    nsteps: int

    # initial step length
    step_init: float
