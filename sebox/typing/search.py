from __future__ import annotations
import typing as tp

from .kernel import Kernel


class Search(Kernel):
    """Gradient optimization."""
    # maximum number of search steps
    nsteps: int

    # initial step length
    step_init: float

    # index and length of the final step
    step_final: tp.Optional[float]

    # current step length
    step: float
