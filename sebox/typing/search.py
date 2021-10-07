from __future__ import annotations
import typing as tp

from .kernel import Node, _Kernel


class _Search(_Kernel):
    # maximum number of search steps
    nsteps: int

    # initial step length
    step_init: float

    # index and length of the final step
    step_final: tp.Optional[float]

    # current step length
    step: float


class Search(Node['Search'], _Search):
    """Gradient optimization."""
