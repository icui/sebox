from __future__ import annotations
import typing as tp

from .search import Search


class Optimizer(Search):
    """Gradient optimization."""
    # total number of iterations
    niters: int

    # initial iteration
    iteration: tp.Optional[int]
