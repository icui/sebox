from __future__ import annotations
import typing as tp

from .search import Node, _Search


class _Optimizer(_Search):
    """Gradient optimization."""
    # total number of iterations
    niters: int

    # initial iteration
    iteration: tp.Optional[int]


class Optimizer(Node['Optimizer'], _Optimizer):
    """Gradient optimization."""
