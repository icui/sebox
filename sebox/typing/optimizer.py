from __future__ import annotations
import typing as tp

from .search import Node, _Search


class _Optimizer(_Search):
    """Gradient optimization."""
    # total number of iterations
    niters: int

    # current iteration
    iteration: int

    # initial iteration
    iteration_start: int

    # history of iteration_start
    iteration_breakpoints: tp.Set[int]

    # restart iteration after n iterations
    iteration_restart: tp.Optional[int]


class Optimizer(Node['Optimizer'], _Optimizer):
    """Gradient optimization."""
