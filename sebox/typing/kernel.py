from __future__ import annotations
import typing as tp

from .solver import Node, _Solver


class _Kernel(_Solver):
    # current iteration
    iteration: tp.Optional[int]

    # initial iteration
    iteration_start: tp.Optional[int]

    # length of a time step
    dt: float

    # period range
    period_range: tp.List[float]

    # skip computing adjoint kernels
    misfit_only: bool

    # inherit from an existing Kernel
    inherit_kernel: Kernel


class Kernel(Node['Kernel'], _Kernel):
    """Compute kernel and / or misfit."""
