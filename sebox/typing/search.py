from __future__ import annotations
import typing as tp

from sebox import Node
from .kernel import Kernel


class Search(Node):
    """Gradient optimization."""
    # maximum number of search steps
    nsteps: int

    # initial step length
    step_init: float

    # index and length of the final step
    step_final: tp.Tuple[int, float]

    # current step length
    step: float

    # kernel node
    inherit_kernel: Kernel
