from __future__ import annotations

from sebox import Node
from .kernel import Kernel


class Search(Node):
    """Gradient optimization."""
    # maximum number of search steps
    nsteps: int

    # initial step length
    step_init: float

    # current step length
    step: float

    # kernel node
    inherit_kernel: Kernel
