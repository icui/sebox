from __future__ import annotations
import typing as tp

from .kernel import Kernel
from .search import Search


class Optimizer(Kernel, Search):
    """Gradient optimization."""
    # total number of iterations
    niters: int
