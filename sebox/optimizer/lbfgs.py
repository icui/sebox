from __future__ import annotations

from sebox.utils.adios import xgd, xlbfgs
from .gd import *


def direction(node: Optimizer):
    """Momentum direction."""
    if node.iteration == node.iteration_start:
        xgd(node)

    else:
        xlbfgs(node)