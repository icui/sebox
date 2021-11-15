from __future__ import annotations

from sebox.utils.adios import xgd, xlbfgs, xlbfgs2
from .gd import *


def direction(node: Optimizer):
    """Momentum direction."""
    if node.iteration == node.iteration_start:
        xgd(node)
    
    elif node.test_lbfgs:
        xlbfgs2(node)

    else:
        xlbfgs(node)
