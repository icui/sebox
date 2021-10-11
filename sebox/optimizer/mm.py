from __future__ import annotations

from sebox.utils.adios import xgd, xmm
from .gd import *


def direction(node: Optimizer):
    """Momentum direction."""
    if node.iteration == 0:
        xgd(node)

    else:
        xmm(node)
