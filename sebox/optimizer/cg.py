from __future__ import annotations

from sebox.utils.adios import xgd, xcg
from .gd import *


def direction(node: Optimizer):
    """Momentum direction."""
    if node.iteration == 0:
        xgd(node)

    else:
        xcg(node)