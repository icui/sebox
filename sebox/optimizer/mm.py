from sebox.utils.adios import xcg
from .gd import main, iterate, check, Optimizer


def direction(node: Optimizer):
    """Momentum direction."""
    xcg(node)
