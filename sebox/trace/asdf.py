import typing as tp

from sebox import root

if tp.TYPE_CHECKING:
    from sebox.trace import Trace


def gather(ws: Trace):
    """Convert MPI trace to ASDF trace."""


def scatter(ws: Trace):
    """Convert ASDF trace to MPI trace."""
