from __future__ import annotations
from functools import partial
import typing as tp

if tp.TYPE_CHECKING:
    from sebox.solver import Mesh


def echo(f):
    print(f, 1)


async def mesh(ws: Mesh):
    """Generate mesh."""
    await ws.mpiexec(partial(echo, 2), 4)
