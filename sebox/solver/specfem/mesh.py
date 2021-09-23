from functools import partial

from sebox.solver import Mesh


def echo(f):
    print(f, 1)


async def mesh(ws: Mesh):
    """Generate mesh."""
    await ws.mpiexec(partial(echo, 2), 4)
