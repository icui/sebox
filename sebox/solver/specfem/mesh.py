from sebox.solver import Mesh


def echo():
    print(1)


async def mesh(ws: Mesh):
    """Generate mesh."""
    await ws.mpiexec(echo, 4)
