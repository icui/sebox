from sebox.solver import Mesh


def echo(ws: Mesh):
    print(ws.path_model, 1)


async def mesh(ws: Mesh):
    """Generate mesh."""
    await ws.mpiexec(echo, 4)
