from __future__ import annotations
import typing as tp

from sebox import Workspace
from .utils import getsize, probe_mesher, probe_solver

if tp.TYPE_CHECKING:
    from sebox.solver import Forward, Mesh

    class Par_file(tp.TypedDict, total=False):
        """DATA/Par_file in specfem."""
        # 0 for forward simulation, 3 for adjoint simulation
        SIMULATION_TYPE: int

        # save forward wavefield
        SAVE_FORWARD: bool

        # use monochromatic source time function
        USE_MONOCHROMATIC_CMT_SOURCE: bool

        # simulation duration
        RECORD_LENGTH_IN_MINUTES: float

        # model name
        MODEL: str

        # use high order time scheme
        USE_LDDRK: bool

        # number of processors in XI direction
        NPROC_XI: int

        # number of processors in ETA direction
        NPROC_ETA: int

        # number of chunks
        NCHUNKS: int

        # compute steady state kernel for source encoded FWI
        STEADY_STATE_KERNEL: bool

        # steady state duration for source encoded FWI
        STEADY_STATE_LENGTH_IN_MINUTES: float

        # sponge absorbing boundary
        ABSORB_USING_GLOBAL_SPONGE: bool

        # center latitude of sponge
        SPONGE_LATITUDE_IN_DEGREES: float

        # center longitude of sponge
        SPONGE_LONGITUDE_IN_DEGREES: float

        # radius of the sponge
        SPONGE_RADIUS_IN_DEGREES: float


async def _xspecfem(ws: Workspace):
    """Call xspecfem3D."""
    await ws.mpiexec('bin/xspecfem3D', getsize(ws), 1)


def xspecfem(ws: Workspace):
    """Add a task to call xspecfem3D."""
    ws.add(_xspecfem, { 'prober': probe_solver })


async def _xmeshfem(ws: tp.Union[Forward, Mesh]):
    """Call xmeshfem3D."""
    if ws.path_mesh:
        ws.ln(ws.abs(ws.path_mesh, 'DATABASES_MPI/*.bp'))
    
    else:
        await ws.mpiexec('bin/xmeshfem3D', getsize(ws))


def xmeshfem(ws: Workspace):
    """Add a task to call xspecfem3D."""
    ws.add(_xmeshfem, { 'prober': probe_mesher })
