from __future__ import annotations
import typing as tp

from sebox.core.workspace import Workspace, Directory

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


async def xspecfem(ws: Workspace):
    """Call xspecfem3D."""
    await ws.mpiexec('bin/xspecfem3D', getsize(ws), 1)


async def xmeshfem(ws: tp.Union[Forward, Mesh]):
    """Call xmeshfem3D."""
    if ws.path_mesh:
        ws.ln(ws.abs(ws.path_mesh, 'DATABASES_MPI/*.bp'))
    
    else:
        await ws.mpiexec('bin/xmeshfem3D', getsize(ws))


def getpars(d: tp.Optional[Directory] = None) -> Par_file:
    """Get entries in Par_file."""
    from sebox import root
    
    if d is None:
        d = Directory(tp.cast(str, root.path_specfem))

    pars: Par_file = {}

    for line in d.readlines('DATA/Par_file'):
        if '=' in line:
            keysec, valsec = line.split('=')[:2]
            key = keysec.split()[0]
            val = valsec.split('#')[0].split()[0]

            if val == '.true':
                pars[key] = True
            
            elif val == '.false.':
                pars[key] = False
            
            elif val.isnumeric():
                pars[key] = int(val)
            
            else:
                try:
                    pars[key] = float(val.replace('D', 'E').replace('d', 'e'))
                
                except:
                    pars[key] = val
    
    return pars


def setpars(d: Directory, pars: Par_file):
    """Set entries in Par_file."""
    lines = d.readlines('DATA/Par_file')

    # update lines from par
    for i, line in enumerate(lines):
        if '=' in line:
            keysec = line.split('=')[0]
            key = keysec.split()[0]

            if key in pars and pars[key] is not None:
                val = pars[key]

                if isinstance(val, bool):
                    val = f'.{str(val).lower()}.'

                elif isinstance(val, float):
                    if len('%f' % val) < len(f'{val}'):
                        val = '%fd0' % val

                    else:
                        val = f'{val}d0'

                lines[i] = f'{keysec}= {val}'

    d.writelines(lines, 'DATA/Par_file')


def getsize(d: tp.Optional[Directory] = None):
    """Number of processors to run the solver."""
    pars = getpars(d)

    if 'NPROC_XI' in pars and 'NPROC_ETA' in pars and 'NCHUNKS' in pars:
        return pars['NPROC_XI'] * pars['NPROC_ETA'] * pars['NCHUNKS']
    
    raise RuntimeError('not dimension in Par_file')
