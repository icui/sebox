import typing as tp

if tp.TYPE_CHECKING:
    from nnodes import Node, Directory


class Par_file(tp.TypedDict, total=False):
    """DATA/Par_file in specfem."""
    # 1 for forward simulation, 3 for adjoint simulation
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

    # output seismograms in 3D array
    OUTPUT_SEISMOS_3D_ARRAY: bool

    # downsample output seismograms
    NTSTEP_BETWEEN_OUTPUT_SAMPLE: int


def getpars(node: Directory) -> Par_file:
    """Get entries in Par_file."""
    pars: Par_file = {}

    if not node.has('DATA/Par_file'):
        from nnodes import Directory

        node = Directory(node.path_specfem) # type: ignore

    for line in node.readlines('DATA/Par_file'):
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


def getsize(node: Directory) -> int:
    """Number of processors to run the solver."""
    pars = getpars(node)

    if 'NPROC_XI' in pars and 'NPROC_ETA' in pars and 'NCHUNKS' in pars:
        return pars['NPROC_XI'] * pars['NPROC_ETA'] * pars['NCHUNKS']
    
    raise RuntimeError('not dimension in Par_file')


def xspecfem(node: Node):
    """Add task to call xspecfem3D."""
    node.add_mpi('bin/xspecfem3D', getsize, 1, data={'prober': probe_solver})


def xmeshfem(node: Node):
    """Add task to call xmeshfem3D."""
    if node.path_mesh:
        node.add(node.ln, name='link_mesh', args=(node.rel(node.path_mesh, 'DATABASES_MPI/*'), 'DATABASES_MPI'))
    
    else:
        node.add_mpi('bin/xmeshfem3D', getsize, data={'prober': probe_mesher})


def setpars(node: Directory, pars: Par_file):
    """Set entries in Par_file."""
    lines = node.readlines('DATA/Par_file')

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

    node.writelines(lines, 'DATA/Par_file')


def probe_solver(node: Directory) -> float:
    """Prober of solver progress."""
    from math import ceil

    if not node.has(out := 'OUTPUT_FILES/output_solver.txt'):
        return 0.0
    
    lines = node.readlines(out)
    lines.reverse()

    for line in lines:
        if 'End of the simulation' in line:
            return 1.0

        if 'We have done' in line:
            words = line.split()
            done = False

            for word in words:
                if word == 'done':
                    done = True

                elif word and done:
                    return ceil(float(word)) / 100

    return 0.0


def probe_mesher(node: Directory) -> float:
    """Prober of mesher progress."""
    ntotal = 0
    nl = 0

    if not node.has(out_file := 'OUTPUT_FILES/output_mesher.txt'):
        return 0.0
    
    lines = node.readlines(out_file)

    for line in lines:
        if ' out of ' in line:
            if ntotal == 0:
                ntotal = int(line.split()[-1]) * 2

            if nl < ntotal:
                nl += 1

        if 'End of mesh generation' in line:
            return 1.0

    if ntotal == 0:
        return 0.0

    return (nl - 1) / ntotal
