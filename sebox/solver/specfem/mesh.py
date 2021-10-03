from __future__ import annotations
import typing as tp

from sebox import Directory
from .shared import setpars, getsize

if tp.TYPE_CHECKING:
    from .typing import Par_file, Specfem


def setup(ws: Specfem):
    """Create mesher workspace."""
    src = ws.path_specfem
    d = Directory(src)

    # specfem directories
    ws.mkdir('DATA')
    ws.mkdir('OUTPUT_FILES')
    ws.mkdir('DATABASES_MPI')

    # link binaries and copy data files
    ws.ln(ws.rel(src, 'bin'))
    ws.cp(ws.rel(src, 'DATA/Par_file'), 'DATA')
    ws.cp(ws.rel(ws.path_event or d.path('DATA/CMTSOLUTION')), 'DATA/CMTSOLUTION')
    ws.cp(ws.rel(ws.path_stations or d.path('DATA/STATIONS')), 'DATA/STATIONS')

    # link specfem model directories
    for subdir in d.ls('DATA', isdir=True):
        if subdir != 'GLL':
            ws.ln(ws.rel(src, 'DATA', subdir), 'DATA')
    
    # link model file to run mesher
    if ws.path_model:
        ws.mkdir('DATA/GLL')
        ws.ln(ws.rel(ws.path_model), 'DATA/GLL/model_gll.bp')

    # update Par_file
    pars: Par_file = { 'MODEL': 'GLL' }
    
    if ws.lddrk is not None:
        pars['USE_LDDRK'] = ws.lddrk

    if ws.catalog_boundary is not None:
        pars['ABSORB_USING_GLOBAL_SPONGE'] = True
        pars['SPONGE_LATITUDE_IN_DEGREES'] = ws.catalog_boundary[0]
        pars['SPONGE_LONGITUDE_IN_DEGREES'] = ws.catalog_boundary[1]
        pars['SPONGE_RADIUS_IN_DEGREES'] = ws.catalog_boundary[2]
    
    else:
        pars['ABSORB_USING_GLOBAL_SPONGE'] = False
    
    setpars(ws, pars)


def xmeshfem(ws: Specfem):
    """Add task to call xmeshfem3D."""
    if ws.path_mesh:
        ws.add(ws.ln, name='link_mesh', args=(ws.rel(ws.path_mesh, 'DATABASES_MPI/*.bp'), 'DATABASES_MPI'))
    
    else:
        ws.add_mpi('bin/xmeshfem3D', getsize, data={'prober': _probe})


async def mesh(ws: Specfem):
    """Generate mesh."""
    ws.add(setup)
    xmeshfem(ws)


def _probe(d: Specfem) -> float:
    """Prober of mesher progress."""
    ntotal = 0
    nl = 0

    if not d.has(out_file := 'OUTPUT_FILES/output_mesher.txt'):
        return 0.0
    
    lines = d.readlines(out_file)

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
