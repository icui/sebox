from __future__ import annotations
import typing as tp

from sebox import Directory
from .specfem import xmeshfem
from .utils import setpars

if tp.TYPE_CHECKING:
    from .specfem import Par_file
    from .solver import Forward


def setup(ws: Forward):
    """Create mesher workspace."""
    d = Directory(ws.path_specfem)

    # specfem directories
    ws.mkdir('DATA')
    ws.mkdir('OUTPUT_FILES')
    ws.mkdir('DATABASES_MPI')

    # link binaries and event files
    ws.ln(d.abs('bin'))
    ws.cp(d.abs('DATA/Par_file'), 'DATA')
    ws.cp(ws.path_event or d.abs('DATA/CMTSOLUTION'), 'DATA/CMTSOLUTION')
    ws.cp(ws.path_stations or d.abs('DATA/STATIONS'), 'DATA/STATIONS')

    # link specfem model directories
    for subdir in d.ls('DATA', isdir=True):
        if subdir != 'GLL':
            ws.ln(d.abs('DATA', subdir), 'DATA')
    
    # link model file to run mesher
    if ws.path_model:
        ws.mkdir('DATA/GLL')
        ws.ln(ws.path_model, 'DATA/GLL/model_gll.bp')

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


async def mesh(ws: Forward):
    """Generate mesh."""
    ws.test = 'abc'
    ws.add(setup)
    ws.add(xmeshfem)
