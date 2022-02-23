from __future__ import annotations
import typing as tp

from nnodes import Node, Directory
from .shared import setpars, xmeshfem, Par_file


def setup(node: Node):
    """Create mesher node."""
    src = tp.cast(str, node.path_specfem)
    d = Directory(src)

    # specfem directories
    node.mkdir('DATA')
    node.mkdir('OUTPUT_FILES')
    node.mkdir('DATABASES_MPI')

    # link binaries and copy data files
    node.ln(node.rel(src, 'bin'))
    node.cp(node.rel(src, 'DATA/Par_file'), 'DATA')
    node.cp(node.rel(node.path_event or d.path('DATA/CMTSOLUTION')), 'DATA/CMTSOLUTION')
    node.cp(node.rel(node.path_stations or d.path('DATA/STATIONS')), 'DATA/STATIONS')

    # link specfem model directories
    for subdir in d.ls('DATA', isdir=True):
        if subdir != 'GLL':
            node.ln(node.rel(src, 'DATA', subdir), 'DATA')
    
    # link model file to run mesher
    if node.path_model:
        node.mkdir('DATA/GLL')
        node.ln(node.rel(node.path_model), 'DATA/GLL/model_gll.bp')

    # update Par_file
    pars: Par_file = { 'MODEL': 'GLL' }
    
    if node.lddrk is not None:
        pars['USE_LDDRK'] = node.lddrk

    if node.catalog_boundary is not None:
        pars['ABSORB_USING_GLOBAL_SPONGE'] = True
        pars['SPONGE_LATITUDE_IN_DEGREES'] = node.catalog_boundary[0]
        pars['SPONGE_LONGITUDE_IN_DEGREES'] = node.catalog_boundary[1]
        pars['SPONGE_RADIUS_IN_DEGREES'] = node.catalog_boundary[2]
    
    else:
        pars['ABSORB_USING_GLOBAL_SPONGE'] = False
    
    setpars(node, pars)


async def mesh(node: Node):
    """Generate mesh."""
    node.add(setup)
    xmeshfem(node)
