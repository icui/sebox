from __future__ import annotations
import typing as tp

from sebox import Directory
from .specfem import Par_file, xmeshfem, xspecfem
from .utils import setpars

if tp.TYPE_CHECKING:
    from sebox.solver import Forward

    class SpecfemForward(Forward):
        """Forward simulation with specfem-specific configuraitons."""
        # specfem directory
        path_specfem: str

        # use LDDRK time scheme
        lddrk: bool


def setup(ws: SpecfemForward):
    """Create forward workspace."""
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
    
    if not ws.path_mesh:
        # link model file to run mesher
        if ws.path_model:
            ws.mkdir('DATA/GLL')
            ws.ln(ws.path_model, 'DATA/GLL/model_gll.bp')
        
        else:
            raise AttributeError('path_mesh or path_model is required')

    # update Par_file
    pars: Par_file = { 'SIMULATION_TYPE': 1, 'MODEL': 'GLL' }

    if ws.save_forward is not None:
        pars['SAVE_FORWARD'] = ws.save_forward
    
    if ws.monochromatic_source is not None:
        pars['USE_MONOCHROMATIC_CMT_SOURCE'] = ws.monochromatic_source
    
    if ws.duration is not None:
        pars['RECORD_LENGTH_IN_MINUTES'] = ws.duration
    
    if ws.transient_duration is not None:
        if ws.duration is None:
            raise ValueError('solver duration must be set if transient_duration exists')

        pars['STEADY_STATE_KERNEL'] = True
        pars['STEADY_STATE_LENGTH_IN_MINUTES'] = ws.duration - ws.transient_duration
    
    else:
        pars['STEADY_STATE_KERNEL'] = False
    
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


def finalize(ws: SpecfemForward):
    """Move generated forward traces."""
    ws.mv('OUTPUT_FILES/synthetic.h5', 'traces.h5')


def forward(ws: SpecfemForward):
    """Forward simulation."""
    ws.add(setup)
    ws.add(xmeshfem)
    ws.add(xspecfem)
    ws.add(finalize)
