from __future__ import annotations
import typing as tp

from .specfem import xmeshfem, xspecfem, setpars
from .mesh import setup as setup_mesh

if tp.TYPE_CHECKING:
    from .specfem import Par_file
    from sebox import typing

    class Forward(typing.Forward):
        """Forward simulation with specfem-specific configuraitons."""
        # specfem directory
        path_specfem: str

        # use LDDRK time scheme
        lddrk: bool


def setup(ws: Forward):
    """Create forward workspace."""
    if not ws.path_mesh and not ws.path_model:
        raise AttributeError('path_mesh or path_model is required')
    
    setup_mesh(ws)

    # update Par_file
    pars: Par_file = { 'SIMULATION_TYPE': 1 }

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
    
    setpars(ws, pars)


def forward(ws: Forward):
    """Forward simulation."""
    ws.add(setup)
    xmeshfem(ws)
    xspecfem(ws)
    ws.add(('sebox.dataset.asdf_npy', 'scatter'), path_bundle='OUTPUT_FILES/synthetic.h5', path_mpi='traces')
