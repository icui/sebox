from __future__ import annotations
import typing as tp

from .mesh import setup as setup_mesh, xmeshfem
from .shared import setpars, getsize

if tp.TYPE_CHECKING:
    from .typing import Par_file, Specfem


def setup(ws: Specfem):
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


def xspecfem(ws: Specfem):
    """Add task to call xspecfem3D."""
    ws.add_mpi('bin/xspecfem3D', getsize, 1, data={'prober': _probe})


def forward(ws: Specfem):
    """Forward simulation."""
    ws.add(setup)
    xmeshfem(ws)
    xspecfem(ws)
    ws.add(('sebox.utils.asdf', 'scatter'),
        path_input=ws.path('OUTPUT_FILES/synthetic.h5'), path_output=ws.path('traces'),
        stats={'cmps': ['N', 'E', 'Z']})


def _probe(d: Specfem) -> float:
    """Prober of solver progress."""
    from math import ceil

    if not d.has(out := 'OUTPUT_FILES/output_solver.txt'):
        return 0.0
    
    lines = d.readlines(out)
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
