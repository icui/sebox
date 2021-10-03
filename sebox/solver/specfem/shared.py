from __future__ import annotations
import typing as tp

from sebox import Directory

if tp.TYPE_CHECKING:
    from .typing import Par_file


def getsize(d: Directory):
    """Number of processors to run the solver."""
    pars = getpars(d)

    if 'NPROC_XI' in pars and 'NPROC_ETA' in pars and 'NCHUNKS' in pars:
        return pars['NPROC_XI'] * pars['NPROC_ETA'] * pars['NCHUNKS']
    
    raise RuntimeError('not dimension in Par_file')


def getpars(d: Directory) -> Par_file:
    """Get entries in Par_file."""
    pars: Par_file = {}

    if not d.has('DATA/Par_file'):
        d = Directory(tp.cast(tp.Any, d).path_specfem)

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
