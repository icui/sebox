from __future__ import annotations
import typing as tp

from sebox import Directory

if tp.TYPE_CHECKING:
    from .specfem import Par_file


def getpars(d: Directory) -> Par_file:
    """Get entries in Par_file."""
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


def getsize(d: Directory):
    """Number of processors to run the solver."""
    pars = getpars(d)

    if 'NPROC_XI' in pars and 'NPROC_ETA' in pars and 'NCHUNKS' in pars:
        return pars['NPROC_XI'] * pars['NPROC_ETA'] * pars['NCHUNKS']
    
    raise RuntimeError('not dimension in Par_file')


def probe_mesher(d: Directory) -> float:
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


def probe_solver(d: Directory) -> float:
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


def _format_station(lines: dict, ll: tp.List[str]):
    """Format a line in STATIONS file."""
    # location of dots for floating point numbers
    dots = 28, 41, 55, 62

    # line with station name
    line = ll[0].ljust(13) + ll[1].ljust(5)

    # add numbers with correct indentation
    for i in range(4):
        num = ll[i + 2]

        if '.' in num:
            nint, _ = num.split('.')
        
        else:
            nint = num

        while len(line) + len(nint) < dots[i]:
            line += ' '
        
        line += num
    
    lines['.'.join(ll[:2])] = line


def merge_stations(d: Directory, dst: str, use_catalog: bool = False):
    """Merge multiple stations into one."""
    lines = {}

    if use_catalog:
        # exclude events and stations that are not in the catalog
        from sebox.utils.catalog import getevents, getstations
        events = getevents()
        stations = getstations()
    
    else:
        # include all events and stations
        events = None
        stations = None

    for src in d.ls():
        event = src.split('.')[1]

        if events and event not in events:
            continue

        for line in d.readlines(src):
            if len(ll := line.split()) == 6:
                station = ll[1] + '.' + ll[0]

                if station in lines:
                    continue

                if stations and station not in stations:
                    continue

                _format_station(lines, ll)
    
    d.writelines(lines.values(), dst)


def extract_stations(d: Directory, dst: str):
    """Extract STATIONS from ASDFDataSet."""
    from os.path import join

    from pyasdf import ASDFDataSet

    for src in d.ls():
        event = src.split('.')[0]
        lines = {}
        out = join(dst, f'STATIONS.{event}')

        if d.has(out):
            continue

        with ASDFDataSet(src, mode='r', mpi=False) as ds:
            for station in ds.waveforms.list():
                if not hasattr(ds.waveforms[station], 'StationXML'):
                    print('  ' + station)
                    continue

                sta = ds.waveforms[station].StationXML.networks[0].stations[0] # type: ignore

                ll = station.split('.')
                ll.reverse()
                ll.append(f'{sta.latitude:.4f}')
                ll.append(f'{sta.longitude:.4f}')
                ll.append(f'{sta.elevation:.1f}')
                ll.append(f'{sta.channels[0].depth:.1f}')

                _format_station(lines, ll)
        
        d.writelines(lines.values(), join(dst, f'STATIONS.{event}'))
