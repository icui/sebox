def process(node):
    """Process downloaded data."""
    node.concurrent = True
    node.add(process_observed)
    node.add(process_synthetic)


def process_observed(node):
    node.concurrent = True

    for event in node.ls('events'):
        node.add(process_event, mode='obs', event=event, name=event,
            src=f'raw_obs/{event}.bp', dst=f'proc_obs/{event}.bp')
        break


def process_synthetic(node):
    node.concurrent = True

    for event in node.ls('events'):
        node.add(process_event, mode='syn', event=event, name=event,
            src=f'raw_syn/{event}.bp', dst=f'proc_syn/{event}.bp')


def process_event(node):
    from seisbp import SeisBP

    with SeisBP(node.src, 'r') as bp:
        stations = bp.stations

    node.add_mpi(_process, node.np, args=(node.src, node.dst + '_'), mpiarg=stations, group_mpiarg=True, cwd=f'log_{node.mode}')
    node.add(node.mv, args=(node.dst + '_', node.dst), name='move_output')


def _select(stream):
    from obspy import Stream

    # select 3 components from the stream
    for trace_z in stream.select(component='Z'):
        for cmps in [['N', 'E'], ['1', '2']]:
            traces = [trace_z]

            for cmp in cmps:
                if len(substream := stream.select(component=cmp, location=trace_z.stats.location)):
                    traces.append(substream[0])
            
            if len(traces) == 3:
                return Stream(traces)


def _detrend(stream, taper):
    """Detrend and taper."""
    stream.detrend('linear')
    stream.detrend('demean')

    if taper:
        stream.taper(max_percentage=None, max_length=taper*60)


def _process(stas, src, dst):
    from seisbp import SeisBP

    with SeisBP(src, 'r', True) as raw_bp, SeisBP(dst, 'w', True) as proc_bp:
        for sta in stas:
            stream = raw_bp.stream(sta)
            origin = raw_bp.read(raw_bp.events[0]).preferred_origin()
            inv = raw_bp.read(sta)

            try:
                if proc_stream := _process_stream(stream, origin, inv, 'obs'):
                    proc_bp.write(proc_stream)
            
            except:
                print(sta)


def _process_stream(st, origin, inv, mode):
    import numpy as np
    from sebox.catalog import catalog
    from pytomo3d.signal.process import rotate_stream, sac_filter_stream

    if (stream := _select(st)) is None:
        return

    proc = catalog.process
    taper = proc.get('taper')

    # resample and align
    stream.interpolate(1/catalog.dt, starttime=origin.time)
        
    # detrend and apply taper after filtering
    _detrend(stream, taper)

    # attach response
    stream.attach_response(inv)

    # period anchors
    cl = proc['corner_left']
    cr = proc['corner_right']
    pmin = proc['period_min']
    pmax = proc['period_max']
    pre_filt = [1/pmax*cr*cr, 1/pmax*cr, 1/pmin/cl, 1/pmin/cl/cl]

    # remove instrument response
    if mode == 'obs':
        stream.remove_response(output="DISP", zero_mean=False, taper=False,
            water_level=catalog.process.get('water_level'), pre_filt=pre_filt)
    
    else:
        sac_filter_stream(stream, pre_filt)

    # detrend and apply taper
    _detrend(stream, taper)
    
    # pad and rotate
    nt = int(np.round(catalog.duration * 60 / catalog.dt))

    for trace in stream:
        data = np.zeros(nt)
        data[:min(nt, trace.stats.npts)] = trace[:min(nt, trace.stats.npts)]
        trace.data = data

    stream = rotate_stream(stream, origin.latitude, origin.longitude, inv)

    # make sure stream has 1 radial, 1 transverse and 1 vertical trace
    if len(stream) != 3 or any(len(stream.select(component=cmp)) != 1 for cmp in ['R', 'T', 'Z']):
        return

    return stream
