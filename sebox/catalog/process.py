def process(node):
    """Process downloaded data."""
    node.concurrent = True
    node.add(process_observed)
    node.add(process_synthetic)


def process_observed(node):
    node.concurrent = True

    for event in node.ls('events'):
        if not node.has(f'bp_obs/{event}.bp'):
            continue

        node.add(process_event, mode='obs', event=event, name=event,
            src=f'bp_obs/{event}.bp', dst=f'proc_obs2/{event}.bp')
        break


def process_synthetic(node):
    node.concurrent = True

    for event in node.ls('events'):
        if node.has(f'proc_syn/{event}.bp'):
            continue

        node.add(process_event, mode='syn', event=event, name=event,
            src=f'raw_syn/{event}.bp', dst=f'proc_syn/{event}.bp')


def process_event(node):
    from seisbp import SeisBP

    with SeisBP(node.src, 'r') as bp:
        stations = bp.stations

    node.add_mpi(_process, node.np, args=(node.src, node.dst, node.mode),
        mpiarg=stations, group_mpiarg=True, cwd=f'log_{node.mode}', name=node.event)


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


def _process(stas, src, dst, mode):
    from nnodes import root
    from seisbp import SeisBP

    print(dst)

    with SeisBP(src, 'r', True) as raw_bp, SeisBP(dst, 'w', True) as proc_bp:
        evt = raw_bp.read(raw_bp.events[0])
        origin = evt.preferred_origin()

        if root.mpi.rank == 0:
            proc_bp.write(evt)

        for sta in stas:
            # proc_bp.write(raw_bp.stream(sta))
            try:
                stream = raw_bp.stream(sta)
                inv = raw_bp.read(sta)

                proc_bp.write(inv)
                proc_bp.write(stream)

                # if proc_stream := _process_stream(stream, origin, inv, mode):
                #     proc_bp.write(inv)
                #     proc_bp.write(proc_stream)
            
            except:
                print('?', sta)


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

    # period anchors
    cl = proc['corner_left']
    cr = proc['corner_right']
    pmin = proc['period_min']
    pmax = proc['period_max']
    pre_filt = [1/pmax*cr*cr, 1/pmax*cr, 1/pmin/cl, 1/pmin/cl/cl]

    # remove instrument response
    if mode == 'obs':
        stream.attach_response(inv)
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
