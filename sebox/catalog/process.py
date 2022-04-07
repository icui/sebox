def process(node):
    """Process downloaded data."""
    node.concurrent = True
    node.add(process_observed)
    node.add(process_synthetic)


def process_observed(node):
    node.concurrent = True

    for event in node.ls('events'):
        node.add(process_event, mode='obs', event=event, name=event,
            src=f'raw_obs/{event}.h5', dst=f'proc_obs/{event}.bp')
        break


def process_synthetic(node):
    node.concurrent = True

    for event in node.ls('events'):
        node.add(process_event, mode='syn', event=event, name=event,
            src=f'raw_syn/{event}.h5', dst=f'proc_syn/{event}.bp')


def process_event(node):
    from functools import partial
    from asdfy import ASDFProcessor

    mode = node.mode
    src = f'{node.event}.h5'

    ap = ASDFProcessor(f'raw_{mode}/{src}', None,
        partial(_process, mode), input_type='stream', accessor=True)
    
    node.add_mpi(partial(_proc, ap, mode, node.event), node.np, name='process', fname=node.event, cwd=f'log_{mode}')
    node.add(node.mv, args=(f'proc_{mode}/_{src}', f'proc_{mode}/{src}'), name='move_traces')


def _proc(ap, mode, event):
    from mpi4py import MPI
    import adios2

    comm = MPI.COMM_WORLD

    # with-as will call adios2.close on fh at the end
    with adios2.open(f"proc_obs/{event}.bp", "w", comm) as fh:

        for acc in ap.access():
            st = _process(acc, mode)

            for tr in st:
                fh.write(acc.station + '.' + tr.stats.channel, tr.data)


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


def _process(mode, acc):
    import numpy as np
    from nnodes import root
    from sebox.catalog import catalog
    from pytomo3d.signal.process import rotate_stream, sac_filter_stream

    print(acc.station, root.mpi.rank)

    if (stream := _select(acc.stream)) is None:
        return
    
    origin = acc.origin

    if origin is None:
        from obspy import read_events
        origin = read_events(f'events/{acc.event}')[0].preferred_origin()

    proc = catalog.process
    taper = proc.get('taper')

    # resample and align
    stream.interpolate(1/catalog.dt, starttime=origin.time)
        
    # detrend and apply taper after filtering
    _detrend(stream, taper)

    # attach response
    stream.attach_response(acc.inventory)

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

    stream = rotate_stream(stream, origin.latitude, origin.longitude, acc.inventory)

    # make sure stream has 1 radial, 1 transverse and 1 vertical trace
    if len(stream) != 3 or any(len(stream.select(component=cmp)) != 1 for cmp in ['R', 'T', 'Z']):
        return

    return stream
