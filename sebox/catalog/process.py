def process(node):
    """Process downloaded data."""
    node.concurrent = True
    node.add(process_observed, concurrent=True)
    node.add(process_synthetic, concurrent=True)


def process_observed(node):
    _process_traces(node, 'obs')


def process_synthetic(node):
    _process_traces(node, 'syn')


def _process_traces(node, mode: str):
    node.mkdir(f'proc_{mode}')

    for event in node.ls(f'events'):
        if node.has(f'raw_{mode}/{event}.h5') and not node.has(f'proc_{mode}/{event}.h5'):
            node.add(process_event, mode=mode, event=event, name=event)


def process_event(node):
    from functools import partial
    from asdfy import ASDFProcessor

    mode = node.mode
    src = f'{node.event}.h5'

    ap = ASDFProcessor(f'raw_{mode}/{src}', f'proc_{mode}/_{src}',
        partial(_process, mode=='obs'), input_type='stream', output_tag=f'proc_{mode}', accessor=True)
    
    node.add_mpi(ap.run, node.np, name='process', fname=node.event, cwd=f'log_{mode}')
    node.add(node.mv, args=(f'proc_{mode}/_{src}', f'proc_{mode}/{src}'), name='move_traces')


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


def _process(obs, acc):
    import numpy as np
    from nnodes import root
    from sebox.catalog import catalog
    from pytomo3d.signal.process import rotate_stream, sac_filter_stream

    print(acc.station, root.mpi.rank)

    if (stream := _select(acc.stream)) is None:
        return
    
    origin = acc.origin
    taper = catalog.process.get('taper')
    pre_filt = catalog.process.get('remove_response')

    # detrend and apply taper
    _detrend(stream, taper)

    # remove instrument response
    if obs:
        stream.attach_response(acc.inventory)
        stream.remove_response(output="DISP", zero_mean=False, taper=False,
            water_level=catalog.process.get('water_level'), pre_filt=pre_filt)
    
    else:
        sac_filter_stream(stream, pre_filt)
    
    # detrend and apply taper after filtering
    _detrend(stream, taper)

    # resample and align
    stream.interpolate(1/catalog.dt, starttime=origin.time)
    
    # pad and rotate
    nt = int(np.round(catalog.duration * 60 / catalog.dt))

    for trace in stream:
        data = np.zeros(nt)
        data[:min(nt, trace.stats.npts)] = trace[:min(nt, trace.stats.npts)]
        trace.data = data

    stream = rotate_stream(stream, origin.latitude, origin.longitude, acc.inventory)

    if len(stream) != 3:
        return
    
    for cmp in ['R', 'T', 'Z']:
        if len(stream.select(component=cmp)) != 1:
            return

    return stream
