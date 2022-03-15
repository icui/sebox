def process_traces(node):
    """Process downloaded data."""
    from functools import partial
    from asdfy import ASDFProcessor

    node.mkdir('process')

    for mode in ('obs', 'syn'):
        for src in node.ls(f'raw_{mode}'):
            ap = ASDFProcessor(f'raw_{mode}/{src}', f'proc_{mode}/{src}',
                partial(_process, mode=='obs'), 'stream', f'raw_{mode}', f'proc_{mode}', True)
            node.add_mpi(ap.run, name=src.split('.')[0] + '_' + mode, cwd='process')


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
    from sebox import catalog
    from pytomo3d.signal.process import rotate_stream

    nt = int(np.round(catalog.duration * 60 / catalog.dt))
    print(acc.station, nt)

    if (stream := _select(acc.stream)) is None:
        return
    
    taper = catalog.processing.get('taper')

    # remove instrument response
    if obs:
        _detrend(stream, taper)
        stream.attach_response(acc.inventory)
        stream.remove_response(output="DISP", zero_mean=False, taper=False,
            water_level=60, pre_filt=catalog.processing.get('remove_response'))

    # resample and align
    origin = acc.origin
    stream.interpolate(1/catalog.dt, starttime=origin.time)
    
    # detrend and apply taper
    _detrend(stream, taper)
    
    # pad and rotate
    for trace in stream:
        data = np.zeros(nt)
        data[:min(nt, trace.stats.npts)] = trace[:min(nt, trace.stats.npts)]

    stream = rotate_stream(stream, origin.latitude, origin.longitude, acc.inventory)

    if len(stream) != 3:
        return
    
    for cmp in ['R', 'T', 'Z']:
        if len(stream.select(component=cmp)) != 1:
            return

    return stream
