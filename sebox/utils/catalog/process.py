def process_traces(node):
    """Process downloaded data."""
    from nnodes import root

    node.mkdir('process')

    for src in node.ls('raw_obs'):
        node.rm(f'proc_obs/{src}')
        node.add_mpi(process_trace, 1, (root.job.cpus_per_node, 0),
            arg=src, name=src.split('.')[0] + '_obs', cwd='process')


def process_trace(src: str):
    from functools import partial
    from pyasdf import ASDFDataSet

    with ASDFDataSet(f'raw_obs/{src}', mode='r', mpi=False, compression=None) as ds:
        origin = ds.events[0].preferred_origin()
        inventory = ds.waveforms[f'{stream[0].stats.network}.{stream[0].stats.station}'].StationXML # type: ignore
        ds.process(partial(_process, origin, inventory), f'proc_obs/{src}', {'raw_obs': 'proc_obs'})


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


def _process(origin, inventory, stream):
    from traceback import format_exc
    from sebox import catalog
    from pytomo3d.signal.process import rotate_stream

    try:
        if (stream := _select(stream)) is None:
            return
        
        taper = catalog.processing.get('taper')

        # remove instrument response
        _detrend(stream, taper)
        stream.attach_response(inventory)
        stream.remove_response(output="DISP", zero_mean=False, taper=False,
            water_level=60, pre_filt=catalog.processing.get('remove_response'))

        # resample and align
        stream.interpolate(1/catalog.processing['dt'], starttime=origin.time)
        
        # detrend and apply taper
        _detrend(stream, taper)
        
        # rotate
        stream = rotate_stream(stream, origin.latitude, origin.longitude, inventory)

        if len(stream) != 3:
            return
        
        for cmp in ['R', 'T', 'Z']:
            if len(stream.select(component=cmp)) != 1:
                return

        return stream
        
    except Exception:
        print(format_exc())


