def prepare_observed(node):
    node.add(download_events)
    node.add(download_traces)
    node.add(process_trace)


def download_events(node):
    """Download events."""


def download_traces(node):
    """Download observed data."""
    for event in node.ls('events'):
        node.add(download_trace, event=event, name=event, cwd=f'downloads/{event}')


def download_trace(node):
    """Download observed data of an event."""
    node.add(request_data)
    node.add(convert_h5)


def request_data(node):
    from sebox import catalog
    from obspy import read_events
    from obspy.clients.fdsn.mass_downloader import GlobalDomain, Restrictions, MassDownloader

    event = read_events(f'events/{node.event}')[0]
    node.mkdir('mseed')
    node.mkdir('xml')

    gap = catalog.download['gap']
    eventtime = event.preferred_origin().time
    starttime = eventtime - gap * 60
    endtime = eventtime + (catalog.duration + gap) * 60

    rst = Restrictions(starttime=starttime, endtime=endtime, **catalog.download['restrictions'])
    mdl = MassDownloader()
    mdl.download(GlobalDomain(), rst,
        threads_per_client=catalog.download.get('threads') or 3,
        mseed_storage=node.path('mseed'),
        stationxml_storage=node.path('xml'))


def convert_h5(node):
    from traceback import format_exc
    from pyasdf import ASDFDataSet
    from obspy import read, read_events
    from .index import format_station

    with ASDFDataSet(node.path(f'{node.event}.h5'), mode='w', mpi=False, compression=None) as ds:
        try:
            ds.add_quakeml(read_events((f'events/{node.event}')))
        
        except Exception:
            node.write(format_exc(), 'error.log', 'a')

        stations = set()
        station_lines = ''

        for src in node.ls(f'mseed'):
            station = '.'.join(src.split('.')[:2])
            stations.add(station)

            try:
                ds.add_waveforms(read(node.path(f'mseed/{src}')), 'raw_obs')

            except Exception:
                node.write(format_exc(), 'error.log', 'a')
        
        for station in stations:
            try:
                ds.add_stationxml(node.path(f'xml/{station}.xml'))
                sta = ds.waveforms[station].StationXML.networks[0].stations[0] # type: ignore
                ll = station.split('.')
                ll.reverse()
                ll += [f'{sta.latitude:.4f}', f'{sta.longitude:.4f}', f'{sta.elevation:.1f}', f'{sta.channels[0].depth:.1f}']
                station_lines += format_station(ll)
            
            except Exception:
                node.write(format_exc(), 'error.log', 'a')
        
        node.write(station_lines, f'STATIONS.{node.event}')


def process_traces(node):
    """Process downloaded data."""
    from nnodes import root

    for src in node.ls('raw_obs'):
        node.add_mpi(process_trace, arg=src)


def process_trace(src: str):
    from functools import partial
    from pyasdf import ASDFDataSet

    with ASDFDataSet(f'raw_obs/{src}', mode='r', mpi=False, compression=None) as ds:
        ds.process(partial(_process, ds), f'proc_obs/{src}', {'raw_obs': 'proc_obs'})


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


def _process(ds, stream):
    from traceback import format_exc
    from sebox import catalog
    from pytomo3d.signal.process import rotate_stream

    try:
        if (stream := _select(stream)) is None:
            return
        
        origin = ds.events[0].preferred_origin()
        taper = catalog.processing.get('taper')
        inventory = ds.waveforms[f'{stream[0].stats.network}.{stream[0].stats.station}'].StationXML # type: ignore

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


