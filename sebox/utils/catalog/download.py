def download(node):
    node.add(download_events)
    node.add(download_traces)


def download_events(node):
    """Download events."""


def download_traces(node):
    """Download observed data."""
    for event in node.ls('events'):
        if not node.has(f'raw_obs/{event}.h5'):
            node.add(download_trace, event=event, name=event, cwd=f'downloads/{event}')


def download_trace(node):
    """Download observed data of an event."""
    node.add_mpi(request_data)
    node.add_mpi(convert_h5)


def request_data():
    from traceback import format_exc
    from nnodes import root
    from sebox import catalog
    from obspy import read_events
    from obspy.clients.fdsn.mass_downloader import GlobalDomain, Restrictions, MassDownloader

    node = root.mpi

    try:
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
    
    except:
        node.write(format_exc(), 'error_download.log')


def convert_h5():
    from traceback import format_exc
    from nnodes import root
    from pyasdf import ASDFDataSet
    from obspy import read, read_events
    from .index import format_station

    node = root.mpi

    if not node.has('error_download.log'):
        return

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
    
    node.mv(f'{node.event}.h5', '../../raw_obs')
