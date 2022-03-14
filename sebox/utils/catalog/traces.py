def prepare_traces(node):
    node.add(download_events)
    node.add(download_traces)
    node.add(compute_synthetic)


def download_events(node):
    """Download events."""


def download_traces(node):
    """Download observed data."""
    for event in node.ls('events'):
        node.add(download_trace, event=event, cwd=f'downloads/{event}')


def download_trace(node):
    """Download observed data of an event."""
    # node.add(_download_fdsn)
    node.add(_convert_h5)


def _download_fdsn(node):
    from sebox import catalog
    from obspy import read_events
    from obspy.clients.fdsn.mass_downloader import GlobalDomain, Restrictions, MassDownloader

    event = read_events(f'../../events/{node.event}')[0]
    node.mkdir('mseed')
    node.mkdir('xml')

    gap = catalog.config['download_gap']
    eventtime = event.preferred_origin().time
    starttime = eventtime - gap * 60
    endtime = eventtime + (catalog.duration[0] + gap) * 60

    rst = Restrictions(starttime=starttime, endtime=endtime, **catalog.config['download'])
    mdl = MassDownloader()
    mdl.download(GlobalDomain(), rst, mseed_storage=node.abs('mseed'), stationxml_storage=node.abs('xml'))


def _convert_h5(node):
    from pyasdf import ASDFDataSet
    from obspy import read, read_events

    from .index import format_station

    with ASDFDataSet(f'{node.event}.h5', mode='w', mpi=False, compression=None) as ds:
        try:
            ds.add_quakeml(read_events(f'events/{node.event}'))
        
        except Exception as e:
            node.write(str(e), 'error.log', 'a')

        stations = set()
        station_lines = ''

        for src in node.ls(f'mseed'):
            station = '.'.join(src.split('.')[:2])
            stations.add(station)

            try:
                ds.add_waveforms(read(f'mseed/{src}'), 'raw_obs')

            except Exception as e:
                node.write(str(e), 'error.log', 'a')
        
        for station in stations:
            try:
                ds.add_stationxml(f'downloads/xml/{station}.xml')
                sta = ds.waveforms[station].StationXML.networks[0].stations[0]
                ll = station.split('.')
                ll.reverse()
                ll += [f'{sta.latitude:.4f}', f'{sta.longitude:.4f}', f'{sta.elevation:.4f}', f'{sta.depth:.4f}']
                station_lines += format_station(ll)
            
            except Exception as e:
                node.write(str(e), 'error.log', 'a')
        
        node.write(station_lines, f'STATIONS.{event}')


def compute_synthetic(node):
    """Compute synthetic data."""
