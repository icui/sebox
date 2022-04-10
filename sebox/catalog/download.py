def download(node):
    node.add(download_events)
    node.add(download_traces)


def download_events(node):
    """Download events."""


def download_traces(node):
    """Download observed data."""
    for event in node.ls('events'):
        # if not node.has(f'raw_obs/{event}.h5'):
            node.add(download_trace, event=event, name=event, cwd=f'downloads/{event}')
            break


def download_trace(node):
    """Download observed data of an event."""
    node.ln('../../catalog.toml')
    arg = (node.event, f'downloads/{node.event}')
    # node.add_mpi(request_data, arg=arg, use_multiprocessing=True)
    # node.add_mpi(convert_h5, arg=arg, use_multiprocessing=True)
    # node.add_mpi(convert_xml, arg=arg, use_multiprocessing=True)


def request_data(arg):
    from traceback import format_exc
    from nnodes import root
    from sebox.catalog import catalog
    from obspy import read_events
    from obspy.clients.fdsn.mass_downloader import GlobalDomain, Restrictions, MassDownloader

    event = arg[0]
    node = root.subdir(arg[1])

    try:
        evt = read_events(f'events/{event}')[0]
        node.mkdir('mseed')
        node.mkdir('xml')

        gap = catalog.download['gap']
        eventtime = evt.preferred_origin().time
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


def convert_h5(arg):
    from traceback import format_exc
    from nnodes import root
    from pyasdf import ASDFDataSet
    from obspy import read, read_events
    from .index import format_station

    event = arg[0]
    node = root.subdir(arg[1])

    if node.has('error_download.log'):
        return

    with ASDFDataSet(node.path(f'{event}.h5'), mode='w', mpi=False, compression=None) as ds:
        try:
            ds.add_quakeml(read_events(node.path(f'../../events/{event}')))
        
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
        
        node.write(station_lines, f'STATIONS.{event}')


def convert_bp(node):
    node.add_mpi(_convert_bp, 786, args=('obs',), mpiarg=node.ls('events'))
    # node.add_mpi(_convert_bp, 786, args=('syn'), mpiarg=node.ls('events'))


def _convert_bp(event, mode):
    from nnodes import root
    from pyasdf import ASDFDataSet
    from seisbp import SeisBP
    from obspy import read_events

    with ASDFDataSet(f'raw_{mode}/{event}.h5', mode='r', mpi=False) as h5, \
        SeisBP(f'bp_{mode}/{event}.bp', 'w', True) as bp:
        if root.mpi.rank == 0:
            bp.write(read_events(f'events/{event}'))

        invs = root.load(f'inventories/{event}.pickle')

        for sta in invs:
            if len(tags := h5.waveforms[sta].get_waveform_tags()):
                bp.write(invs[sta])
                bp.write(h5.waveforms[sta][tags[0]])


    # for event in node.ls('events'):
    #     with ASDFDataSet(f'raw_syn/{event}.h5', mode='r', mpi=False) as ds:
    #         stas = ds.waveforms.list()

    #     node.add_mpi(_convert_bp, node.np, args=(event,), mpiarg=stas, group_mpiarg=True)
    #     break


def _convert_bp_(stas, event):
    from nnodes import root
    from pyasdf import ASDFDataSet
    from seisbp import SeisBP
    from obspy import read_events
    with ASDFDataSet(f'raw_obs/{event}.h5', mode='r', mpi=False) as h5, \
        SeisBP(f'bp_obs/{event}.bp', 'w', True) as bp:
        if root.mpi.rank == 0:
            bp.write(read_events(f'events/{event}'))

        invs = root.load(f'inventories/{event}.pickle')

        for sta in stas:
            bp.write(invs[sta])
            bp.write(h5.waveforms[sta].raw_obs)


# def convert_xml(arg):
#     from pyasdf import ASDFDataSet
#     from obspy import Inventory
#     from nnodes import 

#     event = arg[0]
    
#     with ASDFDataSet(f'{arg[1]}/{event}.h5', mode='r', mpi=False) as ds:
#         inv = Inventory()


#     from traceback import format_exc
#     from nnodes import root
#     from pyasdf import ASDFDataSet
#     from obspy import read, read_events, read_inventory
#     from .index import format_station

