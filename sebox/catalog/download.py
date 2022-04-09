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
    from pyasdf import ASDFDataSet

    for event in node.ls('events'):
        with ASDFDataSet(f'raw_syn/{event}.h5', mode='r', mpi=False) as ds:
            stas = ds.waveforms.list()

        node.add_mpi(_convert_bp, 1, args=(event,), mpiarg=stas, group_mpiarg=True)
        break


def _convert_bp(stas, event):
    import adios2
    import numpy as np
    from nnodes import root
    from pyasdf import ASDFDataSet
    
    with ASDFDataSet(f'raw_obs/{event}.h5', mode='r', mpi=False) as obs_h5, \
        ASDFDataSet(f'raw_syn/{event}.h5', mode='r', mpi=False) as syn_h5, \
        ASDFDataSet(f'bp_obs/{event}.h5', mode='w', mpi=False, compression=None) as h5:
        # adios2.open(f'bp_obs/{event}.bp', 'w', root.mpi.comm) as bp:
        for sta in stas:
            h5.add_waveforms(obs_h5.waveforms[sta].raw_obs, 'raw_obs')

            # for tr in obs_h5.waveforms[sta].raw_obs:
            #     bp.write(f'{sta}.{tr.stats.channel}', tr.data, count=tr.data.shape)
        
        # bps = [obs_bp, syn_bp]
        # lines = root.readlines(f'events/{event}')[2:13]
        # edata = np.array([float(l.split()[-1]) for l in lines])

        # for bp in bps:
        #     bp.write(event, edata, count=[11], end_step=True)

        #     for sta in stas:
        #         s = syn_h5.waveforms[sta].StationXML.networks[0].stations[0] # type: ignore
        #         bp.write(sta, np.array([s.latitude, s.longitude, s.elevation, s.channels[0].depth]), count=[4], end_step=(sta==stas[-1]))



    #     bp.end_step()
        
    #     print('step 2:', root.mpi.rank)
        
    #     for sta in stas:
    #         bp.write(sta, obs_h5.waveforms[sta].raw_obs)
        
    #     print('step 3:', root.mpi.rank)
    #     bp.end_step()


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

