def prepare_traces(node):
    node.add(download_events)
    node.add(download_traces)
    node.add(compute_synthetic)


def download_events(node):
    """Download events."""


def download_traces(node):
    """Download observed data."""
    for event in node.ls('events'):
        node.add(download_trace, event=event)


def download_trace(node):
    """Download observed data of an event."""
    node.add(_download_fdsn)
    node.add(_convert_h5)


def _download_fdsn(node):
    from sebox import catalog
    from obspy import read_events
    from obspy.clients.fdsn.mass_downloader import GlobalDomain, Restrictions, MassDownloader

    event = read_events(f'events/{node.event}')[0]
    node.mkdir(mdir := f'downloads/{node.event}/xml')
    node.mkdir(xdir := f'downloads/{node.event}/mseed')

    gap = catalog.config['download_gap']
    eventtime = event.preferred_origin().time
    starttime = eventtime - gap * 60
    endtime = eventtime + (catalog.duration[0] + gap) * 60

    rst = Restrictions(starttime=starttime, endtime=endtime, **catalog.config['download'])
    mdl = MassDownloader()
    mdl.download(GlobalDomain(), rst, mseed_storage=mdir, stationxml_storage=xdir)


def _convert_h5(node):
    pass


def compute_synthetic(node):
    """Compute synthetic data."""
