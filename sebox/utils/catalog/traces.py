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
    node.add(_download_mseed)
    node.add(_convert_h5)


def _download_mseed(node):
    from obspy import read_events
    from obspy.clients.fdsn.mass_downloader import GlobalDomain, Restrictions, MassDownloader

    event = read_events(f'events/{node.event}')
    node.mkdir(mdir := f'downloads/{node.event}/xml')
    node.mkdir(xdir := f'downloads/{node.event}/mseed')

    eventtime = event.preferred_origin().time
    starttime = eventtime - node.download_gap * 60
    endtime = eventtime + (node.duration[0] + node.download_gap) * 60

    rst = Restrictions(starttime=starttime, endtime=endtime, **node.download)
    mdl = MassDownloader()
    mdl.download(GlobalDomain(), rst, mseed_storage=mdir, stationxml_storage=xdir)


def _convert_h5(node):
    pass


def compute_synthetic(node):
    """Compute synthetic data."""
