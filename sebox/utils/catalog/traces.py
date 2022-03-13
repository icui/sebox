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
    from obspy import read_events

    event = read_events(f'events/{node.event}')
    print(event)


def compute_synthetic(node):
    """Compute synthetic data."""
