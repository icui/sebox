def blend(node):
    node.mkdir('blend_obs')

    for event in node.ls('events')[1:]:
        if node.has(f'proc_obs/{event}.h5') and node.has(f'proc_syn/{event}.h5'):
            node.add(blend_event, event=event)


def blend_event(node):
    from collections import namedtuple
    from pyasdf import ASDFDataSet

    event = node.event
    
    ds1 = ASDFDataSet(f'proc_obs/{event}.h5', mode='r', mpi=False)
    ds2 = ASDFDataSet(f'proc_syn/{event}.h5', mode='r', mpi=False)
    A = namedtuple('A', ['event', 'station', 'trace'])

    sta = 'TA.C24K'
    # sta = 'AZ.BZN'
    obs = ds1.waveforms[sta].proc_obs[2]
    syn = ds2.waveforms[sta].proc_syn[2]

    node.mkdir(f'blend/{event}')
    _blend(A(event, sta, obs), A(event, sta, syn))


def blend_eventx(node):
    from asdfy import ASDFProcessor

    event = node.event
    node.mkdir(f'blend/{event}')
    
    ap = ASDFProcessor((f'proc_obs/{event}.h5', f'proc_syn/{event}.h5'), f'blend_obs/{event}.h5',
        _blend, output_tag='blend_obs', accessor=True)
    node.add_mpi(ap.run, node.np, name=f'blend_{event}')


def _blend(obs_acc, syn_acc):
    from pyflex import Config, WindowSelector
    from sebox import catalog
    import matplotlib.pyplot as plt

    station = syn_acc.station
    event = syn_acc.event
    obs = obs_acc.trace
    syn = syn_acc.trace

    config = Config(min_period=catalog.period_min, max_period=catalog.period_max)
    ws = WindowSelector(obs, syn, config)
    wins = ws.select_windows()
    ratio = sum(sum(syn.data[win.left: win.right] ** 2) for win in wins) / sum(syn.data ** 2)
    plt.title(f'ratio: {ratio:.2f}')
    ws.plot()
    

