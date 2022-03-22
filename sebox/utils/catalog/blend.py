def blend(node):
    node.mkdir('blend_obs')

    for event in node.ls('events')[1:]:
        if node.has(f'proc_obs/{event}.h5') and node.has(f'proc_syn/{event}.h5'):
            node.add(blend_event, event=event)


def blend_event(node):
    from pyasdf import ASDFDataSet

    event = node.event
    
    ds1 = ASDFDataSet(f'proc_obs/{event}.h5', mode='r', mpi=False)
    ds2 = ASDFDataSet(f'proc_syn/{event}.h5', mode='r', mpi=False)

    sta = 'TA.C24K'
    # sta = 'AZ.BZN'
    obs = ds1.waveforms[sta].proc_obs[2]
    syn = ds2.waveforms[sta].proc_syn[2]

    _blend(event, obs, syn)


def blend_event_(node):
    from functools import partial
    from asdfy import ASDFProcessor

    event = node.event
    node.mkdir(f'blend/{event}')
    
    ap = ASDFProcessor((f'proc_obs/{event}.h5', f'proc_syn/{event}.h5'), f'blend_obs/{event}.h5',
        partial(_blend, event), output_tag='blend_obs')
    node.add_mpi(ap.run, node.np, name=f'blend_{event}')


def _blend(event, obs, syn):
    from pyflex import Config, select_windows
    from sebox import catalog

    config = Config(min_period=catalog.period_min, max_period=catalog.period_max)
    win = select_windows(obs, syn, config, plot=True, plot_filename=f'blend/{event}/window.png')

    print(win)
