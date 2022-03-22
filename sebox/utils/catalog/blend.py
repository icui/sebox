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
    l1 = ds1.waveforms.list()
    l2 = ds2.waveforms.list()

    for sta in l1[1:2]:
        if sta in l2:
            obs = ds1.waveforms[sta].proc_obs[2]
            syn = ds2.waveforms[sta].proc_syn[2]

            node.mkdir(f'blend/{event}')
            _blend(A(event, sta, obs), A(event, sta, syn))


def blend_eventx(node):
    from asdfy import ASDFProcessor

    event = node.event
    node.mkdir(f'blend/{event}')
    
    ap = ASDFProcessor((f'proc_obs/{event}.h5', f'proc_syn/{event}.h5'), f'blend_obs/{event}.h5', _blend, output_tag='blend_obs', accessor=True)
    node.add_mpi(ap.run, node.np, name=f'blend_{event}')


def _blend(obs_acc, syn_acc):
    from pyflex import Config, WindowSelector, select_windows
    from sebox import catalog
    from nnodes import root
    from scipy.fft import fft
    import numpy as np
    import matplotlib.pyplot as plt

    station = syn_acc.station
    event = syn_acc.event
    obs = obs_acc.trace
    syn = syn_acc.trace

    config = Config(min_period=catalog.period_min, max_period=catalog.period_max)
    ws = WindowSelector(obs, syn, config)
    wins = ws.select_windows()
    ratio = sum(sum(syn.data[win.left: win.right] ** 2) for win in wins) / sum(syn.data ** 2)

    if ratio > catalog.window['energy_threshold']:
        print(f'{station} {ratio:.2f}')

        d = root.subdir(f'blend/{event}/{station}')
        d.mkdir()
        ws.plot(filename=d.path('windows.png'))
        plt.clf()

        nt = int(catalog.period_max / catalog.dt / 2)
        taper = np.hanning(nt * 2)

        d1 = obs.data
        d2 = syn.data

        df = 1 / float(syn.stats.endtime - syn.stats.starttime)
        print(df)
        imin = int(np.ceil(1 / catalog.period_max / df))
        imax = int(np.floor(1 / catalog.period_min / df)) + 1

        f1 = fft(d1)[imin: imax]
        f2 = fft(d2)[imin: imax]

        for i, win in enumerate(wins):
            fl = 0 if i == 0 else wins[i-1].right + nt + 1
            fr = len(d1) - 1 if i == len(wins) - 1 else wins[i+1].left - nt - 1

            if win.left - fl >= nt:
                l = win.left - nt
                r = win.left
                d1[fl: l] = d2[fl: l]
                d1[l: r] += (d2[l: r] - d1[l: r]) * taper[nt:]
            
            if fr - win.right >= nt:
                l = win.right + 1
                r = win.right + nt + 1
                d1[r: fr + 1] = d2[r: fr + 1]
                d1[l: r] += (d2[l: r] - d1[l: r]) * taper[:nt]

        select_windows(obs, syn, config, plot=True, plot_filename=d.path('windows_blended.png'))
        f3 = fft(d1)[imin: imax]
        
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(np.angle(f1), label='obs')
        plt.plot(np.angle(f2), label='syn')
        plt.plot(np.angle(f1 / f2), label='diff')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(np.angle(f3), label='obs_glue')
        plt.plot(np.angle(f2), label='syn')
        plt.plot(np.angle(f3 / f2), label='diff')
        # plt.savefig(d.path('frequency.png'))
        plt.show()
        exit()

