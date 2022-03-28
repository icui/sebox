import typing as tp


def window(node):
    node.concurrent = True
    node.mkdir('blend_obs')

    for event in node.ls('events'):
        if node.has(f'proc_obs/{event}.h5') and node.has(f'proc_syn/{event}.h5') and not node.has(f'blend_obs/{event}.h5'):
            node.add(window_event, name=event, event=event)


def window_event(node):
    from asdfy import ASDFProcessor

    src = f'{node.event}.h5'
    node.mkdir(f'blend/{node.event}')
    
    ap = ASDFProcessor((f'proc_obs/{src}', f'proc_syn/{src}'), f'blend_obs/_{src}',
        _blend, output_tag='blend_obs', accessor=True)
    
    node.add_mpi(ap.run, node.np, name=f'blend', fname=node.event, cwd=f'log_blend')
    node.add(node.mv, args=(f'blend_obs/_{src}', f'blend_obs/{src}'), name='move_traces')


def _blend(obs_acc, syn_acc):
    from pyflex import Config, WindowSelector, select_windows
    from nnodes import root
    from scipy.fft import fft
    import numpy as np

    from .catalog import catalog

    station = syn_acc.station
    event = syn_acc.event
    obs = obs_acc.trace
    syn = syn_acc.trace
    cha = obs.stats.channel
    savefig = catalog.window.get('savefig')
    
    cfg = catalog.window['flexwin']
    config = Config(min_period=catalog.period_min, max_period=catalog.period_max, **{**cfg['default'], **cfg[cha[-1]]})
    ws = WindowSelector(obs, syn, config, syn_acc.ds.events[0], syn_acc.invenntory)
    wins = ws.select_windows()
    ratio1 = sum(sum(syn.data[win.left: win.right] ** 2) for win in wins) / sum(syn.data ** 2)
    ratio2 = sum(sum(obs.data[win.left: win.right] ** 2) for win in wins) / sum(obs.data ** 2)

    if min(ratio1, ratio2) > catalog.window['energy_threshold']:
        print(f'{station} {ratio1:.2f} {ratio2:.2f}')

        d = root.subdir(f'blend/{event}/{station}')
        d.mkdir()

        if savefig:
            import matplotlib

            matplotlib.use('Agg')

        ws.plot(filename=d.path(f'{cha}_windows.png') if savefig else None)

        nt = int(catalog.period_max / catalog.dt / 2)
        taper = np.hanning(nt * 2)

        d1 = obs.data
        d2 = syn.data

        df = 1 / float(syn.stats.endtime - syn.stats.starttime)
        imin = int(np.ceil(1 / catalog.period_max / df))
        imax = int(np.floor(1 / catalog.period_min / df)) + 1

        f1 = tp.cast(np.ndarray, fft(d1)[imin: imax])
        f2 = tp.cast(np.ndarray, fft(d2)[imin: imax])

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

        if savefig:
            import matplotlib.pyplot as plt
            from cartopy.crs import PlateCarree
            from cartopy.feature import LAND
            
            plt.clf()
            select_windows(obs, syn, config, plot=True, plot_filename=d.path(f'{cha}_blended.png'))

            f3 = tp.cast(np.ndarray, fft(d1)[imin: imax])
            
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.plot(np.angle(f1), label='obs')
            plt.plot(np.angle(f2), label='syn')
            plt.plot(np.angle(f1 / f2), label='diff')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(np.angle(f3), label='obs_glue')
            plt.plot(np.angle(f2), label='syn')
            plt.plot(np.angle(f3 / f2), label='diff')
            plt.legend()

            plt.title(f'{station}.{cha} {ratio1:.2f} {ratio2:.2f}')
            plt.savefig(d.path(f'{cha}_frequency.png'))

            if not d.has('location.png'):
                plt.clf()
                
                e = syn_acc.origin
                s = syn_acc.inventory[0][0]
                fig = plt.figure()
                crs = PlateCarree()
                ax = fig.add_subplot(1, 1, 1, projection=crs)
                ax.gridlines()
                ax.add_feature(LAND, zorder=0, edgecolor='black', facecolor=(0.85, 0.85, 0.85))

                ax.scatter(e.longitude, e.latitude, s=80, color="r", marker="*", edgecolor="k", linewidths=0.7, transform=PlateCarree())
                ax.scatter(s.longitude, s.latitude, s=60, color="steelblue", marker=".", edgecolor="k", linewidths=0.7, transform=PlateCarree())
                plt.title(f'{event} {station}  lat: {s.latitude:.2f}  lon: {s.longitude:.2f}')
                plt.savefig(d.path('location.png'))

        # return obs
        exit()

