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
    from pyflex import Config, WindowSelector
    from nnodes import root
    from scipy.fft import fft
    from pytomo3d.signal.process import sac_filter_trace
    import numpy as np

    from .catalog import catalog

    station = syn_acc.station
    event = syn_acc.event
    cmp = syn_acc.trace.stats.component
    savefig = catalog.window.get('savefig')

    df = 1 / catalog.duration_ft / 60
    imin = int(np.ceil(1 / catalog.period_max / df))
    imax = int(np.floor(1 / catalog.period_min / df)) + 1
    fincr = (imax - imin) // 3
    imax = imin + fincr * 3

    ft_obs = np.full(imax - imin, np.nan)

    for i in range(imin, imax, fincr):
        obs = obs_acc.trace.copy()
        syn = syn_acc.trace.copy()

        cl = catalog.process['corner_left']
        cr = catalog.process['corner_right']
        fmin = i * df
        fmax = fmin + df * fincr
        pre_filt = [fmin * cr, fmin, fmax, fmax / cl]
        
        sac_filter_trace(obs, pre_filt)
        sac_filter_trace(syn, pre_filt)
    
        cfg = catalog.window['flexwin']
        config = Config(min_period=catalog.period_min, max_period=catalog.period_max, **{**cfg['default'], **cfg[cmp]})
        ws = WindowSelector(obs, syn, config, syn_acc.ds.events[0], syn_acc.inventory)
        wins = ws.select_windows()

        diff = syn.data - obs.data
        ratio1 = sum(sum(syn.data[win.left: win.right] ** 2) for win in wins) / sum(syn.data ** 2)
        ratio2 = sum(sum(obs.data[win.left: win.right] ** 2) for win in wins) / sum(obs.data ** 2)
        ratio3 = sum(sum(diff[win.left: win.right] ** 2) for win in wins) / sum(diff ** 2)

        # if min(ratio1, ratio2, ratio3) > catalog.window['energy_threshold']:
        if ratio3 > catalog.window['energy_threshold']:
            cha = f'{obs.stats.channel}#{int(1/fmax)}-{int(1/fmin)}'
            print(f'{station} {cha} {ratio1:.2f} {ratio2:.2f} {ratio3:.2f}')

            d = root.subdir(f'blend/{event}/{station}')

            if savefig:
                # use non-interactive backend
                import matplotlib
                matplotlib.use('Agg')

                d.mkdir()
                ws.plot(filename=d.path(f'{cha}_windows.png'))

            # nt = int(catalog.period_max / catalog.dt / 2)
            # taper = np.hanning(nt * 2)

            # d1 = obs.data
            # d2 = syn.data

            # df = 1 / float(syn.stats.endtime - syn.stats.starttime)
            # imin = int(np.ceil(1 / catalog.period_max / df))
            # imax = int(np.floor(1 / catalog.period_min / df)) + 1

            # f1 = tp.cast(np.ndarray, fft(d1)[imin: imax])
            # f2 = tp.cast(np.ndarray, fft(d2)[imin: imax])

            # for i, win in enumerate(wins):
            #     fl = 0 if i == 0 else wins[i-1].right + nt + 1
            #     fr = len(d1) - 1 if i == len(wins) - 1 else wins[i+1].left - nt - 1

            #     if win.left - fl >= nt:
            #         l = win.left - nt
            #         r = win.left
            #         d1[fl: l] = d2[fl: l]
            #         d1[l: r] += (d2[l: r] - d1[l: r]) * taper[nt:]
                
            #     if fr - win.right >= nt:
            #         l = win.right + 1
            #         r = win.right + nt + 1
            #         d1[r: fr + 1] = d2[r: fr + 1]
            #         d1[l: r] += (d2[l: r] - d1[l: r]) * taper[:nt]

            # if savefig:
            #     import matplotlib.pyplot as plt
                
            #     plt.clf()
            #     plt.figure()
            #     plt.title(f'{station}.{cha} {ratio1:.2f} {ratio2:.2f}')
                
            #     plt.subplot(3, 1, 1)
            #     plt.plot(d1, label='obs_blend')
            #     plt.plot(d2, label='syn')
            #     plt.legend()

            #     plt.subplot(3, 1, 2)
            #     plt.plot(np.angle(f1), label='obs')
            #     plt.plot(np.angle(f2), label='syn')
            #     plt.plot(np.angle(f1 / f2), label='diff')
            #     plt.legend()

            #     plt.subplot(3, 1, 3)
            #     f3 = tp.cast(np.ndarray, fft(d1)[imin: imax])
            #     plt.plot(np.angle(f3), label='obs_blend')
            #     plt.plot(np.angle(f2), label='syn')
            #     plt.plot(np.angle(f3 / f2), label='diff')
            #     plt.legend()
                
            #     plt.savefig(d.path(f'{cha}_blended.png'))


def plot_stations(node):
    from pyasdf import ASDFDataSet
    from cartopy.crs import PlateCarree
    from cartopy.feature import LAND
    import matplotlib.pyplot as plt

    for event in node.ls('blend'):
        with ASDFDataSet(f'blend_obs/{event}.h5', mode='r', mpi=False) as ds:
            for station in node.ls(f'blend/{event}'):
                e = ds.events[0].preferred_origin()
                s: tp.Any = ds.waveforms[station].StationXML[0][0]

                plt.clf()
                fig = plt.figure()
                crs = PlateCarree()
                ax = fig.add_subplot(1, 1, 1, projection=crs)
                ax.gridlines()
                ax.add_feature(LAND, zorder=0, edgecolor='black', facecolor=(0.85, 0.85, 0.85))

                ax.plot([s.longitude, e.longitude], [s.latitude, e.latitude], color='black', alpha=0.5)
                ax.scatter(e.longitude, e.latitude, s=80, color="r", marker="*", edgecolor="k", linewidths=0.7, transform=PlateCarree())
                ax.scatter(s.longitude, s.latitude, s=60, color="b", marker=".", edgecolor="k", linewidths=0.7, transform=PlateCarree())
                plt.title(f'{event} {station}  lat: {s.latitude:.2f}  lon: {s.longitude:.2f}')
                plt.savefig(dst := f'blend/{event}/{station}/location.png')
                print(dst)
