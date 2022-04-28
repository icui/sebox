import typing as tp


def window2(node):
    events = node.ls('events')
    node.add_mpi(_blend2, len(events), mpiarg=events)


def _blend2(event) -> tp.Any:
    from seisbp import SeisBP
    from nnodes import root
    import logging
    import warnings

    logging.disable()
    warnings.filterwarnings("ignore")
    dst = f'blend_obs/{event}'

    with SeisBP(f'proc_obs/{event}.bp', 'r') as obs_bp, SeisBP(f'proc_syn/{event}.bp', 'r') as syn_bp:
        evt = syn_bp.read(syn_bp.events[0])

        for sta in obs_bp.stations:
            if sta in syn_bp.stations:
                if root.has(f'{dst}/{sta}.pickle'):
                    continue

                output = {}

                inv = syn_bp.read(sta)

                for cmp in ('R', 'T', 'Z'):
                    try:
                        obs_tr = obs_bp.trace(sta, cmp)
                        syn_tr = syn_bp.trace(sta, cmp)
                    
                    except:
                        output[cmp] = [[], [], []]

                    else:
                        output[cmp] = _window(obs_tr, syn_tr, evt, inv, cmp)
                
                root.dump(output, f'{dst}/{sta}.pickle')
                print(f'{dst}/{sta}.pickle')


                # inv = bp_syn.read(sta)

                # for cmp in ('R', 'T', 'Z'):
                #     obs_tr = bp_obs.trace(sta, cmp)
                #     syn_tr = bp_syn.trace(sta, cmp)

                #     if output := _blend_trace(obs_tr, syn_tr, evt, inv, cmp, bp_syn.events[0], sta):
                #         for tag, data in output.items():
                #             bp_w.put(f'{sta}.{cmp}:{tag}', data)
                #             print(event, sta)


def window(node):
    node.concurrent = True
    node.mkdir('blend_obs')

    for event in node.ls('events'):
        # if node.has(f'proc_obs/{event}.bp') and node.has(f'proc_syn/{event}.bp') and not node.has(f'blend_obs/{event}.bp'):
        if node.has(f'proc_obs/{event}.bp') and node.has(f'proc_syn/{event}.bp'):
            node.add(window_event, name=event, event=event)


def window3(node):
    from seisbp import SeisBP

    node.concurrent = True

    for e in node.ls('blend_obs'):
        if node.has(f'done/{e}'):
            continue

        with SeisBP(f'proc_syn/{e}.bp', 'r') as bp:
            ratio = len(node.ls(f'blend_obs/{e}')) / len(bp.stations)

            if ratio != 1:
                node.add(window_event, name=e, event=e)
            
            else:
                node.write(str(len(bp.stations)), f'done/{e}')


def ft(node):
    events = node.ls('events')[:2]
    node.add_mpi(_ft, len(events), mpiarg=events)


def _ft(event):
    from pyasdf import ASDFDataSet
    from seisbp import SeisBP
    from nnodes import root
    from sebox.catalog import catalog
    import numpy as np

    nbands = catalog.nbands

    df = 1 / catalog.duration_ft / 60
    imin = int(np.ceil(1 / catalog.period_max / df))
    imax = int(np.floor(1 / catalog.period_min / df)) + 1
    fincr = (imax - imin) // nbands
    nf = fincr * nbands

    measurements = {}

    with SeisBP(f'proc_obs/{event}.bp', 'r') as obs_bp, SeisBP(f'proc_syn/{event}.bp', 'r', True) as syn_bp, \
        ASDFDataSet(f'ft_obs/{event}.h5', mode='w', mpi=False) as obs_h5, ASDFDataSet(f'ft_syn/{event}.h5', mode='w', mpi=False) as syn_h5, \
        ASDFDataSet(f'ft_win/{event}.h5', mode='w', mpi=False) as win_h5:
        for sta in syn_bp.stations:
            if not root.has(pkl := f'blend_obs/{event}/{sta}.pickle'):
                continue
            
            wins_rtz = root.load(pkl)

            if not any(len(wins) for wins in wins_rtz):
                continue
            
            print(sta)
            
            output = {}

            for cmp in ('R', 'T', 'Z'):
                try:
                    obs_tr = obs_bp.trace(sta, cmp)
                    syn_tr = syn_bp.trace(sta, cmp)
                
                except:
                    pass
                
                else:
                    output[cmp] = _ft_trace(obs_tr, syn_tr, wins_rtz[cmp])
            
            if len(output):
                measurements[sta] = {}

                for cmp in ('R', 'T', 'Z'):
                    if cmp in output and output[cmp] is not None:
                        measurements[sta][cmp] = {}
                        ft_obs = output[cmp]['obs']
                        ft_syn = output[cmp]['syn']
                        ft_win = output[cmp]['win']

                        measurements[sta]['obs'] = output[cmp]['obs_bands']
                        measurements[sta]['syn'] = output[cmp]['syn_bands']
                        measurements[sta]['win'] = output[cmp]['win_bands']
                    
                    else:
                        ft_obs = np.zeros(nf, dtype=complex)
                        ft_syn = np.zeros(nf, dtype=complex)
                        ft_win = np.zeros(nf, dtype=complex)
                    
                    obs_h5.add_auxiliary_data(ft_obs, 'FT', sta.replace('.', '_') + '_MX' + cmp, {}) # type: ignore
                    syn_h5.add_auxiliary_data(ft_syn, 'FT', sta.replace('.', '_') + '_MX' + cmp, {}) # type: ignore
                    win_h5.add_auxiliary_data(ft_win, 'FT', sta.replace('.', '_') + '_MX' + cmp, {}) # type: ignore

                    print(sta, cmp)
        
        root.dump(measurements, f'bands/{event}.pickle')


def _ft_trace(obs_tr, syn_tr, wins_all):
    from scipy.fft import fft
    from pytomo3d.signal.process import sac_filter_trace
    import numpy as np

    from .catalog import catalog

    nbands = catalog.nbands
    df = 1 / obs_tr.stats.npts / obs_tr.stats.delta
    imin = int(np.ceil(1 / catalog.period_max / df))
    imax = int(np.floor(1 / catalog.period_min / df)) + 1
    fincr = (imax - imin) // nbands
    imax = imin + fincr * nbands
    print(imin, imax)

    cl = catalog.process['corner_left']
    cr = catalog.process['corner_right']

    try:
        fobs = tp.cast(np.ndarray, fft(obs_tr.data))
        fsyn = tp.cast(np.ndarray, fft(syn_tr.data))
    
    except:
        return

    output = {
        'syn': np.full(imax - imin, np.nan, dtype=complex),
        'syn_bands': np.zeros(nbands, dtype=int),
        'obs': np.full(imax - imin, np.nan, dtype=complex),
        'obs_bands': np.zeros(nbands, dtype=int),
        'win': np.full(imax - imin, np.nan, dtype=complex),
        'win_bands': np.zeros(nbands, dtype=int)
    }

    for iband in range(nbands):
        wins = wins_all[iband]
        i1 = imin + iband * fincr
        i2 = i1 + fincr
        print('>', i1, i2)

        obs = obs_tr.copy()
        syn = syn_tr.copy()

        fmin = i1 * df
        fmax = (i2 - 1) * df
        pre_filt = [fmin * cr, fmin, fmax, fmax / cl]
        
        sac_filter_trace(obs, pre_filt)
        sac_filter_trace(syn, pre_filt)

        diff = syn.data - obs.data
        ratio_syn = sum(sum(syn.data[win.left: win.right] ** 2) for win in wins) / sum(syn.data ** 2)
        ratio_obs = sum(sum(obs.data[win.left: win.right] ** 2) for win in wins) / sum(obs.data ** 2)
        ratio_diff = sum(sum(diff[win.left: win.right] ** 2) for win in wins) / sum(diff ** 2)

        has_full = ratio_diff > catalog.window['threshold_diff']
        has_blended = ratio_syn > catalog.window['threshold_syn'] and ratio_obs > catalog.window['threshold_obs']

        if has_full or has_blended:
            output['syn'][i1-imin: i2-imin] = fsyn[i1: i2]
            output['obs'][i1-imin: i2-imin] = fobs[i1: i2]
            output['syn_bands'][iband] = 1
        
        if has_full:
            output['obs_bands'][iband] = 1

        if has_blended:
            output['win_bands'][iband] = 1
            nt = int(catalog.period_max / catalog.dt / 2)
            taper = np.hanning(nt * 2)

            d1 = obs.data
            d2 = syn.data

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
            
            output['win'][i1-imin: i2-imin] = fft(d1)[i1: i2]

    if any(output['syn_bands']):
        return output


def window_event(node):
    from seisbp import SeisBP

    src = f'{node.event}.bp'
    # node.mkdir(f'blend/{node.event}')

    with SeisBP(f'proc_syn/{src}', 'r') as bp:
        stations = bp.stations
    
    node.add_mpi(_blend, node.np, name=f'blend_{node.event}',
        args=(f'proc_obs/{src}', f'proc_syn/{src}', f'blend_obs/{node.event}'),
        mpiarg=stations, group_mpiarg=True, cwd=f'log_blend')


def _blend3(np, stas, obs, syn, dst):
    from multiprocessing import Pool
    from functools import partial
    from nnodes.mpiexec import splitargs

    with Pool(processes=np) as pool:
        pool.map(partial(_blendx, obs, syn, dst), splitargs(stas, np))


def _blendx(obs, syn, dst, stas):
    _blend(stas, obs, syn, dst)


def _blend(stas, obs, syn, dst) -> tp.Any:
    from seisbp import SeisBP
    from nnodes import root
    import logging
    import warnings

    logging.disable()
    warnings.filterwarnings("ignore")

    with SeisBP(obs, 'r', True) as obs_bp, SeisBP(syn, 'r', True) as syn_bp:
        # SeisBP(dst, 'w', True) as dst_bp:
        evt = syn_bp.read(syn_bp.events[0])

        # if root.mpi.rank == 0:
        #     dst_bp.write(evt)

        for sta in stas:
            if root.has(f'{dst}/{sta}.pickle'):
                continue

            output = {}

            inv = syn_bp.read(sta)

            for cmp in ('R', 'T', 'Z'):
                try:
                    obs_tr = obs_bp.trace(sta, cmp)
                    syn_tr = syn_bp.trace(sta, cmp)
                
                except:
                    output[cmp] = [[], [], []]

                else:
                    output[cmp] = _window(obs_tr, syn_tr, evt, inv, cmp)
            
            root.dump(output, f'{dst}/{sta}.pickle')
            print(f'{dst}/{sta}.pickle')
    
    print(root.mpi.rank, 'done')


def _window(obs_tr, syn_tr, evt, inv, cmp):
    from pyflex import Config, WindowSelector
    from pytomo3d.signal.process import sac_filter_trace
    import numpy as np

    from .catalog import catalog

    nbands = catalog.nbands

    df = 1 / catalog.duration_ft / 60
    imin = int(np.ceil(1 / catalog.period_max / df))
    imax = int(np.floor(1 / catalog.period_min / df)) + 1
    fincr = (imax - imin) // nbands
    imax = imin + fincr * nbands

    cl = catalog.process['corner_left']
    cr = catalog.process['corner_right']

    output = [[]] * nbands

    for iband in range(nbands):
        i1 = imin + iband * fincr
        i2 = i1 + fincr

        obs = obs_tr.copy()
        syn = syn_tr.copy()

        fmin = i1 * df
        fmax = (i2 - 1) * df
        pre_filt = [fmin * cr, fmin, fmax, fmax / cl]
        
        sac_filter_trace(obs, pre_filt)
        sac_filter_trace(syn, pre_filt)
    
        cfg = catalog.window['flexwin']
        config = Config(min_period=1/fmax, max_period=1/fmin, **{**cfg['default'], **cfg[cmp]})
        ws = WindowSelector(obs, syn, config, evt, inv)

        try:
            output[iband] = ws.select_windows()
        
        except Exception:
            pass
    
    return output


def _blend_trace(obs_tr, syn_tr, evt, inv, cmp, event, station):
    from pyflex import Config, WindowSelector
    from nnodes import root
    from scipy.fft import fft
    from pytomo3d.signal.process import sac_filter_trace
    import numpy as np

    from .catalog import catalog

    savefig = catalog.window.get('savefig')
    nbands = catalog.nbands

    df = 1 / catalog.duration_ft / 60
    imin = int(np.ceil(1 / catalog.period_max / df))
    imax = int(np.floor(1 / catalog.period_min / df)) + 1
    fincr = (imax - imin) // nbands
    imax = imin + fincr * nbands

    cl = catalog.process['corner_left']
    cr = catalog.process['corner_right']

    fobs = tp.cast(np.ndarray, fft(obs_tr.data))
    fsyn = tp.cast(np.ndarray, fft(syn_tr.data))

    output = {
        'syn': np.full(imax - imin, np.nan, dtype=complex),
        'syn_bands': np.zeros(nbands, dtype=int),
        'obs': np.full(imax - imin, np.nan, dtype=complex),
        'obs_bands': np.zeros(nbands, dtype=int),
        'blend': np.full(imax - imin, np.nan, dtype=complex),
        'blend_bands': np.zeros(nbands, dtype=int)
    }

    for iband in range(nbands):
        i1 = imin + iband * fincr
        i2 = i1 + fincr

        obs = obs_tr.copy()
        syn = syn_tr.copy()

        fmin = i1 * df
        fmax = (i2 - 1) * df
        tag = f'{obs.stats.channel}_{int(1/fmax)}-{int(1/fmin)}'
        pre_filt = [fmin * cr, fmin, fmax, fmax / cl]
        
        sac_filter_trace(obs, pre_filt)
        sac_filter_trace(syn, pre_filt)
    
        cfg = catalog.window['flexwin']
        config = Config(min_period=1/fmax, max_period=1/fmin, **{**cfg['default'], **cfg[cmp]})
        ws = WindowSelector(obs, syn, config, evt, inv)

        try:
            wins = ws.select_windows()
        
        except Exception:
            continue

        diff = syn.data - obs.data
        ratio_syn = sum(sum(syn.data[win.left: win.right] ** 2) for win in wins) / sum(syn.data ** 2)
        ratio_obs = sum(sum(obs.data[win.left: win.right] ** 2) for win in wins) / sum(obs.data ** 2)
        ratio_diff = sum(sum(diff[win.left: win.right] ** 2) for win in wins) / sum(diff ** 2)

        has_full = ratio_diff > catalog.window['threshold_diff']
        has_blended = ratio_syn > catalog.window['threshold_syn'] and ratio_obs > catalog.window['threshold_obs']
        d = root.subdir(f'blend/{event}/{station}')

        if has_full or has_blended:
            output['syn'][i1-imin: i2-imin] = fsyn[i1: i2]
            output['syn_bands'][iband] = 1

            if savefig:
                # use non-interactive backend
                import matplotlib
                matplotlib.use('Agg')

                d.mkdir()
                ws.plot(filename=d.path(f'{tag}.png'))
        
        if has_full:
            output['obs'][i1-imin: i2-imin] = fobs[i1: i2]
            output['obs_bands'][iband] = 1

        if has_blended:
            nt = int(catalog.period_max / catalog.dt / 2)
            taper = np.hanning(nt * 2)

            d1 = obs.data
            d2 = syn.data

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
            
            output['blend'][i1-imin: i2-imin] = fft(d1)[i1: i2]
            output['blend_bands'][iband] = 1

            if savefig:
                import matplotlib.pyplot as plt

                f1 = fobs[i1: i2]
                f2 = fsyn[i1: i2]
                f3 = output['blend'][i1-imin: i2-imin]
                
                plt.clf()
                plt.figure()
                plt.title(f'{station}.{tag} {ratio_obs:.2f} {ratio_syn:.2f}')
                
                plt.subplot(3, 1, 1)
                plt.plot(d1, label='obs_blend')
                plt.plot(d2, label='syn')
                plt.legend()

                plt.subplot(3, 1, 2)
                plt.plot(np.angle(f1), label='obs')
                plt.plot(np.angle(f2), label='syn')
                plt.plot(np.angle(f1 / f2), label='diff')
                plt.legend()

                plt.subplot(3, 1, 3)
                plt.plot(np.angle(f3), label='obs_blend')
                plt.plot(np.angle(f2), label='syn')
                plt.plot(np.angle(f3 / f2), label='diff')
                plt.legend()
                
                plt.savefig(d.path(f'{tag}_blend.png'))

    if any(output['syn_bands']):
        return output


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
