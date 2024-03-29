import typing as tp


def index(node):
    event_stations = {}

    for e in node.ls('events'):
        pkl = node.load(f'bands/{e}.pickle')
        event_stations[e] = list(pkl.keys())
    
    node.dump(event_stations, 'event_stations.pickle')


def index2(node):
    stations = node.load('stations.pickle')
    station_locations = {}
    station_lines = {}

    for e in node.ls('events'):
        for line in node.readlines(f'../ns/stations/STATIONS.{e}'):
            if len(ll := line.split()) == 6:
                sta = ll[1] + '.' + ll[0]

                if sta in stations and sta not in station_locations:
                    slat = float(ll[2])
                    slon = float(ll[3])
                    station_locations[sta] = slat, slon
                    _format_station(station_lines, ll)
    
    node.dump(station_locations, 'station_locations.pickle')
    node.dump(station_lines, 'station_lines.pickle')


def index3(node):
    import numpy as np

    es = node.load('events.pickle')
    ss = node.load('stations.pickle')
    cs = ('R', 'T', 'Z')

    m = np.zeros([len(es), len(ss), 3, 3], dtype=int)

    for i, e in enumerate(es):
        print(e)
        
        bands = node.load(f'bands/{e}.pickle')

        for s in bands:
            j = ss.index(s)

            for c in bands[s]:
                k = cs.index(c)

                m[i, j, k, :] = bands[s][c]['syn']
    
    node.dump(m, 'measurements.npy')



def _format_station(lines: dict, ll: tp.List[str]):
    """Format a line in STATIONS file."""
    # location of dots for floating point numbers
    dots = 28, 41, 55, 62

    # line with station name
    line = ll[0].ljust(13) + ll[1].ljust(5)

    # add numbers with correct indentation
    for i in range(4):
        num = ll[i + 2]

        if '.' in num:
            nint, _ = num.split('.')
        
        else:
            nint = num

        while len(line) + len(nint) < dots[i]:
            line += ' '
        
        line += num
    
    lines[ll[1] + '.' + ll[0]] = line


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
    events = node.ls('events')
    node.add_mpi(_ft, len(events), mpiarg=events)


def ft2(node):
    for event in node.ls('events'):
        node.add_mpi(_ft2, node.np, args=(event,), mpiarg=node.ls(f'blend_obs/{event}'), group_mpiarg=True)


def _ft2(stas, event):
    from seisbp import SeisBP
    from nnodes import root
    from sebox.catalog import catalog
    import numpy as np

    nbands = catalog.process['nbands']

    nt_se = int(round((catalog.process['duration_encoding']) * 60 / catalog.process['dt']))
    df = 1 / catalog.process['dt'] / nt_se

    imin = int(np.ceil(1 / catalog.process['period_max'] / df))
    imax = int(np.floor(1 / catalog.process['period_min'] / df)) + 1
    fincr = (imax - imin) // nbands

    measurements = {}

    with SeisBP(f'proc_obs/{event}.bp', 'r') as obs_bp, SeisBP(f'proc_syn/{event}.bp', 'r') as syn_bp, \
        SeisBP(f'proc_49/{event}.bp', 'r') as syn2_bp:
        for stap in stas:
            sta = '.'.join(stap.split('.')[:2])
            
            try:
                wins_rtz = root.load(f'blend_obs/{event}/{sta}.pickle')
            
            except:
                # print(event, sta)
                continue

            if not any(len(wins) for wins in wins_rtz):
                continue

            for cmp in ('R', 'T', 'Z'):
                try:
                    obs_tr = obs_bp.trace(sta, cmp)
                    syn_tr = syn_bp.trace(sta, cmp)
                    syn2_tr = syn2_bp.trace(sta, cmp)
                
                except:
                    pass
                
                else:
                    _ft_trace(obs_tr, syn_tr, syn2_tr, wins_rtz[cmp], sta, cmp)


def _ft(event):
    from pyasdf import ASDFDataSet
    from seisbp import SeisBP
    from nnodes import root
    from sebox.catalog import catalog
    import numpy as np

    nbands = catalog.process['nbands']

    nt_se = int(round((catalog.process['duration_encoding']) * 60 / catalog.process['dt']))
    df = 1 / catalog.process['dt'] / nt_se

    imin = int(np.ceil(1 / catalog.process['period_max'] / df))
    imax = int(np.floor(1 / catalog.process['period_min'] / df)) + 1
    fincr = (imax - imin) // nbands
    nf = fincr * nbands

    measurements = {}

    with SeisBP(f'proc_obs/{event}.bp', 'r') as obs_bp, SeisBP(f'proc_syn/{event}.bp', 'r') as syn_bp, \
        ASDFDataSet(f'ft_obs/{event}.h5', mode='w', mpi=False) as obs_h5, ASDFDataSet(f'ft_syn/{event}.h5', mode='w', mpi=False) as syn_h5, \
        ASDFDataSet(f'ft_win/{event}.h5', mode='w', mpi=False) as win_h5:
        for sta in syn_bp.stations:
            if not root.has(pkl := f'blend_obs/{event}/{sta}.pickle'):
                continue
            
            try:
                wins_rtz = root.load(pkl)
            
            except:
                # print(event, sta)
                continue

            if not any(len(wins) for wins in wins_rtz):
                continue
            
            output = {}

            for cmp in ('R', 'T', 'Z'):
                try:
                    obs_tr = obs_bp.trace(sta, cmp)
                    syn_tr = syn_bp.trace(sta, cmp)
                
                except:
                    pass
                
                else:
                    m = _ft_trace(obs_tr, syn_tr, wins_rtz[cmp], sta, cmp)
                    # try:
                    #     m = _ft_trace(obs_tr, syn_tr, wins_rtz[cmp], cmp)
                    
                    # except:
                    #     print(event, sta, cmp, obs_tr.data)
                    #     continue

                    if m is not None:
                        output[cmp] = m
            
            if len(output):
                measurements[sta] = {}

                for cmp in ('R', 'T', 'Z'):
                    if cmp in output:
                        measurements[sta][cmp] = {}
                        ft_obs = output[cmp]['obs']
                        ft_syn = output[cmp]['syn']
                        ft_win = output[cmp]['win']

                        measurements[sta][cmp]['obs'] = output[cmp]['obs_bands']
                        measurements[sta][cmp]['syn'] = output[cmp]['syn_bands']
                        measurements[sta][cmp]['win'] = output[cmp]['win_bands']
                    
                    else:
                        ft_obs = np.full(nf, np.nan + 0j, dtype=complex)
                        ft_syn = np.full(nf, np.nan + 0j, dtype=complex)
                        ft_win = np.full(nf, np.nan + 0j, dtype=complex)
                    
                    obs_h5.add_auxiliary_data(ft_obs, 'FT', sta.replace('.', '_') + '_MX' + cmp, {}) # type: ignore
                    syn_h5.add_auxiliary_data(ft_syn, 'FT', sta.replace('.', '_') + '_MX' + cmp, {}) # type: ignore
                    win_h5.add_auxiliary_data(ft_win, 'FT', sta.replace('.', '_') + '_MX' + cmp, {}) # type: ignores
        
        root.dump(measurements, f'bands/{event}.pickle')


def _pad(data, nt):
    import numpy as np
    from sebox.catalog import catalog

    shape = data.shape
    dt = catalog.process['dt']

    if nt > shape[-1]:
        # expand observed data with zeros
        pad = list(shape)
        pad[-1] = nt - shape[-1]

        taper = np.zeros(pad, dtype=data.dtype)
        ntaper = int(catalog.process['taper'] * 60 / dt)
        data[..., -ntaper:] *= np.hanning(2 * ntaper)[ntaper:]
        
        return np.concatenate([data, taper], axis=-1)
    
    # ntaper = int(catalog.process['taper'] * 60 / dt)
    # data[3000: 3000 + ntaper] *= np.hanning(2 * ntaper)[ntaper:]
    # data[3000 + ntaper:] = 0.0
    
    return data[:nt]

def _ft_trace(obs_tr, syn_tr, syn2_tr, wins_all, sta, cmp):
    from scipy.fft import fft
    from pytomo3d.signal.process import sac_filter_trace
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')
    np.seterr(all='raise')

    from sebox.catalog import catalog

    nbands = catalog.process['nbands']
    bnames = ['#17-23s', '#23-40s', '#40-100s']
    dtx = obs_tr.stats.delta / 60
    ntx = obs_tr.stats.npts
    t = np.linspace(0, (ntx-1) * dtx, ntx)
    
    nt_se = int(round((catalog.process['duration_encoding']) * 60 / catalog.process['dt']))
    df = 1 / catalog.process['dt'] / nt_se

    imin = int(np.ceil(1 / catalog.process['period_max'] / df))
    imax = int(np.floor(1 / catalog.process['period_min'] / df)) + 1
    fincr = (imax - imin) // nbands
    imax = imin + fincr * nbands

    cl = catalog.process['corner_left']
    cr = catalog.process['corner_right']

    # print(nt_se, imin, imax, fincr)

    fobs = tp.cast(np.ndarray, fft(_pad(obs_tr.data, nt_se)))
    fsyn = tp.cast(np.ndarray, fft(_pad(syn_tr.data, nt_se)))
    fsyn2 = tp.cast(np.ndarray, fft(_pad(syn2_tr.data, nt_se)))
    # fobs = tp.cast(np.ndarray, fft(obs_tr.data))
    # fsyn = tp.cast(np.ndarray, fft(syn_tr.data))

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

        if len(wins) == 0:
            continue

        i1 = imin + iband * fincr
        i2 = i1 + fincr

        obs = obs_tr.copy()
        syn = syn_tr.copy()
        
        if sum(win.right - win.left + 1 for win in wins) / len(syn.data) < catalog.window['threshold_duration']:
            continue

        fmin = i1 * df
        fmax = (i2 - 1) * df
        pre_filt = [fmin * cr, fmin, fmax, fmax / cl]
        
        try:
            # obs.filter('bandpass', freqmin=fmin, freqmax=fmax)
            # syn.filter('bandpass', freqmin=fmin, freqmax=fmax)
            # print(obs.stats.npts, obs.stats.delta, syn.stats.npts, syn.stats.delta)
            sac_filter_trace(obs, pre_filt)
            sac_filter_trace(syn, pre_filt)
        
        except:
            # print('?', syn.data, obs.data)
            return

        diff = syn.data - obs.data
        syn_sum = sum(syn.data ** 2)
        obs_sum = sum(obs.data ** 2)
        diff_sum = sum(diff ** 2)

        def emean(data):
            return sum(data ** 2) / len(data)

        ratio_syn = sum(sum(syn.data[win.left: win.right] ** 2 / syn_sum) for win in wins)
        ratio_obs = sum(sum(obs.data[win.left: win.right] ** 2 / obs_sum) for win in wins)
        ratio_diff = sum(sum(diff[win.left: win.right] ** 2 / diff_sum) for win in wins)

        has_full = ratio_diff > catalog.window['threshold_diff']
        has_blended = ratio_syn > catalog.window['threshold_syn'] and ratio_obs > catalog.window['threshold_obs']

        # if has_full or has_blended:
        if has_full:
            output['syn'][i1-imin: i2-imin] = fsyn[i1: i2]
            output['obs'][i1-imin: i2-imin] = fobs[i1: i2]
            output['win'][i1-imin: i2-imin] = fobs[i1: i2]
            output['syn_bands'][iband] = 1

            # plt.figure(figsize=(12, 8))
            plt.figure(figsize=(20, 15))
            plt.plot(t, obs.data, label='obs')
            plt.plot(t, syn.data, label='syn')
            plt.plot(t, syn2_tr.data, label='syn_new')
            
            for win in wins:
                plt.axvspan(win.left * dtx, win.right * dtx, facecolor='lightgray')
            
            plt.legend()
            plt.savefig(f'plots2/{sta}.{cmp}{bnames[iband]}.pdf')
        
        else:
            continue
        
        if has_full:
            output['obs_bands'][iband] = 1

        if has_blended:
            output['win_bands'][iband] = 1
        
        nt = int(catalog.process['period_max'] / catalog.process['dt'] / 2)
        taper = np.hanning(nt * 2)

        d1 = obs.data.copy()
        d1_2 = obs.data.copy()
        d1_3 = obs.data.copy()
        d2 = syn.data
        d3 = syn2_tr.data

        bwins = []
        bwins2 = []
        title = ' '

        syn_mean = 0.0
        for win in wins:
            syn_mean += sum(syn.data[win.left:win.right+1] ** 2)
        syn_mean /= len(syn.data)

        for i, win in enumerate(wins):
            fl = 0 if i == 0 else wins[i-1].right + nt + 1
            fr = len(d1) - 1 if i == len(wins) - 1 else wins[i+1].left - nt - 1

            if win.left - fl >= nt:
                l = win.left - nt
                r = win.left
                d1[fl: l] = d2[fl: l]
                d1[l: r] += (d2[l: r] - d1[l: r]) * taper[nt:]
                bwins.append((fl,r))
                
                rb = emean(d2[fl:r]) / syn_mean
                title += f'{rb:.2f} '

                if rb < catalog.window['threshold_blend']:
                    d1_2[fl:r] = d1[fl:r]
                    d1_3[fl: l] = d3[fl: l]
                    d1_3[l: r] += (d3[l: r] - d1_3[l: r]) * taper[nt:]
                    bwins2.append((fl,r))

            
            if fr - win.right >= nt:
                l = win.right + 1
                r = win.right + nt + 1
                d1[r: fr + 1] = d2[r: fr + 1]
                d1[l: r] += (d2[l: r] - d1[l: r]) * taper[:nt]
                bwins.append((l,fr+1))

                rb = emean(d2[l:fr+1]) / syn_mean
                title += f'{rb:.2f} '

                if rb < catalog.window['threshold_blend']:
                    d1_2[l:fr+1] = d1[l:fr+1]
                    d1_3[r: fr + 1] = d3[r: fr + 1]
                    d1_3[l: r] += (d3[l: r] - d1_3[l: r]) * taper[:nt]
                    bwins2.append((l,fr+1))
        
        # if len(bwins):
        #     # plt.figure(figsize=(12, 8))
        #     plt.figure(figsize=(20, 15))
        #     plt.plot(t, d1, label='obs')
        #     plt.plot(t, d2, label='syn')

        #     for win in bwins:
        #         plt.axvspan(win[0] * dtx, win[1] * dtx, facecolor='lightgray')
            
        #     plt.legend()
        #     plt.title(title)
        #     plt.savefig(f'plots/{sta}.{cmp}{bnames[iband]}_w1.pdf')
        
        if len(bwins2):
            plt.figure(figsize=(20, 15))
            plt.plot(t, obs.data, label='obs')
            plt.plot(t, d1_2, label='obs_win')
            plt.plot(t, syn.data, label='syn')
            plt.plot(t, syn2_tr.data, label='syn_new')
            
            for win in wins:
                plt.axvspan(win.left * dtx, win.right * dtx, facecolor='lightgray')
            
            plt.legend()
            plt.savefig(f'plots/{sta}.{cmp}{bnames[iband]}.pdf')

            # # plt.figure(figsize=(12, 8))
            # plt.figure(figsize=(20, 15))
            # plt.plot(t, d1_2, label='obs')
            # plt.plot(t, d2, label='syn')
            # plt.plot(t, syn2_tr.data, label='syn_new')
            
            # for win in bwins2:
            #     plt.axvspan(win[0] * dtx, win[1] * dtx, facecolor='palegreen')
            
            # plt.legend()
            # plt.savefig(f'plots/{sta}.{cmp}{bnames[iband]}_w.pdf')
        
        plt.figure(figsize=(20, 15))
        plt.plot(t, obs.data, label='obs')
        plt.plot(t, d1_2, label='win')
        plt.plot(t, d1_3, label='win(49)')
        
        for win in wins:
            plt.axvspan(win.left * dtx, win.right * dtx, facecolor='lightgray')
        
        plt.legend()
        plt.savefig(f'plots2/{sta}.{cmp}{bnames[iband]}.win.pdf')
        
        # output['win'][i1-imin: i2-imin] = fft(_pad(d1, nt_se))[i1: i2]
        plt.figure(figsize=(12, 8))
        plt.plot(np.angle(fobs[i1: i2] / fsyn[i1: i2]), label='original')
        # if len(bwins):
        #     plt.plot(np.angle(fft(_pad(d1, nt_se))[i1: i2] / fsyn[i1: i2]), label='w1')
        #     pname += '1'
        if len(bwins2):
            plt.plot(np.angle(fft(_pad(d1_2, nt_se))[i1: i2] / fsyn[i1: i2]), label='windowed')
            plt.plot(np.angle(fft(_pad(d1_3, nt_se))[i1: i2] / fsyn[i1: i2]), label='windowed(49)')
            plt.legend()
            plt.savefig(f'plots/{sta}.{cmp}{bnames[iband]}.pdf')
        else:
            plt.legend()
            plt.savefig(f'plots3/{sta}.{cmp}{bnames[iband]}.pdf')

        
        if len(bwins2):
            plt.figure(figsize=(12, 8))
            plt.plot(np.angle(fft(_pad(d1_2, nt_se))[i1: i2] / fsyn[i1: i2]), label='old')
            plt.plot(np.angle(fft(_pad(d1_2, nt_se))[i1: i2] / fsyn2[i1: i2]), label='new')
            plt.plot(np.angle(fft(_pad(d1_3, nt_se))[i1: i2] / fsyn2[i1: i2]), label='new(49)')
            plt.legend()
            plt.savefig(f'plots/{sta}.{cmp}{bnames[iband]}.iter.pdf')


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

    ###### FIXME
    traces = root.load('traces.pickle')
    root.mkdir(f'{dst}/plots')
    ######

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

            ###### FIXME
            for cmp in ('Z',):
            # for cmp in ('R', 'T', 'Z'):
                try:
                    obs_tr = obs_bp.trace(sta, cmp)
                    syn_tr = syn_bp.trace(sta, cmp)
                
                except:
                    output[cmp] = [[], [], []]

                else:
                    output[cmp] = _window(obs_tr, syn_tr, evt, inv, cmp, traces[sta][cmp], f'{dst}/plots/{sta}')
            
            root.dump(output, f'{dst}/{sta}.pickle')
            print(f'{dst}/{sta}.pickle')
    
    print(root.mpi.rank, 'done')


def _window(obs_tr, syn_tr, evt, inv, cmp, bands, dst):
    from pyflex import Config, WindowSelector
    from pytomo3d.signal.process import sac_filter_trace
    import numpy as np

    from sebox.catalog import catalog

    nbands = catalog.process['nbands']

    df = 1 / catalog.process['duration_encoding'] / 60
    imin = int(np.ceil(1 / catalog.process['period_max'] / df))
    imax = int(np.floor(1 / catalog.process['period_min'] / df)) + 1
    fincr = (imax - imin) // nbands
    imax = imin + fincr * nbands

    cl = catalog.process['corner_left']
    cr = catalog.process['corner_right']

    output = [[]] * nbands

    for iband in range(nbands):
        ###### FIXME: remove parameter `bands` and `dst`
        savefig = catalog.window.get('savefig')
        bnames = ['#17-23s', '#23-40s', '#40-100s']
        if bands[iband] == 0:
            output[iband] = []
            continue
        ######

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

            if savefig and len(output[iband]):
                # use non-interactive backend
                import matplotlib
                matplotlib.use('Agg')

                ws.plot(filename=dst + bnames[iband] + '.png')
        
        except Exception:
            pass
    
    return output


def _blend_trace(obs_tr, syn_tr, evt, inv, cmp, event, station):
    from pyflex import Config, WindowSelector
    from nnodes import root
    from scipy.fft import fft
    from pytomo3d.signal.process import sac_filter_trace
    import numpy as np

    from sebox.catalog import catalog

    savefig = catalog.window.get('savefig')
    nbands = catalog.process['nbands']

    df = 1 / catalog.process['duration_encoding'] / 60
    imin = int(np.ceil(1 / catalog.process['period_max'] / df))
    imax = int(np.floor(1 / catalog.process['period_min'] / df)) + 1
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
            nt = int(catalog.process['period_max'] / catalog.process['dt'] / 2)
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
