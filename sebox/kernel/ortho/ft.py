from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir, getstations, getcomponents

if tp.TYPE_CHECKING:
    from numpy import ndarray
    from .typing import Ortho


def ft_syn(node: Ortho, data: ndarray):
    from scipy.fft import fft
    return fft(data[..., node.nt_ts: node.nt_ts + node.nt_se])[..., node.imin: node.imax] # type: ignore


def ft_obs(node: Ortho, data: ndarray):
    import numpy as np
    from scipy.fft import fft

    shape = data.shape

    if (nt := node.kf * node.nt_se) > len(data):
        # expand observed data with zeros
        pad = list(shape)
        pad[-1] = nt - len(data)
        data = np.concatenate([data, pad])
    
    else:
        data = data[..., :nt]

    return fft(data)[..., ::node.kf][..., node.imin: node.imax] # type: ignore


def ft(node: Ortho, _):
    import numpy as np
    from sebox import root
    from sebox.mpi import pid

    root.restore(node)
    stats = node.load('forward/traces/stats.pickle')
    data = node.load(f'forward/traces/{pid}.npy')
    
    # resample if necessary
    if not np.isclose(stats['dt'], node.dt):
        from scipy.signal import resample
        print('resample:', stats['dt'], '->', node.dt)
        resample(data, int(round(stats['nt'] * stats['dt'] / node.dt)), axis=-1)

    # FFT
    data_nez = ft_syn(node, data)
    data_rtz = rotate_frequencies(node, data_nez, stats['cmps'], True)
    node.dump(data_rtz, f'enc_syn/{pid}.npy', mkdir=False)


def mf(node: Ortho, stas: tp.List[str]):
    import numpy as np
    from scipy.fft import ifft
    from sebox.mpi import pid, rank
    from mpi4py.MPI import COMM_WORLD as comm

    # read data
    root.restore(node)
    stats = node.load('forward/traces/stats.pickle')
    syn = node.load(f'enc_syn/{pid}.npy')
    obs = node.load(f'enc_obs/{pid}.npy')
    ref = node.load(f'enc_diff/{pid}.npy')
    weight = node.load(f'enc_weight/{pid}.npy')

    # clip phases
    weight[np.where(abs(ref) > np.pi)] = 0.0

    # compute diff
    phase_diff = np.angle(syn / obs) * weight
    amp_diff = np.log(np.abs(syn) / np.abs(obs)) * weight

    if node.double_difference:
        # unwrap or clip phases
        phase_sum = sum(comm.allgather(np.nansum(phase_diff, axis=0)))
        amp_sum = sum(comm.allgather(np.nansum(amp_diff, axis=0)))
        weight_sum = sum(comm.allgather(np.nansum(weight, axis=0)))
        weight_sum[np.where(weight_sum == 0.0)] = 1.0 # type: ignore

        # sum of phase and amplitude differences
        phase_diff -= phase_sum / weight_sum
        amp_diff -= amp_sum / weight_sum

    # apply measurement weightings
    omega = np.arange(node.imin, node.imax) / node.imin
    phase_diff *= weight * node.phase_factor / omega
    amp_diff *= weight * node.amplitude_factor

    # save misfit value
    fincr = node.frequency_increment

    def save_misfit(diff, name):
        mf = np.zeros([syn.shape[0], syn.shape[1], node.nbands_used])

        for i in range(node.nbands_used):
            mf[..., i] = np.nansum(diff[..., i * fincr: (i + 1) * fincr] ** 2, axis=-1)
        
        mf_sum = comm.gather(mf.sum(axis=0), root=0)

        if rank == 0:
            node.dump(sum(mf_sum), f'{name}_mf.npy', mkdir=False)

    save_misfit(phase_diff, 'phase')

    if node.amplitude_factor > 0:
        save_misfit(amp_diff, 'amp')

    if not node.misfit_only:
        from pyasdf import ASDFDataSet

        # compute adjoint sources
        nan = np.where(np.isnan(phase_diff))
        syn[nan] = 1.0

        phase_adj = phase_diff * (1j * syn) / abs(syn) ** 2
        phase_adj[nan] = 0.0

        if node.amplitude_factor > 0:
            amp_adj = amp_diff * syn / abs(syn) ** 2
            amp_adj[nan] = 0.0
        
        else:
            amp_adj = np.zeros(syn.shape)

        # fourier transform of adjoint source time function
        ft_adj = rotate_frequencies(node, phase_adj + amp_adj, stats['cmps'], False)

        # fill full frequency band
        ft_adstf = np.zeros([len(stas), 3, node.nt_se], dtype=complex)
        ft_adstf[..., node.imin: node.imax] = ft_adj
        ft_adstf[..., -node.imin: -node.imax: -1] = np.conj(ft_adj)

        # stationary adjoint source
        adstf_tau = -ifft(ft_adstf).real # type: ignore

        # repeat to fill entrie adjoint duration
        nt = stats['nt']
        adstf_tile = np.tile(adstf_tau, int(np.ceil(nt / node.nt_se)))
        adstf = adstf_tile[..., -nt:]

        if node.taper:
            ntaper = int(node.taper * 60 / node.dt)
            adstf[..., -ntaper:] *= np.hanning(2 * ntaper)[ntaper:]
        
        with ASDFDataSet('adjoint.h5', mode='w', compression=None, mpi=True) as ds:
            data_type = 'AdjointSources'

            def get_info(data, sta, cmp):
                path = sta.replace('.', '_') + '_MX' + cmp
                return ds._add_auxiliary_data_get_collective_information(
                    data=data,
                    data_type=data_type,
                    tag_path=[path],
                    parameters={}
                )

            # write collective trace meta data
            for i, sta in enumerate(getstations()):
                for j, cmp in enumerate(stats['cmps']):
                    data = adstf[i, j]
                    info = get_info(data, sta, cmp)
                    ds._add_auxiliary_data_write_collective_information(info=info)
            
            comm.barrier()
            
            # write independent trace data
            for i, sta in enumerate(stas):
                for j, cmp in enumerate(stats['cmps']):
                    data = adstf[i, j]
                    info = get_info(data, sta, cmp)
                    ds._add_auxiliary_data_write_independent_information(info=info, data=data)

        node.dump((adstf, stas, stats['cmps']), f'enc_mf/{pid}.pickle', mkdir=False)


def gather(node: Ortho):
    # get total misfit
    mf = node.load('phase_mf.npy').sum()

    if node.amplitude_factor > 0:
        mf += node.load('amp_mf.npy').sum()

    node.parent.parent.misfit_kl = mf


def rotate_frequencies(node: Ortho, data: ndarray, cmps_ne: tp.Tuple[str, str, str], direction: bool):
    import numpy as np
    from sebox.mpi import pid

    cmps_rt = getcomponents()
    data_rot = np.zeros(data.shape, dtype=complex)
    baz = getdir().load(f'baz_p{root.task_nprocs}/{pid}.pickle')

    if direction:
        # rotate from NE to RT
        cmps_from = cmps_ne
        cmps_to = cmps_rt
        c1, c2 = cmps_ne.index('N'), cmps_ne.index('E')
        c3, c4 = cmps_rt.index('R'), cmps_rt.index('T')
    
    else:
        # rotate from RT to NE
        cmps_from = cmps_rt
        cmps_to = cmps_ne
        c1, c2 = cmps_rt.index('R'), cmps_rt.index('T')
        c3, c4 = cmps_ne.index('N'), cmps_ne.index('E')

    data_rot[:, cmps_from.index('Z'), :] = data[:, cmps_to.index('Z'), :]

    for event, slots in node.fslots.items():
        if len(slots) == 0:
            continue
        
        ba = baz[event] if direction else 2 * np.pi - baz[event]

        for slot in slots:
            a = data[:, c1, slot]
            b = data[:, c2, slot]
            data_rot[:, c3, slot] = - b * np.sin(ba) - a * np.cos(ba)
            data_rot[:, c4, slot] = - b * np.cos(ba) + a * np.sin(ba)
            
    return data_rot
