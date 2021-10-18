from __future__ import annotations
import typing as tp

from sebox import root
from sebox.utils.catalog import getdir, getcomponents

if tp.TYPE_CHECKING:
    from numpy import ndarray
    from .typing import Encoding


def ft_syn(enc: Encoding, data: ndarray):
    """Get Fourier coefficients of synthetic data."""
    from scipy.fft import fft
    return fft(data[..., enc['nt_ts']: enc['nt_ts'] + enc['nt_se']])[..., enc['imin']: enc['imax']] # type: ignore


def ft_obs(enc: Encoding, data: ndarray):
    """Get Fourier coefficients of observed data."""
    import numpy as np
    from scipy.fft import fft

    shape = data.shape
    nt = enc['kf'] * enc['nt_se']

    if nt > shape[-1]:
        # expand observed data with zeros
        pad = list(shape)
        pad[-1] = nt - shape[-1]
        data = np.concatenate([data, pad])
    
    else:
        data = data[..., :nt]

    return fft(data)[..., ::enc['kf']][..., enc['imin']: enc['imax']] # type: ignore


def ft(enc: Encoding, _):
    """Process synthetic traces."""
    import numpy as np
    from sebox import root

    stats = root.mpi.load('../forward/traces/stats.pickle')
    data = root.mpi.mpiload(f'../forward/traces')
    
    # resample if necessary
    dt = enc['dt']
    if not np.isclose(stats['dt'], dt):
        from scipy.signal import resample
        print('resample:', stats['dt'], '->', dt)
        resample(data, int(round(stats['nt'] * stats['dt'] / dt)), axis=-1)

    # FFT
    data_nez = ft_syn(enc, data)
    data_rtz = rotate_frequencies(enc, data_nez, stats['cmps'], True)
    root.mpi.mpidump(data_rtz)


def mfadj(enc: Encoding, stas: tp.List[str]):
    """Compute misfit and adjoint source."""
    from pyasdf import ASDFDataSet

    # raise error last to ensure comm.barrier() succeeds
    err = None

    try:
        adstf, cmps = mf(enc, stas, False)
    
    except Exception as e:
        err = e
    
    # write to adjoint.h5 in sequence
    for k in range(root.mpi.size):
        try:
            if k == root.mpi.rank:
                with ASDFDataSet(root.mpi.path('../adjoint.h5'), mode='a' if k else 'w', compression=None, mpi=False) as ds:
                    for i, sta in enumerate(stas):
                        for j, cmp in enumerate(cmps): # type: ignore
                            ds.add_auxiliary_data(adstf[i, j], 'AdjointSources', sta.replace('.', '_') + '_MX' + cmp, {}) # type: ignore
        
        except Exception as e:
            err = e
        
        root.mpi.comm.barrier()
    
    if err:
        raise err


def mf(enc: Encoding, stas: tp.List[str], misfit_only: bool = True):
    import numpy as np
    from scipy.fft import ifft

    comm = root.mpi.comm

    # read data
    stats = root.mpi.load('../forward/traces/stats.pickle')
    syn = root.mpi.mpiload(f'../enc_syn')
    obs = root.mpi.mpiload(f'../enc_obs')
    # ref = root.mpi.mpiload(f'../enc_diff')
    weight = root.mpi.mpiload(f'../enc_weight')

    if enc['test_encoding']:
        weight[...] = 1.0

    # unwrap phases
    # weight[np.where(abs(ref) > np.pi)] = 0.0

    # compute diff
    phase_diff = np.angle(syn / obs) * weight
    amp_diff = np.log(np.abs(syn) / np.abs(obs)) * weight

    if enc['double_difference']:
        # unwrap or clip phases
        phase_sum = sum(comm.allgather(np.nansum(phase_diff, axis=0)))
        amp_sum = sum(comm.allgather(np.nansum(amp_diff, axis=0)))
        weight_sum = sum(comm.allgather(np.nansum(weight, axis=0)))
        weight_sum[np.where(weight_sum == 0.0)] = 1.0 # type: ignore

        # sum of phase and amplitude differences
        phase_diff -= phase_sum / weight_sum
        amp_diff -= amp_sum / weight_sum

    # apply measurement weightings
    phase_diff *= weight * enc['phase_factor']
    amp_diff *= weight * enc['amplitude_factor']

    if enc.get('normalize_frequency'):
        omega = np.arange(enc['imin'], enc['imax']) / enc['imin']
        phase_diff /= omega

    # save misfit value
    fincr = enc['frequency_increment']

    def write(diff, name):
        mf = np.zeros([syn.shape[0], syn.shape[1], enc['nbands_used']])

        for i in range(enc['nbands_used']):
            mf[..., i] = np.nansum(diff[..., i * fincr: (i + 1) * fincr] ** 2, axis=-1)
        
        mf_sum = comm.gather(mf.sum(axis=0), root=0)

        if root.mpi.rank == 0:
            root.mpi.dump(sum(mf_sum), f'../{name}_mf.npy', mkdir=False)

    write(phase_diff, 'phase')

    if enc['amplitude_factor'] > 0:
        write(amp_diff, 'amp')

    if misfit_only:
        return None, None

    # compute adjoint sources
    nan = np.where(np.isnan(phase_diff))
    syn[nan] = 1.0

    phase_adj = phase_diff * (1j * syn) / abs(syn) ** 2
    phase_adj[nan] = 0.0

    if enc['amplitude_factor'] > 0:
        amp_adj = amp_diff * syn / abs(syn) ** 2
        amp_adj[nan] = 0.0
    
    else:
        amp_adj = np.zeros(syn.shape)

    # fourier transform of adjoint source time function
    ft_adj = rotate_frequencies(enc, phase_adj + amp_adj, stats['cmps'], False)

    # fill full frequency band
    ft_adstf = np.zeros([len(stas), 3, enc['nt_se']], dtype=complex)
    ft_adstf[..., enc['imin']: enc['imax']] = ft_adj
    ft_adstf[..., -enc['imin']: -enc['imax']: -1] = np.conj(ft_adj)

    # stationary adjoint source
    adstf_tau = -ifft(ft_adstf).real # type: ignore

    # repeat to fill entrie adjoint duration
    nt = stats['nt']
    adstf_tile = np.tile(adstf_tau, int(np.ceil(nt / enc['nt_se'])))
    adstf = adstf_tile[..., -nt:]

    if enc['taper']:
        ntaper = int(enc['taper'] * 60 / enc['dt'])
        adstf[..., -ntaper:] *= np.hanning(2 * ntaper)[ntaper:]
    
    return adstf, stats['cmps']


def rotate_frequencies(enc: Encoding, data: ndarray, cmps_ne: tp.Tuple[str, str, str], direction: bool):
    import numpy as np

    cmps_rt = getcomponents()
    data_rot = np.zeros(data.shape, dtype=complex)
    baz = getdir().load(f'baz_p{root.task_nprocs}/{root.mpi.pid}.pickle')

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

    for event, slots in enc['fslots'].items():
        if len(slots) == 0:
            continue
        
        ba = baz[event] if direction else 2 * np.pi - baz[event]

        for slot in slots:
            a = data[:, c1, slot]
            b = data[:, c2, slot]
            data_rot[:, c3, slot] = - b * np.sin(ba) - a * np.cos(ba)
            data_rot[:, c4, slot] = - b * np.cos(ba) + a * np.sin(ba)
            
    return data_rot
