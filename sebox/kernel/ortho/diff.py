from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from .typing import Kernel

from sebox import root
from sebox.utils.catalog import getstations
from .ft import rotate_frequencies


async def diff(ws: Kernel):
    ws.mkdir('misfit')
    await ws.mpiexec(_diff, arg=ws, arg_mpi=getstations())
    ws.add(_gather)


async def _diff(ws: Kernel, stas: tp.List[str]):
    import numpy as np
    from scipy.fft import ifft
    from sebox.mpi import pid
    from mpi4py.MPI import COMM_WORLD as comm

    # read data
    stats = ws.load('forward/traces/stats.pickle')
    syn = ws.load(f'enc_syn/{pid}.npy')
    obs = ws.load(f'enc_obs/{pid}.npy')
    ref = ws.load(f'enc_diff/{pid}.npy')
    weight = ws.load(f'enc_weight/{pid}.npy')

    # clip phases
    weight[np.where(abs(ref) > np.pi)] = 0.0

    # compute diff
    phase_diff = np.angle(syn / obs) * weight
    amp_diff = np.log(np.abs(syn) / np.abs(obs)) * weight

    if ws.double_difference:
        # unwrap or clip phases
        phase_sum = sum(comm.allgather(np.nansum(phase_diff, axis=0)))
        amp_sum = sum(comm.allgather(np.nansum(amp_diff, axis=0)))
        weight_sum = sum(comm.allgather(np.nansum(weight, axis=0)))

        # sum of phase and amplitude differences
        phase_diff = phase_diff - phase_sum / weight_sum
        amp_diff = amp_diff - amp_sum / weight_sum

    
        if 'II.OBN' in stas:
            print('II.OBN')
            print(np.nanmax(phase_sum), np.nanmax(amp_sum), np.nanmax(weight_sum))

    # apply measurement weightings
    omega = np.arange(ws.imin, ws.imax) / ws.imin
    phase_diff *= weight * ws.phase_factor / omega
    amp_diff *= weight * ws.amplitude_factor

    # misfit values and adjoint sources
    nan = np.where(np.isnan(phase_diff))
    syn[nan] = 1.0

    # misfit value
    mf = np.nansum(phase_diff ** 2, axis=-1)

    if ws.amplitude_factor > 0:
        mf += np.nansum(amp_diff ** 2, axis=-1)

    ws.dump(mf, f'misfit/{pid}.npy', mkdir=False)

    # compute adjoint source
    if not ws.misfit_only:
        phase_adj = phase_diff * (1j * syn) / abs(syn) ** 2
        phase_adj[nan] = 0.0

        if ws.amplitude_factor > 0:
            amp_adj = amp_diff * syn / abs(syn) ** 2
            amp_adj[nan] = 0.0
        
        else:
            amp_adj = np.zeros(syn.shape)

        # fourier transform of adjoint source time function
        ft_adj = rotate_frequencies(ws, phase_adj + amp_adj, stats['cmps'], False)

        # fill full frequency band
        ft_adstf = np.zeros([len(stas), 3, ws.nt_se], dtype=complex)
        ft_adstf[..., ws.imin: ws.imax] = ft_adj
        ft_adstf[..., -ws.imin: -ws.imax: -1] = np.conj(ft_adj)

        # stationary adjoint source
        adstf_tau = -ifft(ft_adstf).real # type: ignore

        # repeat to fill entrie adjoint duration
        nt = stats['nt']
        adstf_tile = np.tile(adstf_tau, int(np.ceil(nt / ws.nt_se)))
        adstf = adstf_tile[..., -nt:]

        if ws.taper:
            ntaper = int(ws.taper * 60 / ws.dt)
            adstf[..., -ntaper:] *= np.hanning(2 * ntaper)[ntaper:]
        
        ws.dump((adstf, stas, stats['cmps']), f'misfit/{pid}.pickle', mkdir=False)


def _gather(ws: Kernel):
    from pyasdf import ASDFDataSet

    with ASDFDataSet(ws.path('adjoint.h5'), mode='w', mpi=False) as ds:
        for pid in ws.ls('misfit', '*.pickle'):
            adstf, stas, cmps = ws.load(f'misfit/{pid}')

            for i, sta in enumerate(stas):
                for j, cmp in enumerate(cmps):
                    ds.add_auxiliary_data(adstf[i, j], 'AdjointSources', sta.replace('.', '_') + '_MX' + cmp, {})
