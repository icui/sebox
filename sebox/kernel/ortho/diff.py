from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from .typing import Kernel

from sebox.utils.catalog import getstations
from .ft import rotate_frequencies


async def diff(ws: Kernel):
    ws.mkdir('misfit')
    ws.mkdir('adjoint')
    stas = getstations()
    await ws.mpiexec(_diff, arg=(ws, len(stas)), arg_mpi=stas)
    exit()


async def _diff(arg: tp.Tuple[Kernel, int], stas: tp.List[str]):
    import numpy as np
    from scipy.fft import ifft
    from scipy.signal import resample
    from sebox.mpi import pid
    from mpi4py.MPI import COMM_WORLD as comm

    ws, nsta = arg

    # read data
    stats = ws.load('forward/traces/stats.pickle')
    syn = ws.load(f'enc_syn/{pid}.npy')
    obs = ws.load(f'enc_obs/{pid}.npy')
    ref = ws.load(f'enc_diff/{pid}.npy')
    weight = ws.load(f'enc_weight/{pid}.npy')

    # compute diff
    phase_diff = np.angle(syn / obs) * weight
    phase_diff[np.where(abs(ref) > np.pi)] = 0.0
    amp_diff = np.log(np.abs(syn) / np.abs(obs) * weight)

    if ws.double_difference:
        # unwrap or clip phases
        nsta = len(getstations())
        phase_sum = sum(comm.allgather(phase_diff.sum(axis=0)))
        amp_sum = sum(comm.allgather(amp_diff.sum(axis=0)))
        npairs = sum(comm.allgather(np.invert(np.isnan(phase_diff)).astype(int).sum(axis=0)))

        # sum of phase and amplitude differences
        phase_diff = (nsta * phase_diff - phase_sum) / npairs
        amp_diff = (nsta * amp_diff - amp_sum) / npairs

    
        if 'II.OBN' in stas:
            print('II.OBN', nsta)
            print(npairs.shape, phase_sum.shape, amp_sum.shape, phase_sum.max(), amp_sum.max())
            print(npairs.max())
            print(npairs.sum(axis=0))

    # apply measurement weightings
    omega = np.arange(ws.imin, ws.imax) / ws.imin
    phase_diff *= ws.phase_factor / omega
    amp_diff *= ws.amplitude_factor

    # misfit values and adjoint sources
    nan = np.squeeze(np.where(np.isnan(phase_diff)))
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
        ft_adstf[ws.imin: ws.imax] = ft_adj
        ft_adstf[-ws.imin: -ws.imax: -1] = np.conj(ft_adj)

        # stationary adjoint source
        adstf_tau = -ifft(ft_adstf).real # type: ignore

        # repeat to fill entrie adjoint duration
        nt = stats['npts']
        adstf_tile = np.tile(adstf_tau, int(np.ceil(nt / ws.nt_se)))
        adstf = adstf_tile[-nt:]

        if ws.taper:
            ntaper = int(ws.taper * 60 / ws.dt)
            adstf[..., -ntaper:] *= np.hanning(2 * ntaper)[ntaper:]
        
        ws.dump(adstf, f'adjoint/{pid}.npy', mkdir=False)
