from __future__ import annotations
import asyncio
import typing as tp
from functools import partial
from math import ceil
from time import time
from datetime import timedelta

from .root import root
from .directory import Directory


# pending tasks
_pending: tp.Dict[asyncio.Lock, int] = {}

# running tasks
_running: tp.Dict[asyncio.Lock, int] = {}


def _dispatch(lock: asyncio.Lock, nnodes: int) -> bool:
    """Execute a task if resource is available."""
    ntotal = root.job_nnodes

    if nnodes > ntotal:
        raise RuntimeError(f'Insufficient nodes ({nnodes} / {ntotal})')

    if nnodes <= ntotal - sum(_running.values()):
        _running[lock] = nnodes
        return True
    
    return False


def _name(cmd: tp.Union[str, tp.Callable]) -> str:
    """Get file name to store pickled function and / or stdout."""
    if isinstance(cmd, str):
        return 'mpiexec_' + cmd.split(' ')[0].split('/')[-1]

    func = cmd

    while isinstance(func, partial):
        func = func.func

        return 'mpiexec_' + func.__name__.lstrip('_')
    
    return 'mpiexec'


async def mpiexec(d: Directory, cmd: tp.Union[str, tp.Callable],
    nprocs: int, cpus_per_proc: int, gpus_per_proc: int,
    name: tp.Optional[str], arg: tp.Any, arg_mpi: tp.Optional[list]):
    """Schedule the execution of MPI task"""
    # task queue controller
    lock = asyncio.Lock()

    # error occurred
    err = None
    
    try:
        # remove unused proceessors
        if arg_mpi:
            nprocs = min(len(arg_mpi), nprocs)

        # calculate node number
        nnodes = int(ceil(nprocs * cpus_per_proc  / (root.cpus_per_node or root.sys.cpus_per_node)))

        if gpus_per_proc > 0:
            nnodes = max(nnodes, int(ceil(nprocs * gpus_per_proc  / (root.gpus_per_node or root.sys.gpus_per_node))))

        # wait for node resources
        await lock.acquire()

        if not _dispatch(lock, nnodes):
            _pending[lock] = nnodes
            await lock.acquire()

        # save function as pickle to run in parallel
        if name is None:
            name = _name(cmd)

        if callable(cmd):
            if arg_mpi:
                # assign a chunk of arg_mpi to each processor
                arg_mpi = sorted(arg_mpi)
                args = []
                chunk = int(ceil(len(arg_mpi) / nprocs))
                
                # adjust number of processors
                if nprocs * chunk > len(arg_mpi):
                    nprocs -= (nprocs * chunk - len(arg_mpi)) // chunk

                for i in range(nprocs - 1):
                    args.append(arg_mpi[i * chunk: (i + 1) * chunk])
                
                args.append(arg_mpi[(nprocs - 1) * chunk:])
            
            else:
                args = None

            cwd = None
            d.rm(f'{name}.*')
            d.dump((cmd, arg, args), f'{name}.pickle')
            cmd = f'python -m "sebox.mpi" {d.path(name)}'
        
        else:
            cwd = d.path()
        
        # wrap with parallel execution command
        cmd = root.sys.mpiexec(cmd, nprocs, cpus_per_proc, gpus_per_proc)
        
        # create subprocess to execute task
        with open(d.path(f'{name}.out'), 'a') as f:
            # write command
            f.write(f'{cmd}\n\n')
            time_start = time()

            # execute in subprocess
            process = await asyncio.create_subprocess_shell(cmd, cwd=cwd, stdout=f, stderr=f)
            await process.communicate()

            # write elapsed time
            f.write(f'\nelapsed: {timedelta(seconds=int(time()-time_start))}\n')

        if d.has(f'{name}.error'):
            raise RuntimeError(d.read(f'{name}.error'))

        elif process.returncode:
            raise RuntimeError(f'{cmd}\nexit code: {process.returncode}')
    
    except Exception as e:
        err = e
    
    # clear entry
    if lock in _pending:
        del _pending[lock]
    
    if lock in _running:
        del _running[lock]
    
    # sort entries by their node number
    pendings = sorted(_pending.items(), key=lambda item: item[1], reverse=True)

    # execute tasks if resource is available
    for lock, nnodes in pendings:
        if _dispatch(lock, nnodes):
            del _pending[lock]
            lock.release()

    if err:
        raise err
