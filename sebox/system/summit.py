from os import environ
from subprocess import check_call

from sebox import root


# number of CPUs per node
cpus_per_node = 42

# number of GPUs per node
gpus_per_node = 6


def submit(cmd: str, dst: str):
    """Write and submit job script."""
    # hours and minutes
    walltime = root.job_walltime
    hh = int(walltime // 60)
    mm = int(walltime - hh * 60)

    # job script
    lines = [
        '#!/bin/bash',
        f'#BSUB -J {root.job_name}',
        f'#BSUB -P {root.job_account}',
        f'#BSUB -W {hh:02d}:{mm:02d}',
        f'#BSUB -nnodes {root.job_nnodes}',
        f'#BSUB -o lsf.%J.o',
        f'#BSUB -e lsf.%J.e'
    ]

    if root.job_debug:
        lines.append('#BSUB -q debug')

    # add main command
    lines.append(cmd + '\n')

    # write job script and submit
    root.writelines(lines, dst + '/job.bash')
    check_call('bsub job.bash', shell=True, cwd=dst)

def requeue():
    """Run current job again."""
    if environ.get('LSB_INTERACTIVE') != 'Y':
        check_call('brequeue ' + environ['LSB_JOBID'], shell=True)
        exit()

    else:
        root.job_paused = False


def mpiexec(cmd: str, nprocs: int, cpus_per_proc: int = 1, gpus_per_proc: int = 0):
    """Get the command to call MPI."""
    flags = ' --smpiargs="off"' if nprocs == 1 else ''

    return f'jsrun{flags} -n {nprocs} -a 1 -c {cpus_per_proc} -g {gpus_per_proc} {cmd}'
