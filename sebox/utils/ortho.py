from sebox import root, Node
from sebox.kernel.ortho.ft import ft
from .catalog import getdir, getstations


def scatter_obs(node: Node):
    """Generate observed frequencies."""
    node.concurrent = True
    cdir = getdir()

    for src in cdir.ls('raw_obs'):
        event = src.split('.')[0]
        dst = f'raw_obs_p{root.task_nprocs}/{event}'
        
        if not cdir.has(dst):
            node.add(('sebox.utils.asdf', 'scatter'), event, stats={'cmps': ['N', 'E', 'Z']},
                path_input=cdir.path(f'raw_obs/{src}'), path_output=cdir.path(dst))


def ft_obs(node: Node):
    node.concurrent = True
    cdir = getdir()
    stas = getstations()

    enc = {
        'kf': 1,
        'nt_se': 90000,
        'taper': 5.0,
        'dt': 0.16,
        'nfreq': 750,
        'imax': 846,
        'imin': 96
    }

    for event in cdir.ls(f'raw_obs_p{root.task_nprocs}'):
        src = cdir.path(f'raw_obs_p{root.task_nprocs}/{event}')
        dst = cdir.path(f'ft_obs_p{root.task_nprocs}/{event}')
        node.rm(dst)
        node.mkdir(dst)
        node.add_mpi(ft, name=event, arg=({
            **enc,
            'fslots': {event: list(range(enc['nfreq']))}
        }, src, dst, False), arg_mpi=stas)
