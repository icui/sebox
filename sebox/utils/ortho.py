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
    enc = node.load('encoding.pickle')
    stas = getstations()

    for event in cdir.ls(f'raw_obs_p{root.task_nprocs}'):
        src = f'raw_obs_p{root.task_nprocs}/{event}'
        dst = f'ft_obs_p{root.task_nprocs}/{event}'
        node.cp('stats.pickle', cdir.path(dst))
        node.add_mpi(ft, arg=(enc, src, dst, False), arg_mpi=stas)
