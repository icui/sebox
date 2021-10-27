from __future__ import annotations

from sebox import root, Node
from sebox.kernel.ortho.ft import ft
from sebox.kernel.ortho.preprocess import getenc, Ortho
from .catalog import getdir, getstations


def scatter_obs(node: Ortho):
    """Generate observed frequencies."""
    node.concurrent = True
    cdir = getdir()

    for src in cdir.ls('raw_obs'):
        event = src.split('.')[0]
        dst = f'raw_obs_p{root.task_nprocs}/{event}'
        
        if not cdir.has(dst):
            node.add(('sebox.utils.asdf', 'scatter'), event, stats={'cmps': ['N', 'E', 'Z']},
                path_input=cdir.path(f'raw_obs/{src}'), path_output=cdir.path(dst))


def ft_obs(node: Ortho):
    node.concurrent = True
    cdir = getdir()
    stas = getstations()
    enc = getenc(node)
    node.dump(enc, 'encoding.pickle')

    for event in cdir.ls(f'raw_obs_p{root.task_nprocs}'):
        src = cdir.path(f'raw_obs_p{root.task_nprocs}/{event}')
        dst = cdir.path(f'ft_obs_p{root.task_nprocs}/{event}')
        node.rm(dst)
        node.mkdir(dst)
        node.add_mpi(ft, name=event, arg=({
            **enc,
            'fslots': {event: list(range(enc['nfreq']))}
        }, src, dst, False), arg_mpi=stas)
