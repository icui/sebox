from sebox import root, Node
from .catalog import getdir


def scatter_obs(node: Node):
    """Generate observed frequencies."""
    node.concurrent = True
    cdir = getdir()

    for src in cdir.ls('raw_obs'):
        event = src.split('.')[0]
        dst = f'raw_p{root.task_nprocs}/{event}'
        
        if not cdir.has(dst):
            node.add(('sebox.utils.asdf', 'scatter'), event, stats={'cmps': ['N', 'E', 'Z']},
                path_input=cdir.path(f'raw_obs/{src}'), path_output=cdir.path(dst))
