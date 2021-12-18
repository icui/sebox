from __future__ import annotations
import typing as tp

from sebox.utils.adios import xgd

if tp.TYPE_CHECKING:
    from sebox.typing import Optimizer

def main(node: Optimizer):
    if len(node) == 0:
        if node.iteration_breakpoints is None:
            node.iteration_breakpoints = set()

        if node.iteration_start is not None:
            # inherit iterations from an existing workspace
            node.iteration_breakpoints.add(node.iteration_start)

            for i in range(node.iteration_start):
                node.add(None, f'iter_{i:02d}', iteration=i)
            
            node[-1].add('optimizer.check')

        else:
            # initialize iterations
            node.iteration_breakpoints.add(0)
            node.iteration_start = 0
            node.add('optimizer.iterate', 'iter_00', iteration=0)


def iterate(node: Optimizer):
    """Add an iteration."""
    node.ln(node.rel(node.path_model), 'model_init.bp')

    # compute kernels
    kl = node.add('kernel')

    # compute direction
    node.add('optimizer.direction')

    # line search
    node.add('search', inherit_kernel=kl)

    # add new iteration
    node.add('optimizer.check')


def direction(node: Optimizer):
    """Compute direction."""
    xgd(node)


def check(node: Optimizer):
    """Add a new iteration if necessary."""
    optim = node.parent.parent
    i = node.iteration + 1

    if i < optim.niters and node.parent is optim[-1]:
        restart = False

        if len(node.parent) >= 4 and node.parent[-2].failed:
            restart = True
        
        elif i - node.iteration_start == node.iteration_restart:
            restart = True
        
        if restart:
            optim.iteration_start = i
            optim.iteration_breakpoints.add(i)

        optim.add('optimizer.iterate', f'iter_{i:02d}',
            iteration=i,
            path_model=optim.path(f'iter_{node.iteration:02d}/model_new.bp'),
            path_mesh=optim.path(f'iter_{node.iteration:02d}/mesh_new'))
