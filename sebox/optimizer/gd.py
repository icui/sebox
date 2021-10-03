from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from sebox.typing import Optimizer

def main(ws: Optimizer):
    if len(ws) == 0:
        ws.add(iterate, 'iter_00', iteration=0)


def iterate(ws: Optimizer):
    """Add an iteration."""
    ws.add(ws.ln, args=(ws.rel(ws.path_model), 'model_init.bp'))


def _add(ws: Optimizer):
    ws.add(iterate, f'iter_{len(ws):02d}', path_model=ws.path(f'iter_{len(ws)-1:02d}/model_new.bp'))
