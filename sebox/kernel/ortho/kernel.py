from __future__ import annotations
import typing as tp

from sebox import Directory

if tp.TYPE_CHECKING:
    from sebox import typing

    class Kernel(typing.Kernel):
        """Source encoded kernel computation."""
        # path to encoded observed traces
        path_encoded: tp.Optional[str]


def kernel(ws: Kernel):
    """Compute kernels."""
    if ws.path_encoded:
        pass

    else:
        