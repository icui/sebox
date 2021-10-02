from sebox import Task
from .solver import *


modules = {
    'solver': {
        # generate mesh
        'mesh': Task[Forward],

        # forward simulation
        'forward': Task[Forward],

        # adjoint simulation
        'adjoint': Task[Adjoint],

        # sum and smooth kernels
        'sum': Task[Sum]
    }
}