from sys import argv, stderr
from traceback import format_exc
import typing as tp
import asyncio

from sebox import root


def comm() -> tp.Tuple[int, int]:
    """Get MPI rank and size."""
    from mpi4py.MPI import COMM_WORLD

    return COMM_WORLD.Get_rank(), COMM_WORLD.Get_size()


if __name__ == '__main__':
    try:
        (func, arg, arg_mpi) = root.load(f'{argv[1]}.pickle')

        args = []

        if arg is not None:
            args.append(arg)

        if arg_mpi:
            rank, _ = comm()
            args.append(arg_mpi[rank])

        if asyncio.iscoroutine(result := func(*args)):
            asyncio.run(result)
    
    except Exception:
        err = format_exc()
        print(err, file=stderr)
        root.write(err, f'{argv[1]}.error', 'a')
