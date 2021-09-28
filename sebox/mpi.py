from sys import argv, stderr
from traceback import format_exc
from mpi4py.MPI import COMM_WORLD as comm
import asyncio

from sebox import root


# MPI rank and number of processes
rank: int = comm.Get_rank()
size: int = comm.Get_size()

# output file name of current process
pid = f'p{"0" * (len(str(size - 1)) - len(str(rank)))}{rank}'

if __name__ == '__main__':
    try:
        # saved function and arguments from main process
        (func, arg, arg_mpi) = root.load(f'{argv[1]}.pickle')

        args = []

        if arg is not None:
            args.append(arg)

        if arg_mpi is not None:
            args.append(arg_mpi[rank])

        if asyncio.iscoroutine(result := func(*args)):
            asyncio.run(result)
    
    except Exception:
        err = format_exc()
        print(err, file=stderr)
        root.write(err, f'{argv[1]}.error', 'a')
