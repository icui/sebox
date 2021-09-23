from sys import argv, stderr
from traceback import format_exc
import asyncio

from sebox import root

def add_error(e: Exception):
    """Save error message.."""
    err = format_exc()
    print(err, file=stderr)
    root.write(err, f'{argv[1]}.error', 'a')


if __name__ == '__main__':
    try:
        func = root.load(f'{argv[1]}.pickle')

        if asyncio.iscoroutine(result := func()):
            asyncio.run(result)
    
    except Exception as e:
        add_error(e)
