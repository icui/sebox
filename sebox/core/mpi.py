from sys import argv
import asyncio

from sebox import root

if __name__ == '__main__':
    try:
        func = root.load(f'{argv[1]}.pickle')

        if asyncio.iscoroutine(result := func()):
            asyncio.run(result)
    
    except Exception as e:
        root.write(f'{argv[1]}.error', str(e), 'a')
