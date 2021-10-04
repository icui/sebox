from sys import argv

from sebox import root

if __name__ == '__main__':
    # determine target directory
    if len(argv) > 1:
        dst = f'job.{argv[1]}'

        if root.has(dst):
            raise FileExistsError(f'{dst} already exists')

    else:
        i = 0

        while root.has(dst := f'job.{i:04d}'):
            i += 1
    
    # copy config.toml
    config = root.load('config.toml')
    root.dump(config, dst + '/config.toml')
    
    # submit job
    root.restore()
    root.submit(dst)
    print('submitted', dst)
