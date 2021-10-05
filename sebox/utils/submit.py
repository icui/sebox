from sys import argv

from sebox import root


def mkws():
    """Create a new workspace."""
    # determine target directory
    if len(argv) > 1:
        dst = argv[1]

        if root.has(dst):
            raise FileExistsError(f'{dst} already exists')

    else:
        i = 0

        while root.has(dst := f'job.{i:04d}'):
            i += 1
    
    # copy config.toml
    config = root.load('config.toml')
    root.dump(config, dst + '/config.toml')

    return dst


def submit(run: bool):
    """Create and submit job."""
    root.restore()
    root.sys.submit('python -c "from sebox import root; root.run()"', mkws(), run)
