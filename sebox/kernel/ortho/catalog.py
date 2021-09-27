from sebox.utils.catalog import getdir, merge_stations


async def merge(_):
    cdir = getdir()
    merge_stations(cdir.subdir('stations'), cdir, True)


async def scatter_obs(_):
    cdir = getdir()

async def scatter_diff(_):
    cdir = getdir()
