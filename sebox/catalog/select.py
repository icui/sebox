import numpy as np


pmin = 17.0
pmax = 100.0
duration = 288.0
nbands_encoding = 12
nbands = 3
cl = 0.8
cr = 0.75

df = 1 / duration / 60
imin = int(np.ceil(1 / pmax / df))
imax = int(np.floor(1 / pmin / df)) + 1
fincr = (imax - imin) // nbands_encoding * (nbands_encoding // nbands)
imax = imin + fincr * nbands


def select(node):
    node.add(select_event, event='C201912201139A')


def select_event(node):
    pass
