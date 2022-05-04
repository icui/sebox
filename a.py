from nnodes import root as node
import numpy as np

nmax = 0
for l in range(42):
    p = node.load(f'p{l:02d}.npy')
    for i in range(p.shape[0]):
        for j in range(3):
            z = np.count_nonzero(np.invert(np.isnan(p[i][j])))
            if z > nmax:
                nmax = z
                print(l,i,j,z)
