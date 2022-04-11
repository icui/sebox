from nnodes import root
from seisbp import SeisBP

for e in root.ls('events'):
    b1 = SeisBP(f'proc_obs/{e}.bp', 'r')
    b2 = SeisBP(f'../nsb/proc_m00/{e}.bp', 'r')

    if len(b1.stations) != len(b2.stations):
        print(e, len(b1.stations), len(b2.stations))
