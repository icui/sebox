from nnodes import root
from obspy import read_inventory
from pyasdf import ASDFDataSet

def get_xml(event):
    if root.has(dst := f'inventories/{event}.pickle'):
        return

    with ASDFDataSet(f'raw_obs/{event}.h5', mode='r', mpi=False) as obs_h5, \
        ASDFDataSet(f'raw_m25/{event}.h5', mode='r', mpi=False) as syn_h5:
        invs = {}

        for sta in syn_h5.waveforms.list():
            try:
                inv = obs_h5.waveforms[sta].StationXML
            
            except:
                inv = read_inventory(f'../sebox2/downloads/{event}/xml/{sta}.xml')
            
            invs[sta] = inv
        
        root.dump(invs, dst)


def main(node):
    node.add_mpi(get_xml, 42, mpiarg=root.ls('events'))
