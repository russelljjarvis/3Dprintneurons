"""
Demo of automatically generating a 3D-printable WRL file from a SWC file.
"""

#import urllib3
#import urllib
from neuron import h
from prepare_3dprintable import ctng
from mayavi import mlab
import sys
sys.path.append('geometry3d')
from surface import surface
from triangularMesh import TriangularMesh
from voxelize import voxelize
from scalarField import ScalarField
from mpi4py import MPI
from scoop import futures

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
import os
os.system('ls *.swc > swc_names.txt')
f = open('swc_names.txt')

nfs = [line.strip() for line in open('swc_names.txt', 'r')]

nfsw = [line.strip() for line in open('swc_names.txt', 'r')]
f.close()

#morphology_url = 'http://neuromorpho.org/dableFiles/amaral/CNG%20version/c91662.CNG.swc'
#
#wrl_filename = 'c91662.wrl'

"""
download c91662 morphology from NeuroMorpho.Org

Ascoli, G. A., Donohue, D. E., & Halavi, M. (2007). NeuroMorpho. Org: a central
   resource for neuronal morphologies. The Journal of Neuroscience, 27(35),
   9247-9251.
Ishizuka, N., Cowan, W. M., & Amaral, D. G. (1995). A quantitative analysis of
    the dendritic organization of pyramidal cells in the rat hippocampus. Journal
    of Comparative Neurology, 362(1), 17-45.
with open(morphology_filename, 'w') as f:

    import urllib.request
    with urllib.request.urlopen(morphology_url) as url:
        s = url.read()
        #I'm guessing this would output the html source code?
        print(s)

        f.write(str(s))
    #f.write(urllib3.urlopen(morphology_url, timeout=10).read())

Load it into NEURON

"""
print('cleared 1')

h.load_file('stdlib.hoc')
h.load_file('import3d.hoc')
print('cleared 2')


#for (i=25+pc.id; i < 51 ;/*in standard dir 4228;*/ i +=pc.nhost) { //0, 4, 8, 12

#itergids = iter( i for i in range(25+RANK, 4228, SIZE) )

#for i in itergids:
#    print(i,nfs[i])

iteratorgid=[i for i in xrange(25,2300)]

def func2map(iteratorgid):    
    cell = h.Import3d_SWC_read()
    print(len(nfs))
    print(i)
    morphology_filename=nfs[i]    
    #morphology_filename = '2007-12-02-A.CNG.swc'
    cell = h.Import3d_SWC_read()
    cell.input(morphology_filename)
    i3d = h.Import3d_GUI(cell, 0)
    i3d.instantiate(None)
    file_name=str(nfs[i])+str('.wrl')
    print(file_name)
    ctng(show=False, magnification=200,file_name=file_name)
    #mlab.savefig(str(i)+'.wrl')   
    
def main():    
    futures.map(func2map,iteratorgid)    

if __name__ == "__main__":
    main()
    #import pdb
    #pdb.set_trace()
    #print('cleared 3')

    #info_swc=utils.gcs(utils.NCELL)

