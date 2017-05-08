"""
Demo of automatically generating a 3D-printable WRL file from a SWC file.
"""
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

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

from scoop import futures

import os
os.system('ls *.swc > swc_names.txt')
f = open('swc_names.txt')

nfs = [line.strip() for line in open('swc_names.txt', 'r')]

#nfs=[]
#nfs.append('04b_spindle3aFI.CNG.swc')
#nfs.append('re80.CNG.swc')

f.close()

#morphology_url = 'http://neuromorpho.org/dableFiles/amaral/CNG%20version/c91662.CNG.swc'
#
#wrl_filename = 'c91662.wrl'


h.load_file('stdlib.hoc')
h.load_file('import3d.hoc')
#h.load_file('check_tree.hoc')
#h('a = new Morphologies()')
#h('a.check_tree()')
#for (i=25+pc.id; i < 51 ;/*in standard dir 4228;*/ i +=pc.nhost) { //0, 4, 8, 12

itergids = iter( i for i in range(RANK, len(nfs), SIZE-1) )

#for i in itergids:
#    print(i,nfs[i])

#iteratorgid=[i for i in xrange(25,2300)]

def func2map(i):
#for i in iteratorgid:
    cell = h.Import3d_SWC_read()
    print(len(nfs))
    print(i)
    morphology_filename=nfs[i]
    #morphology_filename = '2007-12-02-A.CNG.swc'
    cell = h.Import3d_SWC_read()

    cell.input(morphology_filename)
    i3d = h.Import3d_GUI(cell, 0)

    i3d.instantiate(None)
    #h('check_tree()')
    file_name=str(nfs[i])+str('.wrl')
    print(file_name)
    ctng(show=False, magnification=200,file_name=file_name)
    mlab.savefig(str(nfs[i])+'.wrl')

#def main():
#    futures.map(func2map,iteratorgid)
for i in itergids:
    func2map(i)
#if __name__ == "__main__":
 #   main()
    #import pdb
    #pdb.set_trace()
    #print('cleared 3')

    #info_swc=utils.gcs(utils.NCELL)
