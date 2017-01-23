
from allensdk.model.biophys_sim.neuron.hoc_utils import HocUtils
import logging
import glob
from mpi4py import MPI
#import btmorph
import numpy

#pc = h.ParallelContext()
#h('objref pc')
#h.pc = pc
#s = "mpi4py thinks I am %d of %d,\
# NEURON thinks I am %d of %d\n"
#cw = MPI.COMM_WORLD
##print s % (cw.rank, cw.size, pc.id(), pc.nhost())
#h('time_start=pc.time()')
#PROJECTROOT=os.getcwd()
from neuron import h

#from neuronpy.nrnobjects import cell
##print dir(cell.Cell.soma)
class Utils(HocUtils):

#class Utils(HocUtils):
    _log = logging.getLogger(__name__)

    def __init__(self):

        #super(Utils, self).__init__()
        self.stim = None
        self.stim_curr = None
        self.sampling_rate = None
        self.cells = []
        self.gidlist=[]
        self.NCELL=0
        self.celldict={}
        self.COMM = MPI.COMM_WORLD
        self.SIZE = self.COMM.Get_size()
        self.RANK = self.COMM.Get_rank()
        self.allsecs=None #global list containing all NEURON sections, initialized via mkallsecs
        self.coordict=None
        self.celldict={}
        self.cellmorphdict={}
        self.nclist = []
        self.h=h
        h('objref pc')
        h('pc = new ParallelContext()')
        self.pc=h.pc


    def prep_list(self):
        import pickle
        allrows = pickle.load(open('allrows.p', 'rb'))
        allrows.remove(allrows[0])#The first list element are the column titles.
        allrows2 = [i for i in allrows if int(len(i))>9 ]
        return allrows2



    def register_gid(self, gid, source, section=None):
        """Register a global ID with the global `ParallelContext` instance."""
        ####print "registering gid %s to %s (section=%s)" % (gid, source, section)
        self.parallel_context.set_gid2node(gid, self.mpi_rank) # assign the gid to this node
        if is_point_process(source):
            nc = h.NetCon(source, None)                          # } associate the cell spike source
        else:
            nc = h.NetCon(source, None, sec=section)
        self.parallel_context.cell(gid, nc)                     # } with the gid (using a temporary NetCon)
        self.gid_sources.append(source) # gid_clear (in _State.reset()) will cause a
                                        # segmentation fault if any of the sources
                                        # registered using pc.cell() no longer exist, so
                                        # we keep a reference to all sources in the
                                        # global gid_sources list. It would be nicer to
                                        # be able to unregister a gid and have a __del__
                                        # method in ID, but this will do for now.



    def gcsneuroml(self,NCELL):
        import neuroml
        from neuroml.loaders import SWCLoader
        #/usr/local/lib/python2.7/dist-packages/neuron/neuroml/biophysics.py
        #/home/russell/git/libNeuroML/neuroml/examples/ion_channel_generation.py
        #23550003583330
        NCELL=self.NCELL
        SIZE=self.SIZE
        RANK=self.RANK
        from neuron import h
        pc=h.ParallelContext()
        h=self.h
        h('objref pc, nc')
        h('pc = new ParallelContext()')
        h('py = new PythonObject()')
        swcdict={}
        NFILE = 3175
        fit_ids = self.description.data['fit_ids'][0] #excitatory
        self.cells_data = self.description.data['biophys'][0]['cells']
        info_swc=self.prep_list()
        d = { x: y for x,y in enumerate(info_swc)}
        import os
        os.chdir(os.getcwd() + '/main')
        #itergids = iter( i for i in range(RANK, NCELL, SIZE) )
        #morphxmldoc= [ SWCLoader.load_swc_single(d[i][3]) for i in itergids ]
        #return morphxmldoc
        #cell = h.mkcell(d[i][3])
        #morphxmldoc=SWCLoader.load_swc_single(d[i][3])
        ##print morphxmldoc
        i=0
        #for i in itergids:
        cell = h.mkcell(d[i][3])
        #morphxmldoc=SWCLoader.load_swc_single(d[i][3])
        cell.geom_nseg()
        cell.gid1=i #itergids.next()
        if 'pyramid' in d[i]:
            self.load_cell_parameters(cell, fit_ids[self.cells_data[0]['type']])
            cell.polarity=1
        else:
        #inhibitory type stained cell.
            self.load_cell_parameters(cell, fit_ids[self.cells_data[2]['type']])
            cell.polarity=0

        secnames=str(h.cas().name()) #cas=the currently accessed section.
        #print(secnames)
        cellind = int(secnames[secnames.find('Cell[') + 5:secnames.find('].')])
        #print(cellind)
        h('forsec cell.all{ #print secname()}')
        h('Cell[0].soma[0] nc =  new NetCon(&v(0.5), nil)')
        pc.set_gid2node(i,RANK)
        h('pc.cell('+str(i)+', nc)')  # //
        cell1=pc.gid2cell(i)
        #print(cell1)
        self.celldict[i]=cell#(cell,swc_tree)
        self.cells.append(cell)
        return morphxmldoc


        #print(len(h.List('NetCon')))

        pol=[ a.polarity for a in self.cells ]
        import numpy as np
        #print np.sum(pol)
        os.chdir(os.getcwd() + '/../')
        self.h.define_shape()

        self.h('forall{ for(x,0){ insert xtra }}')
        self.h('forall{ for(x,0){ insert extracellular}}')
        self.h('xopen("interpxyz.hoc")')
        self.h('grindaway()')
        #self.h('xopen("seclists.hoc")')
        ##print gids
        ##print len(gids)

    def read_local_swc(self):
        h=self.h
        NCELL=self.NCELL
        SIZE=self.SIZE
        RANK=self.RANK
        #from neuron import h
        pc=h.ParallelContext()

        morphs=[]
        cells1=[]
        #self.initialize_hoc()
        self.h.xopen('load_swc4.hoc')
        #import use_local_files
        swclist=glob.glob('*.swc')
        #itergids = iter( i for i in range(RANK, 1, SIZE) )

        #for swcf, i in enumerate(swclist):
        #for i in itergids:
            #morphology = swc.read_swc(swcf)
        cell = h.mkcell(swclist[0])
                #self.generate_morphology(cell, d[i][3])
        self.generate_morphology(cell,swclist[0])
            #self.load_cell_parameters(cell, fit_ids[utils.cells_data[i]['type']])
        cells1.append(cell)
            #cell1=self.load_cell_parameters()
            #cells1.append(cell1)
            #print type(cells1)
            #print type(cell1)
            #morphology.root
            #morphs.append(morphology)
        return swclist,cells1

    def gcs(self,NCELL):
        NCELL=self.NCELL
        SIZE=self.SIZE
        RANK=self.RANK


        h=self.h
        #from neuron import h
        pc=h.ParallelContext()

        h('objref pc, py, nc, cells')
        h('pc = new ParallelContext()')
        h('py = new PythonObject()')
        #h('')

        swcdict={}
        NFILE = 3175
        #fit_ids = self.description.data['fit_ids'][0] #excitatory
        #self.cells_data = self.description.data['biophys'][0]['cells']
        info_swc=self.prep_list()
        d = { x: y for x,y in enumerate(info_swc)}
        import os
        os.chdir(os.getcwd() + '/main')
        #itergids = iter( i for i in range(RANK, NCELL, SIZE) )

        #itergids=iter(gids)
        #iterd=iter(d)
        #for i in itergids:
            #print i, 'itergids'
        i=0
        #cell = self.h.Cell() #use self.h.cell() for allen brain cell.
        cell = h.mkcell(d[i][3])
        #self.generate_morphology(cell, d[i][3])#iterd.next())#iterswc.next())
        #h('cell.geom_nseg()')

        #print cell, 'cell'
        cell.geom_nseg()
        cell.gid1=i #itergids.next()

        #print int(cell.gid1)

        '''
        #print type(cell), 'this does not always work'
         '''
        if 'pyramid' in d[i]:
            self.load_cell_parameters(cell, fit_ids[self.cells_data[0]['type']])
            cell.polarity=1
        else:
        #inhibitory type stained cell.
            self.load_cell_parameters(cell, fit_ids[self.cells_data[2]['type']])
            cell.polarity=0

        #h('objref nc')
        #h('pc.set_gid2node('+str(i)+','+str(RANK)+')')  # // associate gid i with this host
        #h('cell.soma[0] nc =  new NetCon(&v(0.5), nil)')
        #h('pc.set_gid2node('+str(i)+','+str(RANK)+')')  # // associate gid i with this host
        #h('pc.cell('+str(i)+', nc)')


        #h('pc.set_gid2node(int(py.i), pc.id)')  # // associate gid i with this host
        #h.cell.soma[0] netcon = new NetCon(&v(x), target)
        ##print dir(h.cell.soma)
        ##print dir(h.cell)

        #nc = h.NetCon(h.cell.soma(1)._ref_v, null)
        #h('topology()')
        #nc = h.NetCon(self.h.Cell[0].soma(1)._ref_v, null, sec = self.h.Cell[0].soma[0])


        #secnames = sec.name()
        #cellind = int(secnames[secnames.find('Cell[') + 5:secnames.find('].')])
        #h('Cell['+str(cellind)+'].soma[0] nc =  new NetCon(&v(0.5), nil)')

        #h('forsec cell.somatic{ #print secname()}')

        #h('crash here please')
        ##print this can't be write
        h('Cell[0].soma[0] nc =  new NetCon(&v(0.5), nil)')
        #h('cell.soma nc =  new NetCon(&v(0.5), nil)')

        #h('pc.cell('+str(int(i))+', nc)')
        #pc.cell(i,h.nc)


        pc.set_gid2node(i,RANK)
        h('pc.cell('+str(i)+', nc)')  # //

        #h('pc.set_gid2node('+str(i)+', pc.id)')  # // associate gid i with this host
        #h('#print Cell[0]')
        #h('#print Cell[0].soma[0]')
        #h('#print cell')
        #h.cell=cell
        ##print dir(cell.soma)

        #h.cell.soma[0] h.nc
        #h('cell.soma[0] nc =  new NetCon(&v(0.5), nil)')

        #h.pc.cell(int(i),h.nc)

        #h('cells.append(cell)')
        #h('gidvec.append(py.i)')
        cell1=pc.gid2cell(i)
        #print cell1
        #swc_tree = btmorph.STree2()
        #swc_tree.read_SWC_tree_from_file(d[i][3])
        self.celldict[i]=cell#(cell,swc_tree)
        #stats=btmorph.BTStats(self.celldict[i][1])
        #Returns 3 lists for soma bifurcations and leafs.
        #Typicall synapses would only occur at leaf node collisions, this is something I had not considered.
        #soma,bifs,ends=stats.get_points_of_interest()
        #end[0].get_content()['pd3'].type
        #axonterminals=[i for i in ends if (i.get_content()['p3d'].type==)]
        #dendriteterminals=[i for i in ends if (i.get_content()['p3d'].type==)]
        #soma=[i for i in ends if (i.get_content()['p3d'].type==)]
        self.cells.append(cell)


        len(h.List('NetCon'))
        pol=[ a.polarity for a in self.cells ]
        import numpy as np
        #print np.sum(pol)
        os.chdir(os.getcwd() + '/../')
        self.h.define_shape()
        self.h('forall{ for(x,0){ insert xtra }}')
        self.h('forall{ for(x,0){ insert extracellular}}')
        self.h('xopen("interpxyz.hoc")')
        self.h('grindaway()')


 # no .clear() command

    def htype (obj): st=obj.hname(); sv=st.split('['); return sv[0]
    def secname (obj): obj.push(); #print self.h.secname() ; self.h.pop_section()
    def psection (obj): obj.push(); #print self.h.psection() ; self.h.pop_section()

    # still need to generate a full allsecs
    def mkallsecs():
        """ mkallsecs - make the global allsecs variable, containing
        all the NEURON sections.
        """
        #global allsecs
        allsecs=self.h.SectionList()
        return allsecs

    def matrix_reduce(self,ecm,icm):
        import numpy as np
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        icm = np.zeros((NCELL, NCELL))
        ecm = np.zeros((NCELL, NCELL))
        COMM.Barrier()
        my_icm = np.zeros_like(icm)
        COMM.Reduce([icm, MPI.DOUBLE], [my_icm, MPI.DOUBLE], op=MPI.SUM,
                    root=0)
        my_ecm = np.zeros_like(ecm)
        COMM.Reduce([ecm, MPI.DOUBLE], [my_ecm, MPI.DOUBLE], op=MPI.SUM,
                    root=0)
        return ecm, icm


    def prun(self,tstop):
        h=self.h
        pc=h.ParallelContext()
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        checkpoint_interval = 50000.

        #This code is from:
        #http://senselab.med.yale.edu/ModelDB/ShowModel.asp?model=151681
        cvode = h.CVode()
        cvode.cache_efficient(1)

      # pc.spike_compress(0,0,1)

        pc.setup_transfer()
        mindelay = pc.set_maxstep(10)
        if RANK == 0:
            print('mindelay = %g' % mindelay)
        runtime = h.startsw()
        exchtime = pc.wait_time()

        inittime = h.startsw()
        h.stdinit()
        inittime = h.startsw() - inittime
        if RANK == 0:
            print('init time = %g' % inittime)

        while h.t < tstop:
            told = h.t
            tnext = h.t + checkpoint_interval
            if tnext > tstop:
                tnext = tstop
            pc.psolve(tnext)
            if h.t == told:
                if RANK == 0:
                    print('psolve did not advance time from t=%.20g to tnext=%.20g\n' \
                        % (h.t, tnext))
                break

        # if h.t%2==0: The problem is h.t is float multiple not integer multiple

            #print 'working', h.t
        runtime = h.startsw() - runtime
        comptime = pc.step_time()
        splittime = pc.vtransfer_time(1)
        gaptime = pc.vtransfer_time()
        exchtime = pc.wait_time() - exchtime
        if RANK == 0:
            print('runtime = %g' % runtime)
        #print comptime, exchtime, splittime, gaptime


    '''
    def innerloop(self,i,data,ecm, icm):
        h=self.h
        pc=h.ParallelContext()
        RANK=self.RANK

        secnames = ''# sec.name()
        cellind =0 #int(secnames[secnames.find('Cell[') + 5:secnames.find('].')])  # This is the index of the post synaptic cell.
        polarity = 0
        h('objref coords')
        h('coords = new Vector(3)')
        cnt=0
        if i in self.celldict.keys():

            for sec in self.celldict[i].spk_rx_ls.allsec():
            #for sec in j.spk_rx_ls.allsec():
                for seg in sec:
                    #print seg.x, sec.name(), RANK, data['hostfrom']

                    ##print sec.name()
                    h(str('coords2.x[2]=') + str('z_xtra(')
                      + str(seg.x) + ')')
                    h(str('coords2.x[1]=') + str('y_xtra(')
                      + str(seg.x) + ')')
                    h(str('coords2.x[0]=') + str('x_xtra(')
                    + str(seg.x) + ')')

                    h('coordsx=0.0')
                    h.coordsx = data['coords'][0]
                    h('coordsy=0.0')
                    h.coordsy = data['coords'][1]
                    h('coordsz=0.0')
                    h.coordsz = data['coords'][2]

                    coordsx = float(data['coords'][0])
                    coordsy = float(data['coords'][1])
                    coordsz = float(data['coords'][2])
                    r = 0.
                    import math
                    r=math.sqrt((h.coords2.x[0] - coordsx)**2+(h.coords2.x[1] - coordsy)**2+(h.coords2.x[2] - coordsz)**2)
                    gidn=data['gid']
                    #if front.parent == None or o_front.parent == None:
                    #    D = np.sqrt(np.sum((front.xyz-o_front.xyz)**2))
                    #else:
                    #    D = dist3D_segment_to_segment (front.xyz,front.parent.xyz,o_front.parent.xyz,o_front.xyz)

                    #h('py.r = sqrt((coords2.x[0] - coordsx)^2 + (coords2.x[1] - coordsy)^2 + (coords2.x[2] - coordsz)^2)')
                    r = float(r)
                    if r < 1:
                        #print r,# 'this is not hopefuly wiring everything to everything'
                        gidcompare = ''

                        secnames = sec.name()
                        cellind = int(secnames[secnames.find('Cell[') + 5:secnames.find('].')])  # This is the index of the post synaptic cell.

                        polarity = 0


                        #cellind is a cell index, that is relative to the host. So the identifier repeats on different hosts.
                        #gidn is a global identifier. These numbers are not repeated on different hosts.
                        polarity=int(h.Cell[int(cellind)].polarity)

                        #print polarity
                        h('objref syn_')
                        if int(polarity) == int(0):
                            post_syn = secnames + ' ' + 'syn_ = new GABAa(' + str(seg.x) + ')'
                            icm[i][gidn] = icm[i][gidn] + 1
                        else:

                            post_syn = secnames + ' ' + 'syn_ = new AMPA(' + str(seg.x) + ')'
                            ecm[i][gidn] = ecm[i][gidn] + 1

                        h(post_syn)
                        #print post_syn
                        h('#print syn_')
                        syn_=h.syn_
                        h.syn_.cid=i
                        h.Cell[cellind].ampalist.append(h.syn_)
                        h.Cell[cellind].div.append(data['gid'])
                        h.Cell[cellind].gvpre.append(data['gid'])
                        nc=pc.gid_connect(data['gid'],syn_)
                        h.nc.delay=1+r/0.4
                        h.nc.weight[0]=r/0.4
                        self.nclist.append(nc)





        return self.nclist, ecm, icm
    '''
    def inner1(self,j):
        from neuron import h

        pc=h.ParallelContext()
        h=self.h

        import numpy as np
        coordictlist=[]
        coordict={}
        pc=h.pc
        h('objref coords')
        h('coords = new Vector(3)')

        #from collections import defaultdict
        #coordict2 = defaultdict(dict)
        if j in self.celldict.keys():
            seglist= iter( (seg, sec, self.celldict[j]) for sec in self.celldict[j].spk_trig_ls for seg in sec )
            for (seg,sec, cellc) in seglist:
                    get_cox = str('coords.x[0]=x_xtra('
                                  + str(seg.x) + ')')
                    h(get_cox)


                    get_coy = str('coords.x[1]=y_xtra('
                                  + str(seg.x) + ')')
                    h(get_coy)
                    get_coz = str('coords.x[2]=z_xtra('
                                  + str(seg.x) + ')')
                    h(get_coz)


                    #coordict={}
                    coordict['hostfrom']=pc.id()
                    coordict['coords'] = np.array(h.coords.to_python(),
                                              dtype=np.float64)
                    coordict['gid']= int(j)
                    ##print coordict['gid'], cellc.gid1
                    coordict['seg']= seg.x

                    secnames = sec.name()  # h.secnames
                    coordict['secnames'] = str(secnames)
                    ##print seg.x, sec.name(), 'below x_xtra'
                    h('#print x_xtra('+ str(seg.x) +')')
                    coordictlist.append(coordict)

        #print len(coordictlist)#, len(seglist), 'length comparison'
        return coordictlist

    def tracenet(self):
        ncsize=len(self.h.NetCon)
        import numpy as np
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        icm = np.zeros((NCELL, NCELL))
        ecm = np.zeros((NCELL, NCELL))
        COMM.Barrier()
        my_icm = np.zeros_like(icm)
        COMM.Reduce([icm, MPI.DOUBLE], [my_icm, MPI.DOUBLE], op=MPI.SUM,
                    root=0)
        my_ecm = np.zeros_like(ecm)
        COMM.Reduce([ecm, MPI.DOUBLE], [my_ecm, MPI.DOUBLE], op=MPI.SUM,
                    root=0)
        #return ecm, icm
        #make a list of tuples where each list element contains (srcid,tgtid,srcpop,tgtpop)
        #for s in xrange(0,SIZE):
        for i in xrange(0,ncsize-1):
            #srcs.append(int(self.h.NetCon[i].srcgid()))
            #tgts.append(int(self.h.NetCon[i].postcell().gid1))
            srcind=int(self.h.NetCon[i].srcgid())
            tgtind=int(self.h.NetCon[i].postcell().gid1)
            #print int(utils.h.NetCon[i].srcgid()),int(utils.h.NetCon[i].postcell().gid1),utils.celldict[srcind],utils.celldict[tgtind]
                ##print strlist[tgtind]==dic[tgtind], ' sanity check '
                #add to list of tuples, netcon src, netcon tgt, src index, target index.
            lsoftup.append((int(utils.h.NetCon[i].srcgid()),int(utils.h.NetCon[i].postcell().gid1),utils.celldict[srcind],utils.celldict[tgtind]))
        return lsoftup
        #The broadcasting and gathering should happen on a different host.
        #Actually this should all be reduced to rank0
        #    data = COMM.bcast(lsoftup, root=s)  # ie root = rank
        #return data



    def wirecells_s(self):
        '''wire cells on the same hosts'''
        import numpy as np
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        icm = np.zeros((NCELL, NCELL))
        ecm = np.zeros((NCELL, NCELL))
        #self.nclist
        from neuron import h
        pc=h.ParallelContext()

        h=self.h


        celliter= iter( (j, i) for j,l in self.celldict.iteritems() for i,t in self.celldict.iteritems() if i!=j )
        for (j,i) in celliter:
            #print i==j
            cell1=pc.gid2cell(i)

            coordictlist=self.inner1(j)



            #This can be functionalised
            for coordict in coordictlist:

                seglist= iter( (seg, sec, self.celldict[j]) for sec in self.celldict[j].spk_rx_ls for seg in sec )

                for (seg, sec, cellc) in seglist:
                    secnames = sec.name()
                    cellind = int(secnames[secnames.find('Cell[') + 5:secnames.find('].')])  # This is the index of the post synaptic cell.
                    h('objref coords2')
                    h('coords2 = new Vector(3)')


                    h(str('coords2.x[2]=') + str('z_xtra(')
                      + str(seg.x) + ')')
                    h(str('coords2.x[1]=') + str('y_xtra(')
                      + str(seg.x) + ')')
                    h(str('coords2.x[0]=') + str('x_xtra(')
                    + str(seg.x) + ')')

                    h('coordsx=0.0')


                    ##print coordict['coords'][0], coordict['coords'][1],coordict['coords'][2],type(coordict['coords'][0])
                    h.coordsx = coordict['coords'][0]
                    h('coordsy=0.0')
                    h.coordsy = coordict['coords'][1]
                    h('coordsz=0.0')
                    h.coordsz = coordict['coords'][2]

                    coordsx = float(coordict['coords'][0])
                    coordsy = float(coordict['coords'][1])
                    coordsz = float(coordict['coords'][2])
                    r = 0.
                    import math
                    r=math.sqrt((h.coords2.x[0] - coordsx)**2+(h.coords2.x[1] - coordsy)**2+(h.coords2.x[2] - coordsz)**2)
                    gidn=coordict['gid']
                    #if front.parent == None or o_front.parent == None:
                    #    D = np.sqrt(np.sum((front.xyz-o_front.xyz)**2))
                    #else:
                    #    D = dist3D_segment_to_segment (front.xyz,front.parent.xyz,o_front.parent.xyz,o_front.xyz)

                    #h('py.r = sqrt((coords2.x[0] - coordsx)^2 + (coords2.x[1] - coordsy)^2 + (coords2.x[2] - coordsz)^2)')
                    r = float(r)
                    if r < 1:

                        #print r,# 'this is not hopefuly wiring everything to everything'
                        gidcompare = ''


                        polarity = 0


                        #cellind is a cell index, that is relative to the host. So the identifier repeats on different hosts.
                        #gidn is a global identifier. These numbers are not repeated on different hosts.
                        polarity=int(h.Cell[int(cellind)].polarity)
                        ##print seg.x, coordict['seg'], coordict['secnames'], sec.name(), RANK, coordict['hostfrom'], coordict['gid'], int(h.Cell[int(cellind)].gid1)

                        ##print polarity
                        h('objref syn_')
                        if int(polarity) == int(0):
                            post_syn = secnames + ' ' + 'syn_ = new GABAa(' + str(seg.x) + ')'
                            icm[i][gidn] = icm[i][gidn] + 1
                        else:

                            post_syn = secnames + ' ' + 'syn_ = new AMPA(' + str(seg.x) + ')'
                            ecm[i][gidn] = ecm[i][gidn] + 1

                        h(post_syn)
                        ##print post_syn
                        h('#print syn_')
                        syn_=h.syn_
                        h.syn_.cid=i
                        h.Cell[cellind].ampalist.append(h.syn_)
                        h.Cell[cellind].div.append(coordict['gid'])
                        h.Cell[cellind].gvpre.append(coordict['gid'])
                        #nc=pc.gid_connect(i,syn_)

                        h('objref nc')
                        #target
                        #This object assignment syntax is a bit wrong.
                        ls=str(coordict['secnames'])+' nc =  new NetCon(&v('+str(coordict['seg'])+'),'+str(sec.name())+')'
                        #print str(sec.name()), coordict['secnames'] , coordict['seg']
                        ##print cellc.gid1, "gid"
                        ##print ls
                        h(ls)
                        nc=h.nc
                        ##print i, coordict['gid'], coordict['gid']==i, 'gid system consistant?'
                        ##print t.gid1, cell1.gid1, 'gids consistant? in wiring?'
                        ##print 'pre gids consistant?'
                        ##print j, coordict['gid'], coordict['gid']==j, 'gid system consistant?', l.gid1

                        #nc=pc.gid_connect(coordict['gid'],syn_)
                        nc.delay=1+r/0.4
                        nc.weight[0]=r/0.4
                        ##print nc, 'connected!'
                        self.nclist.append(nc)

        ecm,icm = self.matrix_reduce(ecm,icm)
        return (self.nclist, ecm, icm)


        #for j,l in self.celldict.iteritems(): #pre synapses
            #for i,t in self.celldict.iteritems(): #post synapses.
                #    if pc.gid_exists(i):#if the postsynaptic cell even exists grab a reference to it.


                            #for sec in t.spk_rx_ls:#post synapses.

                                ##print (int(t.gid1) != int(coordict['gid'])), int(t.gid1),  int(coordict['gid'])
                                #h('objref cell1')


                                    ##print 'fails here'
                                    #h('cell1=pc.gid2cell('+str(i)+')')
                                    #h('#print cell1, "cell1"')

                            #for sec in self.celldict[i].spk_rx_ls.allsec():
                            ##print 'gid ',int(self.celldict[i].gid1), i, int(t.gid1)
                                ##print int(h.Cell[int(cellind)].gid1), int(self.cells[int(cellind)].gid1), int(coordict['gid']), sec.name(), int(cellind)
        #                                        #print 'these numbers are not reliable', int(h.Cell[int(cellind)].gid1), int(self.cells[int(cellind)].gid1), t.gid1, t.soma[0].name()
                                ##print 'these numbers are not reliable', int(h.Cell[int(cellind)].gid1), int(self.cells[int(cellind)].gid1), int(t.gid1), i#, t

                                        #if(int(h.Cell[int(cellind)].gid1)!=coordict['gid'])

                                    #for sec in j.spk_rx_ls.allsec():
                             #   for seg in sec:

                                                #break
                                        #data=None
                #data=None

            #if RANK == s:

            ##print 'can I use a function decorator for everything inside this part?'

            #coordict=self.inner1(j)

            #coordictlist=[]

                #for sec in j.spk_trig_ls.allsec():



    def innermost(self,data):
        from segment_distance import dist3D_segment_to_segment
        import numpy as np
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        icm = np.zeros((NCELL, NCELL))
        ecm = np.zeros((NCELL, NCELL))
        #self.nclist
        from neuron import h
        pc=h.ParallelContext()
        h=self.h

        secnames = ''
        cellind =0
        polarity = 0
        h('objref coords')
        h('coords = new Vector(3)')
        h('objref pc')
        h('pc = new ParallelContext()')

        #This can be functionalised
        h('objref coords2')
        h('coords2 = new Vector(3)')



        iterdata=iter( (k,i,t) for k in data for i,t in self.celldict.iteritems() if i in self.celldict.keys() if int(t.gid1) != int(k['gid']))
#For ever GID
#For every coordinate thats received from a broadcast.
              #for i,t in self.celldict.iteritems():
#For ever GID thats on this host (in the dictionary)
                 #if i in self.celldict.keys():

        for k,i,t in iterdata :
               #:  # if the gids are not the same.
#Rule out self synapsing neurons (autopses), with the condition
#pre GID != post GID
#If the putative post synaptic gid exists on this CPU, the reference
#to the corresponding cell object is the element 't' in celldict gid-> cell dictionary
#Iterate through the post synaptic
#tree.

            iterseg=iter( (seg,sec) for sec in t.spk_rx_ls for seg in sec)
            for seg,sec in iterseg:
                #print seg.x, sec.name(), k['secnames']
                h('objref cell1')
                h('cell1=pc.gid2cell('+str(i)+')')
                secnames = sec.name()
                cellind = int(secnames[secnames.find('Cell[') + 5:secnames.find('].')])  # This is the index of the post synaptic cell.


                h(str('coords2.x[2]=') + str('z_xtra(')
                  + str(seg.x) + ')')
                h(str('coords2.x[1]=') + str('y_xtra(')
                  + str(seg.x) + ')')
                h(str('coords2.x[0]=') + str('x_xtra(')
                + str(seg.x) + ')')

                h('coordsx=0.0')
                h.coordsx = k['coords'][0]
                h('coordsy=0.0')
                h.coordsy = k['coords'][1]
                h('coordsz=0.0')
                h.coordsz = k['coords'][2]

                coordsx = float(k['coords'][0])
                coordsy = float(k['coords'][1])
                coordsz = float(k['coords'][2])

    #Find the euclidian distance between putative presynaptic segments,
    #and putative post synaptic segments.

    #If the euclidian distance is below an allowable threshold in micro
    #meters, continue on with code responsible for assigning a
    #synapse, and a netcon. Neurons parallel context class can handle the actual message passing associated with sending and receiving action potentials on different hosts.


                r = 0.
                import math
                r=math.sqrt((h.coords2.x[0] - coordsx)**2+(h.coords2.x[1] - coordsy)**2+(h.coords2.x[2] - coordsz)**2)
                gidn=k['gid']
                r = float(r)
                if r < 1:

                    #print r,# 'this is not hopefuly wiring everything to everything'
                    polarity = 0
                    polarity=int(h.Cell[int(cellind)].polarity)
                    #print seg.x, k['seg'], k['secnames'], sec.name(), RANK, k['hostfrom'], k['gid'], int(h.Cell[int(cellind)].gid1)

                    #print polarity
                    h('objref syn_')
                    if int(polarity) == int(0):
                        post_syn = secnames + ' ' + 'syn_ = new GABAa(' + str(seg.x) + ')'
                        icm[i][gidn] = icm[i][gidn] + 1
                    else:

                        post_syn = secnames + ' ' + 'syn_ = new AMPA(' + str(seg.x) + ')'
                        ecm[i][gidn] = ecm[i][gidn] + 1

                    h(post_syn)
                    #print post_syn
                    h('#print syn_')
                    syn_=h.syn_
                    h.syn_.cid=i
                    h.Cell[cellind].ampalist.append(h.syn_)
                    h.Cell[cellind].div.append(k['gid'])
                    h.Cell[cellind].gvpre.append(k['gid'])
                    nc=pc.gid_connect(k['gid'],syn_)
                    nc.delay=1+r/0.4
                    nc.weight[0]=r/0.4
                    self.nclist.append(nc)
            h('uninsert xtra')
        return self.nclist, ecm, icm

    def wirecells4(self):
        '''wire cells between hosts, but don't wire cells on the same host.'''

        #def wirecells(RANK,NCELL,SIZE,h,icm,ecm):
        from segment_distance import dist3D_segment_to_segment
        import numpy as np
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        icm = np.zeros((NCELL, NCELL))
        ecm = np.zeros((NCELL, NCELL))
        #self.nclist
        h=self.h
        pc=h.ParallelContext()
        secnames = ''
        cellind =0
        polarity = 0
        h('objref coords')
        h('coords = new Vector(3)')
        h('objref pc')
        h('pc = new ParallelContext()')

        coordict=None
        coordictlist=None

    #Iterate over all CPU ranks, iterate through all GIDs (global
    #identifiers, stored in the python dictionary).
        for s in xrange(0, SIZE):
            celliter= iter( (i, j) for i,j in self.celldict.iteritems() )
            for (i,j) in celliter:
                ##print i==j
                cell1=pc.gid2cell(i)
                coordictlist=self.inner1(i)
                #print 'entered parallel wiring now'
            data = COMM.bcast(coordictlist, root=s)  # ie root = rank

            if len(data) != 0:
                self.nclist, ecm, icm = self.innermost(data)


        data=None
        ecm,icm = self.matrix_reduce(ecm,icm)
        return (self.nclist, ecm, icm)

        #This can be functionalised
        '''
        h('objref coords2')
        h('coords2 = new Vector(3)')



        if len(data) != 0:
           for k in data:
        #For every coordinate thats received from a broadcast.
                          for i,t in self.celldict.iteritems():
        #For ever GID thats on this host (in the dictionary)
                             if i in self.celldict.keys():
                                 if int(t.gid1) != int(k['gid']):  # if the gids are not the same.
        #Rule out self synapsing neurons (autopses), with the condition
        #pre GID != post GID
        #If the putative post synaptic gid exists on this CPU, the reference
        #to the corresponding cell object is the element 't' in celldict gid-> cell dictionary
        #Iterate through the post synaptic
        #tree.

                            for sec in t.spk_rx_ls:
                                h('objref cell1')
                                h('cell1=pc.gid2cell('+str(i)+')')
                                secnames = sec.name()
                                cellind = int(secnames[secnames.find('Cell[') + 5:secnames.find('].')])  # This is the index of the post synaptic cell.

                                for seg in sec:

                                    h(str('coords2.x[2]=') + str('z_xtra(')
                                      + str(seg.x) + ')')
                                    h(str('coords2.x[1]=') + str('y_xtra(')
                                      + str(seg.x) + ')')
                                    h(str('coords2.x[0]=') + str('x_xtra(')
                                    + str(seg.x) + ')')

                                    h('coordsx=0.0')
                                    h.coordsx = k['coords'][0]
                                    h('coordsy=0.0')
                                    h.coordsy = k['coords'][1]
                                    h('coordsz=0.0')
                                    h.coordsz = k['coords'][2]

                                    coordsx = float(k['coords'][0])
                                    coordsy = float(k['coords'][1])
                                    coordsz = float(k['coords'][2])

#Find the euclidian distance between putative presynaptic segments,
#and putative post synaptic segments.

#If the euclidian distance is below an allowable threshold in micro
#meters, continue on with code responsible for assigning a
#synapse, and a netcon. Neurons parallel context class can handle the actual message passing associated with sending and receiving action potentials on different hosts.


                                    r = 0.
                                    import math
                                    r=math.sqrt((h.coords2.x[0] - coordsx)**2+(h.coords2.x[1] - coordsy)**2+(h.coords2.x[2] - coordsz)**2)
                                    gidn=k['gid']
                                    r = float(r)
                                    if r < 10:

                                        #print r,# 'this is not hopefuly wiring everything to everything'
                                        polarity = 0
                                        polarity=int(h.Cell[int(cellind)].polarity)
                                        #print seg.x, k['seg'], k['secnames'], sec.name(), RANK, k['hostfrom'], k['gid'], int(h.Cell[int(cellind)].gid1)

                                        #print polarity
                                        h('objref syn_')
                                        if int(polarity) == int(0):
                                            post_syn = secnames + ' ' + 'syn_ = new GABAa(' + str(seg.x) + ')'
                                            icm[i][gidn] = icm[i][gidn] + 1
                                        else:

                                            post_syn = secnames + ' ' + 'syn_ = new AMPA(' + str(seg.x) + ')'
                                            ecm[i][gidn] = ecm[i][gidn] + 1

                                        h(post_syn)
                                        #print post_syn
                                        h('#print syn_')
                                        syn_=h.syn_
                                        h.syn_.cid=i
                                        h.Cell[cellind].ampalist.append(h.syn_)
                                        h.Cell[cellind].div.append(k['gid'])
                                        h.Cell[cellind].gvpre.append(k['gid'])
                                        nc=pc.gid_connect(k['gid'],syn_)
                                        nc.delay=1+r/0.4
                                        nc.weight[0]=r/0.4
                                        self.nclist.append(nc)
                            h('uninsert xtra')
    # Remove the mechanism, since the only point of these mechanisms, is to get accurate coordinates
    # once this has been achieved they can be uninserted from the spike trigger list.
        '''





    def wirecells3(self):
        '''wire cells between hosts, but don't wire cells on the same host.'''
        #def wirecells(RANK,NCELL,SIZE,h,icm,ecm):
        from segment_distance import dist3D_segment_to_segment
        import numpy as np
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        icm = np.zeros((NCELL, NCELL))
        ecm = np.zeros((NCELL, NCELL))
        self.nclist=[]
        h=self.h
        pc=h.ParallelContext()
        secnames = ''
        cellind =0
        polarity = 0
        h('objref coords')
        h('coords = new Vector(3)')
        h('objref pc')
        h('pc = new ParallelContext()')

        coordict=None
        coordictlist=None

    #Iterate over all CPU ranks, iterate through all GIDs (global
    #identifiers, stored in the python dictionary).
        for s in xrange(0, SIZE):
            for j,l in self.celldict.iteritems():


                if pc.gid_exists(j):
                    coordictlist=[]

    #if the required GID is found on this CPU, get a reference to the cell
    #object and iterate through sections comprising its
    #presynaptic tree (axonal).

                    for sec in self.celldict[j].spk_trig_ls:
                        for seg in sec:


    #inside the Python HOC object 'h', the variable x_xtra denotes a
    #segment dependent x coordinate associated with a mechanism in the
    #section being iterated through, y and z coordinates can be obtained
    #via the same means.

                            get_cox = str('coords.x[0]=x_xtra('
                                          + str(seg.x) + ')')

    #Use NEURONs self reflection to execute a string (data type), as an
    #instruction inside a call to the NEURON HOC object 'h'.


                            get_coy = str('coords.x[1]=y_xtra('
                                          + str(seg.x) + ')')
                            h(get_coy)
                            get_coz = str('coords.x[2]=z_xtra('
                                          + str(seg.x) + ')')
                            h(get_coz)
                            coordict={}

                            secnames = sec.name()
                            cellind = int(secnames[secnames.find('Cell[') + 5:secnames.find('].')])

	#coordict is a dictionary that contains an array of coordinates, the GID of
	#the putative presynaptic neuron, the segment of the putative
	#presynaptic axon connection source and the section string name
	#correspoding to the presynaptic location of those coorinates, this will all be
     #appended to a list which is updated in this nested for loop.

                            coordict['gid']=j
                            coordict['seg']= seg.x

                            secnames = sec.name()  # h.secnames
                            coordict['secnames'] = str(secnames)
                            coordict['coords'] = np.array(h.coords.to_python(),
                                              dtype=np.float64)
                            coordict['hostfrom']=RANK
                            ##print i,j, ' i,j', seg.x, sec.name(), RANK

                            coordictlist.append(coordict)
     #Remove mechanisms, since the only point of these mechanisms, is to get accurate coordinates
     #once this has been achieved they can be uninserted from the spike trigger list.
                    h('uninsert xtra')
                    #h('uninsert extracellular')

  	#After exiting the nested for loop broadcast those coordinates to all hosts/ranks other than the one currently on.

                    data = COMM.bcast(coordictlist, root=s)  # ie root = rank

                    h('objref coords2')
                    h('coords2 = new Vector(3)')

                    if len(data) != 0:
                       for k in data:
    #For every coordinate thats received from a broadcast.
                          for i,t in self.celldict.iteritems():
    #For ever GID thats on this host (in the dictionary)
                             if i in self.celldict.keys():
                                 if int(t.gid1) != int(k['gid']):  # if the gids are not the same.
    #Rule out self synapsing neurons (autopses), with the condition
    #pre GID != post GID
    #If the putative post synaptic gid exists on this CPU, the reference
    #to the corresponding cell object is the element 't' in celldict gid-> cell dictionary
    #Iterate through the post synaptic
    #tree.

                                    for sec in t.spk_rx_ls:
                                        h('objref cell1')
                                        h('cell1=pc.gid2cell('+str(i)+')')
                                        secnames = sec.name()
                                        cellind = int(secnames[secnames.find('Cell[') + 5:secnames.find('].')])  # This is the index of the post synaptic cell.

                                        for seg in sec:

                                            h(str('coords2.x[2]=') + str('z_xtra(')
                                              + str(seg.x) + ')')
                                            h(str('coords2.x[1]=') + str('y_xtra(')
                                              + str(seg.x) + ')')
                                            h(str('coords2.x[0]=') + str('x_xtra(')
                                            + str(seg.x) + ')')

                                            h('coordsx=0.0')
                                            h.coordsx = k['coords'][0]
                                            h('coordsy=0.0')
                                            h.coordsy = k['coords'][1]
                                            h('coordsz=0.0')
                                            h.coordsz = k['coords'][2]

                                            coordsx = float(k['coords'][0])
                                            coordsy = float(k['coords'][1])
                                            coordsz = float(k['coords'][2])

    #Find the euclidian distance between putative presynaptic segments,
    #and putative post synaptic segments.

    #If the euclidian distance is below an allowable threshold in micro
    #meters, continue on with code responsible for assigning a
    #synapse, and a netcon. Neurons parallel context class can handle the actual message passing associated with sending and receiving action potentials on different hosts.


                                            r = 0.
                                            import math
                                            r=math.sqrt((h.coords2.x[0] - coordsx)**2+(h.coords2.x[1] - coordsy)**2+(h.coords2.x[2] - coordsz)**2)
                                            gidn=k['gid']
                                            r = float(r)
                                            if r < 10:

                                                #print r,# 'this is not hopefuly wiring everything to everything'
                                                polarity = 0
                                                polarity=int(h.Cell[int(cellind)].polarity)
                                                #print seg.x, k['seg'], k['secnames'], sec.name(), RANK, k['hostfrom'], k['gid'], int(h.Cell[int(cellind)].gid1)

                                                #print polarity
                                                h('objref syn_')
                                                if int(polarity) == int(0):
                                                    post_syn = secnames + ' ' + 'syn_ = new GABAa(' + str(seg.x) + ')'
                                                    icm[i][gidn] = icm[i][gidn] + 1
                                                else:

                                                    post_syn = secnames + ' ' + 'syn_ = new AMPA(' + str(seg.x) + ')'
                                                    ecm[i][gidn] = ecm[i][gidn] + 1

                                                h(post_syn)
                                                #print post_syn
                                                h('#print syn_')
                                                syn_=h.syn_
                                                h.syn_.cid=i
                                                h.Cell[cellind].ampalist.append(h.syn_)
                                                h.Cell[cellind].div.append(k['gid'])
                                                h.Cell[cellind].gvpre.append(k['gid'])
                                                nc=pc.gid_connect(k['gid'],syn_)
                                                h.nc.delay=1+r/0.4
                                                h.nc.weight[0]=r/0.4
                                                self.nclist.append(nc)
                                    h('uninsert xtra')
    # Remove the mechanism, since the only point of these mechanisms, is to get accurate coordinates
    # once this has been achieved they can be uninserted from the spike trigger list.


            data=None
        ecm,icm = self.matrix_reduce(ecm,icm)
        return (self.nclist, ecm, icm)



    def mb(RANK, NCELL, SIZE, allrows2, gidvec, h,s1):
        #make soff and boff optional parameters in python function.
        #pdb.set_trace()
        #h.load_file("stdlib.hoc")
        #global s1
        #global i, cnt, cnti
        #h('chdir(workingdir)')
        #os.chdir(os.getcwd() + '/main')
        h('py = new PythonObject()')
        h('cnt=0')
        ie0 = np.zeros((NCELL, 2))
        ie1 = np.zeros((NCELL, 2))

        i=0
        cnti = 0
        cnt=0
        s1=''
        storename=''
        h('cnt=pc.id')

        for i in range(RANK, NCELL,SIZE):#NCELL-1, SIZE):  # 20 was int(len
            h.py.i=i
            s1 = allrows2[i]
            h.py.s1=s1

            storename = str(s1[3])  # //simply being inside a loop, may be the main problem
            if re.search('.swc', storename):
                h.cell = h.mkcell(storename)


                h('cell.geom_nseg()')
                #h.cell.geom_nseg()
                h('cell.gid1=py.i')
                h('cell.gvpre.#printf')
                h('cell.gvpost.#printf')




                h.cell.nametype=str(s1[5])
                #h('cell.nametype=py.str(py.s1[5])')
                h.cell.num_type=int(s1[6])
                #h('cell.num_type=py.int(py.s1[6])')
                h('cell.population=py.str(py.s1[7])')
                #h('cell.reponame=py.str(py.storename)')
                h.cell.reponame=str(storename)
                h('cell.div.resize(py.int(py.NCELL))')
                h('cell.conv.resize(py.int(py.NCELL))')
                #h('cell.nametype=py.str(py.s1[5])')

                h('if(strcmp("pyramid",py.s1[5])==0){pyr_list.append(cell)}')

                h('if(strcmp(cell.population,"neocortex")==0){ if(strcmp("pyramid",py.s[5])==0){ cell.pyr() }}')



                h('if(strcmp(cell.population,"hippocampus")==0){ if(strcmp("pyramid",py.s[5])==0){ cell.pyr2() }}')

                #if 'hippocampus' in s1:
                #    h('cell.pyr2()')


                if 'interneuron' in s1:
                    ie0[cnti] = i
                    ie1[cnti] = 1
                    cnti += 1
                    h.cell.basket()
                    h('cell.polarity=1')
                if 'pyramid' in s1:
                    ie0[cnti] = i
                    ie1[cnti] = 0
                    cnti += 1
                    h.cell.pyr()
                    h('cell.polarity=0')
                #h.cell.basket()
                #h('cell.polarity=1')


                h('if(strcmp("interneuron",py.s1[5])==0){ inter_list.append(cell) }')
                h('if(strcmp("interneuron",py.s1[5])==0){ cell.basket()}')
                h('if(strcmp("interneuron",py.s1[5])==0){ #print "inter neuron"}')

                h('if(strcmp("aspiny",py.s1[5])==0){ aspiny_list.append(cell) }')

                h('if(strcmp(cell.population,"hippocampus")==0){ hipp.append(cell)  }')

                h('if(strcmp(cell.population,"neocortex")==0){ neoc.append(cell) }')

                h('strdef cellposition')
                h('s#print(cellposition,"%s%d%s%d%s%d%s","cell.position(",py.float(py.s1[0]),",",py.float(py.s1[1]),",",py.float(py.s1[2]),")")')
                h('#print cellposition')
                h('execute(cellposition)')

                #h.pc.set_gid2node(int(i),RANK)
                h('pc.set_gid2node(int(py.i), pc.id)')  # // associate gid i with this host

                h('cell.soma[0] nc =  new NetCon(&v(0.5), nil)')

                h('pc.cell(int(py.i), nc)')  # //
                #h.pc.cell(int(i),h.nc)

                h('cells.append(cell)')
                h('gidvec.append(py.i)')
                #h.gidvec.append(i)
                gidvec.append(i)
                #print i
                h('#print py.i, " py.i", cnt, " cnt"')
                h('cnt+=pc.nhost')

                cnt += 1
                #strangley i can update in this environment but not in wirecells.
        #destroy the allrows list to free up memory, then return the empty list.
        #I can't seem to destroy allrows here without wrecking something in rigp later.
        #allrows=[]
        #pdb.set_trace()
        return (h.cells, allrows2, ie0, ie1)

    def generate_morphology(self, cell, morph_filename):
        h = self.h

        swc = self.h.Import3d_SWC_read()
        swc.input(morph_filename)
        imprt = self.h.Import3d_GUI(swc, 0)
        imprt.instantiate(cell)

        for seg in cell.soma[0]:
            seg.area()

        for sec in cell.all:
            sec.nseg = 1 + 2 * int(sec.L / 40)

        #cell.simplify_axon()
        #for sec in cell.axonal:
        #    sec.L = 30
        #    sec.diam = 1
        #    sec.nseg = 1 + 2 * int(sec.L / 40)
        #cell.axon[0].connect(cell.soma[0], 0.5, 0)
        #cell.axon[1].connect(cell.axon[0], 1, 0)
        h.define_shape()

    def load_cell_parameters(self, cell, type_index):
        h=self.h
        passive = self.description.data['fit'][type_index]['passive'][0]
        conditions = self.description.data['fit'][type_index]['conditions'][0]
        genome = self.description.data['fit'][type_index]['genome']

        # Set passive properties
        cm_dict = dict([(c['section'], c['cm']) for c in passive['cm']])
        for sec in cell.all:
            sec.Ra = passive['ra']
            sec.cm = cm_dict[sec.name().split(".")[1][:4]]
            sec.insert('pas')
            for seg in sec:
                seg.pas.e = passive["e_pas"]

        # Insert channels and set parameters
        for p in genome:
            sections = [s for s in cell.all if s.name().split(".")[1][:4] == p["section"]]
            for sec in sections:
                if p["mechanism"] != "":
                    sec.insert(p["mechanism"])
                    h('#print psection()')
                setattr(sec, p["name"], p["value"])

        # Set reversal potentials
        for erev in conditions['erev']:
            sections = [s for s in cell.all if s.name().split(".")[1][:4] == erev["section"]]
            for sec in sections:
                sec.ena = erev["ena"]
                sec.ek = erev["ek"]

    def connect_cells(self):
        self.synlist = []
        self.nclist = []
        connections = self.description.data["biophys"][0]["connections"]

        for connection in connections:
            for target in connection["targets"]:
                source_cell = self.cells[connection["source"]]
                target_cell = self.cells[target]

                syn = self.h.Exp2Syn(0.5, sec=target_cell.dend[0])
                syn.e = connection["erev"]
                source_section = source_cell.soma[0]
                nc = self.h.NetCon(source_section(0.5)._ref_v, syn, sec=source_section)
                nc.weight[0] = connection["weight"]
                nc.threshold = -20
                nc.delay = 2.0

                self.synlist.append(syn)
                self.q.append(nc)


    def connect_ring(self):
        self.synlist = []
        self.nclist = []
        connections = self.description.data["biophys"][0]["connections"]

        #for connection in connections:
        #    for target
            #in connection["targets"]:
        for i,discard in enumerate(self.cells):
            for j, discard in enumerate(self.cells):
                if i!=j:
                    #print type(i), type(j)

                    source_cell = self.cells[i]
                    target_cell = self.cells[j]
                    syn = self.h.Exp2Syn(0.5, sec=target_cell.dend[0])
                    self.nclist[0].syn.Section
                    #syn.e = connection["erev"]
                    #syn.e = -0.6
                    source_section = source_cell.soma[0]
                    nc = self.h.NetCon(source_section(0.5)._ref_v, syn, sec=source_section)
                    nc.weight[0]=0.005
                    #nc.weight[0] = connection["weight"]
                    nc.threshold = -20
                    nc.delay = 2.0
                    self.synlist.append(syn)
                    self.nclist.append(nc)


    #from neuromac.segment_distance import dist3D_segment_to_segment


    def setup_iclamp_step(self, target_cell, amp, delay, dur):
        self.stim = self.h.IClamp(target_cell.soma[0](0.5))
        self.stim.amp = amp
        self.stim.delay = delay
        self.stim.dur = dur

    def record_values(self):
        vec = { "v": [],
                "t": self.h.Vector() }

        for i, cell in enumerate(self.cells):
            vec["v"].append(self.h.Vector())
            vec["v"][i].record(cell.soma[0](0.5)._ref_v)
        vec["t"].record(self.h._ref_t)

        return vec
