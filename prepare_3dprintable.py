"""
prepare_3dprintable.py
Robert A McDougal

Invokes ctng on a selected morphology after scaling it in a way that is suitable
for 3D printing.

help(prepare_3dprintable.ctng) for more.
"""

def ctng(secs=None, dx=0.5, cable_diam=3, somascale=3, special_all_diam={},
         magnification=200, show=True, color=(1, 0, 0),file_name=str('')):
    """
    ctng: prepare NEURON morphology for 3D printing

    Parameters (all are optional):

        secs = which sections to include (default: all)
        dx = mesh discretization after scaling before magnification in um (default: 0.5 um)
            decrease dx if neuron appears disconnected
            increase dx to reduce the number of triangles
        cable_diam = diameter to set non-soma sections to (default: 1 um)
        somascale = scale factor for soma (default: 1)
            if dendrites are magnified, they may hide the soma if it is not
            also scaled
        special_all_diam = exceptions for cable_diam, a dictionary keyed by
            section names whose values are the diameters for those sections
        magnification = magnification factor (default: 200)
        show = show the surface when done (default: True)
        color = what color to show the surface in (default: red)

    Returns:
        mesh, tri_mesh where:
        mesh is a mlab.triangular_mesh
        tri_mesh is a geometry3d.triangularMesh.TriangularMesh

    Note:
        surface area is tri_mesh.area
        enclosed volume is tri_mesh.enclosed_volume
        number of triangles is (len(tri_mesh.data) / 9.)
        The data in the WRL is in mm.
    """

    from neuron import h
    if secs is None:
        secs = h.allsec()

    import sys

    nouniform = False

    print('phase 1')

    from mayavi import mlab
    #mlab.options.offscreen = True

    import geometry3d
    import time
    import numpy

    xs, ys, zs = [], [], []

    for sec in secs:
        if 'soma' not in sec.name():
            for i in range(int(h.n3d(sec=sec))):
                d = h.diam3d(i, sec=sec)
                xs.append(h.x3d(i, sec=sec))
                ys.append(h.y3d(i, sec=sec))
                zs.append(h.z3d(i, sec=sec))

                if sec.name() not in special_all_diam:
                    h.pt3dchange(i, d*5+cable_diam, sec=sec)
                else:
                    # change this if min diam instead
                    h.pt3dchange(i, special_all_diam[sec.name()], sec=sec)
        elif 'soma' in sec.name():
            x, y, z, diam = [], [], [], []
            for i in range(int(h.n3d(sec=sec))):
                x.append(h.x3d(i, sec=sec))
                y.append(h.y3d(i, sec=sec))
                z.append(h.z3d(i, sec=sec))
                d = h.diam3d(i, sec=sec)
                diam.append(h.diam3d(i, sec=sec))
            h.pt3dclear(sec=sec)
            x, y, z, diam = d*5 + somascale * numpy.array(x), somascale * numpy.array(y), somascale * numpy.array(z), somascale * numpy.array(diam)
            i = int(len(x) / 2)
            midptx, midpty, midptz = x[i], y[i], z[i]
            x -= midptx / 2.
            y -= midpty / 2.
            z -= midptz / 2.
            for xpt, ypt, zpt, diampt in zip(x, y, z, diam):
                h.pt3dadd(xpt, ypt, zpt, diampt, sec=sec)


    print('bounding box: [%g, %g] x [%g, %g] x [%g, %g]' % (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))

    print('phase 2')

    h.load_file('stdlib.hoc')
    print('phase 3')
    start = time.time()
    tri_mesh = geometry3d.surface(secs, dx, n_soma_step=100, nouniform=nouniform)
    print('phase 4')
    magnification /= 1000.

    if color is not None:
        mesh = mlab.triangular_mesh(tri_mesh.x * magnification, tri_mesh.y * magnification, tri_mesh.z * magnification, tri_mesh.faces, color=color)
    else:
        mesh = mlab.triangular_mesh(tri_mesh.x * magnification, tri_mesh.y * magnification, tri_mesh.z * magnification, tri_mesh.faces, representation='wireframe', opacity=0)
    print('time to construct mesh:', time.time() - start)

    start = time.time()
    area = tri_mesh.area
    print('area: ', area)
    print('time to compute area:', time.time() - start)

    start = time.time()
    vol = tri_mesh.enclosed_volume
    print('volume: ', vol)
    print('time to compute volume:', time.time() - start)

    print('number of triangles: %g' % (len(tri_mesh.data) / 9.))
    #mlab.options.offscreen = True

    #if show:
    #   mlab.show()


    mlab.savefig(file_name)

    return mesh, tri_mesh
