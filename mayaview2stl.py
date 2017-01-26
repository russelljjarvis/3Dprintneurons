#source code from:
#http://stackoverflow.com/questions/25934891/export-a-mayavi-surface-as-a-stl
from mayavi import mlab
import vtk

# numpy array of points which define the vertices of the surface
points = np.array([[0,0,0],[1,0,0],[1,1,0]]) 
# numpy array defining the triangle which connects those points
element = np.array([[0,1,2]]) 

# mlab surface defined with the points and element
surface = mlab.pipeline.triangular_mesh_source(points[:,0], points[:,1], points[:,2], element) 

# For readability, define a variable as the _vtk_obj from the surface.
# This will be the surface data in the .stl file.
surface_vtk = surface.outputs[0]._vtk_obj

stlWriter = vtk.vtkSTLWriter()
# Set the file name
stlWriter.SetFileName('test_surface.stl')
# Set the input for the stl writer. surface.output[0]._vtk_obj is a polydata object
stlWriter.SetInput(surface_vtk)
# Write the stl file
stlWriter.Write()

# View the .stl surface that was just written-----------------------------------------------
from mayavi.core.api import Engine

engine = Engine()
# Create a new figure and add that figure to the engine
fig = mlab.figure(engine = engine)
# Open the stl file
surface_data = engine.open('test_surface.stl')
# Add the opened surface to the pipeline
opened_surface = mlab.pipeline.surface(surface_data)
# Add a module to show the opened surface
mlab.pipeline.surface(opened_surface, figure = fig)
# Show the scene
mlab.show()
