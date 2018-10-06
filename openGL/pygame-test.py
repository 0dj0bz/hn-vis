import pygame
import math
import numpy
from pygame.locals import *

import OpenGL.GL as GL
# from OpenGL.GL import *
from OpenGL.arrays import vbo

from OpenGL.GL import shaders

from OpenGL.GLU import *
from OpenGL.GLUT import *


DISPLAYSURF = None
BLUE  = (  0,   0, 255)

# vertices = (
#     (1, -1, 1),
#     (1, 1, -1),
#     (-1, 1, -1),
#     (-1, -1, -1),
#     (1, -1, 1),
#     (1, 1, 1),
#     (-1, -1, 1),
#     (-1, 1, 1),
#     (1, -1, -1),
#     (-1, -1, -1),
#     (1, -1, 1),
#     (1, -1, -1)    
#     )

vertices = [ 0. ,  0. , 0., 1.,
             0.6,  0.6, 0., 1.,    
            -0.6,  0.6, 0., 1.,
            -0.6,  0.6, 0., 1.,
            -0.6,  0.6, 0., 1.,
            -0.6, -0.6, 0., 1. ]


fg_verts = [
    0.5, 1.6532864348661747e-06, 0., 2., 
    0.13095797576723478, 0.166743295493802, 0., 2., 
    0.3493377365344924, 0.6175118341321838, 0., 2., 
    0.5500652783610165, 0.3611895373439849, 0., 2., 
    0.7477586475062714, 0.06835036177218468, 0., 2., 
    0.36415028075504324, 0.02886529979539565, 0., 2., 
    0.8311180770115469, 0.4795188445618137, 0., 2., 
    0.7682453403975932, 0.8753861684379545, 0., 2., 
    0.5, 0.999999236436107, 0., 2., 
    0.9999997998063339, 0.5, 0., 2., 
    1.1359673577349128e-06, 0.5, 0., 2. ]

fg = {
    '0': [0.9481861022281417, 0.28324669394264734, 0., 2.], 
    '1': [0.702171117848851, 0.45784324387835584, 0., 2.], 
    '2': [0.91306817503928, 0.7753899856321336, 0., 2.], 
    '3': [0.9999920281444343, 0.5, 0., 2.], 
    '4': [0.6019830729716555, 0.6374772421943264, 0., 2.], 
    '5': [0.29251043365742907, 0.6423783460576596, 0., 2.], 
    '6': [0.7552703260067156, 0.9298552776222673, 0., 2.], 
    '7': [0.5, 0.9999980782402845, 0., 2.], 
    '8': [1.3493470302217148e-07, 0.5, 0., 2.], 
    '9': [0.4544289237644432, 0.0024650025925041508, 0., 2.], 
    '10': [0.5, 1.1127675970223905e-05, 0., 2.]}

edge_array = None
node_array = None

edges0 = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )

NODE_RADIUS = 5.0

CIRCLE_POINTS = 15

verticies2d = (
    (1, -1),
    (1,  1),
    (-1, 1),
    (-1, -1)
    )

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7),
    (7,8),
    (1,9),
    (3,10)
    )

edges2 = (
    (0,1),
    (0,3),
    (2,1),
    (2,3),
    )

surfaces = (
    (0,1,2,3),
    (3,2,7,6),
    (6,7,5,4),
    (4,5,1,0),
    (1,5,7,2),
    (4,0,3,6)
    )

colors = (
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (0,1,0),
    (1,1,1),
    (0,1,1),
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (1,0,0),
    (1,1,1),
    (0,1,1),
    )


class OpenGLDraw:
    """Creates a simple vertex shader..."""

    global edge_array
    global node_array

    def __init__( self ):

        vertex_shader = """
        #version 450

        in vec4 position;
        void main()
        {
           gl_Position = position;
        }
        """

        fragment_shader = """
        #version 450

        void main()
        {
           gl_FragColor = vec4(0.0f, 0.8f, 1.0f, 1.0f);
        }
        """
        self.nodes_shader = GL.shaders.compileProgram(
            GL.shaders.compileShader(vertex_shader, GL.GL_VERTEX_SHADER),
            GL.shaders.compileShader(fragment_shader, GL.GL_FRAGMENT_SHADER)
            )
        self.vao = None    
        self.vaoLen = 0

        edge_vertex_shader = """
        #version 450

        in vec4 position;
        void main()
        {
           gl_Position = position;
        }
        """

        edge_fragment_shader = """
        #version 450

        void main()
        {
           gl_FragColor = vec4(0.3f, 0.4f, 0.0f, 0.5f);
        }
        """
        self.edges_shader = GL.shaders.compileProgram(
            GL.shaders.compileShader(edge_vertex_shader, GL.GL_VERTEX_SHADER),
            GL.shaders.compileShader(edge_fragment_shader, GL.GL_FRAGMENT_SHADER)
            )
        self.edge_vao = None    
        self.edge_vaoLen = 0


    def createObject(self, data):
        # ---Nodes
        # Create a new VAO (Vertex Array Object) and bind it
        vertex_array_object = GL.glGenVertexArrays(1)
        GL.glBindVertexArray( vertex_array_object )
        
        # Generate buffers to hold our vertices
        vertex_buffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vertex_buffer)
        
        # Get the position of the 'position' in parameter of our shader and bind it.
        position = GL.glGetAttribLocation(self.nodes_shader, 'position')
        GL.glEnableVertexAttribArray(position)
        
        # Describe the position data layout in the buffer
        GL.glVertexAttribPointer(position, 4, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))
        
        # Send the data over to the buffer
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(node_array)*sizeof(ctypes.c_float), node_array, GL.GL_STATIC_DRAW)
        self.vaoLen = len(node_array)

        # Unbind the VAO first (Important)
        GL.glBindVertexArray( 0 )
        
        # Unbind other stuff
        GL.glDisableVertexAttribArray(position)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        
        self.vao = vertex_array_object

        # ---Edges
        # Create a new VAO (Vertex Array Object) and bind it
        edge_vertex_array_object = GL.glGenVertexArrays(1)
        GL.glBindVertexArray( edge_vertex_array_object )
        
        # Generate buffers to hold our vertices
        vertex_buffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vertex_buffer)
        
        # Get the position of the 'position' in parameter of our shader and bind it.
        position = GL.glGetAttribLocation(self.edges_shader, 'position')
        GL.glEnableVertexAttribArray(position)
        
        # Describe the position data layout in the buffer
        GL.glVertexAttribPointer(position, 4, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))
        
        print(edge_array)

        # Send the data over to the buffer
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(edge_array)*sizeof(ctypes.c_float), edge_array, GL.GL_STATIC_DRAW)
        self.edge_vaoLen = len(edge_array)

        # Unbind the VAO first (Important)
        GL.glBindVertexArray( 0 )
        
        # Unbind other stuff
        GL.glDisableVertexAttribArray(position)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        
        self.edge_vao = edge_vertex_array_object        


        return vertex_array_object

    def displayObject(self):
        """Render the geometry for the scene."""
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glUseProgram(self.nodes_shader)

        GL.glBindVertexArray(self.vao)
        GL.glPointSize(NODE_RADIUS)
        GL.glDrawArrays(GL.GL_POINTS, 0, self.vaoLen)

        GL.glBindVertexArray(0)
        shaders.glUseProgram(0)

        GL.glUseProgram(self.edges_shader)

        GL.glBindVertexArray(self.edge_vao)
        GL.glPointSize(3.0)
        GL.glDrawArrays(GL.GL_LINES, 0, self.edge_vaoLen)

        GL.glBindVertexArray(0)
        shaders.glUseProgram(0)



def main():
    # TestContext.ContextMainLoop()

    global CIRCLE_POINTS
    global NODE_RADIUS
    global node_array
    global edge_array

    circleVerts = [[0,0]]

    # this creates a base vertex circle; this should be translatable to the center of each vertex
    for i in range(0,32):
        piDivPts = (math.pi/16)
        mx = 1
        my = 1
        circleVerts += [[math.cos(piDivPts*i*mx)*NODE_RADIUS, math.sin(piDivPts*i*my)*NODE_RADIUS]]
        circleVerts += [[math.cos(piDivPts*(i+1)*mx)*NODE_RADIUS, math.sin(piDivPts*(i+1)*my)*NODE_RADIUS]]   

    circleTemplate = numpy.array(circleVerts, 'f')   

    display = (1200,1200)
    pygame.init()
    screen = pygame.display.set_mode(display, pygame.OPENGL|pygame.DOUBLEBUF)
    GL.glClearColor(0.0, 0.0, 0.0, 1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)

    clock = pygame.time.Clock()

    arry = []
    # g1 = numpy.array(vertices, 'f')
    for i in fg:
        arry += fg[str(i)]

    node_array = numpy.array(arry, 'f')

    print('main: node_array', node_array)

    n1 = numpy.array(node_array, 'f')
    print(n1)


    d = OpenGLDraw()


    arry = []
    # create edges array
    for e in edges:
        arry += fg[str(e[0])]
        arry += fg[str(e[1])]

    edge_array = numpy.array(arry, 'f')

    print('main: edge_array', edge_array)

    d.createObject(n1)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                return


        # GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT)
        d.displayObject()

        pygame.display.flip()
        # pygame.time.wait(10)


    # idx = 0
    # for i in iter(g1):
    #     idx1 = 0

    #     vi = [ [i[0], 0], [0, i[1]] ]
    #     print('vi', vi)
    #     vbo = numpy.copy(circleTemplate)
    #     idx2 = 0
    #     for i2 in iter(circleTemplate):
    #         vbo[idx2] = i2.dot(vi)
    #         idx2 += 1

    #         # print(vbo)
    #         d.vbo = d.createVBO(vbo)

    #     idx1 += 1

    # print("vbo: ", d.vbo)
    # vbo = d.createVBO(g1)



if __name__ == '__main__':
    try:
        main()
    finally:
        pygame.quit()