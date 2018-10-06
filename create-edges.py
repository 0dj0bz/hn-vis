#python 3.6

import sys
import argparse
import uuid
import networkit
import json
from multiprocessing import Process
import matplotlib.pyplot as plt
import math
from random import random
from numpy import arange
import pickle

import pygame
import math
import numpy
from pygame.locals import *

import OpenGL.GL as GL
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
from OpenGL.GLU import *
from OpenGL.GLUT import *


#-----------------------------------------------------------------------------
# Globals
#Graph
cSize = 0

#OpenGL
NODE_RADIUS = 5.0
CIRCLE_POINTS = 15

edge_array = None
node_array = None


#-----------------------------------------------------------------------------
# BEGIN Graph

class Edge:

    def __init__(self, edgeId=None, parentGraph=None, srcNode=None, 
        dstNode=None, weight=1.0, isDirected=False, label=None, 
        properties={}):

        if (edgeId is None):
            self.edgeId = 'E_' + str(srcNode.nodeId) + '_' + str(dstNode.nodeId)
        else:
            self.edgeId = edgeId

        #print('Adding Edge: ', self.edgeId, ' src.nID: ', srcNode.nID, ' dst.nID: ', dstNode.nID)
        
        self.src = srcNode
        self.dst = dstNode

        self.label = label
        self.weight = weight
        self.isDirected = isDirected
        self.properties = properties
        self.eId = parentGraph.addEdge(srcNode.nId, dstNode.nId, weight)

        return

    def __del__(self):

        return



#-----------------------------------------------------------------------------

class Node:

    def __init__(self, nodeId=None, parentGraph=None, label=None, 
        properties={}):

        self.coords = {'x' : 0, 'y' : 0, 'dx' : 0, 'dy' : 0}
        
        self.nId = parentGraph.addNode()

        if (nodeId is None):
            self.nodeId = 'N_' + str(uuid.uuid4())
        else:
            self.nodeId = nodeId

        self.label = label
        self.properties = properties

        return

    def __del__(self):

        return

#-----------------------------------------------------------------------------



#-----------------------------------------------------------------------------

class Graph:

    def __init__(self, graphId=None):

        self.graph = networkit.graph.Graph()

        if (graphId is None):
            self.graphId = 'G_' + str(uuid.uuid4())
        else:
            self.graphId = graphId

        self.graph.setName(self.graphId)

        self.nodes = {}
        self.edges = {}

        return

    def __del__(self):

        self.nodes = None
        self.edges = None
        self.graph = None

        return

    def addNode(self, nodeId=None, label=None, properties={}):

        n = Node(nodeId=nodeId, parentGraph=self.graph, label=label, 
            properties=properties)

        nDict = {n.nodeId : {'node': n, 'properties': properties}}

        self.nodes.update(nDict)
        
        if (self.graph.numberOfNodes() != len(self.nodes.keys())):
            print("graph / dict out of balance")
            
        return n

    def addEdge(self, src=None, dst=None, isDirected=False, weight=1.0, 
        label='', properties={}):

        e = Edge(parentGraph=self.graph, srcNode=src, dstNode=dst, 
            weight=weight, isDirected=isDirected, label=label,
            properties=properties)

        eDict = {e.edgeId : {'edge':e, 'properties':properties}}
        self.edges.update(eDict)

        return e    

    def copyNode(self, srcNodeId, dstGraph):
        try:
            n = Node(nodeId=srcNodeId, parentGraph=dstGraph.graph, 
                label=self.nodes[srcNodeId]['node'].label, 
                properties=self.nodes[srcNodeId]['properties'])
    
            nDict = {n.nodeId : {'node': n, 'properties': n.properties}}
    
            dstGraph.nodes.update(nDict)
            
            if (dstGraph.graph.numberOfNodes() != len(dstGraph.nodes.keys())):
                print("graph / dict out of balance")

        except:
            n = None
            
        finally:            
            return n


    # attractive force
    def f_a(self, d,k):
        return d*d/k

    # repulsive force
    def f_r(self, d,k):
        return k*k/d

    def fruchterman_reingold(self, iteration=50):
        W = 1
        L = 1
        area = W*L
        k = math.sqrt(area/len(self.nodes))  #nx.number_of_nodes(G))

        # initial position
        for v in iter(self.nodes):  #nx.nodes_iter(G):
            # G.node[v]['x'] = W*random()
            # G.node[v]['y'] = L*random()
            self.nodes[v]['node'].coords['x'] = W * random()
            self.nodes[v]['node'].coords['y'] = L *random()

        t = W/10
        dt = t/(iteration+1)

        print("area:{0}".format(area))
        print("k:{0}".format(k))
        print("t:{0}, dt:{1}".format(t,dt))

        for i in range(iteration):
            print("iter {0}".format(i))

            pos = {}
            # for v in G.nodes_iter():
            #    pos[v] = [G.node[v]['x'],G.node[v]['y']]
            for v in iter(self.nodes):
                pos[v] = [self.nodes[v]['node'].coords['x'], self.nodes[v]['node'].coords['y']]

            # plt.close()
            # plt.ylim([-0.1,1.1])
            # plt.xlim([-0.1,1.1])
            # plt.axis('off')

            # TODO:  resume work here
            # nx.draw_networkx(G,pos=pos,node_size=10,width=0.1,with_labels=False)
            # plt.savefig("fig/{0}.png".format(i))

            # calculate repulsive forces
            for v in iter(self.nodes):
                self.nodes[v]['node'].coords['dx'] = 0
                self.nodes[v]['node'].coords['dy'] = 0
                for u in iter(self.nodes):
                    if v != u:
                        dx = self.nodes[v]['node'].coords['x'] - self.nodes[u]['node'].coords['x']
                        dy = self.nodes[v]['node'].coords['y'] - self.nodes[u]['node'].coords['y']
                        delta = math.sqrt(dx*dx+dy*dy)
                        if delta != 0:
                            d = self.f_r(delta,k)/delta
                            self.nodes[v]['node'].coords['dx'] += dx*d
                            self.nodes[v]['node'].coords['dy'] += dy*d

            # calculate attractive forces
            for e in iter(self.edges):
                v = self.edges[e]['edge'].dst
                u = self.edges[e]['edge'].src
                dx = v.coords['x'] - u.coords['x']
                dy = v.coords['y'] - u.coords['y']
                delta = math.sqrt(dx*dx+dy*dy)
                if delta != 0:
                    d = self.f_a(delta,k)/delta
                    ddx = dx*d
                    ddy = dy*d
                    v.coords['dx'] += -ddx
                    u.coords['dx'] += +ddx
                    v.coords['dy'] += -ddy
                    u.coords['dy'] += +ddy

            # limit the maximum displacement to the temperature t
            # and then prevent from being displace outside frame
            for v in iter(self.nodes):
                dx = self.nodes[v]['node'].coords['dx']
                dy = self.nodes[v]['node'].coords['dy']
                disp = math.sqrt(dx*dx+dy*dy)
                if disp != 0:
                    # cnt += 1
                    d = min(disp,t)/disp
                    x = self.nodes[v]['node'].coords['x'] + dx*d
                    y = self.nodes[v]['node'].coords['y'] + dy*d
                    x =  min(W,max(0,x)) - W/2
                    y =  min(L,max(0,y)) - L/2
                    self.nodes[v]['node'].coords['x'] = min(math.sqrt(W*W/4-y*y),max(-math.sqrt(W*W/4-y*y),x)) + W/2
                    self.nodes[v]['node'].coords['y'] = min(math.sqrt(L*L/4-x*x),max(-math.sqrt(L*L/4-x*x),y)) + L/2

            # cooling
            t -= dt

        pos = {}
        for v in iter(self.nodes):
            pos[v] = [self.nodes[v]['node'].coords['x'],self.nodes[v]['node'].coords['y'], 0., 1.2]
            
        # plt.close()
        # plt.ylim([-0.1,1.1])
        # plt.xlim([-0.1,1.1])
        # plt.axis('off')
        # nx.draw_networkx(G,pos=pos,node_size=10,width=0.1,with_labels=False)
        # plt.savefig("fig/{0}.png".format(i+1))

        return pos


#-----------------------------------------------------------------------------

class HnGraph:

    def __init__(self):

        self.graph = Graph()

        return

    def __del__(self):

        return


#-----------------------------------------------------------------------------

class Story(object):
    def __init__(self, graph=None):
        self.id_ = None
        self.type_ = None
        self.dead = False
        self.deleted = False
        self.parent = None
        self.title = None
        self.url = None
        self.by = None
        self.kids = None
        self.sourcePath = None
        self.parent = None
        self.node = None
        self.graph = graph

        return

    def makeFqnFromIdx(self, idx):
        # we need to calculate the subdirectory for a given index.
        # e.g., /0/         for 1-999999
        #       /1000000/   for 1000000-1999999
        #       etc.

        subdir = str(int(idx / 1000000)*1000000)

        filename = str(idx) + '.json' 
                
        fqn = self.sourcePath + '/' + subdir + '/' + filename

        return fqn

    def makeDict(self):
        d = {}

        id_         = {'id_' : self.id_}
        type_       = {'type': self.type_}
        by          = {'by'  : self.by }
        kids        = {'kids': self.kids}


        d.update(id_)
        d.update(type_)
        d.update(by)
        d.update(kids)

        return d

    def processStory(self, fn, isReply=False):

        try:
            f=open(fn, 'r')
            data = json.load(f)
            # print(data['type'])
            if (data['type'] != 'story' and data['type'] != 'comment'):
                print("type: not a story or comment")
                self.id_ = None
                return None
            else:
                #print("type: is a story or comment")
                self.type_ = data['type']

                if (self.type_ == 'comment' and isReply == False):
                    self.id_ = None
                    return

                if (data['type'] == 'story'):
                    print("type is story")
                    if 'url' in data:
                        self.url = data['url']
                    if 'title' in data:
                        self.title = data['title']
                else:
                    #self.parent = data['parent']
                    print("----------type is comment")
                    print("------------refers to " + str(data['parent']))

                self.id_ = data['id']

                if 'by' in data:
                    self.by = data['by']

                if 'deleted' in data:
                    self.deleted = data['deleted']
                else:
                    self.deleted = False

                if 'dead' in data:
                    self.dead = data['dead']
                else:
                    self.dead = False

                if 'kids' in data:
                    self.kids = data['kids']
                    # cSize = cSize + len(self.kids)

                else:
                    self.kids = None

                n = self.graph.addNode(nodeId=self.id_, 
                    properties=self.makeDict())

                self.node = n


        except:
            print("Error processing story.")
            self.id_ = None

        return

    def processReplies(self, parentNode, level = 0):

        print((level+1) * "--" + "----Processing replies for story: " 
            + str(self.id_) + "...")

        print("parentNode:" + str(parentNode))
        print("parentNode.nodeId", parentNode.nodeId)
        print("parentNode.nId", parentNode.nId)

        for reply in range(0, len(self.kids)):
            print((level+1) * "--" + "------" + "Processing reply: ", 
                self.kids[reply])

            fn = self.makeFqnFromIdx(self.kids[reply])

            s = Story(self.graph)
            s.sourcePath = self.sourcePath
            s.processStory(fn, isReply=True)

            print('src', s.node.nId, 'dst', parentNode.nId)
            self.graph.addEdge(s.node, parentNode)

            #check to see if it has replies; if so, recurse
            if s.kids is not None:
                ritem = s.processReplies(s.node, level + 1)

        print((level+1) * "--" + "----Replies complete")

        return



# END Graph
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# BEGIN OpenGL
        
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

# END OpenGL
#-----------------------------------------------------------------------------


def drawGL_main():
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
        arry += fg[i]

    node_array = numpy.array(arry, 'f')

    print('main: node_array', node_array)

    n1 = numpy.array(node_array, 'f')
    print(n1)


    d = OpenGLDraw()


    arry = []
    # create edges array
    for e in edges:
#        arry += fg[revNodeIdx2[e[0]]]
#        arry += fg[revNodeIdx2[e[1]]]
        arry += fg[revNodeIdx2[e[0]]]
        arry += fg[revNodeIdx2[e[1]]]
    edge_array = numpy.array(arry, 'f')

    print('main: edge_array', edge_array)

    d.createObject(n1)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                pygame.quit()
                return


        # GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT)
        d.displayObject()

        pygame.display.flip()

if __name__ == "__main__":

#     sourceDir = ""
#     graphFn = ""
#     startRec  = 0
#     cSize = 0

    g = Graph()

#     def calcSize(storyId):
#         s = g.nodes[storyId]
#         cSize = 1
#         if s['properties']['kids'] is not None:
#             cSize += len(s['properties']['kids'])
#             for k in s['properties']['kids']:
#                 cSize += calcSize(k)
#         return cSize 


    # first, load the existing graph into memory

    # g.G = networkit.graphio.GraphMLReader.read('g_complete.graphml', networkit.Format.GraphML)

    # try:
    #   fp = file('graph2.json', 'r')
    #   data1 = json.load(fp)
    #   print("data loaded from file")
    #   g.G = json_graph.node_link_graph(data1)
    #   print("graph populated")
    #   fp.close()

    # except:
    #   print("error loading graph from file")
    #   pass

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--sourceDir", type=str,
    #     help="Dirctory from which to read data")
    # parser.add_argument("--startIndex", type=str,
    #     help="Initial index to process")
    # parser.add_argument("--endIndex", type=str,
    #     help="Final index to process")
    # parser.add_argument("--graphFile", type=str,
    #     help="File in which to store this graph.")

    # args = parser.parse_args()

    # if args.sourceDir is None:
    #     print("Error: --sourceDir is a required parameter.")
    #     sys.exit()
    # else:
    #     sourceDir = args.sourceDir

    #     if args.startIndex is None:
    #         print("Error: --startIndex is a required parameter.")
    #         sys.exit()
    #     else:
    #         startIndex = int(args.startIndex)

        # if args.endIndex is None:
        #     print("Error: --endIndex is a required parameter.")
        #     sys.exit()
        # else:
        #     endIndex = int(args.endIndex)

        # if args.graphFile is None:
        #     print("Error: --graphFile is a required parameter.")
        #     sys.exit()
        # else:
        #     graphFn = args.graphFile

        # s = Story(g)

        # index = startIndex
        # # for index in range(startIndex, endIndex):

        # s.sourcePath = sourceDir

        # fqn = s.makeFqnFromIdx(index)

        # cSize = 1
        # print("Processing file: " + fqn)
        # s.processStory(fqn)

        # # TODO:  This block of logic should happen inside of the 
        # # Story class.  Just sayin.
        # if (s.id_ is not None):
        #     # this means it is a valid story, so add a node to the graph

        #     # n = s.graph.addNode(nodeId=s.id_, 
        #     #   properties=s.makeDict())

        #     # s.node = n


        #     if (s.kids is not None):

        #         # t = Process(target=s.processReplies, 
        #         #   kwargs=({'parentNode':s.node}))
        #         # t.start()
        #         # t.join()

        #         cSize += len(s.kids)

        #         s.processReplies(parentNode=s.node)

        # else:
        #     print("Skipped file " + fqn + ".")


        # print(list(g.G.nodes(data=True)))

        # fp = file(graphFn, 'w')
        # data1 = json_graph.node_link_data(g.G)
        # json.dump(data1, fp)
        # fp.close()

# create nkit graph node id to nodes dict element
# idx = 1
# revNodeIdx = {}
# maxIdx = len(g.nodes.keys())
# while (idx <= maxIdx):
#     try:
#         nIdx = { g.nodes[idx]['node'].nId : idx}
#         revNodeIdx.update(nIdx)
#     except:
#         pass
#     idx = idx + 1

# pickle revNodeIdx object
# fp = open('revNodeIdx', 'wb')
# pickle.dump(revNodeIdx, fp)

# unpickle revNodeIdx object
    fp = open('revNodeIdx', 'rb')
    revNodeIdx = pickle.load(fp)
    
    # unpickle g.nodes
    fp = open('graph', 'rb')
    g.nodes = pickle.load(fp)
    
    # load graph
    g.graph = networkit.graphio.readGraph('../final.graphml', networkit.Format.GraphML)
    
    g2 = None
    g2 = Graph()




def processBFSCallback(src, dst, weight, isDirected):
    try:
        print('src: ', revNodeIdx[src], ' dst: ', revNodeIdx[dst], ' weight: ', weight, ' isDirected: ', isDirected)
        n = g.copyNode(     g.nodes[revNodeIdx[dst]]['node'].nodeId, g2)
        
        nIdx = { n.nId : n.nodeId}
        revNodeIdx2.update(nIdx)
        
        g2.addEdge(src=g2.nodes[revNodeIdx[src]]['node'], dst=g2.nodes[revNodeIdx[dst]]['node'])

    except:
        print('error with src: ', src, ' dst: ', dst)
        pass
    
def printem(src, dst, weight, isDirected):
    try:
        print('src: ', revNodeIdx[src], ' dst: ', revNodeIdx[dst], ' weight: ', weight, ' isDirected: ', isDirected)
    except:
        pass
    
"""

gOut = {}
eOut = {}
revNodeIdx2 = {}
g2 = None
g2 = Graph()
top15 = [910, 5434116, 5597765, 5925561, 5266615, 845836, 12180, 4560683, 4288424, 5761191, 2585488, 3620943, 3878810, 4820858, 4017987]
#n = g.copyNode(363, g2)
 
for idx in top15:
    print('processing node ' + str(idx) + '...')
#    revNodeIdx2 = {}

    try:
        n = g.copyNode(revNodeIdx[idx], g2)
        if n is not None:
            
            nIdx = { n.nId : n.nodeId}
            revNodeIdx2.update(nIdx)
            
            g.graph.BFSEdgesFrom(idx, processBFSCallback)
            
    except:
        print('error')
        pass
    
#g.graph.BFSEdgesFrom(910, processBFSCallback)
edges = g2.graph.edges()
fg = g2.fruchterman_reingold(iteration=100)
drawGL_main()

"""    