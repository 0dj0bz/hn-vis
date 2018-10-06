
import sys
import argparse
import uuid
import json
from multiprocessing import Process
import matplotlib.pyplot as plt
import math
from random import random
from numpy import arange



vertices = (
    (1, -1, 1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1),
    (1, -1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, -1, -1)    
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
    (5,7)
    )

#-----------------------------------------------------------------------------

class Edge:

    def __init__(self, edgeId=None, parentGraph=None, srcNode=None, 
        dstNode=None, weight=1.0, isDirected=False, label=None, 
        properties={}):

        if (edgeId is None):
            self.edgeId = 'E_' + str(srcNode.nodeId) + '_' + str(dstNode.nodeId)
        else:
            self.edgeId = edgeId


        self.src = srcNode
        self.dst = dstNode

        self.label = label
        self.weight = weight
        self.isDirected = isDirected
        self.properties = properties
        # self.eId = parentGraph.addEdge(srcNode.nId, dstNode.nId, weight)

        return

    def __del__(self):

        return



#-----------------------------------------------------------------------------

class Node:

    def __init__(self, nodeId=None, parentGraph=None, label=None, 
        properties={}):

        self.coords = {'x' : 0, 'y' : 0, 'dx' : 0, 'dy' : 0}

        # self.nId = parentGraph.addNode()

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

        # self.graph = networkit.graph.Graph()

        if (graphId is None):
            self.graphId = 'G_' + str(uuid.uuid4())
        else:
            self.graphId = graphId

        # self.graph.setName(self.graphId)

        self.nodes = {}
        self.edges = {}

        return

    def __del__(self):

        self.nodes = None
        self.edges = None
        # self.graph = None

        return

    def addNode(self, nodeId=None, label=None, properties={}):

        n = Node(nodeId=nodeId, parentGraph=None, label=label, 
            properties=properties)

        nDict = {n.nodeId : {'node': n, 'properties': properties}}

        self.nodes.update(nDict)
        
        # if (self.graph.numberOfNodes() != len(self.nodes.keys())):
        #     print("graph / dict out of balance")
            
        return n

    def addEdge(self, src=None, dst=None, isDirected=False, weight=1.0, 
        label=None, properties={}):

        e = Edge(parentGraph=None, srcNode=src, dstNode=dst, 
            weight=weight, isDirected=isDirected, label=label,
            properties=properties)

        eDict = {e.edgeId : {'edge':e, 'properties':properties}}
        self.edges.update(eDict)

        return e    

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
            pos[v] = [self.nodes[v]['node'].coords['x'],self.nodes[v]['node'].coords['y']]
            
        # plt.close()
        # plt.ylim([-0.1,1.1])
        # plt.xlim([-0.1,1.1])
        # plt.axis('off')
        # nx.draw_networkx(G,pos=pos,node_size=10,width=0.1,with_labels=False)
        # plt.savefig("fig/{0}.png".format(i+1))

        return pos

if __name__ == "__main__":

    g = Graph()

    print(edges[0][0], edges[0][1])


    for i in range(0,11):
        g.addNode(str(i))

    print(g.nodes[str(0)])


    for j in range(0,11):
        g.addEdge(src=g.nodes[str(edges[j][0])]['node'], dst=g.nodes[str(edges[j][1])]['node'])

    fg = g.fruchterman_reingold()

    for e in edges:
        print('edge: ', e[0], '->', e[1], ': ', fg[str(e[0])], ', ', fg[str(e[1])])
