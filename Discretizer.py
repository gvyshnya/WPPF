#!/usr/bin/python
import numpy as np
import math


class Discretizer(object):

    def __init__(self, inputVector):
        self.N = len(inputVector)
        self.vertices = inputVector
        self.weights = np.zeros((self.N,self.N), dtype = float)
        self.adjacency = np.ones((self.N,self.N), dtype = int)
        mindist = 1e-1
        for i in range(self.N-1):
            for j in range(i+1,self.N):
                distance = self.Euclidean(inputVector[i],inputVector[j])
                if (distance == 0): distance = mindist
                self.weights[i][j] = distance
                self.weights[j][i] = distance
        # average edge weight of the complete graph
        self.avEdgeWt = self.averageEdgeWt(0,self.N-1)
        # complete graphs highest weight
        self.maxWt = self.weights.max()

    def Euclidean(self, a, b):
        return math.sqrt((a-b)*(a-b))

    def getNumVertices(self):
        return self.N

    def getVertices(self):
        return self.vertices

    def getNumEdges(self):
        return self.N*(self.N-1)/2

    def getDistance(self, i, j):
        return self.weights[i][j]

    def getWeights(self):
        return self.weights

    def getAdjacency(self):
        return self.adjacency

    def deg(self,v):
        _deg = 0
        for u in range(self.getNumVertices()):
            if (self.adjacency[v][u] > 0 and not u == v): _deg += 1
        return _deg

    def minDegree(self,i,j): # i and j inclusive
        _minDegree = self.getNumVertices()
        for u in range(i,j+1):
            if (_minDegree > self.deg(u)):
                _minDegree = self.deg(u)
        return _minDegree
            
    def averageEdgeWt(self,i,j): # i and j inclusive
        debug = 0
        avEdgeWt = 0
        for u in range(i,j):
            for v in range(u,j+1):
                avEdgeWt += self.getDistance(u,v)
        if debug: print (avEdgeWt) # update 170711
        avEdgeWt /= (j-i+1)
        return avEdgeWt

    def entropy(self, vec):
        n = len(vec)
        if (n <= 1): return 0
        H = 0
        for i in range(n):
            if (isinstance(vec[i], list)):
                wi = float(len(vec[i]))/sum(len(vec))
            else:
                wi = float(vec[i])/sum(vec)
            # print wi
            if (wi > 0):
                H += wi*math.log(1/wi,2)
        return H

    def removeEdge(self, i, j): # Update 170710
        debug = 0
        res = False
        if debug: print ("[Discretizer:removeEdge] remove (", i, ",", j, ")", (self.adjacency[i][j] > 0))
        if (self.adjacency[i][j] > 0): 
            res = True
        self.adjacency[i][j] = 0
        self.adjacency[j][i] = 0
        if debug: print (self.adjacency)
        return res

    def removeEdgeCluster(self, i, j): # New 170710
        debug = 0
        res = False
        if debug: print ("[Discretizer:removeEdgeCluster] remove (", i, ",", j, ")", (self.adjacency[i][j] > 0))
        if (self.adjacency[i][j] > 0): 
            res = True
        self.adjacency[i][j] = 0
        # self.adjacency[j][i] = 0
        if debug: print (self.adjacency)
        return res

    def removeHeaviest(self): # Update 170710
        debug = 0
        wt = np.zeros((self.N,self.N), dtype = float)
        for i in range(self.N):
            for j in range(self.N):
                wt[i][j] = self.weights[i][j]*self.adjacency[i][j]
        w = wt.max()
        if debug: print ("[Discretizer:removeHeaviest] w = ", w)
        ind = np.where(wt == w)
        if debug: print ("[Discretizer:removeHeaviest] ind: ", ind)
        if debug: print ("[Discretizer:removeHeaviest]: Entering r in range cycle")
        for r in range(len(ind)):
            rm = ind[r].tolist()
            if debug: print ("[Discretizer:removeHeaviest] r:", r)
            if debug: print("[Discretizer:removeHeaviest] rm:", rm)
            if debug: print("[Discretizer:removeHeaviest] rm[0]:", rm[0])
            if debug: print("[Discretizer:removeHeaviest] len(rm)/2:", len(rm)/2)
            if debug: print("[Discretizer:removeHeaviest] rm[len(rm)/2]:", rm[len(rm)/2])
            half_length = int(len(rm)/2)
            maxind = max(rm[0],rm[half_length])
            if debug: print ("[Discretizer:removeHeaviest] rm = ", rm, ", maxind = ", maxind)
            if (not rm[0] == rm[half_length]): self.removeEdge(rm[0],rm[half_length])
            # blockify
            for s in range(maxind+1,self.N):
                if debug: print ("[Discretizer:removeHeaviest] removeEdge(", maxind, ",", s, ")")
                self.removeEdge(maxind,s)        
            for s in range(maxind):
                for t in range(maxind+1,self.N):
                    if (not s == t):
                        if debug: print ("[Discretizer:removeHeaviest] removeEdge(", s, ",", t, ")")
                        self.removeEdge(s,t)
        return w

    def removeHeaviestCluster(self, cluster): # Update 170710
        debug = 0
        start = cluster[0]
        end = cluster[len(cluster)-1]
        wt = np.zeros((end-start+1,end-start+1), dtype = float)
        for u in range(start,end+1):
            for v in range(start,end+1):
                wt[u-start][v-start] = self.weights[u][v]*self.adjacency[u][v]
        w = wt.max()
        ind = np.where(wt == w)
        if debug:
            print ("[Discretizer:removeHeaviestCluster] wt:\n", wt)
            print ("[Discretizer:removeHeaviestCluster] w = ", w)
            print ("[Discretizer:removeHeaviestCluster] ind: ", ind)
        for r in range(len(ind)):
            rm = ind[r].tolist()
            half_length = int(len(rm) / 2)
            maxind = max(start+rm[0],start+rm[half_length])
            if debug: print ("[Discretizer:removeHeaviestCluster] rm: ",rm, ", maxind = ", maxind)
            self.removeEdgeCluster(start+rm[0],start+rm[half_length])
            # blockify
            # for c in range(start+rm[0]):
            #     if (not c == start+rm[len(rm)/2]):
            #         self.removeEdgeCluster(c,start+rm[len(rm)/2])
            # for s in range(maxind):
            #     for t in range(maxind+1,self.N):
            #         if (not s == t):
            #             if debug: print "[Discretizer:removeHeaviestCluster] removeEdge(", s, ",", t, ")"
            #             self.removeEdgeCluster(s,t)
        return w

    def isConnected(self): # Determines whether a graph is connected or not
        n = self.getNumVertices()
        Mark = []
        for v in range(n):
            Mark.append(0) # Initialize mark bits
        for v in range(n):
            if ( Mark[v] == 0 ):
                res = len(self.BFS(v)) # BFS or DFS
        if ( res < n ): return False
        return True

    def isConnectedCluster(self, cluster): # Determines whether a graph is connected or not
        start = cluster[0]
        end = cluster[len(cluster)-1]
        Mark = []
        res = 0 # update 170711
        for v in range(end-start+1):
            Mark.append(0) # Initialize mark bits
        for v in range(end-start+1):
            if ( Mark[v] == 0 ):
                res = len(self.BFScluster(v, cluster)) # BFS or DFS
        if ( res < end-start+1 ): return False
        return True

    def BFS(self, v):
        debug = 0
        cluster = []
        cluster.append(v)
        index = 0
        while (index < len(cluster)):
            if debug: print ("[Discretizer:BFS] BFS index = ", index, ", row = ", cluster[index])
            for j in range(self.N):
                if (self.adjacency[cluster[index]][j] > 0):
                    if (j not in cluster): cluster.append(j)
            index += 1
        if debug: print ("[Discretizer:BFS] cluster: ", cluster)
        return cluster

    def BFScluster(self, v, cluster):
        debug = 0
        start = cluster[0]
        end = cluster[len(cluster)-1]
        cluster = []
        cluster.append(v)
        index = 0
        while (index < len(cluster)):
            if debug: print ("[BFScluster] index = ", index, ", row = ", cluster[index])
            for u in range(end-start+1):
                if (self.adjacency[start+cluster[index]][start+u] > 0):
                    if (u not in cluster): cluster.append(u)
            index += 1
        return cluster

    def discretizedVector(self):
        debug = 0
        vector = []
        clusterStart = 0
        it = 0
        while (clusterStart < self.N and it < self.N-1):
            it += 1
            cluster = self.BFS(clusterStart)
            clusterStart = cluster[len(cluster)-1]+1
            if debug: 
                print ("[discretizedVector] cluster: ", cluster)
                print ("                    clusterStart = ", clusterStart)
            vector.append(cluster[len(cluster)-1]-cluster[0]+1)
        return vector

    def getClusters(self):
        debug = 0
        clusters = []
        clusterStart = 0
        it = 0
        while (clusterStart < self.N and it < self.N-1):
            it += 1
            cluster = self.BFS(clusterStart)
            clusterStart = cluster[len(cluster)-1]+1
            if debug: 
                print ("[getClusters] cluster: ", cluster)
                print ("              clusterStart = ", clusterStart)
                print ("              self.N = ", self.N)
            clusters.append(cluster)
        return clusters

    def disconnectFirst(self): # Update 170710
        debug = 0
        wt = np.zeros((self.N,self.N), dtype = float)
        for i in range(self.N):
            for j in range(self.N):
                wt[i][j] = self.weights[i][j]*self.adjacency[i][j]
        while (self.isConnected()):
            w = self.removeHeaviest()
            ind = np.where(wt == w)
            if debug: print ("[Discretizer:disconnectFirst] w = ", w, ", ind = ", ind)
        clusters = self.getClusters()
        return clusters

    def disconnectStep1(self, cluster): # disconnect further 1
        debug = 0
        if (len(cluster) == 0): return [] # update 170711
        aveEdgeWt = self.averageEdgeWt(cluster[0],cluster[len(cluster)-1])
        if (debug == 1):
            print ("[Discretizer:disconnectStep1] aveEdgeWt = ", aveEdgeWt )
            print ("                              compare: ", self.avEdgeWt/2)
        if (aveEdgeWt > self.avEdgeWt/2):
            w = self.removeHeaviestCluster(cluster)
            if debug: print ("[Discretizer:disconnectStep1] w = ", w)
        return self.getClusters()

    def disconnectStep2(self, cluster): # disconnect further 2
        debug = 0
        if (len(cluster) == 0): return [] # update 170711
        vertexSubset = self.vertices[cluster[0]:cluster[len(cluster)-1]+1]
        if (len(vertexSubset) == 0): return [] # update 170711
        ind0 = vertexSubset.argmin(axis=0)  # GV: modified 14072017
        ind1 = vertexSubset.argmax(axis=0)  # GV: modified 14072017
        if (debug == 1):
            print ("[disconnectStep2] vertexSubset: ", vertexSubset)
            print ("                  min vertex = ", min(vertexSubset), " at ", ind0)
            print ("                  max vertex = ", max(vertexSubset), " at ", ind1)
            print ("                  weight = ", self.weights[ind0][ind1])
            print ("                  connected? ", self.isConnectedCluster(cluster))
        if (self.weights[ind0][ind1] >= self.maxWt/2):
            # self.isConnectedCluster(cluster[0], cluster[len(cluster)-1])
            self.removeHeaviestCluster(cluster)
        return self.getClusters()

    def disconnectStep3(self, cluster): # disconnect further 3
        debug = 0
        if (len(cluster) == 0): return [] # update 170711
        minDeg = self.minDegree(cluster[0],cluster[len(cluster)-1])
        if debug:
            print ("[disconnectStep3] minDeg = ", minDeg )
            print ("                  cluster size - 1 = ", len(cluster)-1)
        if (self.minDegree(cluster[0],cluster[len(cluster)-1]) < len(cluster)-1):
            self.removeHeaviestCluster(cluster)
        return self.getClusters()

    def disconnectStep4(self, cluster): # disconnect further 4
        debug = 0
        if (len(cluster) == 0): return [] # update 170711
        vector = self.discretizedVector()
        entropy1 = self.entropy(vector)
        # if (len(cluster) >= self.N/2):
        size1 = int(math.ceil(len(cluster)/2))
        size2 = len(cluster)-size1
        if debug: 
            print ("[Discretizer:disconnectStep4] input cluster: ", cluster)
            print ("[Discretizer:disconnectStep4] vector: ", vector)
            print ("[Discretizer:disconnectStep4] entropy1 = ", entropy1)
            print ("[Discretizer:disconnectStep4] size1:", size1)
            print ("[Discretizer:disconnectStep4] size2:", size2)
        newVector = []
        vecPos = 0
        for i in range(len(vector)):
            if (vector[i] == len(cluster)):
                vecPos = i
                newVector.append(size1)
                newVector.append(size2)
            else:
                newVector.append(vector[i])
        entropy2 = self.entropy(newVector)
        if debug:
            print ("[Discretizer:disconnectStep4] newVector: ", newVector)
            print ("[Discretizer:disconnectStep4] entropy2 = ", entropy2)
        if (entropy2 > entropy1): # if entropy2 > entropy1: accept
            adjPos = 0
            for j in range(vecPos): adjPos += vector[j]
            for j in range(adjPos,adjPos+size1):
                for k in range(adjPos+size1,adjPos+size1+size2):
                    self.removeEdge(j,k)
            return newVector
        return vector

    def discretize(self):
        debug = 0
        if debug:
            # Tests:
            print ("[Discretizer:discretize] No Vertices = ", self.getNumVertices())
            print ("[Discretizer:discretize] Vertices: ", self.getVertices())
            print ("[Discretizer:discretize] Edges = ", self.getNumEdges())
            print ("[Discretizer:discretize] Weights = \n", self.getWeights())
            print ("[Discretizer:discretize] Adjacency = \n", self.getAdjacency())
            print ("[Discretizer:discretize] deg(3) = ", self.deg(3))
            print ("[Discretizer:discretize] isConnected: ", self.isConnected())
            print ("[Discretizer:discretize] minDegree(2,4) ", self.minDegree(2,4))
            print ("[Discretizer:discretize] averageEdgeWt(2,4) ", self.averageEdgeWt(2,4))
            print ("[Discretizer:discretize] averageEdgeWt(0,5) ", self.averageEdgeWt(0,5))
        # STEP 0
        clusters = self.disconnectFirst()
        vector = self.discretizedVector()
        if debug:
            print ("[Discretizer:discretize] STEP 0")
            print ("[Discretizer:discretize] clusters: ", clusters)
            print ("[Discretizer:discretize] getAdjacency:\n", self.getAdjacency())
            print ("[Discretizer:discretize] isConnected: ", self.isConnected())
            print ("[Discretizer:discretize] vector: ", vector)
            print ("[Discretizer:discretize] entropy: ", self.entropy(vector))

        for c in range(len(clusters)):
            for s in range(len(clusters[c])):
                newClusters = self.disconnectStep1(clusters[c])
                newVector = self.discretizedVector()
                if debug:
                    print ("[Discretizer:discretize] STEP 1")
                    print ("[Discretizer:discretize] newClusters: ", newClusters)
                    print ("[Discretizer:discretize] getAdjacency:\n", self.getAdjacency())
                    print ("[Discretizer:discretize] isConnectedCluster: ", self.isConnectedCluster(clusters[c]))
                    print ("[Discretizer:discretize] newVector: ", newVector)
                    print ("[Discretizer:discretize] entropy: ", self.entropy(newVector))
                if (not self.isConnectedCluster(clusters[c])): break
        clusters = newClusters

        for c in range(len(clusters)):
            for s in range(len(clusters[c])):
                newClusters = self.disconnectStep2(clusters[c])
                newVector = self.discretizedVector()
                if debug:
                    print ("[Discretizer:discretize] STEP 2")
                    print ("[Discretizer:discretize] newClusters: ", newClusters)
                    print ("[Discretizer:discretize] getAdjacency:\n", self.getAdjacency())
                    print ("[Discretizer:discretize] isConnectedCluster: ", self.isConnectedCluster(clusters[c]))
                    print ("[Discretizer:discretize] newVector: ", newVector)
                    print ("[Discretizer:discretize] entropy: ", self.entropy(newVector))
                if (not self.isConnectedCluster(clusters[c])): break
        clusters = newClusters

        for c in range(len(clusters)):
            for s in range(len(clusters[c])):
                newClusters = self.disconnectStep3(clusters[c])
                newVector = self.discretizedVector()
                if debug:
                    print ("[Discretizer:discretize] STEP 3")
                    print ("[Discretizer:discretize] newClusters: ", newClusters)
                    print ("[Discretizer:discretize] getAdjacency:\n", self.getAdjacency())
                    print ("[Discretizer:discretize] isConnectedCluster: ", self.isConnectedCluster(clusters[c]))
                    print ("[Discretizer:discretize] newVector: ", newVector)
                    print ("[Discretizer:discretize] entropy: ", self.entropy(newVector))
                if (not self.isConnectedCluster(clusters[c])): break
        clusters = newClusters

        for c in range(len(clusters)):
            for s in range(len(clusters[c])):
                newClusters = self.disconnectStep4(clusters[c])
                newVector = self.discretizedVector()
                if debug:
                    print ("[Discretizer:discretize] STEP 3")
                    print ("[Discretizer:discretize] newClusters: ", newClusters)
                    print ("[Discretizer:discretize] getAdjacency:\n", self.getAdjacency())
                    print ("[Discretizer:discretize] isConnectedCluster: ", self.isConnectedCluster(clusters[c]))
                    print ("[Discretizer:discretize] newVector: ", newVector)
                    print ("[Discretizer:discretize] entropy: ", self.entropy(newVector))
                if (not self.isConnectedCluster(clusters[c])): break
        clusters = newClusters
        vector = newVector
        return vector

    def discretizeWeather(self): # New 170711
        debug = 0
        quantity = sorted(self.vertices)
        cluster = []
        ind = 0
        if debug:
            print ("[Discretizer:discretizeWeather] ind = ", ind)
            print ("[Discretizer:discretizeWeather] quantity: ", quantity)
        while (ind < len(quantity)):
            val = quantity[ind]
            clusterSize = 0
            if debug:
                print ("[Discretizer:discretizeWeather] ind = ", ind)
                print ("[Discretizer:discretizeWeather] val = ", )
            while (quantity[ind] == val):
                ind += 1
                clusterSize += 1
                if (ind == len(quantity)): break
            cluster.append(clusterSize)
        return cluster

    def getBin(self, vector, val): # New 170711
        debug = 0
        quantity = sorted(self.vertices)
        ind = 0
        while (val > quantity[ind]):
            ind += 1
        if debug:
            print ("[Discretizer:getBin] quantity: ", quantity)
            print ("[Discretizer:getBin] ind = ", ind)
            print ("[Discretizer:getBin] ind = ", ind)
        vecSum = 0
        binIndex = 0 
        for i in range(len(vector)):
            vecSum += vector[i]
            if debug:
                print ("[Discretizer:getBin] i = ", i, ", vecSum = ", (vecSum-0.5), " (", ind, ")")
            binIndex = i
            if (ind < (vecSum - 0.5)): break
        return binIndex
