#!/usr/bin/python
import numpy as np
import pandas as pd
import math
import FractalDimensions

class kNN(object):

    def __init__(self, data, key, weights, k):
        self.data = data # panda object of discretized data
        self.key = key
        self.weights = weights
        self.k = k
        self.knn_data = pd.DataFrame(columns=['kNNind', 'kNNval'])

    def Euclidean(self, a, b, useWeights):
        debug = 0
        n = len(a)
        if debug:
            print ("[kNN:Euclidean] a (", n, "): ", a)
            print ("[kNN:Euclidean] b (", len(b), "): ", b)
        d = 0
        for i in range(n-1):
            if debug: print("[kNN.Euclidean]: step in cycle: ", i)
            if ( useWeights ):
                d += self.weights[i]*(a[i]-b[i])**2
            else:
                d += (a[i]-b[i])**2
        if debug: print ("[kNN:Euclidean] d^2 = ", d)
        return math.sqrt(d)

    def kNearestNeighbor(self):
        debug = 0

        useWeights = 0
        current_row = 0
        total_rows = self.data.shape[0]

        kNNval = []
        kNNind = []

        while current_row < total_rows:
            vector = self.data.iloc[current_row]
            distance = self.Euclidean(self.key, vector, useWeights)
            if debug:
                print("[kNN.kNearestNeighbor]: current datapoint row: ", current_row)
                print("[kNN.kNearestNeighbor]: current datapoint vector: ", vector)
                print("[kNN.kNearestNeighbor]: currently calculated distance: ", distance)
            kNNval.append(distance)
            kNNind.append(current_row)
            current_row += 1

        # populate the internal result df with kNNval and kNNind - it will help to sort by min of kNNval and then select
        # k records as closest neighbours

        self.knn_data['kNNind'] = kNNind
        self.knn_data['kNNval'] = kNNval

        # sort the df by kNNval, ascending
        self.knn_data = self.knn_data.sort_values(['kNNval'], ascending=[1])

        if self.k <= 0:
            df_knn = self.knn_data[0:1]
        else:
            df_knn = self.knn_data[0:self.k]
        if debug:
            print("[kNN.kNearestNeighbor]: dataframe k neighbor subset:", df_knn)
        return df_knn['kNNind'].values

    def kNNinterpolation(self, useWeights): # new 170711
        debug = 0

        # TODO: return back m

        kNNval = []
        kNNind = []
        for i in range(self.k-1):
            kNNval.append(0)
            kNNind.append(0)
        for i in range(self.k-1):
            distance = self.Euclidean(self.key, self.data.iloc[i], useWeights)
            kNNval[i] = distance
            kNNind[i] = i
        for i in range(self.k,m):
            distance = self.Euclidean(self.key, self.data.iloc[i], useWeights)
            if ( max(kNNval) > distance ):
                ind = kNNval.index(max(kNNval))
                kNNval[ind] = distance
                kNNind[ind] = i
        # Sort after increasing distance
        if debug: 
            print ("[kNN:kNNinterpolation] kNNind: ", kNNind)
            print ("[kNN:kNNinterpolation] kNNval: ", kNNval)
        kNNind_sort = []
        kNNval_sort = []
        usedIndices = []
        index = kNNval.index(min(kNNval))
        kNNind_sort.append(kNNind[index])
        kNNval_sort.append(kNNval[index])
        usedIndices.append(index)
        for i in range(1,self.k):
            tmp = sum(kNNval)
            for j in range(self.k):
                if (tmp > kNNval[j] and not j in usedIndices):
                    tmp = kNNval[j]
            index = kNNval.index(tmp)
            kNNind_sort.append(kNNind[index])
            kNNval_sort.append(kNNval[index]) 
            usedIndices.append(index)
        if debug: 
            print ("[kNN:kNNinterpolation] kNNind_sort: ", kNNind_sort)
            print ("[kNN:kNNinterpolation] kNNval_sort: ", kNNval_sort)
        zipfile = zip(kNNind_sort,kNNval_sort)
        return zipfile

    @property
    def knn_distances_from_key(self):
        return self.df_knn