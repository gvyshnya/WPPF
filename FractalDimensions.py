#!/usr/bin/python
import numpy as np
import pandas as pd
import math

class FractalDimensions(object):

    def __init__(self, actualBins, alpha):
            self.actualBins = actualBins 	# input from discretization
            self.alpha = alpha 			# default alpha = 0.08
            self.origin = np.zeros((2,1), dtype=float)
            self.eps = 1e-3

#   Usage:
#     origin = np.zeros((2,1), dtype=float)
#     c_p = pointsk(origin, 0)
#     c_x = c_p['xcoord'].values
#     c_y = c_p['ycoord'].values
#     for k in range(1,level):
#        c_p = pointsmk(c_p, k)
#        c_x = c_p['xcoord'].values
#        c_y = c_p['ycoord'].values
#     rk = radius(k)
#     p = datamap(trace, origin)
#     x_p = p['xd'].values
#     y_p = p['yd'].values
#     hcounts = count_vec(x_p, y_p, c_x, c_y, rk)
#     Dinf = Dinf(hcounts, rk)
#     Dcor = Dcor(x_p, y_p, eps)

    def getActualBins(self):
        return self.actualBins

    def getAlpha(self):
        return self.alpha

    def getEps(self):
        return self.eps

    def radius(self, k): # Update 170710
        debug = 1
        rk = 1/(1-self.alpha)
        if (k == 0): return rk
        else:
            for i in range(1,k+1):
                rk = rk*self.alpha
        return rk

    # Creates mapping of data onto clusters
    def datamap(self, data, z0): # not used
        debug = 0
        xpts = []
        ypts = []
        z_d = z0
        dlength = len(data)
        if ( debug == 1 ): print ("[FractalDimensions:datamap] dlength = ", dlength)
        for i in range(dlength):
            z_d = self.w(data[i], z_d)
            xpts.append(z_d[0])
            ypts.append(z_d[1])
            if ( debug == 1 ): 
                print ("[FractalDimensions:datamap] data[", i, "] = ", data[i], ", xcoord = ", float(z_d[0]), ", ycoord = ", float(z_d[1]))
        coord = {'xd': xpts,
                 'yd': ypts}
        df = pd.DataFrame(coord, columns = ['xd', 'yd'])
        return df

    # Circular transformation on level k >= 1
    def wk(self, i, x, k): # not used
        debug = 0
        if ( debug == 1 ): print ("[FractalDimensions:wk] x: ", x)
        A = np.matrix( ((self.alpha,0), (0,self.alpha)) )
        B = np.zeros((2,1), dtype=float)
        B[0][0] = math.cos(i*2*math.pi/self.actualBins)
        B[1][0] = math.sin(i*2*math.pi/self.actualBins)
        Ak = A 
        for j in range(1,k):
            if ( debug == 1 ): print ("[FractalDimensions:wk] j = ", j)
            B = A*B
            Ak = Ak*A
        z_i = Ak*x+B
        return z_i

    # Find initial data coordinates (level 1)
    def pointsk(self, z0, k): # not used
        debug = 0
        xpts = []
        ypts = []
        origin = np.zeros((2,1), dtype=float)
        for i in range(self.actualBins):
            p = self.wk(i, origin, k)
            xpts.append(p[0]+z0[0])
            ypts.append(p[1]+z0[1])
        if ( debug == 1 ):
            print ("[FractalDimensions:pointsk] OUTPUT xpts: ", xpts)
            print ("[FractalDimensions:pointsk] OUTPUT ypts: ", ypts)
        coord = {'xcoord': xpts,
                 'ycoord': ypts}
        df = pd.DataFrame(coord, columns = ['xcoord', 'ycoord'])
        return df

    # Find data coordinates (level > 1)
    def pointsmk(self, z, k): # not used
        debug = 0
        xpts = []
        ypts = []
        origin = np.zeros((2,1), dtype=float)
        if ( debug ==  1 ): print ("[FractalDimensions:poinstmk] len(z) = ", len(z))
        for s in range(len(z)):
            zi = np.zeros((2,1), dtype=float)
            zi[0] = z['xcoord'].values[s]
            zi[1] = z['ycoord'].values[s]
            for i in range(self.actualBins):
                p = self.wk(i, origin, k+1)
                xpts.append(p[0]+zi[0])
                ypts.append(p[1]+zi[1])
        coord = {'xcoord': xpts,
                 'ycoord': ypts}
        df = pd.DataFrame(coord, columns = ['xcoord', 'ycoord'])
        return df

    # Correlation statistic
    def Dcor(self, x, y, eps):  
        debug = 0
        N = len(x)
        res = 0
        if (not len(y) == N): return res # UPDATE 170712
        C = 0
        for i in range(N):
            for j in range(N):
                if ( math.sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j])) < eps ): C = C+1
        res = 0
        if ( float(C)/(N*N) > 0 and eps > 0 ):
            res = math.log(float(C)/(N*N))/math.log(eps)
        if ( debug == 1 ):
            print ("[FractalDimensions:Dcor] eps = ", eps, ", C = ", C, ", res = ", res)
        return res

    # Cross-correlation statistic
    def Dxcor(self, x1, y1, x2, y2, eps): # not used 
        debug = 0
        N1 = len(x1)
        N2 = len(x2)
        C = 0
        for i in range(N1):
            for j in range(N2):
                if ( math.sqrt((x1[i]-x2[j])*(x1[i]-x2[j])+(y1[i]-y2[j])*(y1[i]-y2[j])) < eps ): C = C+1
        res = 0
        if ( float(C)/(N1*N2) > 0 and eps > 0 ):
            res = math.log(float(C)/(N1*N2))/math.log(eps)
        if ( debug == 1 ):
            print ("[FractalDimensions:Dxcor] eps = ", eps, ", C = ", C, ", res = ", res)
        return res

    # Auto-correlation statistic
    def Dhcor(self, x, y, h, eps): # not used
        debug = 0
        N = len(x)
        C = 0
        for i in range(N-h):
            for j in range(h,N):
                if ( math.sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j])) < eps ): C = C+1
        res = 0
        if ( float(C)/((N-h)*(N-h)) > 0 and eps > 0 ):
            res = math.log(float(C)/((N-h)*(N-h)))/math.log(eps)
        if ( debug == 1 ):
            print ("[FractalDimensions:Dhcor] eps = ", eps, ", C = ", C, ", res = ", res)
        return res

    # Count cluster densities
    def count_vec(self, x, y, x0, y0, r): # not used
        debug = 0
        m = len(x0)
        dlength = len(x)
        if ( debug == 1 ):
            print ("[FractalDimensions:count_vec] len(x) = ", dlength, ", len(x0) = ", m, ", r = ", r )
            print ("[FractalDimensions:count_vec] x:  ", min(x), max(x))
            print ("[FractalDimensions:count_vec] y:  ", min(y), max(y))
            print ("[FractalDimensions:count_vec] x0: ", min(x0), max(x0))
            print ("[FractalDimensions:count_vec] y0: ", min(y0), max(y0))
        nvec = []
        idxmax = 0
        for i in range(m):
            n = 0
            for j in range(dlength):
                ddist = math.sqrt((x[j]-x0[i])*(x[j]-x0[i])+(y[j]-y0[i])*(y[j]-y0[i]))
                if ( ddist <= r ): 
                    n = n+1
            nvec.append(n)
        nmax = 0
        if ( debug == 2 ):
            for i in range(len(nvec)):
                if ( nmax < nvec[i] ):
                    nmax = nvec[i]
                    idxmax = i
            print ("[FractalDimensions:count_vec] [<<<>>>] nvec: ", nvec)
            print ("[FractalDimensions:count_vec] idxmax = ", idxmax)
        return nvec

    # Information statistic
    def Dinf(self, counts, r): 
        m = len(counts)
        n = sum(counts)
        psum = 0
        for i in range(m):
            if ( counts[i] > 0 ):
                psum = psum+n/counts[i]
        Hmm = 0
        if ( n > 0 ): Hmm = self.Hempirical(counts)+(m-1)/(2*n)-(1-psum)/(12*n*n)
        return -Hmm/math.log(r)

    # Upper limit of variance of the information statistic
    def Dvar(self, counts, r):
        N = sum(counts)
        if ( N < self.eps ): return 0
        fac = math.log(N)
        return fac*fac/(N*math.log(r)*math.log(r))

    # Variance of entropy
    def Hvar(self, counts):
        m = len(counts)
        n = sum(counts)
        p = []
        for i in range(m):
            if ( n > 0 ):
                p.append(float(counts[i])/n)
            else:
                p.append(0)
        res = 0
        for i in range(m):
            if ( p[i] > 0 ):
                res = res+p[i]*math.log(p[i])*math.log(p[i])
        return res

    # Box counting statistic
    def Dbc(self, counts, r):
        N = 0
        for i in range(len(counts)):
            if ( counts[i] > 0 ): N += 1
        if ( N < self.eps or r <= 0 ): return 0
        return -math.log(N)/math.log(r)

    # Empirical entropy
    def Hempirical(self, counts):
        m = len(counts)
        n = sum(counts)
        p = []
        for i in range(m):
            if ( n > 0 ):
                p.append(float(counts[i])/n)
            else:
                p.append(0)
        hsum = 0
        for i in range(m):
            if ( counts[i] > 0 ):
                hsum = hsum+counts[i]*math.log(counts[i])
        if ( n > 0 ): res2 = math.log(n)-float(hsum)/n
        else: res2 = 0
        return res2
