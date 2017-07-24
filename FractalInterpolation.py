#!/usr/bin/python
import numpy as np
import pandas as pd
import math


class FractalInterpolation(object):

    def __init__(self, xData, yData):
        self.n = len(xData)
        self.x = xData
        self.y = yData

    # Affine transformation
    def wi(self, i, d): # Update 170710
        debug = 0
        if debug: 
            print ("[FractalInterpolation:wi] x: ", self.x)
            print ("[FractalInterpolation:wi] y: ", self.y)
        denom0 = (self.x[self.n-1]-self.x[0])
        if ( denom0 == 0 ): denom0 = 1
        ai = (self.x[i]-self.x[i-1])/denom0
        ei = (self.x[self.n-1]*self.x[i-1]-self.x[0]*self.x[i])/denom0
        epsi = (self.x[self.n-1]-self.x[i])/denom0
        ci = (self.y[i]-self.y[i-1]-d*(self.y[self.n-1]-self.y[0]))/denom0
        fi = (self.x[self.n-1]*self.y[i-1]-self.x[0]*self.y[i]-d*(self.x[self.n-1]*self.y[0]-self.x[0]*self.y[self.n-1]))/denom0
        if debug:
            print ("[FractalInterpolation:wpq] ai = ", ai)
            print ("[FractalInterpolation:wpq] ei = ", ei)
            print ("[FractalInterpolation:wpq] d = ", d)
            print ("[FractalInterpolation:wpq] ci = ", ci)
            print ("[FractalInterpolation:wpq] fi = ", fi)
        A = np.matrix( ((ai,0), (ci, d)) )
        B = np.zeros((2,1), dtype=float)
        B[0][0] = ei
        B[1][0] = fi
        z = np.zeros((2,1), dtype=float)
        z[0][0] = self.x[i]
        z[1][0] = self.y[i]
        omega_i = A*z+B
        return omega_i

    def eps(self, i):
        denom0 = (self.x[self.n-1]-self.x[0])
        if ( denom0 == 0 ): denom0 = 1
        return (self.x[self.n-1]-self.x[i])/denom0

    def Ai(self, i):
        debug = 0
        Ai = self.y[i] - (self.eps(i)*self.y[0]+self.y[self.n-1]*(1-self.eps(i)))
        if debug: print ("[FractalInterpolation:Ai] Ai = ", Ai)
        return Ai

    def Bi(self, m, i, p, q):
        return self.y[m]-(self.eps(i)*self.y[p]+self.y[q]*(1-self.eps(i)))

    # Affine transformation
    def wpq(self, i, p, q): # Update 170710		
        debug = 1
        if debug: 
            print ("[FractalInterpolation:wpq] x: ", self.x)
            print ("[FractalInterpolation:wpq] y: ", self.y)
        denom0 = (self.x[self.n-1]-self.x[0])
        if ( denom0 == 0 ): denom0 = 1
        ai = (self.x[i]-self.x[i-1])/denom0
        ei = (self.x[self.n-1]*self.x[i-1]-self.x[0]*self.x[i])/denom0
        # epsi = epsi(i)
        # Ai = self.y[i] - (epsi*self.y[0]+self.y[self.n-1]*(1-epsi))
        m = int(ai*self.x[i]+ei)
        # Bi = y[m]-(epsi*y[p]+y[q]*(1-epsi))
        num = 0
        denom = 0
        for j in range(self.n):
            num += self.Ai(j)*self.Bi(m,j,p,q)
            denom += self.Ai(j)*self.Ai(j)
        di = num/denom
        ci = (self.y[q]-self.y[p]-di*(self.y[self.n-1]-self.y[0]))/denom0
        fi = (self.x[self.n-1]*self.y[p]-self.x[0]*self.y[q]-di*(self.x[self.n-1]*self.y[0]-self.x[0]*self.y[self.n-1]))/denom0
        if debug:
            print ("[FractalInterpolation:wpq] ai = ", ai)
            print ("[FractalInterpolation:wpq] ei = ", ei)
            print ("[FractalInterpolation:wpq] di = ", di)
            print ("[FractalInterpolation:wpq] ci = ", ci)
            print ("[FractalInterpolation:wpq] fi = ", fi)
        A = np.matrix( ((ai,0), (ci,di)) )
        B = np.zeros((2,1), dtype=float)
        B[0][0] = ei
        B[1][0] = fi
        z = np.zeros((2,1), dtype=float)
        z[0][0] = self.x[i]
        z[1][0] = self.y[i]
        omega_i = A*z+B
        return omega_i
