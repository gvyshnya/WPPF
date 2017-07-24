#!/usr/bin/python
import numpy as np
import math


class Preprocessor(object):

    def __init__(self, inputVector, Kt):
        self.n = len(inputVector)
        self.X = inputVector
        self.Kh = 1.1
        self.Kl = 0.9
        self.Kt = Kt
        self.alpha = 0.3

    def filter(self,q):
        filtered = []
        for t in range(self.n):
            if ( t <= q ):
                Z = 0
                for j in range(self.n-t):
                    # print (t+j)
                    Z += self.alpha*math.pow((1-self.alpha),j)*self.X[t+j]
                filtered.append(Z)
            elif ( t >= self.n-q ):
                Z = 0
                for j in range(t):
                    Z += self.alpha*math.pow((1-self.alpha),j)*self.X[t-j]
                filtered.append(Z)
            else:    
                Z = 0
                for j in range(-q,q+1):
                    # print "!", t, "-", j, ":", (t+j)
                    Z += self.X[t+j]
                filtered.append(Z/float(2*q+1))
        return filtered

    def smoother(self,Y,q):
        Ybar = self.filter(int(math.floor(self.n/2))) # ???
        smoothed = []
        for t in range(self.n):
            if ( Y[t] > self.Kh*Ybar[t] ):
                Z = 0
                if (t >= q and t <= self.n-q-1):
                    for j in range(-q,q+1):
                        Z += self.Kt*Y[t+j]
                    smoothed.append(Z)
                else:
                    smoothed.append(Y[t])
            elif (Y[t] < self.Kh*Ybar[t]):
                Z = 0
                if (t >= q and t <= self.n-q-1):
                    for j in range(-q,q+1):
                        Z += self.Kt*Y[t+j]
                    smoothed.append(Z)
                else:
                    smoothed.append(Y[t])
            else:
                smoothed.append(Y[t])
        return smoothed
