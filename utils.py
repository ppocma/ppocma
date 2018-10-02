"""
Data Scaling Utilities

Modified from code by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import os


class Scaler(object):
 

    def __init__(self, obs_dim, epsilon=0.001, clip=1e10, useOffset=False):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.xMean=np.zeros(obs_dim)
        self.xSqMean=np.zeros(obs_dim)
        self.scale=np.ones(obs_dim)
        self.offset=np.zeros(obs_dim)
        self.nUpdates=0
        self.epsilon=epsilon
        self.clip=clip
        self.useOffset=False

    def update(self, x):
        self.nUpdates+=1
        newWeight=1/self.nUpdates
        self.xSqMean=(1-newWeight)*self.xSqMean+newWeight*np.mean(np.square(x),axis=0) 
        self.xMean=(1-newWeight)*self.xMean+newWeight*np.mean(x,axis=0) 
        if self.useOffset:
            #var(x)=E((x-mean(x))^2)=E(x^2-2*x*E(x)+mean(x)^2)=E(x^2)-2*E(x)*E(x)+E(x)^2=E(x^2)-E(x)^2
            var=self.xSqMean-np.square(self.xMean)
            var=np.maximum(0.0,var)
        else:
            mean=0
            var=self.xSqMean
        self.scale=np.minimum(self.scale,1.0/(2.0*np.sqrt(var)+self.epsilon))  
        self.offset=self.xMean

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return self.scale,self.offset

    def process(self,x:np.array):
        return np.clip((x-self.offset)*self.scale,-self.clip,self.clip)

    
