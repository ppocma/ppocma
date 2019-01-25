"""
Data Scaling Utilities

Modified from code by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import os


class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (2.0*stddev + epsilon) 
    """

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
        self.useOffset=useOffset

    def update(self, x):
        self.nUpdates+=1
        newWeight=1/self.nUpdates
        self.xSqMean=(1-newWeight)*self.xSqMean+newWeight*np.mean(np.square(x),axis=0) 
        self.xMean=(1-newWeight)*self.xMean+newWeight*np.mean(x,axis=0) 
        if self.useOffset:
            mean=self.xMean
        else:
            mean=0
        #var(x)=E((x-mean(x))^2)=E(x^2-2*x*E(x)+mean(x)^2)=E(x^2)-2*E(x)*E(x)+E(x)^2=E(x^2)-E(x)^2
        var=self.xSqMean-np.square(mean)
        var=np.maximum(0.0,var)
        self.scale=np.minimum(self.scale,1.0/(2.0*np.sqrt(var)+self.epsilon))  
        self.offset=mean

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return self.scale,self.offset

    def process(self,x:np.array):
        return np.clip((x-self.offset)*self.scale,-self.clip,self.clip)
    def unscale(self,x:np.array):
        return x/self.scale+self.offset
    

    
class MinMaxScaler(object):
    """ Generate scale and offset based on low-pass filtered max and min vals
    """

    def __init__(self, obs_dim, filter=0.9, epsilon=0.001, useOffset=False, scalarMode=True):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.minVals=-np.ones(obs_dim)
        self.maxVals=np.zeros(obs_dim)
        self.filteredMinVals=self.minVals.copy()
        self.filteredMaxVals=self.maxVals.copy()
        self.first_pass = True
        self.filter=filter
        self.epsilon=epsilon
        self.scalarMode=scalarMode
        self.useOffset=useOffset
        self.scale=1 if scalarMode else np.ones(obs_dim)
        self.offset=0 if scalarMode else np.zeros(obs_dim)

    def update(self, x):
        if self.first_pass:
            self.minVals=np.min(x,axis=0)
            self.maxVals=np.max(x,axis=0)
            self.filteredMinVals=self.minVals.copy()
            self.filteredMaxVals=self.maxVals.copy()
        else:
            self.minVals=np.minimum([self.minVals,np.min(x,axis=0)])
            self.maxVals=np.maximum([self.maxVals,np.max(x,axis=0)])
            self.filteredMinVals=self.filter*self.filteredMinVals+(1-self.filter)*self.minVals
            self.filteredMaxVals=self.filter*self.filteredMaxVals+(1-self.filter)*self.maxVals
        if self.scalarMode:
            self.scale=np.minimum(self.scale,2.0/np.max((self.filteredMaxVals-self.filteredMinVals)+self.epsilon))
        else:
            self.scale=2.0/((self.filteredMaxVals-self.filteredMinVals)+self.epsilon)
            if self.useOffset:
                self.offset=0.5*(self.filteredMaxVals+self.filteredMinVals)

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return self.scale,self.offset
