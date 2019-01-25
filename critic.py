import numpy as np
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #disable Tensorflow GPU usage, these simple graphs run faster on CPU
import tensorflow as tf
import MLP
from collections import deque

useGradientClipping=False
maxGradientNorm=1

#A function that behaves like abs, but has zero derivative at the origin, which should improve final convergence if this is 
#used in computing a loss 
def softAbs(x:tf.Tensor):
    x=tf.abs(x)
    return tf.where(x > 0.5, x-0.25, x*x)

class Critic:
    def __init__(self,stateDim:int,nHidden:int,networkUnits:int,networkActivation,useSkips=False,learningRate:float=1e-3,nHistory:int=1,lossType="L2"):
        stateIn=tf.placeholder(dtype=tf.float32,shape=[None,stateDim])
        valueIn=tf.placeholder(dtype=tf.float32,shape=[None,1])             #training targets for value network
        critic,criticInit=MLP.mlp(stateIn,nHidden,networkUnits,1,networkActivation,firstLinearLayerUnits=0,useSkips=useSkips)  #need a handle for the DenseNet instance for network switching
        diff=valueIn-critic
        if lossType=="L2":
            loss=tf.reduce_mean(tf.square(diff))    
        elif lossType=="L1":
            loss=tf.reduce_mean(tf.abs(diff))       #L1 loss, can be more stable
        elif lossType=="SoftL1":
            loss=tf.reduce_mean(softAbs(diff))       #L1 loss with zero gradient at optimum
        else:
            raise Exception("Loss type not recognized!")
        def optimize(loss):
            optimizer=tf.train.AdamOptimizer(learning_rate=learningRate)
            if not useGradientClipping:
                return optimizer.minimize(loss)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, maxGradientNorm)
            return optimizer.apply_gradients(zip(gradients, variables))
        optimizeCritic=optimize(loss)
        #remember some of the tensors for later
        self.loss=loss
        self.nHistory=nHistory
        self.history=deque()
        self.criticInit=criticInit
        self.stateIn=stateIn
        self.valueIn=valueIn
        self.initialized=False
        self.stateDim=stateDim
        self.critic=critic
        self.optimize=optimizeCritic

    def train(self,sess,states:np.array,values:np.array,nMinibatch:int,nEpochs:int,nBatches:int=0,verbose=True):
        assert(np.all(np.isfinite(states)))
        assert(np.all(np.isfinite(values)))
        nData=states.shape[0]

        #manage history
        self.history.append([states.copy(),values.copy()])
        if len(self.history)>self.nHistory:
            self.history.popleft()

        #train
        nMinibatch=min([nData,nMinibatch])
        if nBatches==0:
            nBatches=max([1,int(nData*nEpochs/nMinibatch)])
        mbState=np.zeros([nMinibatch,self.stateDim])
        mbValue=np.zeros([nMinibatch,1])
        for batchIdx in range(nBatches):
            historyLen=len(self.history)
            for i in range(nMinibatch):
                histIdx=np.random.randint(0,historyLen)
                h=self.history[histIdx]
                nData=h[0].shape[0]
                dataIdx=np.random.randint(0,nData)
                mbState[i,:]=h[0][dataIdx,:]
                mbValue[i]=h[1][dataIdx]
            if batchIdx==0 and not self.initialized:
                #init the MLP biases to prevent large values
                temp,currLoss=sess.run([self.criticInit,self.loss],feed_dict={self.stateIn:mbState,self.valueIn:mbValue})
                self.initialized=True
            else:
                temp,currLoss=sess.run([self.optimize,self.loss],feed_dict={self.stateIn:mbState,self.valueIn:mbValue})
            if verbose and (batchIdx % 100 == 0):
                print("Training critic, batch {}/{}, loss {}".format(batchIdx,nBatches,currLoss))
    def predict(self,sess,states):
        return sess.run(self.critic,feed_dict={self.stateIn:states})
