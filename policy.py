import numpy as np
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #disable Tensorflow GPU usage, these simple graphs run faster on CPU
import tensorflow as tf
import MLP
from collections import deque


def softClip(x, minVal, maxVal):
    #return minVal+(maxVal-minVal)*(1.0+0.5*tf.tanh(x))
    return minVal+(maxVal-minVal)*tf.sigmoid(x)

class Policy:
    def __init__(self, stateDim:int, actionDim:int, actionMinLimit:np.array, actionMaxLimit:np.array, mode="PPO-CMA"
                 , entropyLossWeight=0, networkDepth=2, networkUnits=64, networkActivation="lrelu"
                 , networkSkips=False, networkUnitNormInit=True, usePPOLoss=False, separateVarAdapt=False
                 , learningRate=0.001, minSigma=0.01, useSigmaSoftClip=True, PPOepsilon=0.2, piEpsilon=0, nHistory=1
                 , globalVariance=False, trainableGlobalVariance=True, useGradientClipping=False
                 , maxGradientNorm=0.5, negativeAdvantageAvoidanceSigma=0):
        self.networkDepth = networkDepth
        self.networkUnits = networkUnits
        self.networkActivation = networkActivation
        self.networkSkips = networkSkips
        self.networkUnitNormInit = networkUnitNormInit
        self.usePPOLoss = usePPOLoss
        self.separateVarAdapt = separateVarAdapt
        self.learningRate = learningRate
        self.minSigma = minSigma
        self.useSigmaSoftClip=useSigmaSoftClip
        self.PPOepsilon = PPOepsilon
        self.piEpsilon = piEpsilon
        self.nHistory = nHistory
        self.globalVariance = globalVariance
        self.trainableGlobalVariance = trainableGlobalVariance
        self.useGradientClipping = useGradientClipping
        self.maxGradientNorm = maxGradientNorm
        self.negativeAdvantageAvoidanceSigma = negativeAdvantageAvoidanceSigma

        maxSigma=1.0*(actionMaxLimit-actionMinLimit)
        self.mode=mode

        #to be able to benchmark with Schulman's original network architecture, we may have to disable the data-dependent init of the DenseNet module
        MLP.useUnitNormInit=self.networkUnitNormInit

        #some bookkeeping
        self.usedSigmaSum=0
        self.usedSigmaSumCounter=0

        #inputs
        stateIn=tf.placeholder(dtype=tf.float32,shape=[None,stateDim],name="stateIn")
        actionIn=tf.placeholder(dtype=tf.float32,shape=[None,actionDim],name="actionIn")    #training targets for policy network
        oldPolicyMean=tf.placeholder(dtype=tf.float32,shape=[None,actionDim],name="oldPolicyMeanIn")    #training targets for policy network
        self.oldPolicyMean=oldPolicyMean
        advantagesIn=tf.placeholder(dtype=tf.float32,shape=[None],name="advantagesIn")     #weights, computed based on action advantages
        logPiOldIn=tf.placeholder(dtype=tf.float32,shape=[None],name="logPiOldIn")             #pi_old(a | s), used for PPO
        initSigmaIn=tf.placeholder(dtype=tf.float32,shape=[1,actionDim],name="initSigmaIn")
        
        #by default, we won't use a linear layer at the beginning of the network to reduce dimensionality
        firstLinearLayerUnits=0

        #First, define the mean and log var tensors, depending on configuration. 
        #Depending on the network architecture, we may also need initialization tensors, fetching which causes a data-dependent initialization of the graph
        policyInit=[]
        if stateDim==0:
            #We don't have state at all => policyMean and variance are simply TensorFlow variables
            policyMean=tf.Variable(initial_value=np.zeros([actionDim]),dtype=tf.float32)
            policyLogVar=tf.Variable(initial_value=np.log(np.square(0.5*(actionMaxLimit-actionMinLimit)))*np.ones([actionDim]),dtype=tf.float32,trainable=self.trainableGlobalVariance)
            self.globalLogVarVariable=policyLogVar
        else:
            #We have state, i.e., need neural networks that output a state-dependent mean and variance
            if self.separateVarAdapt or self.globalVariance:
                #Need separate networks for mean and variance
                policyMean,policyMeanInit=MLP.mlp(stateIn,self.networkDepth,self.networkUnits,actionDim,self.networkActivation,firstLinearLayerUnits,self.networkSkips)
                policyInit.append(policyMeanInit)
                if self.globalVariance:
                    policyLogVar=tf.Variable(initial_value=np.log(np.square(0.5*(actionMaxLimit-actionMinLimit)))*np.ones([actionDim]),dtype=tf.float32,trainable=self.trainableGlobalVariance)
                    self.globalLogVarVariable=policyLogVar
                else:
                    policyLogVar,policyLogVarInit=MLP.mlp(stateIn,self.networkDepth,self.networkUnits,actionDim,self.networkActivation,firstLinearLayerUnits,self.networkSkips)
                    policyInit.append(policyLogVarInit)
            else:
                #Single network that outputs both mean and variance
                policyMeanAndLogVar,policyMeanAndLogVarInit=MLP.mlp(stateIn,self.networkDepth,self.networkUnits,actionDim*2,self.networkActivation,firstLinearLayerUnits,self.networkSkips)
                policyMean=policyMeanAndLogVar[:,:actionDim]
                policyLogVar=policyMeanAndLogVar[:,actionDim:]
                policyInit.append(policyMeanAndLogVarInit)

        #sigmoid-clipping of mean to ensure stability
        policyMean=softClip(policyMean, actionMinLimit,actionMaxLimit)
        if self.useSigmaSoftClip:
            #sigmoid-clipping of log var to ensure stability
            #Note: tanh or hard clipping doesn't work as well due to higher chance of zero or almost zero gradients 
            maxLogVar=np.log(maxSigma*maxSigma)
            minLogVar=np.log(self.minSigma*self.minSigma)
            policyLogVar=softClip(policyLogVar,minLogVar,maxLogVar)
        policyVar=tf.exp(policyLogVar)  
        policySigma=tf.sqrt(policyVar)


        #loss functions
        if self.usePPOLoss:
            def loss(policyMean,policyVar,policyLogVar):
                #1/sqrt(var)=exp(log(1/sqrt(var)))=exp(log(1)-log(var^0.5))=exp(-0.5*log(var))=exp(-log(std))
                logPi=tf.reduce_sum(-0.5*tf.square(actionIn-policyMean)/policyVar-0.5*policyLogVar,axis=1)
                #Some PPO implementations use r=tf.exp(logPi-logPiOldIn). However, we've noticed this to cause NaNs especially
                #with non-saturating policy network activation functions like lrelu and the MuJoCo humanoid env.
                #Thus, we also support using the epsilon below to regularize. 
                if self.piEpsilon==0:
                    r=tf.exp(logPi-logPiOldIn)
                else:
                    r=tf.exp(logPi)/(self.piEpsilon+tf.exp(logPiOldIn))
                perSampleLoss=tf.minimum(r*advantagesIn,tf.clip_by_value(r,1-self.PPOepsilon,1+self.PPOepsilon)*advantagesIn)
                return -tf.reduce_mean(perSampleLoss) #because we want to minimize instead of maximize...
            print("Using PPO clipped surrogate loss with epsilon {}".format(self.PPOepsilon))
            policyLoss=loss(policyMean,policyVar,policyLogVar)
            if entropyLossWeight>0:
                #Entropy of a diagonal Gaussian=0.5*log(det(Cov))=0.5*log(trace(Cov))=0.5*sum(log(diag(Cov)))
                policyLoss-=entropyLossWeight*0.5*tf.reduce_mean(tf.reduce_sum(policyLogVar,axis=1))
            assert(self.separateVarAdapt==False)
            #just to be on the safe side, if some batch has an occasional NaN, set the loss to zero
            policyLoss=tf.where(tf.is_nan(policyLoss), tf.zeros_like(policyLoss),policyLoss)
            policyMeanLoss=policyLoss
            policySigmaLoss=policyLoss
        else:
            #Separate mean and sigma adaptation losses
            policyNoGrad=tf.stop_gradient(policyMean)
            policyVarNoGrad=tf.stop_gradient(policyVar)
            policyLogVarNoGrad=tf.stop_gradient(policyLogVar)
            logpNoMeanGrad=-tf.reduce_sum(0.5*tf.square(actionIn-policyNoGrad)/policyVar+0.5*policyLogVar,axis=1)
            logpNoVarGrad=-tf.reduce_sum(0.5*tf.square(actionIn-policyMean)/policyVarNoGrad+0.5*policyLogVarNoGrad,axis=1) 
            posAdvantages=tf.nn.relu(advantagesIn)
            policySigmaLoss=-tf.reduce_mean(posAdvantages*logpNoMeanGrad)
            policyMeanLoss=-tf.reduce_mean(posAdvantages*logpNoVarGrad)
            if self.negativeAdvantageAvoidanceSigma>0:
                negAdvantages=tf.nn.relu(-advantagesIn)
                mirroredAction=oldPolicyMean-(actionIn-oldPolicyMean)  #mirror negative advantage actions around old policy mean (convert them to positive advantage actions assuming linearity) 
                logpNoVarGradMirrored=-tf.reduce_sum(0.5*tf.square(mirroredAction-policyMean)/policyVarNoGrad+0.5*policyLogVarNoGrad,axis=1) 
                effectiveKernelSqWidth=self.negativeAdvantageAvoidanceSigma*self.negativeAdvantageAvoidanceSigma*policyVarNoGrad
                avoidanceKernel=tf.reduce_mean(tf.exp(-0.5*tf.square(actionIn-oldPolicyMean)/effectiveKernelSqWidth),axis=1)
                policyMeanLoss-=tf.reduce_mean((negAdvantages*avoidanceKernel)*logpNoVarGradMirrored)

            #just to be on the safe side, if some batch has an occasional NaN, set the loss to zero
            policySigmaLoss=tf.where(tf.is_nan(policySigmaLoss), tf.zeros_like(policySigmaLoss),policySigmaLoss)
            policyMeanLoss=tf.where(tf.is_nan(policyMeanLoss), tf.zeros_like(policyMeanLoss),policyMeanLoss)

            #Vanilla Policy Gradient loss
            logp=-tf.reduce_sum(0.5*tf.square(actionIn-policyMean)/policyVar+0.5*policyLogVar,axis=1)
            policyLoss=tf.reduce_mean(-advantagesIn*logp)  

        #loss functions for initialization (pretraining)
        policyInitLoss=tf.reduce_mean(tf.square(actionIn-policyMean))
        policyInitLoss+=tf.reduce_mean(tf.square(initSigmaIn-policySigma))

        #optimizers
        def optimize(loss):
            optimizer=tf.train.AdamOptimizer(learning_rate=self.learningRate)
            if not self.useGradientClipping:
                return optimizer.minimize(loss)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.maxGradientNorm)
            return optimizer.apply_gradients(zip(gradients, variables))
        self.optimizePolicy=optimize(policyLoss)
        self.optimizePolicySigma=optimize(policySigmaLoss)
        self.optimizePolicyMean=optimize(policyMeanLoss)
        self.optimizePolicyInit=optimize(policyInitLoss)

        #cache stuff needed elsewhere
        self.actionMinLimit=actionMinLimit.copy()
        self.actionMaxLimit=actionMaxLimit.copy()
        self.stateDim=stateDim
        self.actionDim=actionDim
        self.policyMean=policyMean
        self.stateIn=stateIn
        self.actionIn=actionIn
        self.policyInit=policyInit
        self.policyInitLoss=policyInitLoss
        self.advantagesIn=advantagesIn
        self.policyLoss=policyLoss
        self.logPiOldIn=logPiOldIn
        self.initSigmaIn=initSigmaIn
        self.history=deque()
        self.policyVar=policyVar
        self.policyLogVar=policyLogVar
        self.policySigma=policySigma
        self.initialized=False  #remember that one has to call init() before training (can't call it here as TF globals might not have been initialized yet)

    #init the policy with random Gaussian state samples, such that the network outputs the desired mean and sd
    def init(self,sess:tf.Session,stateMean:np.array,stateSd:np.array,actionMean:np.array,actionSd:np.array,nMinibatch:int=64,nBatches:int=4000,verbose=True):
        for batchIdx in range(nBatches):
            states=np.random.normal(stateMean,stateSd,size=[nMinibatch,self.stateDim])
            if batchIdx==0 and len(self.policyInit)>0:
                #init the MLP biases to prevent large values
                temp,currLoss=sess.run([self.policyInit,self.policyInitLoss],feed_dict={self.stateIn:states,self.actionIn:np.reshape(actionMean,[1,self.actionDim]),self.initSigmaIn:np.reshape(actionSd,[1,self.actionDim])})
            else:
                #drive output towards the desired mean and sd
                temp,currLoss=sess.run([self.optimizePolicyInit,self.policyInitLoss],feed_dict={self.stateIn:states,self.actionIn:np.reshape(actionMean,[1,self.actionDim]),self.initSigmaIn:np.reshape(actionSd,[1,self.actionDim])})
            if verbose and (batchIdx % 100 ==0):
                print("Initializing policy with random Gaussian data, batch {}/{}, loss {}".format(batchIdx,nBatches,currLoss))
        self.initialized=True
    #init the policy with uniform random state samples, such that the network outputs the desired mean and sd
    def initUniform(self,sess:tf.Session,stateMin:np.array,stateMax:np.array,actionMean:np.array,actionSd:np.array,nMinibatch:int=64,nBatches:int=4000):
        for batchIdx in range(nBatches):
            states=np.random.uniform(stateMin,stateMax,size=[nMinibatch,self.stateDim])
            if batchIdx==0 and len(self.policyInit)>0:
                #init the MLP biases to prevent large values
                temp,currLoss=sess.run([self.policyInit,self.policyInitLoss],feed_dict={self.stateIn:states,self.actionIn:np.reshape(actionMean,[1,self.actionDim]),self.initSigmaIn:np.reshape(actionSd,[1,self.actionDim])})
            else:
                #drive output towards the desired mean and sd
                temp,currLoss=sess.run([self.optimizePolicyInit,self.policyInitLoss],feed_dict={self.stateIn:states,self.actionIn:np.reshape(actionMean,[1,self.actionDim]),self.initSigmaIn:np.reshape(actionSd,[1,self.actionDim])})
            if batchIdx % 100 ==0:
                print("Initializing policy with random Gaussian data, batch {}/{}, loss {}".format(batchIdx,nBatches,currLoss))
        self.initialized=True
    #if nBatches==0, nEpochs will be used
    def train(self,sess:tf.Session,states:np.array,actions:np.array,advantages:np.array,nMinibatch:int,nEpochs:int,nBatches:int=0,stateOffset=0,stateScale=1,verbose=True):
        assert(np.all(np.isfinite(states)))
        assert(np.all(np.isfinite(actions)))
        assert(np.all(np.isfinite(advantages)))
        assert(self.initialized)
        nData=actions.shape[0]

        #reset bookkeeping for next iter
        self.usedSigmaSum=0
        self.usedSigmaSumCounter=0

        #manage history
        self.history.append([states.copy(),actions.copy(),advantages.copy()])
        if len(self.history)>self.nHistory:
            self.history.popleft()

        #safety-check that the observed state distribution is at least roughly zero-mean unit sd
        if self.stateDim>0:
            scaledStates=(states-stateOffset)*stateScale
            stateAbsMax=np.max(np.absolute(scaledStates))
            if stateAbsMax>10:
                print("Warning: states deviate up to {} sd:s from expected!".format(stateAbsMax))
        else:
            scaledStates=states
        #train
        assert(len(advantages.shape)==1)  #to prevent nasty silent broadcasting bugs
        nMinibatch=min([nData,nMinibatch])
        if nBatches==0:
            nBatches=max([1,int(nData*nEpochs/nMinibatch)])
        #nBatches=1000
        nVarAdaptBatches=nBatches
        mbStates=np.zeros([nMinibatch,self.stateDim])
        mbActions=np.zeros([nMinibatch,self.actionDim])
        mbOldMean=np.zeros([nMinibatch,self.actionDim])
        mbAdvantages=np.zeros([nMinibatch])
        logPiOld=np.ones([nData])
        mbLogPiOld=np.ones([nMinibatch])
        if self.usePPOLoss:
            policyMean,policyVar,policyLogVar=sess.run([self.policyMean,self.policyVar,self.policyLogVar],feed_dict={self.stateIn:scaledStates})
            #for i in range(nData):
            #    logPiOld[i]=np.sum(-0.5*np.square(actions[i,:]-policyMean[i,:])/policyVar[i,:]-0.5*policyLogVar[i,:])
            logPiOld=np.sum(-0.5*np.square(actions-policyMean)/policyVar-0.5*policyLogVar,axis=1)
        if self.separateVarAdapt:
            assert(self.usePPOLoss==False)
            #if negativeAdvantageAvoidanceSigma>0:
            oldMeans=sess.run(self.policyMean,{self.stateIn:scaledStates})
            for batchIdx in range(nBatches + nVarAdaptBatches if self.separateVarAdapt else nBatches):
                if batchIdx<nVarAdaptBatches:
                    historyLen=len(self.history)
                    for i in range(nMinibatch):
                        histIdx=np.random.randint(0,historyLen)
                        h=self.history[histIdx]
                        nData=h[1].shape[0]
                        dataIdx=np.random.randint(0,nData)
                        mbStates[i,:]=h[0][dataIdx,:]
                        mbActions[i,:]=h[1][dataIdx,:]
                        mbAdvantages[i]=h[2][dataIdx]
                    advantageMean=np.mean(mbAdvantages)
                    mbStates=(mbStates-stateOffset)*stateScale  #here, we must scale per batch because using the history
                    temp,currLoss=sess.run([self.optimizePolicySigma,self.policyLoss],feed_dict={self.stateIn:mbStates,self.actionIn:mbActions,self.advantagesIn:mbAdvantages})
                    if verbose and (batchIdx % 100 == 0):
                        print("Adapting policy variance, batch {}/{}, mean advantage {:.2f}, loss {}".format(batchIdx,nVarAdaptBatches,advantageMean,currLoss))
                #temp,currLoss=sess.run([self.optimizePolicyMean,self.policyLoss],feed_dict={self.stateIn:mbStates,self.actionIn:mbActions,self.advantagesIn:mbAdvantages})
                else:
                    nData=actions.shape[0]
                    for i in range(nMinibatch):
                        dataIdx=np.random.randint(0,nData)
                        mbStates[i,:]=scaledStates[dataIdx,:]  
                        mbActions[i,:]=actions[dataIdx,:]
                        if self.stateDim>0:
                            mbOldMean[i,:]=oldMeans[dataIdx,:]
                        mbAdvantages[i]=advantages[dataIdx]
                    advantageMean=np.mean(mbAdvantages)
                    temp,currLoss=sess.run([self.optimizePolicyMean,self.policyLoss],feed_dict={self.stateIn:mbStates,self.actionIn:mbActions,self.advantagesIn:mbAdvantages,self.logPiOldIn:mbLogPiOld, self.oldPolicyMean:mbOldMean})
                    if verbose and (batchIdx % 100 == 0):
                        print("Adapting policy mean, batch {}/{}, mean advantage {:.2f}, loss {}".format(batchIdx-nVarAdaptBatches,nBatches,advantageMean,currLoss))

        else:
            for batchIdx in range(nBatches + nVarAdaptBatches if self.separateVarAdapt else nBatches):
                for i in range(nMinibatch):
                    dataIdx=np.random.randint(0,nData)
                    if self.stateDim!=0:
                        mbStates[i,:]=scaledStates[dataIdx,:]
                    mbActions[i,:]=actions[dataIdx,:]
                    mbAdvantages[i]=advantages[dataIdx]
                    mbLogPiOld[i]=logPiOld[dataIdx]
                advantageMean=np.mean(mbAdvantages)
                temp,currLoss=sess.run([self.optimizePolicy,self.policyLoss],feed_dict={self.stateIn:mbStates,self.actionIn:mbActions,self.advantagesIn:mbAdvantages,self.logPiOldIn:mbLogPiOld})
                if verbose and (batchIdx % 100 == 0):
                    print("Training policy, batch {}/{}, mean advantage {:.2f}, loss {}".format(batchIdx,nBatches,advantageMean,currLoss))
    def setGlobalStdev(self,relStdev:float, sess:tf.Session):
        assert(self.globalVariance and (not self.trainableGlobalVariance))
        stdev=relStdev*(self.actionMaxLimit-self.actionMinLimit)
        var=np.square(stdev)
        logVar=np.log(var)
        self.globalLogVarVariable.load(logVar,sess)
        
    def sample(self,sess:tf.Session,observations:np.array,enforcedRelSigma:float=None):
        obs=observations
        nObs=obs.shape[0]
        result=np.zeros([nObs,self.actionDim])
        assert(self.initialized)
        policyMean,policySigma=sess.run([self.policyMean,self.policySigma],feed_dict={self.stateIn:obs})
        if np.any(np.isnan(policyMean)):
            raise Exception("Policy mean is NaN")
        if np.any(np.isnan(policySigma)):
            raise Exception("Policy sigma is NaN")
        #if np.any(policySigma<minSigma):
        #    raise Exception("Policy sigma violates limits")

        for i in range(nObs):
            if self.stateDim==0:
                result[i,:]=np.random.normal(policyMean,policySigma,[self.actionDim])
            else:
                if self.globalVariance:
                    result[i,:]=np.random.normal(policyMean[i,:],policySigma,[self.actionDim])
                else:
                    result[i,:]=np.random.normal(policyMean[i,:],policySigma[i,:],[self.actionDim])

        #bookkeeping for logging
        self.usedSigmaSum+=np.mean(policySigma)
        self.usedSigmaSumCounter+=1
        return result
    def getExpectation(self,sess:tf.Session,observations:np.array):
        return sess.run(self.policyMean,feed_dict={self.stateIn:observations})
    def getSd(self,sess:tf.Session,observations:np.array):
        return sess.run(self.policySigma,feed_dict={self.stateIn:observations})
    #def get2dEllipse(self,observations:np.array):
    #    def logProb(self,state:np.array,action:np.array):
    #    def adapt(self,states:np.array,actions:np.array,advantages:np.array,batchSize:int,nEpochs:int):
