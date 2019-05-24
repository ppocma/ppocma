'''

Known issues (TODO): 

- policy.train() takes as input the scaler scale and offset, and uses them internally. However, the scaler also has a clip parameter, which is
  by default set to a large value, but if one wants to use it, scaler.process() will then produce different results than the scaling in policy.train()
  - fix: policy should take the Scaler instance as an argument, although this creates a dependency

'''

import numpy as np
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #disable Tensorflow GPU usage, these simple graphs run faster on CPU
import tensorflow as tf
from policy import Policy
from critic import Critic
from utils import Scaler

#Data structure for holding experience
class Experience:
    def __init__(self,s:np.array,a:np.array,r:float,s_next:np.array,terminated:bool,timeStep=0):
        self.s=s.copy()
        self.a=a.copy()
        self.r=r
        self.s_next=s_next.copy()
        self.terminated=terminated
        self.timeStep=timeStep     
        self.V=r            #value, will be updated by the agent or the client. note: this is the on-policy value, i.e., averaged over children
        self.Vselect=r      #value used for the UCB
        self.advantage=0    #will be updated by the agent
        self.nonDiscountedRewardSum=0   #used in some applications
        self.fullState=None #used in tree search
        #the following only used when building trajectory trees
        self.parent=None
        self.children=[]
        self.depth=1
        #visitation count used by UCB tree search
        self.n=1

    #Updates value upwards in the tree, called after adding a new trajectory to the tree.
    #Also keeps track of tree depth
    def propagateUpwards(self,gamma:float,bestGamma:float=1):
        node=self
        node.V=node.r
        node.Vselect=node.r
        while node.parent is not None:
            node=node.parent
            node.V=node.r
            node.Vselect=-np.inf
            nChildren=len(node.children)
            node.depth=0
            for child in node.children:
                node.V+=gamma*child.V/nChildren
                node.Vselect=max([node.Vselect,node.r+bestGamma*child.Vselect])
                node.depth=max([node.depth,child.depth+1])
            #node.Vselect=node.V

    #adds a child to this node, updating tree linkage
    def addChild(self,child):
        self.children.append(child)
        child.parent=self

    #selects a child node at specific depth, using the UCB formula to visit the children
    def selectChildAtDepth(self,depth,C_ucb):
        node=self
        self.n+=1       #keep track of visitation count
        while depth>0:
            nChildren=len(node.children)
            if nChildren==1:
                #if only one child, just move forward
                node=node.children[0]
            else:
                #if multiple children, select the best-scoring one based on the UCB-1 formula
                maxScore=-np.inf
                maxScoringChild=None
                for child in node.children:
                    if child.depth>=depth:  #we only accept branches that will take us up to the desired depth
                        score=node.Vselect+C_ucb*math.sqrt(math.log(node.n)/child.n)
                        if score>maxScore:
                            maxScoringChild=child
                            maxScore=score
                node=maxScoringChild
            depth-=1        #keep track of depth
            node.n+=1       #keep track of visitation count
        return node

class Agent:
    #In most cases, one only needs to specify stateDim, actionDim, actionMin, and actionMax.
    #The mode parameter defines the algorithm. PPO-CMA-m is the default, i.e., PPO-CMA using the sample mirroring trick.
    #Other choices are "PPO-CMA", "PPO", "PG" and "PG-pos". The two last modes denote vanilla policy gradient, and the "-pos"
    #means that only positive advantage actions are used. See DidacticExample.py for visualization of different modes in a simple quadratic problem.
    def __init__(self, stateDim:int, actionDim:int, actionMin:np.array, actionMax:np.array, learningRate=0.0005
                 , gamma=0.99, GAElambda=0.95, PPOepsilon=0.2, PPOentropyLossWeight=0, nHidden:int=2
                 , nUnitsPerLayer:int=128, mode="PPO-CMA-m", activation="lrelu", H:int=9, entropyLossWeight:float=0
                 , sdLowLimit=0.01, useScaler:bool=True, criticTimestepScale=0.001,initialMean:np.array=None,initialSd:np.array=None):
        #Create policy network 
        print("Creating policy")
        self.actionMin=actionMin.copy()
        self.actionMax=actionMax.copy()
        self.actionDim=actionDim
        self.stateDim=stateDim
        self.useScaler=useScaler
        if useScaler:
            self.scaler=Scaler(stateDim)
        self.scalerInitialized=False
        self.normalizeAdvantages=True
        self.gamma=gamma
        self.GAElambda=GAElambda
        self.criticTimestepScale=0 if gamma==0 else criticTimestepScale     #with gamma==0, no need for this
        piEpsilon = None
        nHistory = 1
        negativeAdvantageAvoidanceSigma = 0
        if mode=="PPO-CMA" or mode=="PPO-CMA-m":
            usePPOLoss=False           #if True, we use PPO's clipped surrogate loss function instead of the standard -A_i * log(pi(a_i | s_i))
            separateVarAdapt=True
            self.reluAdvantages=True if mode=="PPO-CMA" else False
            nHistory=H             #policy mean adapts immediately, policy covariance as an aggreagate of this many past iterations
            useSigmaSoftClip=True
            negativeAdvantageAvoidanceSigma=1 if mode=="PPO-CMA-m" else 0
        elif mode=="PPO":
            usePPOLoss=True           #if True, we use PPO's clipped surrogate loss function instead of the standard -A_i * log(pi(a_i | s_i))
            separateVarAdapt = False
            # separateSigmaAdapt=False
            self.reluAdvantages=False
            useSigmaSoftClip=True
            piEpsilon=0
        elif mode=="PG":
            usePPOLoss=False           #if True, we use PPO's clipped surrogate loss function instead of the standard -A_i * log(pi(a_i | s_i))
            separateVarAdapt = False
            # separateSigmaAdapt=False
            self.reluAdvantages=False
            useSigmaSoftClip=True
            piEpsilon=0
        elif mode=="PG-pos":
            usePPOLoss=False           #if True, we use PPO's clipped surrogate loss function instead of the standard -A_i * log(pi(a_i | s_i))
            separateVarAdapt = False
            # separateSigmaAdapt=False
            self.reluAdvantages=True
            useSigmaSoftClip=True
            piEpsilon=0
        else:
            raise("Unknown mode {}".format(mode))
        self.policy=Policy(stateDim, actionDim, actionMin, actionMax, entropyLossWeight=PPOentropyLossWeight
                           , networkActivation=activation, networkDepth=nHidden, networkUnits=nUnitsPerLayer
                           , networkSkips=False, learningRate=learningRate, minSigma=sdLowLimit, PPOepsilon=PPOepsilon
                           , usePPOLoss=usePPOLoss, separateVarAdapt=separateVarAdapt, nHistory=nHistory
                           , useSigmaSoftClip=useSigmaSoftClip, piEpsilon=piEpsilon
                           , negativeAdvantageAvoidanceSigma=negativeAdvantageAvoidanceSigma)

        #Create critic network, +1 stateDim because at least in OpenAI gym, episodes are time-limited and the value estimates thus depend on simulation time.
        #Thus, we use time step as an additional feature for the critic.
        #Note that this does not mess up generalization, as the feature is not used for the policy during training or at runtime
        print("Creating critic network")
        self.critic=Critic(stateDim=stateDim+1,learningRate=learningRate,nHidden=nHidden,networkUnits=nUnitsPerLayer,networkActivation=activation,useSkips=False,lossType="L1")

        #Experience trajectory buffers for the memorize() and updateWithMemorized() methods
        self.experienceTrajectories=[]
        self.currentTrajectory=[]

        #Init may take as argument a desired initial action mean and sd. These need to be remembered for the first iteration's act,
        #which samples the initial mean and sd directly instead of utilizing the policy network.
        if initialMean is not None:
            self.initialMean=initialMean.copy()
        else:
            self.initialMean=0.5*(self.actionMin+self.actionMax)*np.ones(self.actionDim)
        if initialSd is not None:
            self.initialSd=initialSd.copy()
        else:
            self.initialSd=0.5*(self.actionMax-self.actionMin)*np.ones(self.actionDim)

    #call this after tensorflow's global variables initializer
    def init(self,sess:tf.Session,verbose=False):
        #Pretrain the policy to output the initial Gaussian for all states
        self.policy.init(sess,0,1,self.initialMean,self.initialSd,256,2000,verbose)
    
    #stateObs is an n-by-m tensor, where n = number of observations, m = number of observation variables
    def act(self,sess:tf.Session,stateObs:np.array,deterministic=False,clipActionToLimits=True):
        #Expand a single 1d-observation into a batch of 1 vectors
        if len(stateObs.shape)==1:
            stateObs=np.reshape(stateObs,[1,stateObs.shape[0]])
        #Query the policy for the action, except for the first iteration where we sample directly from the initial exploration Gaussian
        #that covers the whole action space.
        #This is done because we don't know the scale of state observations a priori; thus, we can only init the state scaler in update(), 
        #after we have collected some experience.
        if self.useScaler and (not self.scalerInitialized):
            actions=np.random.normal(self.initialMean,self.initialSd,size=[stateObs.shape[0],self.actionDim])
            if clipActionToLimits:
                actions=np.clip(actions,np.reshape(self.actionMin,[1,self.actionDim]),np.reshape(self.actionMax,[1,self.actionDim]))
            return actions
        else:
            if self.useScaler:
                scaledObs=self.scaler.process(stateObs)
            else:
                scaledObs=stateObs
            if deterministic:
                actions=self.policy.getExpectation(sess,scaledObs)
            else:
                actions=self.policy.sample(sess,scaledObs)
            if clipActionToLimits:
                actions=np.clip(actions,self.actionMin,self.actionMax)
            return actions
    def memorize(self,observation:np.array,action:np.array,reward:float,nextObservation:np.array,done:bool):
        e = Experience(observation, action, reward, nextObservation, done)
        self.currentTrajectory.append(e)
        if done:
            self.experienceTrajectories.append(self.currentTrajectory)
            self.currentTrajectory=[]

    def getAverageActionStdev(self):
        if self.useScaler and (not self.scalerInitialized):
            return np.mean(0.5*(self.actionMax-self.actionMin))
        else:
            return self.policy.usedSigmaSum/(1e-20+self.policy.usedSigmaSumCounter)

    #If you call memorize() after each action, you can update the agent with this method. 
    #If you handle the experience buffers yourself, e.g., due to a multithreaded implementation, use the update() method instead.
    def updateWithMemorized(self,sess:tf.Session,batchSize:int=512,nBatches:int=100,verbose=True,valuesValid=False,timestepsValid=False):
        self.update(sess,experienceTrajectories=self.experienceTrajectories,batchSize=batchSize,nBatches=nBatches,verbose=verbose,valuesValid=valuesValid,timestepsValid=timestepsValid)
        averageEpisodeReturn=0
        for t in self.experienceTrajectories:
            episodeReturn=0
            for e in t:
                episodeReturn+=e.r
            averageEpisodeReturn+=episodeReturn
        averageEpisodeReturn/=len(self.experienceTrajectories)
        self.experienceTrajectories=[]
        self.currentTrajectory=[]
        return averageEpisodeReturn

    #experienceTrajectories is a list of lists of Experience instances such that each of the contained lists corresponds to an episode simulation trajectory
    def update(self,sess:tf.Session,experienceTrajectories,batchSize:int=512,nBatches:int=100,verbose=True,valuesValid=False,timestepsValid=False):
        trajectories=experienceTrajectories   #shorthand

        #Collect all data into linear arrays for training. 
        nTrajectories=len(trajectories)
        nData=0
        for trajectory in trajectories:
            nData+=len(trajectory)
            #propagate values backwards along trajectory if not already done
            if not valuesValid:
                for i in reversed(range(len(trajectory)-1)):
                    #value estimates, used for training the critic and estimating advantages
                    trajectory[i].V=trajectory[i].r+self.gamma*trajectory[i+1].V
            #update time steps if not updated
            if not timestepsValid:
                for i in range(len(trajectory)):
                    trajectory[i].timeStep=i
        allStates=np.zeros([nData,self.stateDim])
        allActions=np.zeros([nData,self.actionDim])
        allValues=np.zeros([nData])
        allTimes=np.zeros([nData,1])
        k=0
        for trajectory in trajectories:
            for e in trajectory:
                allStates[k,:]=e.s
                allValues[k]=e.V  
                allActions[k,:]=e.a
                allTimes[k,0]=e.timeStep*self.criticTimestepScale 
                k+=1


        #Update scalers
        if self.useScaler:
            self.scaler.update(allStates)
            scale, offset = self.scaler.get()
            self.scalerInitialized=True
        else:
            offset=0
            scale=1
 
        #Scale the observations for training the critic
        scaledStates=self.scaler.process(allStates)

        #Train critic
        def augmentCriticObs(obs:np.array,timeSteps:np.array):
            return np.concatenate([obs,timeSteps],axis=1)
        self.critic.train(sess,augmentCriticObs(scaledStates,allTimes),allValues,batchSize,nEpochs=0,nBatches=nBatches,verbose=verbose)

        #Policy training needs advantages, which depend on the critic we just trained.
        #We use Generalized Advantage Estimation by Schulman et al.
        if verbose:
            print("Estimating advantages...".format(len(trajectories)))
        for t in trajectories:
            #query the critic values of all states of this trajectory in one big batch
            nSteps=len(t)
            states=np.zeros([nSteps+1,self.stateDim])
            timeSteps=np.zeros([nSteps+1,1])
            for i in range(nSteps):
                states[i,:]=t[i].s
                timeSteps[i,0]=t[i].timeStep*self.criticTimestepScale
            states[nSteps,:]=t[nSteps-1].s_next
            states=(states-offset)*scale
            values=self.critic.predict(sess,augmentCriticObs(states,timeSteps))

            #GAE loop, i.e., take the instantaneous advantage (how much value a single action brings, assuming that the
            #values given by the critic are unbiased), and smooth those along the trajectory using 1st-order IIR filter.
            advantage=0
            for step in reversed(range(nSteps)):
                delta_t=t[step].r+self.gamma*values[step+1] - values[step]
                advantage=delta_t+self.GAElambda*self.gamma*advantage
                t[step].advantage=advantage

        #Gather the advantages to linear array and apply ReLU and normalization if needed
        allAdvantages=np.zeros([nData])
        k=0
        for trajectory in trajectories:
            for e in trajectory:
                allAdvantages[k]=e.advantage  
                k+=1

        if self.reluAdvantages:
            allAdvantages=np.clip(allAdvantages,0,np.inf)
        if self.normalizeAdvantages:
            aMean=np.mean(allAdvantages)
            aSd=np.std(allAdvantages)
            if verbose:
                print("Advantage mean {}, sd{}".format(aMean,aSd))
            allAdvantages/=1e-10+aSd
            #Clamp the normalized advantages to 3 sd:s, in case of outliers. 
            #Commented out for now to allow computing additional ICML results, as this was not yet implemented in the ICML version
            #advantageLimit=3
            #allAdvantages=np.clip(allAdvantages,-advantageLimit,advantageLimit)

        #Train policy. Note that this uses original unscaled states, because the PPO-CMA variance training needs a history of
        #states in the same scale
        self.policy.train(sess,allStates,allActions,allAdvantages,batchSize,nEpochs=0,nBatches=nBatches,stateOffset=offset,stateScale=scale,verbose=verbose)
            




