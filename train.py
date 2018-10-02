
import numpy as np
import math
import random
from collections import deque
import matplotlib
import matplotlib.pyplot as pp
import gym
import os
import copy
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #disable Tensorflow GPU usage, these simple graphs run faster on CPU
import tensorflow as tf
import argparse
import policy as pl
from policy import Policy
from critic import Critic
from utils import Scaler
import time
import inspect
from mujoco_py.generated import const


import threading

def main(envName,baseline,saveFileName,mode,gamma,GAELambda,nHistory,minSd,nEpochsPerIter,verbose,plotProgress,useScaler,globalVar,iterSimBudget,maxSimSteps,nBenchmarkRuns,scale,rewardScale,L2Critic,networkWidth,actionRepeat,linearSdAnneal,nEpisodes,entropyLossWeight):
    test=envName
    humanoidTest= test=="Humanoid" or test=="HumanoidStandup"
    if mode=="PPO":
        nHistory=1
    H=nHistory
    pl.globalVariance=globalVar
    if linearSdAnneal:
        pl.globalVariance=True
        pl.trainableGlobalVariance=False

    if saveFileName is None:
        saveFileName="results/{}_{}".format(envName,mode)
        if globalVar:
            saveFileName+="_globalVar"
        if useScaler:
            saveFileName+="_useScaler"
        if entropyLossWeight!=0:
            saveFileName+="_entropyLossWeight={}".format(entropyLossWeight)
        if minSd!=0.01:
            saveFileName+="_minSd={}".format(minSd)
        if nEpochsPerIter!=20:
            saveFileName+="_nEpochs={}".format(nEpochsPerIter)
        if mode=="PPO-CMA":
            saveFileName+="_H={}".format(H)
        if iterSimBudget!=2048*8:
            saveFileName+="_N={}".format(iterSimBudget)
        if nEpisodes is not None:
            saveFileName+="_nEpisodes={}".format(nEpisodes)
        if L2Critic:
            saveFileName+="_L2Critic"
        if networkWidth!=64:
            saveFileName+="_networkWidth={}".format(networkWidth)
        if actionRepeat!=2:
            saveFileName+="_actionRepeat={}".format(actionRepeat)
        if linearSdAnneal:
            saveFileName+="_linearSdAnneal"
        if GAELambda!=0.95:
            saveFileName+="_GAELambda={}".format(GAELambda)
        saveFileName+=".npy"

    #Set some hyperparameters based on mode
    if mode=="PPO-CMA":
        usePPOLoss=False           #if True, we use PPO's clipped surrogate loss function instead of the standard -A_i * log(pi(a_i | s_i))
        separateSigmaAdapt=not linearSdAnneal   #with linearly annealed stdev, don't bother with the variance training passes
        reluAdvantages=True
        pl.nHistory=H             #policy mean adapts immediately, policy covariance as an aggreagate of this many past iterations
        pl.useSigmaSoftClip=True
    elif mode=="PPO":
        usePPOLoss=True           #if True, we use PPO's clipped surrogate loss function instead of the standard -A_i * log(pi(a_i | s_i))
        separateSigmaAdapt=False
        reluAdvantages=False
        pl.entropyLossWeight=entropyLossWeight
        pl.useSigmaSoftClip=not linearSdAnneal #using the clipping prevents exact setting of variance
        pl.piEpsilon=0 #1e-10 # #if not humanoidTest else 1e-10   #the MuJoCo humanoid seems to need a bit of extra regularization, at least with lrelu units
    else:
        raise("Unknown mode {}".format(mode))

    #Hyperparameters not specified as command line arguments
    PPOepsilon=0.2               #PPO's epsilon parameter. 
    useThreads=True
    maxTimeStepsPerThread=2000        #episode terminated after this many steps, environment/test -specific
    trainingEndsAt=maxSimSteps    #train for this many simulation steps
    #gamma=np.power(gamma,actionRepeat)  #to make episode returns etc. compatible when adjusting actionRepeat
    #GAELambda=np.power(GAELambda,actionRepeat)
    nBatchesPerIter=0 #        #if 0, nEpochsPerIter will be used
    minibatchSize=256            #training minibatch size
    learningRate=0.0003        #learning rate
    normalizeAdvantages=True  #if True, advantages of each minibatch are scaled to make step size less dependent on reward function design. 
    offset=0                                #offset to use if useScaler==False

    useDataDependentInit=True
    networkActivation="lrelu"
    pl.networkUnitNormInit=True
    
    networkDepth=2
    networkUnits=networkWidth         #per layer
    networkSkips=False      #Use dense skip-connections?




    #initialize environment (multiple copies for multithreading and parallelizing policy evaluations)
    print("Creating environment")
    if nEpisodes is not None:
        nThreads=nEpisodes
    else:
        nThreads=iterSimBudget//maxTimeStepsPerThread
    sims=[]
    for i in range(nThreads):
        sims.append(gym.make(envName))
        #sims.append(MyEnv(noRender=(i!=0) or useThreads))
    sim=sims[0]
    #test = gym.make("Reacher-v2")
    print("Using {} env, max time steps steps".format(envName,sim._max_episode_steps))
    #input()
    actionMinLimit=sim.action_space.low[0]
    actionMaxLimit=sim.action_space.high[0]
    print("action space limits:")
    print(sim.action_space.low)
    print(sim.action_space.high)
    singleActionDim=sim.action_space.low.shape[0]
    actionDim=singleActionDim
    stateDim=sim.observation_space.low.shape[0]


    #Benchmark data, storing average episode reward, policy sigma, and used simulation steps for each benchmark run and episode
    benchmarkData=np.zeros([nBenchmarkRuns,0,3])
    if baseline is not None:
        baseline=np.load(baseline)
        #baseline[:,:,0]*=2  #correct for error in old baselines...


    #Data structure for holding experience
    class Experience:
        def __init__(self,s:np.array,a:np.array,r:float,s_next:np.array,terminated:bool,timeStep):
            self.s=s.copy()
            self.a=a.copy()
            self.r=r
            self.s_next=s_next.copy()
            self.terminated=terminated
            self.V=r
            self.ret=r
            self.advantage=r
            self.nextStateCriticValue=0
            self.timeStep=timeStep

    #Everything runs multiple times to get reliable performance estimate
    for benchmarkRunIdx in range(nBenchmarkRuns):
        print("Starting training run {} of {} environment, using {} with minSd={}, H={}, iterSimBudget={}".format(benchmarkRunIdx,test,mode,minSd,pl.nHistory,iterSimBudget))


        #for each training run, reset and build the graph again
        tf.reset_default_graph()
        sess=tf.Session()

        #Scaler for ensuring state observations stay zero-mean & unit-variance.
        #This is not strictly necessary for most MuJoCo environments, but the Humanoid has quite wildly scaled states and observations
        if useScaler:
            scaler=Scaler(stateDim)
            #rewardScaler=Scaler(1)
            #Init scaler with some random data
            nScalerInit=maxTimeStepsPerThread*4
            scalerInitObs=np.zeros([nScalerInit,stateDim])
            scalerInitRewards=np.zeros(nScalerInit)
            n=0
            while n<nScalerInit:
                scalerInitObs[n,:]=sim.reset()
                n+=1
                while n<nScalerInit:
                    scalerInitObs[n,:],scalerInitRewards[n],terminated,i=sim.step(sim.action_space.sample())
                    n+=1
                    if terminated:
                        break
            scaler.update(scalerInitObs)
            #rewardScaler.update(scalerInitRewards)
            #rewardScale=rewardScaler.scale
            scale, offset = scaler.get()
            print("Scaler scale {}, offset {} ".format(scale,offset))
            #print("Reward scale {}".format(rewardScale))
            totalSimSteps=nScalerInit
        else:
            totalSimSteps=0

        #Create policy network 
        print("Creating policy")
        pl.networkActivation=networkActivation
        pl.networkDepth=networkDepth
        pl.networkUnits=networkUnits
        pl.networkSkips=networkSkips
        pl.learningRate=learningRate
        pl.minSigma=minSd
        pl.usePPOLoss=usePPOLoss
        pl.PPOepsilon=PPOepsilon
        pl.separateVarAdapt=separateSigmaAdapt
        policy=Policy(stateDim,actionDim,sim.action_space.low,sim.action_space.high)

        #Create critic network, +1 stateDim because our episodes are time-limited and the value estimates thus depend on simulation time,
        #which is used as an additional feature
        #Note that this does not mess up generalization, as the feature is not used for the policy during training or at runtime
        print("Creating critic network")
        critic=Critic(stateDim=stateDim+1,learningRate=learningRate,nHidden=networkDepth,networkUnits=networkUnits,networkActivation=networkActivation,useSkips=networkSkips,lossType="L2" if L2Critic else "L1")
        #helper for augmenting critic observations with timesteps
        def augmentCriticObs(obs:np.array,timeSteps:np.array):
            return np.concatenate([obs,timeSteps],axis=1)

        #initialize networks
        tf.global_variables_initializer().run(session=sess)

        #init policy, assuming zero-mean observations with sd of max 10 (which using an autoscaler or manual scaling should ensure)
        policy.init(sess,0,1,0.5*(actionMinLimit+actionMaxLimit)*np.ones(actionDim),0.5*(actionMaxLimit-actionMinLimit)*np.ones(actionDim),minibatchSize,4000)
 
        #init critic to output zeros 
        #critic.init(sess,0,1,0,minibatchSize,4000)
    
        #expectedStateMinValue=-10
        #expectedStateMaxValue=10
        #if useDataDependentInit:
        #    policy.init(sess,expectedStateMinValue,expectedStateMaxValue,0.5*(actionMinLimit+actionMaxLimit)*np.ones(actionDim),0.5*(actionMaxLimit-actionMinLimit)*np.ones(actionDim),256,8000)
        tf.get_default_graph().finalize()   #for multithreading


        #Class to wrap a list, so that we can pass it by reference to thread functions
        class TrajectoryWrapper:
            def __init__(self):
                self.trajectories=[]

        #Do a number of rollouts, i.e., run the simulation on-policy until termination or max steps.
        #Appends a trajectory (i.e., list of Experience instances) to the trajectories list contained by trajectoryWrapper (container needed for threading).
        #We avoid returning the trajectory, as we want this function to be the target of a Thread class...
        #We also run multiple trajectories in each thread to be able to parallelize policy network evaluations and minimize the overhead of sess.run() calls
        def doRollouts(simIdx,trajectoryWrapper:TrajectoryWrapper,useCritic:bool, render:bool=False,scale:float=1,offset:float=0):
            trajectory=[]
            done=False
            observation=np.zeros([1,stateDim])
            observation[0,:]=sims[simIdx].reset()

            #simulate
            nSteps=maxTimeStepsPerThread//actionRepeat
            nTrajSteps=0
            timeStepLimit=sims[0]._max_episode_steps
            for step in range(nSteps):
                scaledObs=(observation-offset)*scale
                action=policy.sample(sess,scaledObs)
                action=np.clip(action,actionMinLimit,actionMaxLimit)
                reward=0
                for subStep in range(actionRepeat):
                    nextObservation,stepReward,done,info=sims[simIdx].step(action[0,:])
                    nTrajSteps+=1
                    stepReward*=rewardScale
                    if render and t==0:
                        sims[simIdx].render()
                    reward+=stepReward
                    if done:
                        break
                e=Experience(observation[0,:],action[0,:],reward,nextObservation,done,nTrajSteps/timeStepLimit)  
                observation[0,:]=nextObservation
                trajectory.append(e)
                if done or (step==nSteps-1):
                    #this trajectory done, copy it to the output buffer
                    trajectoryWrapper.trajectories.append(trajectory)
                    #if in episodic mode, only run one episode per thread
                    if nEpisodes is not None:
                        break
                    #start forming a new trajectory
                    observation[0,:]=sims[simIdx].reset()
                    done=False
                    trajectory=[]
                    nTrajSteps=0

            #propagate value estimates used for training the critic
            trajectories=trajectoryWrapper.trajectories
            for t in range(len(trajectories)):
                trajectory=trajectories[t]
                #e=trajectory[-1]
                ##if trajectory did not end in termination, query the critic to estimate final experience point value
                #In practice this is never run, as the environments are time-limited, and if this
                #is used at the time step limit, training seems to be unstable especially for very short runs like the reacher.
                #Hence, this code has been commented out and we use the critic time step augmentation 
                #if (not e.terminated) and useCritic:
                #    finalStateCriticValue=critic.predict(sess,augmentCriticObs(np.reshape(e.s_next,[1,stateDim]),len(trajectory)))
                #    finalStateCriticValue=finalStateCriticValue[0]
                #    e.V=e.r+gamma*finalStateCriticValue

                #propagate backwards along trajectory
                for i in reversed(range(len(trajectory)-1)):
                    #value estimates, used for training the critic and estimating advantages
                    trajectory[i].V=trajectory[i].r+gamma*trajectory[i+1].V
                    #non-discounted episode return for benchmarking
                    trajectory[i].ret=trajectory[i].r+trajectory[i+1].ret


        iter=0
        #pp.figure(1)
        ####Experience gathering:
        #Loop over training iterations
        while totalSimSteps<trainingEndsAt:
            ####Collect experience        
            #First, simulate rollouts in separate threads, rendering the first episode of each iteration
            threads=[]
            print("Launching {} threads, {} gathering up to experience samples each...".format(nThreads,maxTimeStepsPerThread))

            if linearSdAnneal:
                policy.setGlobalStdev(0.5*max([0,1-totalSimSteps/maxSimSteps]),sess)
                #We want decay^maxSimSteps=finalSd  =>  decay=exp(log(finalSd)/log(totalSimSteps))
                #finalSd=0.1
                #decay=np.exp(np.log(finalSd)/np.log(maxSimSteps))
                #policy.setGlobalStdev(0.5*np.power(decay,totalSimSteps),sess)

            #Each thread will generate a list of trajectories. To be able to use the threading.Thread class, we need to 
            #wrap the lists in TrajectoryWrapper objects
            trajectoryWrappers=[]
            for threadIdx in range(nThreads):
                trajectoryWrapper=TrajectoryWrapper()
                trajectoryWrappers.append(trajectoryWrapper)
                if useThreads:   
                    thread=threading.Thread(target=doRollouts,args=[threadIdx,trajectoryWrapper,iter>0,False,scale,offset])
                    threads.append(thread)
                    thread.start()
                else:
                    doRollouts(threadIdx,trajectoryWrapper,useCritic=iter>0,render=threadIdx==0,scale=scale,offset=offset)

            #wait for threads to finish
            if useThreads:
                for t in threads:
                    t.join()
            

            #Collect trajectories to one list from the wrappers
            trajectories=[]
            for w in trajectoryWrappers:
                for t in w.trajectories: 
                    trajectories.append(t)
        
            #Collect all experience into one list, also some bookkeeping
            averageEpisodeLength=0
            experience=[]
            for episode in range(len(trajectories)):
                t=trajectories[episode]
                averageEpisodeLength+=len(t)/len(trajectories)
                episodeSimSteps=len(t)*actionRepeat
                totalSimSteps+=episodeSimSteps
                for e in t:
                    experience.append(e)
                if episode % 10 == 0 and verbose:
                    print("Episode {}.{}, return {}, terminated {}, nSim (total) {}".format(iter,episode,t[0].ret,t[-1].terminated,totalSimSteps))
            print("Total experience gathered:",len(experience))

            #Plot results, optionally also baseline
            averageEpisodeReward=0
            for t in trajectories:
                #-1 because have to follow the OpenAI way
                averageEpisodeReward+=t[0].ret/len(trajectories)
            benchmarkResult=averageEpisodeReward/rewardScale 
            if benchmarkData.shape[1]<iter+1:
                benchmarkData=np.concatenate([benchmarkData,np.inf*np.ones([nBenchmarkRuns,1,3])],axis=1)
            benchmarkData[benchmarkRunIdx,iter,0]=totalSimSteps
            benchmarkData[benchmarkRunIdx,iter,1]=benchmarkResult
            benchmarkData[benchmarkRunIdx,iter,2]=policy.usedSigmaSum/policy.usedSigmaSumCounter
            pp.figure(1)
            pp.clf()
            if baseline is not None:
                #loop over how many runs in the baseline
                for i in range(baseline.shape[0]):
                    #find how many valid entries
                    for j in range(baseline.shape[1]):
                        if baseline[i,j,1]==np.inf:
                            break;
                    pp.plot(baseline[i,:j,0],baseline[i,:j,1],color='gray')
            for i in range(benchmarkRunIdx+1):
                #find how many valid entries
                for j in range(benchmarkData.shape[1]):
                    if benchmarkData[i,j,1]==np.inf:
                        break;
                pp.plot(benchmarkData[i,:j,0],benchmarkData[i,:j,1],color='black')
            pp.ylabel("Avg episode reward",color='black')
            pp.xlabel("Experience (simulation steps)")
            pp.title(saveFileName)
            ax=pp.gca()
            ax=ax.twinx()
            ax.set_ylabel("Avg policy sigma",color='blue')
            for i in range(benchmarkRunIdx+1):
                #find how many valid entries
                for j in range(benchmarkData.shape[1]):
                    if benchmarkData[i,j,1]==np.inf:
                        break;
                ax.plot(benchmarkData[i,:j,0],benchmarkData[i,:j,2],color='blue')
            ax.set_ylim(0,0.5*(actionMaxLimit-actionMinLimit))
            pp.draw()
            if plotProgress:
                pp.pause(0.001)
        

            #Collect all data into linear arrays for training
            nData=len(experience)
            allStates=np.zeros([nData,stateDim])
            allActions=np.zeros([nData,actionDim])
            allValues=np.zeros([nData])
            allTimes=np.zeros([nData,1])
            for k in range(nData):
                e=experience[k]
                allStates[k,:]=e.s
                allValues[k]=e.V  
                allActions[k,:]=e.a
                allTimes[k,0]=e.timeStep

            #Update scalers
            #Commented out, as it seems it's enough and more stable to only update them in the beginning
            if useScaler:
                scaler.update(allStates)
            #    rewardScaler.update(allRewards)
                scale, offset = scaler.get()
            #    rScale,rOffset=rewardScaler.get()
            #    print("Scaler scale {}, offset {} ".format(scale,offset))
            #    print("Reward scaler scale {}, offset {} ".format(rScale,rOffset))
 
            #Scale the observations for training
            scaledStates=(allStates-offset)*scale

            #Train critic
            critic.train(sess,augmentCriticObs(scaledStates,allTimes),allValues,minibatchSize,nEpochs=nEpochsPerIter,nBatches=nBatchesPerIter)

            #Policy training needs advantages, which depend on the critic we just trained.
            #We use Generalized Advantage Estimation by Schulman et al.
            print("Estimating advantages...".format(len(trajectories)))
            for t in trajectories:
                #query the critic values of all states in one big batch
                nSteps=len(t)
                states=np.zeros([nSteps+1,stateDim])
                timeSteps=np.zeros([nSteps+1,1])
                for i in range(nSteps):
                    states[i,:]=t[i].s
                    timeSteps[i,0]=t[i].timeStep
                states[nSteps,:]=t[nSteps-1].s_next
                states=(states-offset)*scale
                values=critic.predict(sess,augmentCriticObs(states,timeSteps))

                #if trajectory ends in termination, critic should not be used for the final state
                #Or maybe it should? For terminal states, we are anyway training the critic without the recursive lookup
                #if t[nSteps-1].terminated:
                #    values[nSteps]=0

                #GAE loop, i.e., take the instantaneous advantage (how much value a single action brings, assuming that the
                #values given by the critic are unbiased), and smooth those along the trajectory using 1st-order IIR filter.
                for step in reversed(range(nSteps-1)):
                    delta_t=t[step].r+gamma*values[step+1] - values[step]
                    t[step].advantage=delta_t+GAELambda*gamma*t[step+1].advantage

            #Gather the advantages to linear array and apply ReLU and normalization if needed
            nData=len(experience)
            allAdvantages=np.zeros([nData])
            for k in range(nData):
                allAdvantages[k]=experience[k].advantage  
            if reluAdvantages:
                allAdvantages=np.clip(allAdvantages,0,np.inf)
            if normalizeAdvantages:
                aMean=np.mean(allAdvantages)
                aSd=np.std(allAdvantages)
                print("Advantage mean {}, sd{}".format(aMean,aSd))
                allAdvantages/=1e-10+aSd

            #Train policy
            policy.train(sess,allStates,allActions,allAdvantages,minibatchSize,nEpochs=nEpochsPerIter,nBatches=nBatchesPerIter,stateOffset=offset,stateScale=scale)

            #Finally, increment iteration counter
            iter+=1
            if saveFileName is not None:
                np.save(saveFileName,benchmarkData)
                pp.savefig(saveFileName+".png")

            print("Training run {}, iteration {}, mean episode reward {} after {} experience".format(benchmarkRunIdx,iter,benchmarkResult,totalSimSteps))

    print("All done!")






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using PPO or PPO-CMA'))
    parser.add_argument('envName', type=str, help='OpenAI Gym environment name')
    parser.add_argument('--saveFileName', type=str, help='results file name', default=None)
    parser.add_argument('-m','--mode', type=str, help='optimization mode, one of: PPO, PPO-CMA, PG, PG-POS',default="PPO-CMA")
    parser.add_argument('-b','--baseline', type=str, help='baseline results file name for plotting',default=None)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.99)
    parser.add_argument('-l', '--GAELambda', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.95)
    parser.add_argument('-H', '--nHistory',type=int, help='number of iterations to use for training variance',default=3)
    parser.add_argument('--minSd',type=float, help='Lower clipping limit for action stdev', default=0.01)
    parser.add_argument('-K','--nEpochsPerIter',type=int, help='Number of epochs to train for each iteration', default=20)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--plotProgress', action='store_true')
    parser.add_argument('--useScaler', action='store_true')
    parser.add_argument('--scale', type=float, default=1) #many MuJoCo envs have large observations, better to scale down by default.
    parser.add_argument('--rewardScale', type=float, default=1) 
    parser.add_argument('--entropyLossWeight', type=float, help='weight of PPO entropy regularization loss term', default=0) 
    parser.add_argument('--globalVar', action='store_true')
    parser.add_argument('--linearSdAnneal', action='store_true')
    parser.add_argument('--maxSimSteps', type=int, help='train until this many simulation steps in total',default=1000000)
    parser.add_argument('--actionRepeat', type=int, help='how many simulation steps to repeat each action',default=2)
    parser.add_argument('--iterSimBudget', type=int, help='iteration simulation budget',default=16000)
    parser.add_argument('--nBenchmarkRuns', type=int, help='repeat training this many times',default=20)
    parser.add_argument('--networkWidth', type=int, help='how many neurons per layer',default=128)
    parser.add_argument('--L2Critic', action='store_true')
    parser.add_argument('--nEpisodes', type=int, help='if specified, run this many episodes per iteration instead of until iterSimBudget reached',default=None)

    args = parser.parse_args()
    main(**vars(args))
