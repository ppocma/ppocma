import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as pp
import policy as pl
from policy import Policy
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #disable Tensorflow GPU usage, these simple graphs run faster on CPU
import tensorflow as tf



def computeQ(actions:np.array):
    return -np.sum(np.square(actions),axis=1)


nMinibatch=64
nIter=20
nEpochs=2000
initialSigma=0.25
pl.learningRate=0.01
pl.nHistory=3
pl.minSigma=0.01
plotSkip=2


nModes=2
for mode in range(nModes):
    if mode==0:
        #PPO 
        pl.separateVarAdapt=False
        pl.usePPOLoss=True
        pl.trainLogVar=True
        pl.PPOepsilon=0.2
        title="PPO"
    elif mode==1:
        #CMA-ES style
        pl.separateVarAdapt=True
        pl.usePPOLoss=False
        pl.useEvolutionPath=True
        title="PPO-CMA (ours)"
    #create policy
    tf.reset_default_graph()
    print("Initializing Tensorflow")
    sess=tf.Session()
    print("Creating policy network")
    stateDim=0 #In this stateless test, we use 0-dimensional state, in which case the policy network actually consists of directly trained TF variables instead of interconnected neurons.
                #For testing that everything works with real neural networks too (slower), change stateDim
    actionDim=2
    policy=Policy(stateDim,actionDim,-2*np.ones(actionDim),2*np.ones(actionDim))
    dummyState=np.zeros([nMinibatch,stateDim])

    #initialize variables
    print("Initializing Tensorflow globals")
    tf.global_variables_initializer().run(session=sess)

    #init policy to produce gaussian samples with mean -2,-2 and sd 1  (this test's optimum at 0,0, want to see how the mean and variance adapt)
    print("Initializing policy")
    policy.init(sess,stateMean=0,stateSd=1,actionMean=-1*np.ones([2]),actionSd=initialSigma*np.ones([2]),nMinibatch=256,nBatches=8000)
    for iter in range(nIter):
        nSubiter=1
        for subIter in range(nSubiter):
            print("Iter {}, subiter {}".format(iter,subIter))
            #sample actions
            actions=policy.sample(sess,dummyState)

            #compute Q,V, advantage
            Q=computeQ(actions)
            V=np.mean(Q)
            #best=np.argmax(Q)
            advantages=Q-V
            advantageMean=np.mean(advantages)
            advantages/=np.std(advantages)+1e-10  #normalize advantages

            #visualize
            if iter==nIter-1 or ((iter % plotSkip==0) and iter<9):
                pp.figure(1,figsize=[14,4],tight_layout=True)
                nCols=7
                pp.subplot(nModes,nCols,min([(iter//plotSkip)+1,nCols])+mode*nCols)
                pp.cla()
                if mode==0:
                    pp.title("Iteration {}".format(iter+1))
                if iter==0:
                    pp.ylabel(title)
                ax = pp.gca()
                for i in range(6):
                    circle1 = pp.Circle([0,0],i*0.5, fill=False, color='black')
                    ax.add_artist(circle1)
                for i in range(nMinibatch):
                    pp.scatter(actions[i,0],actions[i,1],color='b',marker='.') #color='g' if advantages[i]>0 else 'r')
                    #if advantages[i]==0:
                    #    actions[i,:]=0
                    #if advantages[i]>0:
                    #    print("Action {} advantage {}".format(actions[i,:],advantages[i]))
                pp.xlim(-2,2)
                pp.ylim(-2,2)
                #means=policy.getExpectation(sess,dummyState)
                #pp.scatter(means[0],means[1],color='black')        
                pp.draw()
                pp.pause(0.001)
            #pp.show()

            #update policy      
            #policy.train(sess,dummyState,np.zeros([nMinibatch,actionDim]),advantages,nMinibatch=nMinibatch,nEpochs=4000)
        
            policy.train(sess,dummyState,actions,advantages,nMinibatch=nMinibatch,nEpochs=nEpochs)
            #advantages=np.zeros(nMinibatch)
            #advantages[0]=nMinibatch
            #policy.train(sess,dummyState,np.zeros([nMinibatch,actionDim]),advantages,nMinibatch=nMinibatch,nEpochs=4000)

pp.show()
