import numpy as np
import matplotlib.pyplot as pp
from Agent import Agent
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #disable Tensorflow GPU usage, these simple graphs run faster on CPU
import tensorflow as tf

#Initialize some parameters. We will visualize 4 algorithms, using 64 actions per iteration
nMinibatch=64
fontSize=12
plotIters=[1,5,9,20]
nIter=plotIters[-1]
nModes=4
modes=["PG","PG-pos","PPO","PPO-CMA-m"]
modeNames=["Policy Gradient","Policy Gradient \n (only pos. advantages)","PPO","PPO-CMA (ours)"]
pp.figure(1,figsize=[len(plotIters)*2,nModes*2],tight_layout=True)

#Loop over the visualized algorithms
for mode in range(nModes):
    #create policy
    tf.reset_default_graph()
    print("Initializing Tensorflow")
    sess=tf.Session()
    print("Creating agent")

    #here, we don't really have a state, but the code still depends on state tensors => use a dummy constant state with dimensionality 1
    stateDim=1 
    dummyState=np.zeros([nMinibatch,stateDim])

    #2D actions for easy visualization
    actionDim=2

    #Create and initialize agent
    agent=Agent(stateDim,actionDim,-2*np.ones(actionDim),2*np.ones(actionDim),
                mode=modes[mode],gamma=0,
                initialMean=-1*np.ones(actionDim),
                initialSd=0.25*np.ones(actionDim),
                H=5) #in this simple problem, we may use a smaller H
    tf.global_variables_initializer().run(session=sess)
    agent.init(sess)  # must be called after TensorFlow global variables init

    #always use the same random seed so that all algorithms start from the same initial action distribution
    np.random.seed(0)   

    #loop over training iterations
    for iter in range(nIter):
        print("Iter {}".format(iter))
        #query actions
        actions=agent.act(sess,dummyState)
        #compute rewards
        rewards=-np.sum(np.square(actions),axis=1)
        #make the agent memorize the episodes, each episode with just one action
        for idx in range(actions.shape[0]):
            agent.memorize(dummyState[idx],actions[idx,:],rewards[idx],dummyState[idx],True)
        #update agent (trains the value function predictor and policy networks)
        agent.updateWithMemorized(sess)

        #visualize
        if (iter+1) in plotIters:
            nCols=len(plotIters)
            plotIdx=plotIters.index(iter+1)
            pp.subplot(nModes,nCols,plotIdx+1+mode*nCols)
            pp.cla()
            if mode==0:
                pp.title("Iteration {}".format(iter+1),fontsize=fontSize)
            if iter==0:
                pp.ylabel(modeNames[mode],fontsize=fontSize)
            ax = pp.gca()
            for i in range(6):
                circle1 = pp.Circle([0,0],i*0.5, fill=False, color='black')
                ax.add_artist(circle1)
            for i in range(nMinibatch):
                pp.scatter(actions[i,0],actions[i,1],color='b',marker='.') #color='g' if advantages[i]>0 else 'r')
            pp.xlim(-2,1)
            pp.ylim(-2,1)
            pp.draw()
            pp.pause(0.001)


# pp.savefig("paper-icml/images/teaser_ICML.png",dpi=200)
pp.show()
