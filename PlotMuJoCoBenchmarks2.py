import numpy as np
import math
import random
from collections import deque
import matplotlib
import matplotlib.pyplot as pp
import matplotlib.patches as mpatches
from os import listdir
from os import walk
from os.path import isfile

decimate=100
keepBestSoFar=False  #if True, we plot the best training iteration in each training run so far. This is reasonable, as one could save the network after each iteration and use the one with highest reward
episodicMode=False


def loadNpyResultsAsContinuous(fileName,maxSteps:int):
    print("Loading and converting ",fileName)
    data=np.load(fileName)
    nRuns=0
    while nRuns<data.shape[0] and data[nRuns,0,1]!=np.inf:
        nRuns+=1
    continuousData=np.zeros([nRuns,int(maxSteps)])
    #loop over how many runs in the baseline
    for i in range(nRuns):
        currentStep=0
        currentReward=-1e20
        #find how many valid entries
        nIter=data.shape[1]
        for j in range(nIter):
            entryStep=data[i,j,0]
            entryStep=min([entryStep,maxSteps])
            entryStep=int(entryStep)
            entryReward=data[i,j,1]
            if entryReward!=np.inf:
                currentReward=max([currentReward,entryReward]) if keepBestSoFar else entryReward
            continuousData[i,currentStep:entryStep]=currentReward
            currentStep=entryStep
        if currentStep<maxSteps:
            continuousData[i,currentStep:]=currentReward
    return continuousData


def loadCsvsAsContinuous(folderName,maxSteps:int):
    print("Loading and converting csv:s from "+folderName)
    csvFileNames=[]
    for (dirpath, dirnames, filenames) in walk(folderName):
        f=dirpath+"/progress.csv"
        if isfile(f):
            csvFileNames.append(f)

    nRuns=len(csvFileNames)
    nRuns=min([nRuns,10]) #because we only have 10 runs of PPOCMA

    continuousData=np.zeros([nRuns,int(maxSteps)])
    #loop over how many runs in the baseline
    for i in range(nRuns):
        data = np.genfromtxt(csvFileNames[i], dtype=float, delimiter=',', names=True) 
        data=np.stack([data['TimestepsSoFar'],data['EpRewMean']],axis=1)

        #some init
        currentStep=0
        currentReward=-1e20

        #find how many valid entries
        nIter=data.shape[0]
        for j in range(nIter):
            entryStep=data[j,0]
            entryStep=min([entryStep,maxSteps])
            entryStep=int(entryStep)
            entryReward=data[j,1]
            if entryReward!=np.inf:
                currentReward=max([currentReward,entryReward]) if keepBestSoFar else entryReward
            continuousData[i,currentStep:entryStep]=currentReward
            currentStep=entryStep
        if currentStep<maxSteps:
            continuousData[i,currentStep:]=currentReward
    return continuousData

#returns a legend patch
def plotMeanAndSd(fileName,displayName,color,maxSteps:int):
    #load data
    if ".npy" in fileName:
        data=loadNpyResultsAsContinuous(fileName,maxSteps)
    else:
        data=loadCsvsAsContinuous(fileName,maxSteps)
    plotSteps=np.arange(0,maxSteps)

    #decimate for faster plotting
    data=data[:,0::10]
    #print("data shape ",data.shape)
    plotSteps=plotSteps[0::10]
    #print("plotSteps shape",plotSteps.shape)

    #plot
    mean=np.mean(data,axis=0)
    sd=np.std(data,axis=0)
    pp.plot(plotSteps,mean,color=color)
    #pp.plot(plotSteps,np.max(data,axis=0),color=color,linestyle='--')
    #pp.plot(plotSteps,np.min(data,axis=0),color=color,linestyle='--')
    pp.fill_between(plotSteps,mean-sd,mean+sd,color=color,alpha=0.5)
    #add a legend patch for this plot
    patch = mpatches.Patch(color=color, label=displayName)
    pp.xlabel("Experience (simulation steps)")
    pp.ylabel("Avg. episode reward")
    return patch

#Plot multiple in the same (TODO: easier to plot just one...)
def comparisonPlot(fileNames,displayNames,maxSteps:int):
    pp.cla()
    colors=['gray','blue','red','green']
    colorIdx=0
    legendPatches=[]
    for fileName in fileNames:
        legendPatches.append(plotMeanAndSd(fileName,displayNames[colorIdx],colors[colorIdx],maxSteps))
        colorIdx+=1
    pp.legend(handles=legendPatches)
    pp.xlabel("Experience (simulation steps)")
    pp.ylabel("Avg. episode reward")

def resultName(envName):
    if episodicMode:
        return 'results_episodic/{}_PPO-CMA_useScaler_H=7.npy'.format(envName)
    else:
#        return 'results/{}_PPO-CMA_useScaler_H=5_N=16000_networkWidth=128_GAELambda=1.0.npy'.format(envName)
         return 'results/{}_PPO-CMA_useScaler_H=3_N=16000_networkWidth=128.npy'.format(envName)
#        return 'results/doubleInitialSd/{}_PPO-CMA_useScaler_H=5_N=16000_networkWidth=128.npy'.format(envName)
         #if envName=="HalfCheetah-v2" or envName=="Swimmer-v2":
         #   return 'results/{}_PPO-CMA_useScaler_H=5_nEpisodes=16_networkWidth=128.npy'.format(envName)
         #else:
         #   return 'results/{}_PPO-CMA_useScaler_H=5_nEpisodes=64_networkWidth=128.npy'.format(envName)
         #return 'results/{}_PPO-CMA_useScaler_nEpochs=5_H=5_N=4000_networkWidth=128.npy'.format(envName)
#        return 'results/{}_PPO-CMA_useScaler_nEpochs=10_H=5_N=16000_networkWidth=128.npy'.format(envName)

def ppoResultName(envName):
    if episodicMode:
        return 'results_episodic/{}_PPO_useScaler.npy'.format(envName)
    else:
#        return 'results/{}_PPO_useScaler_N=16000_L2Critic_networkWidth=128.npy'.format(envName)
        return 'results/{}_PPO_useScaler_N=16000_networkWidth=128.npy'.format(envName)


ppocmaColor='blue'
ppoBaselineColor='gray'
ppoColor='red'

#Hopper
pp.figure(figsize=[14,6.5])
pp.subplot(2,4,5)
legendPatches=[]
legendPatches.append(plotMeanAndSd('OpenAIBaselineResults_ppo1\Hopper-v2','PPO (OpenAI)',ppoBaselineColor,1e6))
legendPatches.append(plotMeanAndSd(ppoResultName('Hopper-v2'),'PPO',ppoColor,1e6))
legendPatches.append(plotMeanAndSd(resultName('Hopper-v2'),'PPO-CMA (ours)',ppocmaColor,1e6))
pp.legend(handles=list(reversed(legendPatches)))
pp.xticks([0,500000,1000000],['0','500k','1M'])
pp.title("Hopper-v2")


#Walker
pp.subplot(2,4,6)
legendPatches=[]
legendPatches.append(plotMeanAndSd('OpenAIBaselineResults_ppo1\Walker2D-v2','PPO (OpenAI)',ppoBaselineColor,1e6))
legendPatches.append(plotMeanAndSd(ppoResultName('Walker2D-v2'),'PPO',ppoColor,1e6))
legendPatches.append(plotMeanAndSd(resultName('Walker2D-v2'),'PPO-CMA (ours)',ppocmaColor,1e6))

#comparisonPlot(['Walker_PPO_logVar_MLP.npy','results/Walker2D_PPO-CMA.npy','OpenAIBaselineResults_ppo1\Walker2d-v2'],['PPO','PPO-CMA',"PPO (OpenAI)"],1e6)
pp.legend(handles=list(reversed(legendPatches)))
pp.xticks([0,500000,1000000],['0','500k','1M'])
pp.title("Walker2D-v2")

#Half cheetah
pp.subplot(2,4,4)
legendPatches=[]
legendPatches.append(plotMeanAndSd('OpenAIBaselineResults_ppo1\HalfCheetah-v2','PPO (OpenAI)',ppoBaselineColor,1e6))
legendPatches.append(plotMeanAndSd(ppoResultName('HalfCheetah-v2'),'PPO',ppoColor,1e6))
legendPatches.append(plotMeanAndSd(resultName('HalfCheetah-v2'),'PPO-CMA (ours)',ppocmaColor,1e6))
pp.legend(handles=list(reversed(legendPatches)))
pp.xticks([0,500000,1000000],['0','500k','1M'])
pp.title("HalfCheetah-v2")

#Reacher
pp.subplot(2,4,3)
legendPatches=[]
legendPatches.append(plotMeanAndSd('OpenAIBaselineResults_ppo1\Reacher-v2','PPO (OpenAI)',ppoBaselineColor,1e6))
legendPatches.append(plotMeanAndSd(ppoResultName('Reacher-v2'),'PPO',ppoColor,1e6))
legendPatches.append(plotMeanAndSd(resultName('Reacher-v2'),'PPO-CMA (ours)',ppocmaColor,1e6))
pp.legend(handles=list(reversed(legendPatches)))
pp.xticks([0,500000,1000000],['0','500k','1M'])
pp.title("Reacher-v2")

#inverted pendulum
pp.subplot(2,4,7)
legendPatches=[]
legendPatches.append(plotMeanAndSd('OpenAIBaselineResults_ppo1\InvertedPendulum-v2','PPO (OpenAI)',ppoBaselineColor,1e6))
legendPatches.append(plotMeanAndSd(ppoResultName('InvertedPendulum-v2'),'PPO',ppoColor,1e6))
legendPatches.append(plotMeanAndSd(resultName('InvertedPendulum-v2'),'PPO-CMA (ours)',ppocmaColor,1e6))
pp.legend(handles=list(reversed(legendPatches)))
pp.xticks([0,500000,1000000],['0','500k','1M'])
pp.title("InvertedPendulum-v2")

#inverted double pendulum
pp.subplot(2,4,8)
legendPatches=[]
legendPatches.append(plotMeanAndSd('OpenAIBaselineResults_ppo1\InvertedDoublePendulum-v2','PPO (OpenAI)',ppoBaselineColor,1e6))
legendPatches.append(plotMeanAndSd(ppoResultName('InvertedDoublePendulum-v2'),'PPO',ppoColor,1e6))
legendPatches.append(plotMeanAndSd(resultName('InvertedDoublePendulum-v2'),'PPO-CMA (ours)',ppocmaColor,1e6))
pp.legend(handles=list(reversed(legendPatches)))
pp.xticks([0,500000,1000000],['0','500k','1M'])
pp.title("InvertedDoublePendulum-v2")

pp.subplot(2,4,2)
legendPatches=[]
legendPatches.append(plotMeanAndSd('OpenAIBaselineResults_ppo1\Swimmer-v2','PPO (OpenAI)',ppoBaselineColor,1e6))
legendPatches.append(plotMeanAndSd(ppoResultName('Swimmer-v2'),'PPO',ppoColor,1e6))
legendPatches.append(plotMeanAndSd(resultName('Swimmer-v2'),'PPO-CMA (ours)',ppocmaColor,1e6))
pp.legend(handles=list(reversed(legendPatches)))
pp.xticks([0,500000,1000000],['0','500k','1M'])
pp.title("Swimmer-v2")

#humanoid
pp.subplot(2,4,1)
legendPatches=[]
legendPatches.append(plotMeanAndSd('OpenAIBaselineResults_ppo1\Humanoid-v2','PPO (OpenAI)',ppoBaselineColor,1e7))
legendPatches.append(plotMeanAndSd('results/Humanoid-v2_PPO_useScaler_N=32000.npy','PPO',ppoColor,1e7))
legendPatches.append(plotMeanAndSd('results/Humanoid-v2_PPO-CMA_useScaler_H=7_N=32000.npy','PPO-CMA (ours)',ppocmaColor,1e7))
pp.legend(handles=list(reversed(legendPatches)))
pp.xticks([0,5000000,10000000],['0','5M','10M'])
pp.title("Humanoid-v2 (3D)")


#comparisonPlot(['Reacher_PPO_logVar_MLP.npy','Reacher_PPO-CMA_logVar_MLP.npy','OpenAIBaselineResults_ppo1\Reacher-v2'],['PPO','PPO-CMA',"PPO (OpenAI)"],1e6)
#pp.title("Reacher-v2")
#pp.subplot(2,4,7)
#comparisonPlot(['OpenAIBaselineResults_ppo1\Swimmer-v2'],["PPO (OpenAI)"],1e6)
#pp.title("Swimmer-v2")

pp.tight_layout()

pp.draw()
pp.show()

