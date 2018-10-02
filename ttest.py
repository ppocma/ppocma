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
import scipy.stats


decimate=100
keepBestSoFar=False  #if True, we plot the best training iteration in each training run so far. This is reasonable, as one could save the network after each iteration and use the one with highest reward
episodicMode=False


def loadNpyResultsAsContinuous(fileName,maxSteps:int):
    #print("Loading and converting ",fileName)
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
    #print("Loading and converting csv:s from "+folderName)
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


def printTTestLine(envName, ppocmaResult: np.array, ppoResult:np.array):
    tstats,pvalue=scipy.stats.ttest_ind(ppocmaResult,ppoResult,equal_var=False)
    ppocmaMean=np.mean(ppocmaResult)
    sppocmaMean="{0:.1f}".format(ppocmaMean)
    ppoMean=np.mean(ppoResult)
    sppoMean="{0:.1f}".format(ppoMean)
    spvalue="{0:.4f}".format(pvalue)
    if pvalue>0.05:
        print("{} & {} & {} & {} \\\\".format(envName,sppocmaMean,sppoMean,spvalue))
    else:
        if ppocmaMean>ppoMean:
            print(envName," & \\textbf{",sppocmaMean,"} & ",sppoMean, " & \\textbf{",spvalue,"} \\\\")
        else:
            print(envName," & ",sppocmaMean," & \\textbf{",sppoMean,"} & \\textbf{",spvalue,"} \\\\")

def genTestResult(envName,maxSteps):
    if envName=="Humanoid-v2":
        ppocmaResult=loadNpyResultsAsContinuous('results/Humanoid-v2_PPO-CMA_useScaler_H=5_N=32000.npy',maxSteps)
    else:
        ppocmaResult=loadNpyResultsAsContinuous('results/{}_PPO-CMA_useScaler_H=3_N=16000_networkWidth=128.npy'.format(envName),maxSteps)
    ppoResult=loadCsvsAsContinuous('OpenAIBaselineResults_ppo1\{}'.format(envName),maxSteps)
    ppocmaResult=ppocmaResult[:,-1]
    ppoResult=ppoResult[:,-1]
    printTTestLine(envName,ppocmaResult,ppoResult)



#check the randomized training results 
ppoData = np.genfromtxt("results/Walker2d-v2_PPO_randomized.log", dtype=float, delimiter=',', names=True)
ppocmaData = np.genfromtxt("results/Walker2d-v2_PPO-CMA_randomized.log", dtype=float, delimiter=',', names=True)
ppoData=ppoData['bestSoFar']
ppocmaData = ppocmaData['bestSoFar']
ppoData=ppoData[:50]
ppocmaData=ppocmaData[:50]

print("")
print("Randomized hyperparameter testing results:")
print("PPO-CMA mean {}, sd {}".format(np.mean(ppocmaData),np.std(ppocmaData)))
print("PPO mean {}, sd {}".format(np.mean(ppoData),np.std(ppoData)))
print("PPO-CMA wins {} times out of {}".format(np.count_nonzero(np.maximum(0,ppocmaData-ppoData)),ppocmaData.shape[0]))
tstats,pvalue=scipy.stats.ttest_ind(ppocmaData,ppoData,equal_var=False)
print("Welch t-test p-value: ",pvalue)
#data=np.stack([data['TimestepsSoFar'],data['EpRewMean']],axis=1)
print("")

#print out a LaTeX table with t-tests 
print("T-test table:")
print("")
print("\\begin{center}")
print("\\begin{tabular}{| c | c | c | c |}")
print("\\hline")
print("Environment & PPO-CMA & PPO (OpenAI baseline) & p-value \\\\" )
print("\\hline")
genTestResult("Hopper-v2",1e6)
genTestResult("Walker2d-v2",1e6)
genTestResult("HalfCheetah-v2",1e6)
genTestResult("Reacher-v2",1e6)
genTestResult("Swimmer-v2",1e6)
genTestResult("InvertedPendulum-v2",1e6)
genTestResult("InvertedDoublePendulum-v2",1e6)
genTestResult("Humanoid-v2",1e7)
print("\\hline")
print("\\end{tabular}")       
print("\\end{center}")       
