Source code for the paper PPO-CMA: Proximal Policy Optimization with Covariance Matrix Adaptation

https://openreview.net/forum?id=B1VWtsA5tQ

Note that the code has been through only a partial clean-up. More polish, comments and refactoring will follow in the near future. 


Reproducing the results:

1. Install OpenAI Gym and MuJoCo 

2. Run TeaserFigure.py to test that the install works and generate the paper teaser image

3. Run train_xx.bat batch files to regenerate results (will take some days, only tested on Windows 10)

4. Run PlotMuJoCoBenchmarks2.py to visualize results using the data in the results folder (note that the results folder also contains some unused data from old algorithm and code versions)

5. Run ttest.py to compute and print out all statistical significance tests


We've also included existing results data, i.e., you may skip step 3 if you only want to explore the data from our simulations.


Code structure:

train.py 	The main file, see command line arguments
policy.py	The policy network
critic.py	The value function predictor network
MLP.py		Neural network helper class
utils.py	The automatic observation scaler

