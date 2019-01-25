"""
A tutorial example of how to use PPO-CMA
"""

import gym
import tensorflow as tf
from Agent import Agent

#Simulation budget (steps) per iteration. This is the main parameter to tune.
#8k works for relatively simple environments like the OpenAI Gym Roboschool 2D Hopper.
#For more complex problems such as 3D humanoid locomotion, try 32k or even 64k.
#Larger values are slower but more robust.
N=2000

# Init tensorflow
sess = tf.InteractiveSession()

# Create environment (replace this with your own simulator)
print("Creating simulation environment")
sim = gym.make("MountainCarContinuous-v0")

# Create the agent
agent=Agent(
    stateDim=sim.observation_space.low.shape[0]
    , actionDim=sim.action_space.low.shape[0]
    , actionMin=sim.action_space.low
    , actionMax=sim.action_space.high
)

# Finalize initialization
tf.global_variables_initializer().run(session=sess)
agent.init(sess)  # must be called after TensorFlow global variables init

# How many simulation steps we've taken in total
totalSimSteps = 0

# Stop training after this many steps
max_steps=1000000

# Main training loop
while totalSimSteps < max_steps:
    #Counter for total simulation steps taken in this iteration
    episodeSimSteps = 0

    #Run episodes until the iteration simulation budget runs out
    while episodeSimSteps < N:
        # Reset the simulation 
        observation = sim.reset()

        # Simulate this episode until done
        while True:
            # Query the agent for action given the state observation
            action = agent.act(sess,observation)

            # Simulate using the action
            # Note: this tutorial does not repeat the same action for two steps, 
            # unlike the Run.py script used for the ICML paper results.
            # Repeating the action for multiple steps seems to yield better exploration in 
            # most cases, possibly because it reduces high-frequency action noise.
            nextObservation, reward, done, info = sim.step(action[0, :])

            # Save the experience point
            agent.memorize(observation,action,reward,nextObservation,done)
            observation=nextObservation

            # Bookkeeping
            episodeSimSteps += 1

            # Episode terminated? (e.g., due to time limit or failure)
            if done:
                break

    #All episodes of this iteration done, print results and update the agent
    totalSimSteps += episodeSimSteps
    averageEpisodeReturn=agent.updateWithMemorized(sess,verbose=False)
    print("Simulation steps {}, average episode return {}".format(totalSimSteps,averageEpisodeReturn))
