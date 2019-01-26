"""
A general script for training using OpenAI Gym and Roboschool environments
"""

try:
    import roboschool
except ImportError:
    print('Failed to import Roboschool, ignored.')
import gym
import tensorflow as tf
from Agent import Agent
from Agent import Experience
import argparse
import logger

def main(env_name, mode, learning_rate, ppo_epsilon, ppo_ent_l_w, max_steps, iter_steps, render, batch_size
         , history_buffer_size, n_updates, verbose, run_suffix):
    suffix = '%s-%s-batch_size=%d,iter_steps=%d' % (mode, env_name, batch_size, iter_steps)
    if mode == "PPO":
        suffix += '-epsilon=%.3f-ppo_ent_l_w=%.2f' % (ppo_epsilon, ppo_ent_l_w)
    else:
        suffix += '-H=%d' % history_buffer_size

    print('Starting run for the settings %s' % suffix)

    logger.configure(dir='%s-%s' % (suffix, run_suffix))

    # Init tensorflow
    sess = tf.InteractiveSession()

    # Create environment
    sim = gym.make(env_name)

    # Create the agent
    agent=Agent(
        mode=mode
        , stateDim=sim.observation_space.low.shape[0]
        , actionDim=sim.action_space.low.shape[0]
        , actionMin=sim.action_space.low
        , actionMax=sim.action_space.high
        , learningRate=learning_rate
        , PPOepsilon=ppo_epsilon
        , PPOentropyLossWeight=ppo_ent_l_w
        , H=history_buffer_size
        , useScaler=True # This makes the agent to try to normalize the scale state observations.
    )

    # Finalize initialization
    tf.global_variables_initializer().run(session=sess)
    # print("Initializing agent")
    agent.init(sess)  # Should only be called after the global variables initializer above

    # How many simulation steps to use the same action (larger values than 1 seem to help in MuJoCo agent exploration)
    actionRepeat = 2

    # Main training loop
    totalSimSteps = 0
    nextObservation = None

    iteration = 0
    while totalSimSteps < max_steps:
        #Counter for total simulation steps taken in this iteration
        nSimSteps = 0

        # A list to hold the experience trajectories
        trajectories = []

        #run episodes until budget runs out, computing the average episode reward
        nEpisodes = 0
        averageEpisodeReward = 0
        # print("Collecting experience...")
        while nSimSteps < iter_steps:
            # Reset the episode
            observation = sim.reset()
            done = False
            episodeReward = 0

            # List to hold the experience of this episode
            trajectory = []

            # Simulate this episode until done
            while not done:
                # Query the agent for action
                action = agent.act(sess,observation)

                # Simulate using the action, repeating the same action for actionRepeat steps.
                # Also, compute the total reward received.
                reward = 0
                for _ in range(actionRepeat):
                    nextObservation, stepReward, done, info = sim.step(action[0, :])

                    # Uncomment the following two lines to enable rendering
                    if render and nEpisodes < 5:  # Only render the first few episodes of each iteration
                        sim.render()

                    nSimSteps += 1
                    totalSimSteps += 1
                    reward += stepReward
                    episodeReward += stepReward
                    if done:
                        break

                # Save the experience point
                e = Experience(observation, action, reward, nextObservation, done)
                trajectory.append(e)
                observation = nextObservation

            # Episode done, bookkeeping
            trajectories.append(trajectory)
            averageEpisodeReward += episodeReward
            nEpisodes += 1

        #All episodes of this iteration done, print results and update the agent
        averageEpisodeReward /= nEpisodes
        iteration += 1
        print('================ Iteration %d ================' % iteration)
        logger.record_tabular("Total iterations", iteration)
        logger.record_tabular("Total timesteps", totalSimSteps)
        logger.record_tabular("Episode reward mean", averageEpisodeReward)
        logger.record_tabular("Average policy std", agent.getAverageActionStdev())
        logger.dump_tabular()
        agent.update(sess, trajectories, batchSize=batch_size, nBatches=n_updates, verbose=verbose)
    sess.close()
    print('Finished run for the settings %s' % suffix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym, or Roboschool environments '
                                                  'using PPO, PPO-CMA, or PPO-CMA-m'))
    parser.add_argument('--env_name', type=str, help='OpenAI Gym or Roboschool environment name'
                        , default="MountainCarContinuous-v0")
    parser.add_argument('-m', '--mode', type=str, help='Optimization mode, one of: PPO, PPO-CMA, or PPO-CMA-m'
                        , default="PPO-CMA-m")
    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=5e-4)
    parser.add_argument('--ppo_epsilon', type=float, help='PPO epsilon', default=0.2)
    parser.add_argument('--ppo_ent_l_w', type=float, help='PPO entropy loss weight', default=0)
    parser.add_argument('--max_steps', type=int, help='Maximum timesteps', default=int(1e6))
    parser.add_argument('--iter_steps', type=int, help='Number of timesteps per iteration', default=int(4000))
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, help='Optimization batch size', default=int(2048))
    parser.add_argument('--history_buffer_size', type=int, help='PPO-CMA-m history buffer size', default=int(9))
    parser.add_argument('--n_updates', type=int, help='Number of updates per iteration', default=int(100))
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--run_suffix', type=str, help='Name suffix of the save directory', default="")

    args = parser.parse_args()
    main(**vars(args))
