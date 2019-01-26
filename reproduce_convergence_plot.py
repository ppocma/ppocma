"""
This script reproduces Figure 8 with the following caption in the paper submitted to ICML 2019:
Training curves from the 9 Roboschool environments used in the hyperparameter search. The plots use the best hyperparameter combinations in Figure 4.
NOTE: This script needs Roboschool to be installed.
NOTE: Final plot is saved in a file called "ConvergencePlot.png".
"""

from matplotlib import pyplot as plt
import numpy as np
from matplotlib import rc
import os
from Run import main

def find_indices(col_names):
	indices = {}
	for i in range(len(col_names)):
		indices[col_names[i]] = i
	return indices

def Reproduce_Results():
	env_list = [
		'RoboschoolInvertedPendulum-v1'
		, 'RoboschoolInvertedPendulumSwingup-v1'
		, 'RoboschoolInvertedDoublePendulum-v1'
		, 'RoboschoolReacher-v1'
		, 'RoboschoolHopper-v1'
		, 'RoboschoolWalker2d-v1'
		, 'RoboschoolHalfCheetah-v1'
		, 'RoboschoolAnt-v1'
		, 'RoboschoolPong-v1'
	]
	n_envs = len(env_list)
	
	n_runs = 5
	max_steps = int(1e6)

	for env in env_list:
		for alg in range(2):
			running_ppo = alg == 0
			for r in range(n_runs):
				main(
					env_name=env
					, mode="PPO" if running_ppo else "PPO-CMA-m"
					, learning_rate=5e-4
					, ppo_epsilon=0.005
					, ppo_ent_l_w=0
					, max_steps=max_steps
					, iter_steps=2000 if running_ppo else 8000
					, render=False
					, batch_size=128 if running_ppo else 512
					, history_buffer_size=9
					, n_updates=100
					, verbose=False
					, run_suffix=str(r + 1)
				)

	rc('figure', figsize=(15, 15))
	rc('text', usetex=False)
	fig = plt.figure()

	for e in range(n_envs):
		ax = fig.add_subplot(3, 3, e + 1)
		ax.set_title(env_list[e])
		ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='x')
		if e % 3 == 0:
			ax.set_ylabel('Average episode return', fontsize=14)
		if e >= 6:
			ax.set_xlabel('Simulation steps', fontsize=14)
		s_env = env_list[e]
		for alg in reversed(range(2)):
			running_ppo = alg == 0
			s_alg = 'PPO' if running_ppo else 'PPO-CMA-m'
			file_str = os.path.join(os.getcwd(), 'Results', s_alg + '-' + s_env + '-')
			file_str += 'batch_size=%d,' % (128 if running_ppo else 512)
			file_str += 'iter_steps=%d-' % (2000 if running_ppo else 8000)
			if running_ppo:
				file_str += 'epsilon=0.005-ppo_ent_l_w=0.00-'
			else:
				file_str += 'H=9-'
			all_rewards_np = np.zeros((n_runs, max_steps))
			for r in range(n_runs):
				print(file_str + str(r + 1))
				path = os.path.join(file_str + str(r + 1), 'progress.csv')
				col_names = np.genfromtxt(path, max_rows=1, delimiter=',', dtype=str)
				# 'Total iterations, Total timesteps, Average policy std, Episode reward mean'
				col_indices = find_indices(col_names)
				new_data = np.genfromtxt(path, skip_header=1, delimiter=',', dtype=np.float32)
				rews = new_data[:, col_indices['Episode reward mean']]
				steps = new_data[:, col_indices['Total timesteps']]
				last_step = int(0)
				for it in range(rews.shape[0]):
					new_step = min(int(steps[it]), max_steps)
					all_rewards_np[r, last_step:new_step] = rews[it]
					last_step = new_step
					if last_step > max_steps:
						break
			all_rewards_mean = np.mean(all_rewards_np, axis=0)
			all_rewards_std = np.std(all_rewards_np, axis=0)
			steps = np.arange(0, max_steps)
			if alg == 0:
				color = 'r'
			else:
				color = 'g'
			ax.plot(steps, all_rewards_mean, color=color, alpha=0.9, label='PPO' if alg == 0 else 'PPO-CMA')
			ax.fill_between(steps, all_rewards_mean - all_rewards_std / 2, all_rewards_mean + all_rewards_std / 2
							, color=color, alpha=0.25)
		leg = ax.legend(loc='lower right')
		for line in leg.get_lines():
			line.set_linewidth(5)

	fig.savefig(os.path.join(os.getcwd(), 'ConvergencePlot.png'), bbox_inches='tight', pad_inches=0, dpi=200)
	# plt.show()


if __name__ == '__main__':
	Reproduce_Results()
