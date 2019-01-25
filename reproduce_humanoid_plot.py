"""
This script reproduces Figure 5 with the following caption in the paper submitted to ICML 2019:
Comparing PPO and PPO-CMA in the MuJoCo Humanoid-v2 environment, showing means and standard deviations of training curves from 3 runs with different random seeds.
NOTE: To make the code faster, only data series PPO (N = 32k) and PPO-CMA (N = 32k) are reproduced in this script.
NOTE: Final plot is saved in a file called "HumanoidPlot.png".
"""

from matplotlib import pyplot as plt
import numpy as np
from matplotlib import rc
import os
from Run import main

def Reproduce_Results():
	for alg in range(2):
		running_ppo = alg == 0
		for r in range(3):
			main(
				env_name="Humanoid-v2"
				, mode="PPO" if running_ppo else "PPO-CMA-m"
				, learning_rate=5e-4
				, ppo_epsilon=0.0025
				, ppo_ent_l_w=0
				, max_steps=1e7
				, iter_steps=32000
				, render=False
				, batch_size=128 if running_ppo else 512
				, history_buffer_size=9
				, n_updates=100
				, verbose=False
				, run_suffix=str(r + 1)
			)

	rc('figure', figsize=(14, 7))
	rc('font', size=14)
	rc('axes.spines', top=False, right=False)
	rc('axes', grid=False)
	rc('axes', facecolor='white')

	data_series = [
		{
			'title': 'PPO-CMA (N = 32k)',
			'color': 'g',
			'draw': True,
			'dir_list': ['PPO-CMA-m-Humanoid-v2-batch_size=512,iter_steps=32000-H=9-1'
				, 'PPO-CMA-m-Humanoid-v2-batch_size=512,iter_steps=32000-H=9-2'
				, 'PPO-CMA-m-Humanoid-v2-batch_size=512,iter_steps=32000-H=9-3']
		}
		,
		{
			'title': 'PPO (N = 32k)',
			'color': 'm',
			'draw': True,
			'dir_list': ['PPO-Humanoid-v2-batch_size=128,iter_steps=32000-epsilon=0.003-ppo_ent_l_w=0.00-1'
				, 'PPO-Humanoid-v2-batch_size=128,iter_steps=32000-epsilon=0.003-ppo_ent_l_w=0.00-2'
				, 'PPO-Humanoid-v2-batch_size=128,iter_steps=32000-epsilon=0.003-ppo_ent_l_w=0.00-3']
		}
	]

	base_dir = os.path.join(os.getcwd(), 'Results')
	n_runs = 3
	max_steps = int(1e7)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel('Simulation steps')
	ax.set_ylabel('Average episode return')
	ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='x')
	x_steps = np.arange(0, max_steps)

	def find_indices(col_names):
		indices = {}
		for i in range(len(col_names)):
			indices[col_names[i]] = i
		return indices

	for data_pack in data_series:
		if data_pack['draw']:
			color = data_pack['color']
			all_rewards_np = np.zeros((n_runs, max_steps))
			for r in range(n_runs):
				path = os.path.join(base_dir, data_pack['dir_list'][r], 'progress.csv')
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
			ax.plot(x_steps, all_rewards_mean, color=color, label=data_pack['title'])
			ax.fill_between(x_steps, all_rewards_mean - all_rewards_std / 2, all_rewards_mean + all_rewards_std / 2
							, color=color, alpha=0.25)

	leg = ax.legend(loc='best')
	for line in leg.get_lines():
		line.set_linewidth(10)
	fig.savefig(os.path.join(os.getcwd(), 'HumanoidPlot.png'), bbox_inches='tight', pad_inches=0, dpi=200)
	plt.show()


if __name__ == '__main__':
	Reproduce_Results()
