import numpy as np

class TemporalDifference:

	def __init__(self, env, policy, mode, alpha):
		
		self.env = env
		self.policy = policy
		self.alpha = alpha

		assert mode == 'sarsa' or mode == 'qlearning'
		self.mode = mode

		self.initialize_tables()

	def initialize_tables(self):
		self.q = self.policy.q.copy()
		self.num_updates = np.zeros_like(self.q)

	def sample_greedy_trajectory(self):

		states = []

		self.env.reset()

		while not self.env.is_episode_terminated():

			s = self.env.current_coord; states.append(s)
			a = self.policy.act_greedily(s)

			s_prime, r = self.env.step(a)

		states.append(self.env.current_coord)

		return states

	def do_td_control_for_one_trajectory(self):

		total_reward = 0

		self.env.reset()

		while not self.env.is_episode_terminated():
		
			s = self.env.current_coord
			a = self.policy.act_softly(s)
			
			s_prime, r = self.env.step(a); total_reward += r
			
			if self.mode == 'sarsa':
				bellman_sample = r + self.q[s_prime][self.policy.act_softly(s)]
			elif self.mode == 'qlearning':
				bellman_sample = r + self.q[s_prime][self.policy.act_greedily(s)]

			self.q[s][a] = self.q[s][a] + self.alpha * (bellman_sample - self.q[s][a])
			self.num_updates[s][a] += 1

			self.policy.q = self.q.copy()  # update the policy automatically

		return total_reward

	def run(self, max_iterations, which_tqdm):

		if which_tqdm == 'terminal':
			from tqdm import tqdm
		elif which_tqdm == 'notebook':
			from tqdm.notebook import tqdm

		total_rewards = []
		for i in tqdm(range(max_iterations), leave=False):
			total_rewards.append(self.do_td_control_for_one_trajectory())

		return total_rewards


		
