import numpy as np

class OneStepTDControl:

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
				bellman_sample = r + self.q[s_prime][self.policy.act_softly(s_prime)]
			elif self.mode == 'qlearning':
				bellman_sample = r + self.q[s_prime][self.policy.act_greedily(s_prime)]

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

class NStepTDPrediction:

	def __init__(self, env, policy, alpha, n, gamma=1, use_td_errors=False, true_v=None):
		
		self.env = env
		self.policy = policy
		self.alpha = alpha
		self.n = n
		self.gamma = gamma
		self.use_td_errors = use_td_errors   # for solution to exercise 7.2
		self.true_v = true_v  # for solution to exercise 7.2

		self.initialize_tables()

	def initialize_tables(self):
		self.v = np.zeros(self.env.state_space_shape)

	def do_td_prediction_for_one_trajectory(self):

		self.useful_updates = []
		self.updates = []

		states = {}
		rewards = {}
		td_errors = {}

		self.env.reset()
		states[0] = self.env.current_coord

		T = np.inf

		t = 0
		while True:

			if t < T:

				a = self.policy.act_softly(states[t])
				s_prime, r = self.env.step(a)
				
				states[t+1] = s_prime
				rewards[t+1] = r
				td_errors[t] = r + self.gamma * self.v[s_prime] - self.v[states[t]]
			
				if self.env.is_episode_terminated():
					T = t + 1
					if self.use_td_errors:
						break  # we won't be updating the value of the terminal state, so no further information needed beyond this point

			if not self.use_td_errors:

				tau = t - self.n + 1  # at timestep t, update the value of the state encountered at timestep tau

				if tau >= 0:  # no changes at all are made during the first n - 1 steps of each episode

					# G_{t : t + n} is
					# - the truncated return for time t up until time t + n (after n actions are taken)
					# - plus the discounted estimate (gamma ** n) * v(S_{t+n}) at the end

					first_index = tau + 1
					final_index = np.min([tau + self.n, T])

					truncated_G = np.sum([
						self.gamma ** (i - tau - 1) * rewards[i] for i in np.arange(first_index, final_index+1)
					])

					if tau + self.n < T:
						corrected_G = truncated_G + (self.gamma ** self.n) * self.v[states[tau + self.n]]
					else:
						corrected_G = truncated_G
	
					error = corrected_G - self.v[states[tau]]

					self.v[states[tau]] = self.v[states[tau]] + self.alpha * error
					# if self.v[states[tau + self.n]] != 0:
					# 	self.useful_updates.append(states[tau][1])

				if tau == T - 1: break  # the terminal state needs no update since no further actions are taken

			t += 1

		if self.use_td_errors:

			t = 0
			while True:

				tau = t - self.n + 1

				if tau >= 0:

					first_index = tau
					final_index = np.min([tau + self.n - 1, T-1])  # td error is zero for the Tth timestep
					error = np.sum([
						self.gamma ** (k - tau) * td_errors[k] for k in np.arange(tau, final_index+1)
					])  # see solution to exercise 7.1 for a derivation of this formula

					self.v[states[tau]] = self.v[states[tau]] + self.alpha * error

					if tau == T - 1: break 

				t += 1

		if self.true_v is not None:
			return np.mean((self.true_v - self.v) ** 2) ** (0.5)

	def run(self, max_iterations, which_tqdm, seed=None):

		if which_tqdm == 'terminal':
			from tqdm import tqdm
		elif which_tqdm == 'notebook':
			from tqdm.notebook import tqdm
		elif which_tqdm is None:
			tqdm = lambda x, leave : x

		if seed is not None:
			np.random.seed(seed)

		if self.true_v is None:

			for i in tqdm(range(max_iterations), leave=False):
				self.do_td_prediction_for_one_trajectory()

		else:

			rms_errors = []
			for i in tqdm(range(max_iterations), leave=False):
				rms_error = self.do_td_prediction_for_one_trajectory()
				rms_errors.append(rms_error)

		return rms_errors


		
