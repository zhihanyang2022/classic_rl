import itertools
import numpy as np 

def vectorize(array:tuple):
	return np.array(array).reshape(-1, 1)

class LinearFeatureConstructor:

	def __init__(self, num_raw_features, lowers, uppers):
		self.num_features = num_raw_features + 1
		self.lowers = lowers
		self.uppers = uppers

	def preprocess(self, state):
		state = (vectorize(state) - vectorize(self.lowers)) / (vectorize(self.uppers) - vectorize(self.lowers))
		return np.vstack([state, np.array([[1.]])])  # append bias

class PolynomialFeatureContructor:

	def __init__(self, num_raw_features, order, lowers, uppers):
		self.num_raw_features = num_raw_features
		self.order = order
		self.num_features = int((order + 1) ** num_raw_features)
		self.c = np.array(list(itertools.product(np.arange(self.order+1), repeat=self.num_raw_features)))

		self.lowers = lowers
		self.uppers = uppers

	def preprocess(self, state):
		state = (vectorize(state) - vectorize(self.lowers)) / (vectorize(self.uppers) - vectorize(self.lowers))
		return np.prod(state ** self.c.T, axis=0).reshape(-1, 1)

class FourierFeatureConstructor:

	def __init__(self, num_raw_features, order, lowers, uppers):

		self.num_raw_features = num_raw_features
		self.order = order
		self.num_features = int((order + 1) ** num_raw_features)
		self.c = np.array(list(itertools.product(np.arange(self.order+1), repeat=self.num_raw_features))).T

		self.lowers = lowers
		self.uppers = uppers

	def preprocess(self, state):
		state = (vectorize(state) - vectorize(self.lowers)) / (vectorize(self.uppers) - vectorize(self.lowers))
		dot_products = state.T @ self.c
		return np.cos(np.pi * dot_products).reshape(-1, 1)

# ========== value representations ==========

class Table:

	def __init__(self, state_space_shape:tuple):
		self.counts = np.zeros(state_space_shape) + 1e-5
		self.sum_of_returns = np.zeros(state_space_shape)

	def update(self, state:tuple, target:float):
		self.counts[state] += 1
		self.sum_of_returns[state] += target

	@property
	def v(self):
		return self.sum_of_returns / self.counts

class LinearApproximator:

	def __init__(self, lr:float, fc, state_space_shape:tuple):
		self.lr = lr
		self.fc = fc
		self.w = np.zeros((self.fc.num_features, 1))
		self.state_space_shape = state_space_shape
		
	def calc_v(self, state:tuple):
		return float(self.w.T @ self.fc.preprocess(state))

	def calc_grad_wrt_w(self, state:tuple):
		return self.fc.preprocess(state)

	def update(self, state:tuple, target:float):
		self.w += self.lr * (target - self.calc_v(state)) * self.calc_grad_wrt_w(state)

	@property
	def v(self):
		v = np.zeros(self.state_space_shape)
		for row_ix in range(self.state_space_shape[0]):
			for col_ix in range(self.state_space_shape[1]):
				state = (row_ix, col_ix)
				v[state] = self.calc_v(state)
		return v

# ===========================================

class MCPredictor:

	def __init__(self, env, policy, discount_factor, vrepr, vtrue:np.array=None, on_policy_dist:np.array=None):
		
		self.env = env
		self.policy = policy
		self.discount_factor = discount_factor
		
		self.vrepr = vrepr

		assert (vtrue is None and on_policy_dist is None) or (vtrue is not None and on_policy_dist is not None)
		
		self.vtrue = vtrue
		self.on_policy_dist = on_policy_dist
		
		if self.vtrue is not None and self.on_policy_dist is not None:
			self.record_error = True
		else:
			self.record_error = False

	def record_one_trajectory(self):

		states = []
		rewards = []

		self.env.reset()

		while not self.env.is_episode_terminated():

			s = self.env.current_coord

			states.append(s)
			
			a = self.policy.act(s)
			s_prime, r = self.env.step(a)
		   
			rewards.append(r)

		return states, rewards

	def learn_from_one_trajectory(self, states, rewards):

		T = len(states) # last index
		
		G = 0  # return
		for t in range(T-1, -1, -1):  # T-1, T-2, ..., 1, 0
			G = self.discount_factor * G + rewards[t]
			self.vrepr.update(states[t], G)

	def calc_root_VE(self) -> float:
		"""Calculate the root value error."""
		return np.sqrt(np.sum(self.on_policy_dist * (self.vtrue - self.vrepr.v) ** 2))

	def run(self, max_iterations, which_tqdm='notebook'):

		if which_tqdm == 'terminal':
			from tqdm import tqdm
		elif which_tqdm == 'notebook':
			from tqdm.notebook import tqdm

		if self.record_error: errors = []
		for _ in tqdm(range(max_iterations), leave=False):
			states, rewards = self.record_one_trajectory()
			self.learn_from_one_trajectory(states, rewards)
			if self.record_error: errors.append(self.calc_root_VE())

		if self.record_error:
			return errors


