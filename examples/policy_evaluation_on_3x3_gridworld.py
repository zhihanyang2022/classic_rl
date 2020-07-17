import sys
sys.path.append('..')

import numpy as np

from env.gridworld import GridWorld
from policy import EpsilonSoftPolicy
from algo.dp import PolicyEvaluation

env = GridWorld(num_rows=3, num_cols=3, start_coord=(0, 0), end_coord=(2, 2))

q_init = np.random.uniform(size=env.shape)

print("===== optimal q-table =====")
print(q_init)

policy = EpsilonSoftPolicy(q=q_init, epsilon=0.3)

algo = PolicyEvaluation(env=env, policy=policy)

print("===== q-table before training =====")
print(algo.q)

algo.train(tol=1e-3)

print("===== q-table after training =====")
print(algo.q)
