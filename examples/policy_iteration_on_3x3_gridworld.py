import sys
sys.path.append('..')

import numpy as np

from env.gridworld import GridWorld
from policy import EpsilonSoftPolicy
from algo.dp import PolicyIteration

env = GridWorld(num_rows=3, num_cols=3, start_coord=(0, 0), end_coord=(2, 2))

arbitrary_q = np.random.uniform(size=env.shape)
arbitrary_q[2, 2] = 0

print("===== initial q-table =====")
print(arbitrary_q)

policy = EpsilonSoftPolicy(q=arbitrary_q, epsilon=0.5)

algo = PolicyIteration(env=env, policy=policy)

print("===== q-table before training =====")
print(algo.q)

algo.train(max_iterations=10000, tol=1e-16, value_focus=True)

print("===== q-table after training =====")
print(algo.q)
