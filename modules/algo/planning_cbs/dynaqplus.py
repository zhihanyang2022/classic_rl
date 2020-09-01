import random
import numpy as np
from .cb_base import CallbackBase

class DynaQPlus(CallbackBase):

    """Assume deterministic environment."""

    def __init__(self, planning_ratio, k, env):

        self.planning_ratio = planning_ratio
        self.k = k
        
        self.states = set()
        self.actions_for_each_state = {}
        self.model = {}

        self.tau = {}

        for state in env.loop_states:
            
            self.states.add(state)
            self.actions_for_each_state[state] = set([action for action in env.loop_actions_for_state(state)])
            
            for action in env.loop_actions_for_state(state):
                self.model[(state, action)] = (0, state)
                self.tau[(state, action)] = 0

    def do_modeling(self, **kwargs):
        
        s = kwargs['s']
        a = kwargs['a']
        r = kwargs['r']
        s_prime = kwargs['s_prime']
        algo = kwargs['algo']

        self.model[(s, a)] = (r, s_prime)
        
        self.tau[(s, a)] = 0

        for sbar in self.states:
            for abar in self.actions_for_each_state[sbar]:
                if (sbar, abar) != (s, a): self.tau[(sbar, abar)] += 1

    def do_learning(self, **kwargs):

        s = kwargs['s']
        a = kwargs['a']
        r = kwargs['r']
        s_prime = kwargs['s_prime']
        algo = kwargs['algo']

        algo.improve_value_function(s, a, r, s_prime)
        algo.improve_policy()

    def sample_state_uniformly(self):
        return random.sample(self.states, 1)[0]

    def sample_action_uniformly(self, s):
        return random.sample(self.actions_for_each_state[s], 1)[0]

    def do_planning(self, **kwargs):

        algo = kwargs['algo']
        assert algo.alpha == 1

        for _ in range(self.planning_ratio):

            s = self.sample_state_uniformly()
            a = self.sample_action_uniformly(s)

            r, s_prime = self.model[(s, a)]
            modified_r = r + self.k * np.sqrt(self.tau[(s, a)])

            algo.improve_value_function(s, a, modified_r, s_prime)
            algo.improve_policy()