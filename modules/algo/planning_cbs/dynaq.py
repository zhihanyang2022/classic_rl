import numpy as np
import random
from .cb_base import CallbackBase

class DynaQ(CallbackBase):

    """Assume deterministic environment."""

    def __init__(self, planning_ratio):
        self.visited_states = set()
        self.actions_for_each_state = {}
        self.model = {}
        self.planning_ratio = planning_ratio

    def do_modeling(self, **kwargs):

        s = kwargs['s']
        a = kwargs['a']
        r = kwargs['r']
        s_prime = kwargs['s_prime']

        self.model[(s, a)] = (s_prime, r)

        self.visited_states.add(s)  # for planning
        if s in self.actions_for_each_state.keys():
            self.actions_for_each_state[s].add(a)
        else:
            self.actions_for_each_state[s] = {a}

    def do_learning(self, **kwargs):

        s = kwargs['s']
        a = kwargs['a']
        r = kwargs['r']
        s_prime = kwargs['s_prime']
        algo = kwargs['algo']

        algo.improve_value_function(s, a, r, s_prime)
        algo.improve_policy()

    def sample_state_uniformly(self):
        return random.sample(self.visited_states, 1)[0]

    def sample_action_uniformly(self, s):
        return random.sample(self.actions_for_each_state[s], 1)[0]

    def do_planning(self, **kwargs):

        for _ in range(self.planning_ratio):

            algo = kwargs['algo']

            s = self.sample_state_uniformly()
            a = self.sample_action_uniformly(s)

            s_prime, r = self.model[(s, a)]

            algo.improve_value_function(s, a, r, s_prime)
            algo.improve_policy()