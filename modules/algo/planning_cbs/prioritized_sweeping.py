import copy
import random
import numpy as np
from .cb_base import CallbackBase

class PrioritizedSweeping(CallbackBase):

    """
    Assume deterministic environment.
    Included value function tracker functionality.
    """

    def __init__(self, planning_ratio, td_error_threshold=1e-5):
        
        self.visited_states = set()
        self.actions_for_each_state = {}
        self.model = {}
        self.planning_ratio = planning_ratio
        
        self.priority_queue = {}
        self.td_error_threshold = td_error_threshold

        # value function tracker functionality

        self.num_episodes_passed = 0

        self.qs = []
        self.q_statuses = []  # 0 represents "after a learning step"; 1 represents "after a planning step"

        self.states = []
        self.actions = []
        self.next_states = []
        self.argmax_next_actions = []

        self.priorities = []

    # ==================================================

    def do_modeling(self, **kwargs):
        
        s = kwargs['s']
        a = kwargs['a']
        r = kwargs['r']
        s_prime = kwargs['s_prime']

        self.model[(s, a)] = (r, s_prime)

        self.visited_states.add(s)  # for planning
        if s in self.actions_for_each_state.keys():
            self.actions_for_each_state[s].add(a)
        else:
            self.actions_for_each_state[s] = {a}

    # ==================================================

    def do_learning(self, **kwargs):

        s = kwargs['s']
        a = kwargs['a']
        r = kwargs['r']
        s_prime = kwargs['s_prime']
        algo = kwargs['algo']

        # ***** one (delayed) learning step *****

        td_error = algo.calculate_td_error(s, a, r, s_prime)
        if td_error > self.td_error_threshold:
            self.priority_queue[(s, a)] = td_error  # delayed learning

        # ***** value function tracker functionality *****

        if self.num_episodes_passed < 2:  # only for the first two episodes; otherwise too much to store

            self.qs.append(algo.q.copy())
            self.q_statuses.append(0)

            self.states.append(s)
            self.actions.append(a)
            self.next_states.append(s_prime)
            self.argmax_next_actions.append(algo.policy.act_greedily(s_prime))

            priority_arr = np.zeros(algo.env.action_space_shape)
            for k, v in self.priority_queue.items():
                s, a = k
                priority_arr[s][a] = v
            self.priorities.append(priority_arr)

    # ==================================================

    def sample_state_uniformly(self):
        return random.sample(self.visited_states, 1)[0]

    def sample_action_uniformly(self, s):
        return random.sample(self.actions_for_each_state[s], 1)[0]

    def pop_sa_pair_with_highest_priority(self):
        sa = [sa for sa, td_error in sorted(self.priority_queue.items(), key=lambda item : item[1], reverse=True)][0]
        del self.priority_queue[sa]
        return sa

    def do_planning(self, **kwargs):

        algo = kwargs['algo']

        for _ in range(self.planning_ratio):

            if len(self.priority_queue) > 0:

                # ***** do planning based on priority *****

                s, a = self.pop_sa_pair_with_highest_priority()

                r, s_prime = self.model[(s, a)]

                algo.improve_value_function(s, a, r, s_prime)
                algo.improve_policy()

                # ***** add state-action pairs that lead into (s, a) to the queue *****

                for sbar in self.visited_states:
                    for abar in self.actions_for_each_state[sbar]:
                        rbar, sbar_prime = self.model[(sbar, abar)]
                        if sbar_prime == s:  # if sbar, abar is predicted to lead to s
                            td_error = algo.calculate_td_error(sbar, abar, rbar, s)
                            if td_error > self.td_error_threshold:
                                self.priority_queue[(sbar, abar)] = td_error

                # ***** value function tracker functionality *****

                if self.num_episodes_passed < 2:  # only for the first two episodes; otherwise too much to store

                    self.qs.append(algo.q.copy())
                    self.q_statuses.append(1)

                    self.states.append(s)
                    self.actions.append(a)
                    self.next_states.append(s_prime)
                    self.argmax_next_actions.append(algo.policy.act_greedily(s_prime))

                    priority_arr = np.zeros(algo.env.action_space_shape)
                    for k, v in self.priority_queue.items():
                        s, a = k
                        priority_arr[s][a] = v
                    self.priorities.append(priority_arr)

            else: break

    def after_each_episode(self, **kwargs):
        self.num_episodes_passed += 1

