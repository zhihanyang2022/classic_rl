import numpy as np


class DeterministicPolicy:
    
    def __init__(self, q):
        """
        :param epsilon: total probability assigned to choosing actions randomly
        :param q: the q-value table used to determine to argmax actions
        """
        self.q = q.copy()
        self.num_actions = self.q.shape[-1]
        
    def get_winning_actions(self, s):
        """
        Helper method to self.act_greedily.
        Determine the best actions(s) to take in a state. 
        
        :param s: state
        :return: the action(s) with the highest q-value in state s
        """
        action_vals = self.q[s]
        winners = np.argwhere(action_vals == np.max(action_vals)).flatten()
        return winners
    
    def act_greedily(self, s):
        """
        Helper method to self.act.
        Return an action that is randomly chosen from the results of self.get_winning_actions(s).
        
        :param s: state
        :return: a greedy action
        """
        return np.random.choice(self.get_winning_actions(s))

    def act(self, s):
        """
        Return the appropriate action to take in state s under a deterministic policy.

        :param s: state
        :return: action
        """
        return self.act_greedily(s)

    def calc_pi_a_given_s(self, a, s):
        """
        Calculate π(a|s), the probability of taking action a in state s under target policy π (deterministic).
        
        :param a: action
        :param s: state

        :return: π(a|s)
        """
        
        winning_actions = self.get_winning_actions(s)
        
        if a in winning_actions: 
            return 1 / len(winning_actions)
        else: 
            return 0  

class EpsilonGreedyPolicy(DeterministicPolicy):

    def __init__(self, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def act_randomly(self):
        """
        Helper method to self.act_softly.
        Return a random action.
        
        :return: a random action
        """
        return np.random.randint(self.num_actions)
        
    def act_softly(self, s):
        """
        Helper method to self.act.
        Return self.act_greedily(s) with probability 1 - self.epsilon and self.act_randomly() with probability self.epsilon.
        
        :param s: state
        :return: a soft action
        """
        return self.act_greedily(s) if np.random.uniform(0, 1) > self.epsilon else self.act_randomly()

    def act(self, s):
        """
        Return the appropriate action to take in state s under a epsilon-soft policy.
        
        :param s: state
        :return: action
        """
        return self.act_softly(s)

    def calc_b_a_given_s(self, a, s):
        """
        Calculate b(a|s), the probability of taking action a in state s under behavior policy b (epsilon-soft).
        
        :param a: action
        :param s: state

        :return: b(a|s)
        """
        
        winning_actions = self.get_winning_actions(s)
        
        if a in winning_actions:
            # can be easily derived using a probability tree
            return (1 - self.epsilon) * (1 / len(winning_actions)) + self.epsilon * (1 / self.num_actions)
        else:
            return self.epsilon * (1 / self.num_actions)
        
    
class BTPolicy(EpsilonGreedyPolicy):
    """A policy class that serves as both the target and behavior policy for off-policy methods."""

    pass
        
    