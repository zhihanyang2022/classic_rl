import numpy as np
from tqdm import tqdm

class PolicyEvaluation:

    def __init__(self, env, policy):
        """
        Args:
            env: any environment available in `classic_rl.env`
            policy: an instance of DeterministicPolicy or EpsilonSoftPolicy from `classic_rl.policy`
        """
        self.env = env
        self.policy = policy
    
        self.init_q()

    def init_q(self):
        self.q = np.zeros(self.env.shape)

    def do_policy_evaluation(self, tol) -> None:

        while True:
                
            error = 0
            optimality_error = 0

            for sa, old_q in np.ndenumerate(self.q):

                state, action = (sa[0], sa[1]), sa[-1]

                if state != self.env.end_coord:

                    self.env.reset()

                    self.env.current_coord = state

                    next_state, reward = self.env.step(action)

                    weights, qs = [], []
                    for next_action in range(self.env.num_actions):
                        weights.append(self.policy.calc_b_a_given_s(next_action, next_state))
                        qs.append(self.q[next_state][next_action])

                    v_next_state = np.sum(np.array(weights) * np.array(qs))

                    self.q[state][action] = reward + v_next_state  # the update step

                    optimal_q = reward + self.q[next_state][self.policy.act_hardly(next_state)]

                    optimality_error = np.max([optimality_error, np.abs(optimal_q - self.q[state][action])])

                    error = np.max([error, np.abs(old_q - self.q[state][action])])

            if optimality_error < tol:
                return True
            elif error < tol: 
                return False

    def train(self, tol=1e-3):

        print(f"==========")
        print(f"Running DP policy evaluation ...")

        self.do_policy_evaluation(tol)

        print("Result: Convergence reached.")
        print(f"==========")


class PolicyIteration(PolicyEvaluation):

    def init_q(self):
        self.q = self.policy.q.copy()
        
    def do_policy_improvement(self) -> bool:

        """
        Do policy improvement once for states that are both non-terminal and reachable.
        """

        policy_stable = True
                
        for row_ix in np.arange(self.q.shape[0]):
            for col_ix in np.arange(self.q.shape[1]):
        
                state = (row_ix, col_ix)
            
                if state != self.env.end_coord:

                    old_action = self.policy.act(state)

                    self.policy.q[state] = self.q[state].copy()  # policy automatically computes the argmax
                    new_action = self.policy.act(state)

                    if old_action != new_action and self.policy.q[state][old_action] != self.policy.q[state][new_action]:
                        policy_stable = False

        return policy_stable

    def train(self, max_iterations, tol=1e-3, epsilon_multiplier=0.95, value_focus=False):

        """
        Run the policy iteration algorithm.

        Args:
            num_iterations: the number of iterations of policy evaluation and policy improvement
            tol: the precision tolerance for policy evaluation
        """

        assert max_iterations >= 1
        assert epsilon_multiplier > 0 and epsilon_multiplier < 1

        print(f"Running DP policy iteration for at most {max_iterations} iterations ...")
            
        for i in range(1, max_iterations+1):

            print(f'Iteration: {i}')
        
            value_stable = self.do_policy_evaluation(tol)
            policy_stable = self.do_policy_improvement()

            if policy_stable and not value_focus:
                print(f"Result: Convergence reached at iteration {i}.")
                return

            if value_stable and value_focus:
                print(f'Result: Convergence reached at iteration {i}')
                return

            self.policy.epsilon *= epsilon_multiplier  # Robin-Monro procedure

        print(f"Result: Convergence not reached after {i} iterations.")

                