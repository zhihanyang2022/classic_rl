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
        self.q = np.random.uniform(size=self.env.shape)

    def do_policy_evaluation(self, tol) -> None:

        while True:
                
            error = 0

            for sa, old_q in np.ndenumerate(self.q):

                state, action = (sa[0], sa[1]), sa[-1]

                if state != self.env.end_coord:

                    self.env.reset()

                    self.env.current_coord = state; next_state, reward = self.env.step(action)

                    self.q[state][action] = reward + self.q[next_state][self.policy.act(next_state)]  # the update step

                    error = np.max([error, np.abs(old_q - self.q[sa])])

            if error < self.tol: 
                break

    def train(self, tol=1e-3):

        print(f"Running DP policy evaluation ...")

        self.do_policy_evaluation(tol)

        print("Result: Convergence reached.")


class PolicyIteration(PolicyEvaluation):
        
    def do_policy_improvement(self) -> bool:

        """
        Do policy improvement once for states that are both non-terminal and reachable.
        """

        policy_stable = True
                
        for row_ix in np.arange(self.q.shape[0]):
            for col_ix in np.arange(self.q.shape[1]):
        
                state = (row_ix, col_ix)
            
                if state != self.env.end_coord:

                    old_action = self.policy.act_softly(state)

                    self.policy.q[state] = self.q[state].copy()  # policy automatically computes the argmax
                    new_action = self.policy.act(state)

                    if old_action != new_action and self.q[state][old_action] != self.q[state][new_action]:
                        policy_stable = False

        return policy_stable

    def train(self, num_iterations, tol=1e-3):

        """
        Run the policy iteration algorithm.

        Args:
            num_iterations: the number of iterations of policy evaluation and policy improvement
            tol: the precision tolerance for policy evaluation
        """

        assert num_iterations >= 1

        print(f"Running DP policy iteration for {num_iterations} iterations ...")
            
        for i in tqdm(range(1, num_iterations+1)):
        
            self.do_policy_evaluation(tol)
            policy_stable = self.do_policy_improvement()

            if policy_stable:
                print(f"Result: Convergence reached at iteration {i}.")

        print(f"Result: Convergence not reached after {i} iterations.")

                