from typing import Generator
import numpy as np


class PolicyEvaluation:

    """
    Implement the policy evaluation algorithm using q-tables for epsilon-greedy policies in simple gridworlds.
    """

    def __init__(self, env, policy, discount_factor, truncate_pe=False, pe_tol=1e-3):
        """
        Args:
            env: any environment available in `classic_rl.env`
            policy: An instance of DeterministicPolicy or EpsilonGreedyPolicy from `classic_rl.policy`. If the policy is
                an instance of DeterministicPolicy, then you must make sure that the terminal state can be reached from
                any state; otherwise policy evaluation won't converge.
            discount_factor: a float in the interval [0, 1]
            truncate_pe: whether to truncate policy iteration
            pe_tol: precision tolerance for policy iteration
        """

        self.env = env
        self.policy = policy
        self.truncate_pe = truncate_pe
        self.discount_factor = discount_factor

        if truncate_pe: 
            assert pe_tol is None
        else: 
            assert pe_tol is not None
            self.pe_tol = pe_tol
    
        self.init_q()

    def init_q(self) -> None:
        """
        Helper method to `self.__init__`.
        Initialize a q-table of all zeros.
        """
        self.q = np.zeros(self.env.action_space_shape)

    def backup_v(self, s) -> float:
        """
        Helper method to `self.backup_q`.
        $$V_{\\pi}(s) = \\sum_{a} \\pi(a \\mid s) Q_{\\pi}(s, a)$$
        """
        return np.sum([self.policy.calc_b_a_given_s(a, s) * self.q[s][a] for a in self.env.get_actions(s)])

    def backup_q(self, s, a) -> float:
        """
        Helper method to `self.do_policy_evaluation`.

        The iterative update rule for `self.q` using the Bellman equation for action-value functions.
        $$Q_{\\pi}(s, a) \\leftarrow \\sum_{s^{\\prime}, r} p(s^{\\prime}, r \\mid s, a) \\left[ r + V_{\\pi}(s^{\\prime}) \\right]$$
        """
        self.env.reset()
        self.env.current_coord = s
        s_prime, r = self.env.step(a)
        return r + self.discount_factor * self.backup_v(s_prime)

    def loop_state_action_qval(self) -> Generator:
        """
        Helper method to `self.do_policy_evaluation`. See code for more details.
        """
        for sa, old_q in np.ndenumerate(self.q):
            s, a = (sa[0], sa[1]), sa[-1]
            if self.env.is_actionable(s):
                yield s, a, old_q

    def do_policy_evaluation(self) -> None:
        """
        Helper method to self.run.

        ## Math behind the scene

        See the docstrings for `self.backup_q` and `self.backup_v` for more details.
        """

        # ========== evaluate policy ==========

        while True:
                
            error = 0

            for s, a, old_q in self.loop_state_action_qval():

                self.q[s][a] = self.backup_q(s, a)

                error = np.max([error, np.abs(old_q - self.q[s][a])])

            if self.truncate_pe:
                return

            if error < self.pe_tol:
                return

    def run(self) -> None:
        """
        Run the policy evaluation algorithm. 
        """

        print(f"==========")
        print(f"Running DP policy evaluation ...")

        self.do_policy_evaluation()

        print("Result: Convergence reached.")
        print(f"==========")


class PolicyIteration(PolicyEvaluation):

    """
    Implement the policy iteration algorithm using q-tables for epsilon-greedy policies in simple gridworlds.
    """

    def __init__(self, conv_tol, **kwargs):
        """
        Args:
            conv_tol: precision tolerance for convergence
        """
        super().__init__(**kwargs)
        self.conv_tol = conv_tol

    def init_q(self) -> None:
        """
        Helper method to `self.__init__`.
        Initialize the q-table by duplicating (not just set to equal!) self.policy's q-table.
        """
        self.q = self.policy.q.copy()

    def check_q_convergence(self) -> bool:
        """
        Helper method to self.run.
        Evaluate whether the q-table has convergenced to the optimal q-table using the Bellman optimality equation for action-value functions (BOE).

        ## Math behind the scene

        For epsilon-greedy policies, the BOEs are intuitively defined as (definition 1 and 2):

        $$V_{\\ast}(s) \\triangleq (1 - \\epsilon) \\max_a Q_{\\ast}(s, a) + \\frac{\\epsilon}{\\mid A \\mid} \\sum_a Q_{\\ast}(s, a)$$
        $$Q_{\\ast}(s, a) \\triangleq \\sum_{s^{\\prime}, r} p(s^{\\prime}, r \\mid s, a) \\left[ r + \\gamma V_{\\ast}(s^{\\prime}) \\right]$$

        For more information on optimal value functions, read section 3.6 of Sutton & Barto 2018.

        In this implementation, we evaluate whether the LHS and the RHS of definition 2 are equal. If the maximum difference between
        the LHS and the RHS over all state-action pairs is smaller than `tol`, then this function returns `True`; otherwise, this 
        function returns `False`.

        For a state-action pair \\((s, a)\\):

        * The LHS can be directly obtained using `self.q[s][a]`.
        * The RHS can be calculated using `self.backup_q(s, a)`, which uses `self.backup_v(s)` as a helpful function. This is because
        this implementation relies on a q-table only, and the RHS can only be calculated by substituting definition 1 into 2. Note that, 
        before convergence check, we must temporarily set the policy to be greedy with respect to `self.q` due to the \\(\\max\\) operation 
        in definition 1. Ater convergence check, we must reset the policy. See code for more details.

        Args:
            tol: precision tolerance

        Returns:
            bool: whether convergence is reached
        """
        
        # step 1: update policy (temporarily)
        
        policy_old_q = self.policy.q.copy()
        self.policy.q = self.q.copy()  # self.policy knows how to act greedily with respect to any q-table

        # step 2: calculate the optimality error (the difference between the LHS and the RHS of definition 2)

        optimality_error = 0

        for s, a, old_q in self.loop_state_action_qval():
                
            lhs = self.q[s][a]

            rhs = self.backup_q(s, a)

            optimality_error = np.max([optimality_error, np.abs(lhs - rhs)])

        # step 3: reset policy

        self.policy.q = policy_old_q.copy()

        # step 4: check whether the optimality error falls below a tolerance level

        if optimality_error < self.conv_tol:
            return True
        else:
            return False
        
    def do_policy_improvement(self) -> None:    
        """
        Helper method to self.run.
        """       
        self.policy.q = self.q.copy()

    def sample_greedy_trajectory(self, old_q, tau):

        sas = []

        self.env.reset()

        while not self.env.is_episode_terminated():

            s = self.env.current_coord
            a = self.policy.act_greedily(s)
            sas.append((s, a))

            tau = tau + 1
            tau[s[1], a] = 0

            curiosity_q = tau * 5e-3

            s_prime, r = self.env.step(a)

            self.q = old_q + curiosity_q

            self.policy.q = self.q.copy()

        return sas, tau

    def run(self, max_iterations, which_tqdm) -> None:
        """
        Run the policy iteration algorithm.

        Args:
            max_iterations: the maximum number of iterations before the algorithm is halted
            which_tqdm: "terminal" or "notebook", depending on whether you are running code in a terminal or a jupyter notebook
        """

        assert max_iterations >= 1

        assert which_tqdm in ['terminal', 'notebook']

        if which_tqdm == 'terminal':
            from tqdm import tqdm
        elif which_tqdm == 'notebook':
            from tqdm.notebook import tqdm

        print(f"Running DP policy iteration for at most {max_iterations} iterations ...")
            
        for i in tqdm(range(1, max_iterations+1)):
        
            self.do_policy_evaluation()
            converged = self.check_q_convergence()
            self.do_policy_improvement()

            if converged:
                print(f'Result: Convergence reached at iteration {i}')
                return

        print(f"Result: Convergence not reached after {i} iterations.")

                