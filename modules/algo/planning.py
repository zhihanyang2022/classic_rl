import numpy as np
import matplotlib.pyplot as plt

class QLearningBase:

    def __init__(self, env, policy, alpha, gamma, cb_handler):
        
        self.env = env
        self.policy = policy
        self.alpha = alpha  # learning rate
        self.gamma = gamma

        self.cb_handler = cb_handler

        self.initialize_tables()

    def initialize_tables(self):
        self.q = np.zeros(self.env.action_space_shape)

    def improve_policy(self):
        self.policy.q = self.q.copy()

    def calculate_td_error(self, s, a, r, s_prime):
        bellman_sample = r + self.gamma * self.q[s_prime][self.policy.act_greedily(s_prime)]
        return bellman_sample - self.q[s][a]

    def improve_value_function(self, s, a, r, s_prime):
        self.q[s][a] = self.q[s][a] + self.alpha * self.calculate_td_error(s, a, r, s_prime)

    def do_control_for_one_trajectory(self):

        self.env.reset()

        while not self.env.is_episode_terminated():
        
            s = self.env.current_coord
            a = self.policy.act_softly(s)
            
            s_prime, r = self.env.step(a)

            self.cb_handler.do_modeling(s=s, a=a, r=r, s_prime=s_prime, algo=self)
            
            self.cb_handler.do_learning(s=s, a=a, r=r, s_prime=s_prime, algo=self)
            self.cb_handler.after_learning(s=s, a=a, r=r, s_prime=s_prime, algo=self)
            
            self.cb_handler.do_planning(algo=self)
            self.cb_handler.after_planning(algo=self)

            # self.cb_handler.after_one_step()

        self.cb_handler.after_each_episode(algo=self)

    def run(self, max_iterations, which_tqdm='notebook'):

        if which_tqdm == 'terminal':
            from tqdm import tqdm
        elif which_tqdm == 'notebook':
            from tqdm.notebook import tqdm

        for i in tqdm(range(max_iterations), leave=False):
            self.do_control_for_one_trajectory()

    def plot_value_function_and_policy(self, savefig_fpath=None):

        plt.matshow(self.q.max(axis=-1), fignum=1)
        learned_policy = self.q.argmax(axis=-1)

        ix_to_str = {
            0 : r'$\uparrow$',
            1 : r'$\rightarrow$',
            2 : r'$\downarrow$',
            3 : r'$\leftarrow$'
        }

        for (i, j), z in np.ndenumerate(learned_policy):
            
            if self.env.is_actionable((i, j)):
            
                plt.text(
                    j, i, ix_to_str[int(z)], 
                    ha='center', va='center', color='black', fontsize=15, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='gold', edgecolor='0')
                )
                
        ax = plt.gca()
        ax.set_xticks(np.arange(-.5, learned_policy.shape[1], 1), minor=True);
        ax.set_yticks(np.arange(-.5, learned_policy.shape[0], 1), minor=True);
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

        if savefig_fpath is not None:
            plt.savefig(savefig_fpath, dpi=300, bbox_inches='tight', pad_inches=0)