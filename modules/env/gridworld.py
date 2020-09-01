import numpy as np

class VanillaGridWorld:
    
    def __init__(self, num_rows, num_cols, start_coord, end_coord, mode=0):
        
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.start_coord = start_coord
        self.end_coord = end_coord
        self.current_coord = start_coord
        
        self.num_actions = 4

        self.mode = mode

    @property
    def state_space_shape(self):
        return self.num_rows, self.num_cols

    @property
    def action_space_shape(self):
        return self.num_rows, self.num_cols, self.num_actions

    def get_actions(self, s):
        return np.arange(self.num_actions)
    
    def propose_next_coord(self, coord, action):
        if action == 0:  # up
            proposed_coord = (self.current_coord[0] - 1, self.current_coord[1])
        elif action == 1:  # right
            proposed_coord = (self.current_coord[0], self.current_coord[1] + 1)
        elif action == 2:  # down
            proposed_coord = (self.current_coord[0] + 1, self.current_coord[1])
        elif action == 3:  # left
            proposed_coord = (self.current_coord[0], self.current_coord[1] - 1)
        return proposed_coord

    def is_out_of_world(self, coord):
        return coord[0] < 0 or coord[0] > self.num_rows - 1 or coord[1] < 0 or coord[1] > self.num_cols - 1

    def is_coord_valid(self, coord):
        if not self.is_out_of_world(coord):
            return True
        else:
            return False

    def get_reward(self, coord):
        if self.mode == 0:
            return -1
        elif self.mode == 1:
            return 1 if coord == self.end_coord else 0

    def step(self, action, coord=None):
        
        if coord is None:  # use the step function to predict the reward and next state for taking action in the current state (memorized by the environment)

            assert not self.is_episode_terminated()
            
            proposed_next_coord = self.propose_next_coord(self.current_coord, action)

            if self.is_coord_valid(proposed_next_coord):
                self.current_coord = proposed_next_coord

            return self.current_coord, self.get_reward(self.current_coord)

        else:  # use the step function to predict the reward and next state for any state-action pair
            
            proposed_next_coord = self.propose_next_coord(coord, action)
        
            if self.is_coord_valid(proposed_next_coord):
                return proposed_next_coord, self.get_reward(proposed_next_coord)
            else:
                return coord, self.get_reward(coord)
        
    def is_episode_terminated(self):
        if self.current_coord == self.end_coord: 
            return True
        else:
            return False
    
    def reset(self):
        self.current_coord = self.start_coord

    def is_actionable(self, s):
        if s == self.end_coord:
            return False
        else:
            return True


    def loop_states(self):
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self.is_actionable((i, j)):
                    return (i, j)

    def loop_actions_for_state(self, s):
        for action in range(self.num_actions):
            return action

    def loop_state_action_pairs(self):
        for state in self.loop_states:
            for action in self.loop_actions_for_state(state):
                return state, action

class GridWorldWithWallsAndTraps(VanillaGridWorld):

    """
    Traps are grids with HUGE negative rewards. Walls are grids that are not walkable; if an agent walks on a wall grid, it is
    immediately sent back to its last grid and receives a reward given by self.reward_array[self.current_coord].
    """

    def __init__(self, 
        grid_type_array, 
        reward_array,
        int_to_action_dict,
        grid_type_to_int_dict={'generic_walkable':0, 'start':1, 'wall':2, 'trap':3, 'end':4},
        ):

        self.grid_type_array = grid_type_array
        self.reward_array = reward_array
        self.int_to_action_dict = int_to_action_dict

        self.start_coord = tuple(np.argwhere(self.grid_type_array == grid_type_to_int_dict['start'])[0])
        self.end_coords = [tuple(ls) for ls in np.argwhere(self.grid_type_array == grid_type_to_int_dict['end']).tolist()]
        self.wall_coords = [tuple(ls) for ls in np.argwhere(self.grid_type_array == grid_type_to_int_dict['wall']).tolist()]

        self.num_rows, self.num_cols = self.grid_type_array.shape
        self.num_actions = len(int_to_action_dict.keys())

        self.current_coord = self.start_coord

    def is_actionable(self, s):
        if s in self.end_coords or s in self.wall_coords:
            return False
        else:
            return True

    def is_coord_valid(self, coord):
        if not self.is_out_of_world(coord) and not coord in self.wall_coords:
            return True
        else:
            return False

    def get_reward(self, coord):
        return self.reward_array[coord]

    def is_episode_terminated(self):
        if self.current_coord in self.end_coords: 
            return True
        else:
            return False

    def propose_next_coord(self, coord, action_int):
        if self.int_to_action_dict[action_int] == 'up':
            proposed_coord = (self.current_coord[0] - 1, self.current_coord[1])
        elif self.int_to_action_dict[action_int] == 'right':
            proposed_coord = (self.current_coord[0], self.current_coord[1] + 1)
        elif self.int_to_action_dict[action_int] == 'down':
            proposed_coord = (self.current_coord[0] + 1, self.current_coord[1])
        elif self.int_to_action_dict[action_int] == 'left':
            proposed_coord = (self.current_coord[0], self.current_coord[1] - 1)
        return proposed_coord

class Corridor(VanillaGridWorld):

    """
    - allows for multiple end states
    - allows for a custom reward table
    - no walls or traps allowed
    - use along with a random policy
    """

    def __init__(self, 
        grid_type_array, 
        reward_array,
        grid_type_to_int_dict={'generic':0, 'start':1, 'end':2},
        ):

        self.grid_type_array = grid_type_array
        
        self.reward_array = reward_array

        assert len(np.argwhere(self.grid_type_array == grid_type_to_int_dict['start'])) == 1, "There can only be one start state."

        self.start_coord = tuple(np.argwhere(self.grid_type_array == grid_type_to_int_dict['start'])[0])
        self.end_coords = [tuple(row) for row in np.argwhere(self.grid_type_array == grid_type_to_int_dict['end'])]

        self.num_rows, self.num_cols = self.grid_type_array.shape
        self.num_actions = 2  # left and right

    def is_actionable(self, s):
        if s in self.end_coords:
            return False
        else:
            return True

    def is_coord_valid(self, coord):
        if not self.is_out_of_world(coord):
            return True
        else:
            return False

    def get_reward(self, coord):
        return self.reward_array[coord]

    def is_episode_terminated(self):
        if self.current_coord in self.end_coords: 
            return True
        else:
            return False

    def propose_next_coord(self, coord, action):
        if action == 0:  # left
            proposed_coord = (self.current_coord[0], self.current_coord[1] - 1)
        elif action == 1: # right
            proposed_coord = (self.current_coord[0], self.current_coord[1] + 1)
        return proposed_coord