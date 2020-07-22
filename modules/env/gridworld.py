import numpy as np

class VanillaGridWorld:
    
    def __init__(self, num_rows, num_cols, start_coord, end_coord):
        
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.start_coord = start_coord
        self.end_coord = end_coord
        self.current_coord = start_coord
        
        self.num_actions = 4
        
        self.episode_terminated = False

    @property
    def shape(self):
        return self.num_rows, self.num_cols, self.num_actions

    def get_actions(self, s):
        return np.arange(self.num_actions)

    def is_further_action_possible(self, s):
        if s == self.env.end_coord:
            return False
        else:
            return True
    
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

    def is_coord_terminal(self, coord):
        if self.current_coord == self.end_coord: 
            return True
        else:
            return False

    def get_reward(self, coord):
        return -1

    def step(self, action):
        
        assert not self.episode_terminated
        
        proposed_next_coord = self.propose_next_coord(self.current_coord, action)
        
        if self.is_coord_valid(proposed_next_coord):
            self.current_coord = proposed_next_coord
        
        self.episode_terminated = self.is_coord_terminal(self.current_coord)
        
        return self.current_coord, self.get_reward(self.current_coord)
    
    def reset(self):
        self.current_coord = self.start_coord
        self.episode_terminated = False


class GridWorldWithWallsAndTraps(VanillaGridWorld):

    """
    Traps are grids with HUGE negative rewards. Walls are grids that are not walkable; if an agent walks on a wall grid, it is
    immediately sent back to its last grid and receives a reward given by self.reward_array[self.current_coord].
    """

    def __init__(self, 
        grid_type_array, 
        trap_reward=-100,
        grid_type_to_int_dict={'generic_walkable':0, 'start':1, 'wall':2, 'trap':3, 'end':4}
        ):

        self.grid_type_array = grid_type_array
        
        self.reward_array = np.zeros_like(self.grid_type_array)
        for k, v in grid_type_to_int_dict.items():
            if k in ['generic_walkable', 'start', 'end']:
                self.reward_array[self.grid_type_array == v] = -1
            elif k == 'trap':
                self.reward_array[self.grid_type_array == v] = trap_reward

        assert len(np.argwhere(self.grid_type_array == grid_type_to_int_dict['start'])) == 1, "There can only be one start state."
        assert len(np.argwhere(self.grid_type_array == grid_type_to_int_dict['end'])) == 1, "There can only be one end state."

        self.start_coord = tuple(np.argwhere(self.grid_type_array == grid_type_to_int_dict['start'])[0])
        self.end_coord = tuple(np.argwhere(self.grid_type_array == grid_type_to_int_dict['end'])[0])

        self.wall_coords = [tuple(ls) for ls in np.argwhere(self.grid_type_array == grid_type_to_int_dict['wall']).tolist()]

        self.num_rows, self.num_cols = self.grid_type_array.shape
        self.num_actions = 4

        self.episode_terminated = False

    def is_actionable(self, s):
        if s == self.end_coord or s in self.wall_coords:
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


