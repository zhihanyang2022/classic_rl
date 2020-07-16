class GridWorld:
    
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
    
    def is_out_of_world(self, coord):
        return coord[0] < 0 or coord[0] > self.num_rows - 1 or coord[1] < 0 or coord[1] > self.num_cols - 1
        
    def step(self, action):
        
        assert not self.episode_terminated
        
        if action == 0:  # up
            proposed_coord = (self.current_coord[0] - 1, self.current_coord[1])
        elif action == 1:  # right
            proposed_coord = (self.current_coord[0], self.current_coord[1] + 1)
        elif action == 2:  # down
            proposed_coord = (self.current_coord[0] + 1, self.current_coord[1])
        elif action == 3:  # left
            proposed_coord = (self.current_coord[0], self.current_coord[1] - 1)
            
        if not self.is_out_of_world(proposed_coord):
            self.current_coord = proposed_coord
        
        if self.current_coord == self.end_coord: self.episode_terminated = True
        
        return self.current_coord, -1
    
    def reset(self):
        self.current_coord = self.start_coord
        self.episode_terminated = False