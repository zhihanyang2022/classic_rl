import numpy as np

class TicTacToe():
    
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.side = None
        self.progress = None
        self.done = None
        
    def reset(self):
        self.__init__()

    @property
    def hash_board(self):
        return tuple([i for i in list(self.board.flatten())])
        
    def _check_align(self, array):
        return 0 not in array and np.unique(array).size == 1
        
    def _eval_board(self):
        
        # 3 circles / crosses align in row
        for row in self.board: 
            if self._check_align(row): return 'win'
        
        # 3 circles / crosses align in col
        for col in self.board.T: 
            if self._check_align(col): return 'win'
        
        # 3 circles / crosses align in diagonal
        diag = np.diagonal(self.board)
        if self._check_align(diag): return 'win'
        
        # 3 circles / crosses align in opposite diagonal
        oppo_diag = np.diagonal(np.fliplr(self.board))
        if self._check_align(oppo_diag): return 'win'
        
        # game board filled
        if 0 not in np.unique(self.board): return 'draw'
        
        return 'continue'
    
    def step(self, action, side):
        
        assert self.board[action] == 0, 'the position is filled'
        
        self.side = side
        
        if side == 'circle':
            self.board[action] = -1
        elif side == 'cross' :
            self.board[action] = 1
        
        self.progress = self._eval_board()
        if self.progress == 'continue':
            reward = 0
            done = False
        elif self.progress == 'win':
            reward = 1
            done = True
        elif self.progress == 'draw':
            reward = 0
            done = True
        
        return self.board, reward, done
    
    @property
    def action_space(self):
        advanced_indices = np.where(self.board == 0)
        actions = []
        for x, y in zip(advanced_indices[0], advanced_indices[1]):
            actions.append((x, y))
        return actions
    
    def sample_action_space(self):
        index = np.random.randint(len(self.action_space))
        return self.action_space[index]