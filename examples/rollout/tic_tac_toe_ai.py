import sys
sys.path.append('../../modules')
from env.tic_tac_toe import TicTacToe
import numpy as np

# ***** welcome messages *****

print('==============================')

print('Welcome to the rollout tic-tac-toe AI.')

# ***** player choice of first-hand or second-hand *****

while True:
	first_hand = input('Who goes first? (type "me" for me, "ai" for AI)\n')
	if first_hand in ['me', 'ai']:
		break
	else:
		print('Please try again.')

if first_hand == 'me':
	print('You are going first and are assigned the "x" stone.')
	current_side = 'cross'
	human_side = 'cross'
	ai_side = 'circle'
elif first_hand == 'ai':
	print('You are going second and are assigned the "o" stone.')
	current_side = 'cross'
	human_side = 'circle'
	ai_side = 'cross'

print('==============================')

# ***** utility functions *****

def board_to_str(board):
    board_str = ''
    board_str += '    0 1 2 \n'
    board_str += '   -------\n'
    for i, row in enumerate(board):
        board_str += f' {i} |'
        for item in row:
            if item == 0:
                board_str += ' '
            elif item == -1:
                board_str += 'o'
            elif item == 1:
                board_str += 'x'
            board_str += '|'
        board_str += '\n'
        if i == len(board) - 1:
            board_str += '   -------'
        else:
            board_str += '   -------\n'
    return board_str

def swap_side(current_side):
	if current_side == 'cross':
		return 'circle'
	elif current_side == 'circle':
		return 'cross'

def rollout(game, action):
	
	mind_game = TicTacToe()
	mind_game.board = game.board.copy()

	first_action = True
	current_side = ai_side
	
	while True:
		
		if first_action:
			_, reward, done = mind_game.step(a, current_side)
			first_action = False
		else:
			_, reward, done = mind_game.step(mind_game.sample_action_space(), current_side)
		
		if done:
			
			if current_side == ai_side:
				
				if reward == 0: return reward
				elif reward == 1: return reward
			
			elif current_side == human_side:
				
				if reward == 0: return 0
				elif reward == 1: return -100  # this was able to fix some trouble; undo it and use human first to see some trouble
		
		current_side = swap_side(current_side)

game = TicTacToe()

while True:

	print('==============================')
	print(board_to_str(game.board))
	print('==============================')

	# *****

	if current_side == human_side:
		
		position = tuple(map(int, input('Your position tuple (row, col): ').split(',')))
		_, reward, done = game.step(position, current_side)
	
	elif current_side == ai_side:
		
		# the true AI part
		print('AI is contemplating ...')
		values = np.zeros((len(game.action_space), ))
		for i, a in enumerate(sorted(game.action_space)):
			returns = []
			for n in range(1500):
				returns.append(rollout(game, a))
			values[i] = np.mean(returns)
		_, reward, done = game.step(sorted(game.action_space)[np.argmax(values)], current_side)

	if done:
		print('========== GAME HAS ENDED ==========')
		if reward == 0:
			print('Result: draw')
			print(board_to_str(game.board))
			print('====================================')
			break
		elif reward == 1:
			winning_player = 'human' if human_side == current_side else 'AI'
			print(f'Result: {current_side} (held by {winning_player}) wins!')
			print(board_to_str(game.board))
			print('====================================')
			break

	current_side = swap_side(current_side)

	

