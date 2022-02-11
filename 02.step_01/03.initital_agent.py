import random  # connect the library for working with random numbers
import numpy as np  # connect the library general mathematical and numerical operations
'''
Helper Functions
- drop_piece: return grid status after player drops a piece
- check_winning_move : used to check if dropping a piece in a column of board leads to a winning move or not
'''

# Gets board at next step if agent drops piece in selected column
def drop_piece(grid, col, piece, config):
    next_grid = grid.copy()
    for row in range(config.rows-1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = piece
    return next_grid

# Returns True if dropping piece in column results in game win
def check_winning_move(obs, config, col, piece):
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    next_grid = drop_piece(grid, col, piece, config)
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(next_grid[row,col:col+config.inarow])
            if window.count(piece) == config.inarow:
                return True
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(next_grid[row:row+config.inarow,col])
            if window.count(piece) == config.inarow:
                return True
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(next_grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if window.count(piece) == config.inarow:
                return True
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(next_grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if window.count(piece) == config.inarow:
                return True
    return False


def my_agent(obs, config):
    opponent_piece = 1 if obs.mark == 2 else 2
    choice = []
    for col in range(config.columns):
        if check_winning_move(obs,config,col,obs.mark):
            return col
        elif check_winning_move(obs,config,col,opponent_piece):
            choice.append(col)
    if len(choice):
        return random.choice(choice)
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return random.choice(valid_moves)