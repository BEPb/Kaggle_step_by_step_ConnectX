# Importing Dependencies
import random  # connect the library for working with random numbers
import numpy as np  # connect the library general mathematical and numerical operations

'''
Helper Functions:
- score_move: calculates score if agent drops piece in selected column
- drop_piece: return grid status after player drops a piece
- get_heuristic: calculates value of heuristic for grid
- check_window: checks if window satisfies heuristic conditions
- count_windows: counts number of windows satisfying specified heuristic conditions
'''


# Calculates score if agent drops piece in selected column
def score_move(grid, col, mark, config):
    next_grid = drop_piece(grid, col, mark, config)  # board at next step
    score = get_heuristic(next_grid, mark, config)  # score in next step
    return score


# Gets board at next step if agent drops piece in selected column
def drop_piece(grid, col, mark, config):  #
    next_grid = grid.copy()  # make a copy of the location of the chips on the playing field for its further transformation
    for row in range(config.rows - 1, -1, -1):  # iterate over all rows in the playing field
        if next_grid[row][col] == 0:  # we are not interested in empty cells
            break  # we skip them if we meet such
    next_grid[row][col] = mark  # mark the cell in which our chip will fall
    return next_grid  # return board at next step


# calculates value of heuristic for grid
def get_heuristic(grid, mark, config):
    # coefficients of the positions
    A = 1e6  # if the agent has four disks in a row (the agent won), he gets  1 000 000 points
    B = 1  # if the agent has filled three spaces and the remaining space is empty (the agent wins on the next turn if he fills one more empty space)
    C = 0.5  # if your agent has filled two spaces and the remaining space is empty (your agent wins by filling the empty space after two more moves)
    D = -0.5  # if the opponent has filled two spaces and the remaining space is empty (the opponent wins by filling the empty space after two more moves)
    E = -1e2  # if the opponent has filled three spaces and the remaining space is empty (the opponent wins by filling the empty space)

    # calculation of all interconnections of the positions of friendly and enemy cells
    num_twos = count_windows(grid, 2, mark, config)  # if your agent has filled two spaces
    num_threes = count_windows(grid, 3, mark, config)  # if the agent has filled three spaces
    num_fours = count_windows(grid, 4, mark, config)  # if the agent has four disks in a row (the agent won)
    num_twos_opp = count_windows(grid, 2, mark % 2 + 1, config)  # if the opponent has filled two spaces
    num_threes_opp = count_windows(grid, 3, mark % 2 + 1, config)  # if the opponent has filled three spaces

    '''after calculating all the interconnections of the positions of the cells of our own and the enemy, 
    we calculate the score of this board by applying the coefficients of the positions'''
    score = A * num_fours + B * num_threes + C * num_twos + D * num_twos_opp + E * num_threes_opp
    return score  # final score of this playing field position


# checks if window satisfies heuristic conditions
def check_window(window, num_discs, piece, config):
    return (window.count(piece) == num_discs and window.count(0) == config.inarow - num_discs)


# counts number of windows satisfying specified heuristic conditions
def count_windows(grid, num_discs, piece, config):
    num_windows = 0

    # horizontal
    for row in range(config.rows):  # iterate over all rows in the playing field
        for col in range(config.columns - (config.inarow - 1)):  # iterate over all columns in the playing field
            window = list(
                grid[row, col:col + config.inarow])  # we form a list of all horozontal lines of chips (2, 3 or 4)
            if check_window(window, num_discs, piece, config):  # if it satisfies the line construction condition
                num_windows += 1  # increase the count of situations that satisfy the given condition

    # vertical
    for row in range(config.rows - (config.inarow - 1)):  # iterate over all rows in the playing field
        for col in range(config.columns):  # iterate over all columns in the playing field
            window = list(
                grid[row:row + config.inarow, col])  # we form a list of all vertical lines of chips (2, 3 or 4)
            if check_window(window, num_discs, piece, config):  # if it satisfies the line construction condition
                num_windows += 1  # increase the count of situations that satisfy the given condition

    # positive diagonal
    for row in range(config.rows - (config.inarow - 1)):  # iterate over all rows in the playing field
        for col in range(config.columns - (config.inarow - 1)):  # iterate over all columns in the playing field
            window = list(grid[range(row, row + config.inarow), range(col,
                                                                      col + config.inarow)])  # we form a list of all pos. diagonal lines of chips (2, 3 or 4)
            if check_window(window, num_discs, piece, config):  # if it satisfies the line construction condition
                num_windows += 1  # increase the count of situations that satisfy the given condition

    # negative diagonal
    for row in range(config.inarow - 1, config.rows):  # iterate over all rows in the playing field
        for col in range(config.columns - (config.inarow - 1)):  # iterate over all columns in the playing field
            window = list(grid[range(row, row - config.inarow, -1), range(col,
                                                                          col + config.inarow)])  # we form a list of all neg. diagonal lines of chips (2, 3 or 4)
            if check_window(window, num_discs, piece, config):  # if it satisfies the line construction condition
                num_windows += 1  # increase the count of situations that satisfy the given condition

    return num_windows  # returns the number of possible wins for the given number of cells


# The agent is always implemented as a Python function that accepts two arguments: obs and config
def agent(obs, config):
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]  # Get list of valid moves
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)  # Convert the board to a 2D grid

    # Use the heuristic to assign a score to each possible board in the next turn
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config) for col in valid_moves]))

    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)