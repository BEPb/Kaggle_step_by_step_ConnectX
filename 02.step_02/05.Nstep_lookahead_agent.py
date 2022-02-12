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
- score_move: Uses minimax to calculate value of dropping piece in selected column
- minimax: Minimax implementation
- is_terminal_window: checks if agent or opponent has four in a row in the window
- is_terminal_node: checks if game has ended
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
    # calculation of all interconnections of the positions of friendly and enemy cells
    num_threes = count_windows(grid, 3, mark, config)  # if the agent has filled three spaces
    num_fours = count_windows(grid, 4, mark, config)  # if the agent has four disks in a row (the agent won)
    num_threes_opp = count_windows(grid, 3, mark % 2 + 1, config)  # if the opponent has filled three spaces
    num_fours_opp = count_windows(grid, 4, mark % 2 + 1,
                                  config)  # if the opponent has four disks in a row (the opponent won)

    '''after calculating all the interconnections of the positions of the cells of our own and the enemy, 
    we calculate the score of this board by applying the coefficients of the positions:    
    1 - if the agent has filled three spaces, he get +1 point
    -100 - if the opponent has filled three spaces,  we gets  -100 points 
    - 1e4 - if the opponent has four disks in a row (the opponent won), we gets  -10 000 points
    1e6 - if the agent has four disks in a row (the agent won), he gets  +1 000 000 points
    '''
    score = num_threes - 1e2 * num_threes_opp - 1e4 * num_fours_opp + 1e6 * num_fours
    return score


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


#  Uses minimax to calculate value of dropping piece in selected column
# Использует минимакс для расчета стоимости падающей части в выбранном столбце
def score_move(grid, col, mark, config, nsteps):
    next_grid = drop_piece(grid, col, mark, config)
    score = minimax(next_grid, nsteps - 1, False, mark, config)
    return score


# Helper function for minimax: checks if agent or opponent has four in a row in the window
# Вспомогательная функция для минимакса: проверяет, есть ли у агента или оппонента четыре подряд в окне
def is_terminal_window(window, config):
    return window.count(1) == config.inarow or window.count(2) == config.inarow


# Helper function for minimax: checks if game has ended
def is_terminal_node(grid, config):
    # Check for draw
    if list(grid[0, :]).count(0) == 0:
        return True
    # Check for win: horizontal, vertical, or diagonal

    # horizontal
    for row in range(config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[row, col:col + config.inarow])
            if is_terminal_window(window, config):
                return True
    # vertical
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns):
            window = list(grid[row:row + config.inarow, col])
            if is_terminal_window(window, config):
                return True
    # positive diagonal
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
            if is_terminal_window(window, config):
                return True
    # negative diagonal
    for row in range(config.inarow - 1, config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
            if is_terminal_window(window, config):
                return True
    return False


# Minimax implementation
def minimax(node, depth, maximizingPlayer, mark, config):
    is_terminal = is_terminal_node(node, config)
    valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
    if depth == 0 or is_terminal:
        return get_heuristic(node, mark, config)
    if maximizingPlayer:
        value = -np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark, config)
            value = max(value, minimax(child, depth - 1, False, mark, config))
        return value
    else:
        value = np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark % 2 + 1, config)
            value = min(value, minimax(child, depth - 1, True, mark, config))
        return value


# How deep to make the game tree: higher values take longer to run!
N_STEPS = 3


def agent(obs, config):
    # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)