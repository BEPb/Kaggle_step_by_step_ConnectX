# Importing Dependencies
import random  # connect the library for working with random numbers
import numpy as np  # connect the library general mathematical and numerical operations

'''
Helper Functions
- drop_piece: return grid status after player drops a piece
- check_winning_move : used to check if dropping a piece in a column of board leads to a winning move or not
'''

# Gets board at next step if agent drops piece in selected column
def drop_piece(grid, col, piece, config):  #
    next_grid = grid.copy()  # make a copy of the location of the chips on the playing field for its further transformation
    for row in range(config.rows-1, -1, -1):  # iterate over all rows in the playing field
        if next_grid[row][col] == 0:  # we are not interested in empty cells
            break  # we skip them if we meet such
    next_grid[row][col] = piece # mark the cell in which our chip will fall
    return next_grid # return board at next step


# Returns True if dropping piece in column results in game win
def check_winning_move(obs, config, col, piece):
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # use our function to predict all possible combinations of the next step
    next_grid = drop_piece(grid, col, piece, config)

    # horizontal
    for row in range(config.rows):  # iterate over all rows in the playing field
        for col in range(config.columns - (config.inarow - 1)):  # iterate over all columns in the playing field
            window = list(next_grid[row,
                          col:col + config.inarow])  # we form a list of all horizontal lines of chips (1, 2, 3, 4 and more ...)
            if window.count(
                    piece) == config.inarow:  # if there is a number of chips in the list equal to the win condition (default = 4 chips)
                return True  # we found a winning combination (move)

    # vertical
    for row in range(config.rows - (config.inarow - 1)):  # iterate over all rows in the playing field
        for col in range(config.columns):  # iterate over all columns in the playing field
            window = list(next_grid[row:row + config.inarow,
                          col])  # we form a list of all vertical lines of chips (1, 2, 3, 4 and more ...)
            if window.count(
                    piece) == config.inarow:  # if there is a number of chips in the list equal to the win condition (default = 4 chips)
                return True  # we found a winning combination (move)

    # positive diagonal
    for row in range(config.rows - (config.inarow - 1)):  # iterate over all rows in the playing field
        for col in range(config.columns - (config.inarow - 1)):  # iterate over all columns in the playing field
            window = list(next_grid[range(row, row + config.inarow), range(col,
                                                                           col + config.inarow)])  # we form a list of all pos. diagonal lines of chips
            if window.count(
                    piece) == config.inarow:  # if there is a number of chips in the list equal to the win condition (default = 4 chips)
                return True  # we found a winning combination (move)

    # negative diagonal
    for row in range(config.inarow - 1, config.rows):  # iterate over all rows in the playing field
        for col in range(config.columns - (config.inarow - 1)):  # iterate over all columns in the playing field
            window = list(next_grid[range(row, row - config.inarow, -1), range(col,
                                                                               col + config.inarow)])  # we form a list of all neg. diagonal lines of chips
            if window.count(
                    piece) == config.inarow:  # if there is a number of chips in the list equal to the win condition (default = 4 chips)
                return True  # we found a winning combination (move)

    '''if none of the victory conditions is met, we go from the opposite, on the next turn no one wins, the game continues'''
    return False  # no one will win on the next turn


'''the main function of our agent'''
def my_agent(obs, config):  # your agent receives information about the state of the game board and the settings of the virtual environment
    opponent_piece = 1 if obs.mark == 2 else 2  # designation of players 1 and 2
    choice = []  # create an empty list
    for col in range(config.columns):  # iterate over all columns in the playing field
        if check_winning_move(obs, config, col, obs.mark):  # check if your agent can win this turn
            return col  # we found a winning combination (move) your agent
        elif check_winning_move(obs, config, col, opponent_piece):  # check if opponent agent can win this turn
            choice.append(col)  # we found a winning combination (move) opponent agent
    if len(choice):  # if we have found combinations when our opponent can win, we will count the number of such combinations
        '''if the winning combination of the opponent is not one, then we choose randomly. 
        If the combination is one, then it will be chosen for the next move, thus we prevent the opponent from winning.'''
        return random.choice(choice)

    # we form a list of columns in which we can make a move, they are not yet fully occupied
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    return random.choice(
        valid_moves)  # if we have not found winning situations, then we make a random move from the possible