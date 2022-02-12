import random
import numpy as np


########################################################
def boardToPatterns(grid, config):
    pats = boardDiagonals(grid, config)
    pats.extend(boardHorizontals(grid, config))
    pats.extend(boardHorizontals(grid.T, config))
    pats = list(filter(lambda x: x.count(0) <= 2, pats))
    return pats


def boardDiagonals(grid, config):
    diags = []
    for col in range(config.columns - (config.inarow - 1)):
        for row in range(config.rows - (config.inarow - 1)):
            w = []
            for i in range(config.inarow):
                w.append(grid[row+i][col+i])
            diags.append(w)
            for row in range(config.inarow - 1, config.rows):
                w = []
                for i in range(config.inarow):
                    w.append(grid[row-i][col+i])
            diags.append(w)
    return diags


def boardHorizontals(grid, config):
    pats = []
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1] - (config.inarow - 1)):
            pats.append(list(grid[row, col:col + config.inarow]))
    return pats

def score_move(grid, col, mark, config, nsteps):
    next_grid = drop_piece(grid, col, mark, config)
    score = minimax(next_grid, nsteps-1, False, mark, config, -np.Inf, np.Inf, boardToPatterns(next_grid, config))
    return score

def drop_piece(grid, col, mark, config):
    next_grid = grid.copy()
    for row in range(config.rows-1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return next_grid

def get_heuristic(grid, mark, config, patterns):
    weights = [1, 1e2, 1e6]
    weights_opp = [1.1, 1.1e2, 1e6]
    score = 0
    for n in range(3):
        score += count_windows_in_pattern(patterns, n + 2, mark) * weights[n]
        score -= count_windows_in_pattern(patterns, n + 2, (mark % 2) + 1) * weights_opp[n]
    return score

def count_windows_in_pattern(patterns, num, piece):
    return sum([window.count(piece) == num and window.count(0) == (config.inarow - num) for window in patterns])

def is_terminal_node(grid, config, patterns):
    #draw
    if list(grid[0, :]).count(0) == 0:
        return True
    #win
    return count_windows_in_pattern(patterns, config.inarow, 1) > 0 or count_windows_in_pattern(patterns, config.inarow, 2) > 0


########################################################

# Minimax, ab pruning
def minimax(node, depth, maximizingPlayer, mark, config, A, B, patterns):
    if depth == 0 or is_terminal_node(node, config, patterns):
        return get_heuristic(node, mark, config, patterns)

    valid_moves = [c for c in range(config.columns) if node[0][c] == 0]

    if maximizingPlayer:
        value = -np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark, config)
            value = max(value, minimax(child, depth-1, False, mark, config, A, B, boardToPatterns(child, config)))
            if value >= B:
                break
            A = max(A, value)
        return value
    else:
        value = np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark%2+1, config)
            value = min(value, minimax(child, depth-1, True, mark, config, A, B, boardToPatterns(child, config)))
            if value <= A:
                break
            B = min(B, value)
        return value

def innermost(arr):
    mid = (config.columns - 1) / 2
    distance = [-abs(c-mid) for c in arr]
    return arr[np.argmax(distance)]


def N_step_lookahead_fast(obs, config):
    order = [config.columns//2 - i//2 - 1 if i%2 else config.columns//2 + i//2 for i in range(config.columns)]
    valid_moves = [c for c in order if obs.board[c] == 0]
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    return innermost(max_cols)