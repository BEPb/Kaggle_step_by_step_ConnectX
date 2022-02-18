def my_agent(observation, configuration):
    from random import choice

    q_table = {'1':0,'99d0':0,'9af9':0,'f668d28':0,'f76c50d':0,'8393b05070':0,'23248e0c59f71':0,'23257804066e4':0,'2af2e02c7625b':5}
    board = observation.board[:]
    board.append(observation.mark)
    state_key = list(map(str, board))
    state_key = hex(int(''.join(state_key), 3))[2:]

    if state_key not in q_table.keys():
        return choice([c for c in range(configuration.columns) 
                       if observation.board[c] == 0])
    action = q_table[state_key]

    if observation.board[action] != 0:
        return choice([c for c in range(configuration.columns) 
                       if observation.board[c] == 0])
    return action 