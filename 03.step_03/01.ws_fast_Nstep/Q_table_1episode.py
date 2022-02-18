def my_agent(observation, configuration):
    from random import choice

    q_table = {'1':0,'30aa80':0,'30acb7':0,'2e628344':1,'18abb0ee115':0,'18abb0f7ae4':0,'d2c2a81dccf27':4}
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