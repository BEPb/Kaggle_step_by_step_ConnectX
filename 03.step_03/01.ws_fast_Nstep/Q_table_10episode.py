def my_agent(observation, configuration):
    from random import choice

    q_table = {'1':0,'22':1,'3bef':0,'1b620ce':0,'120a':0,'1519':0,'31a368':0,'323d37':0,'df6b644e':0,'df7b9c33':0,'4a0cac74fe0':0,'4e29205b743':0,'29b6872a1a9874':0,'12e8e72de8f86663':0,'12e8e72de8fdceaa':0,'12e8f50cf2245f69':0,'bba1a95f1b447acc':0,'a72':1,'4df41ab51':0,'4df41ab8a':0,'4df49c81d':5,'4df4378be':0,'4df4378cd':0,'569d8fc74':0,'4a50fddff1f':1,'19737b2525f6':0,'279952c7dab69d':6,'6c7':0,'f65f8f6':0,'f9ebc8d':0,'f9ebcf0':0,'fa20ab1':0,'e497d4c4':0,'175a8424c5':0,'4f8c2a5dd78':0,'12d7b4b4f68f':0,'3fb6313e83fd6e':0,'1fa69d327bb999aa7':0,'2e':0,'6d3':1,'1dcb0':0,'744f7':0,'8a9cc89e':0,'2505d9fe24d':0,'251fd606d42':0,'2522b8f8fd7':0,'7688df8ab774a4':3,'3a':0,'f65f269':0,'111bf0e0':0,'8395556395':0,'4640e2c1c53dc':0,'99d0':0,'9a8d':0,'267e2':0,'7d2b1':4,'56881':0,'2e348dec':0,'2e3674df':0,'18ac2954fe8':0,'6975e28ed002b':1,'6975e28ed0a9c':1,'6975e29463df7':0,'1b5feb1':0,'1b69880':0,'6d849db':0,'1a698d4d0':0,'71cf65e2521':0}
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