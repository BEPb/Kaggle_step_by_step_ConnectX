def my_agent(observation, configuration):
    from random import choice

    q_table = {'1':1,'22':0,'211':0,'21c46':0,'523cb19':0,'15f0881b5a':0,'15f0986cdd':0,'16923d65dc':1,'16923d7419':1,'1a11f32c4d8':0,'696f5e049a8bf':0,'696f5f089a6b0':0,'75246513e0f9b':0,'70827569841ab892':0,'7d01eb10c99db54b':6,'99d0':0,'9a75':1,'5225182':0,'5328a57':0,'2bdbf0748e':2,'3a':0,'10390f':0,'15a156':0,'178849':0,'26942518':0,'30aa80':1,'30b3ad':0,'314d7c':0,'ddfc2146883':0,'ddfc2a65f78':1,'de23222e56d':0,'de26054bbfa':0,'e0e39a363d1':0,'7748ede38e419c':1,'1dadc960105f003':0,'1dadc9602bbd5cc':0,'1fc04fb170ac7e375':0,'20251e08a71b9b4ea':1,'7f0d0377c80c2ce45':3,'604':1,'57685':0,'576c4':0,'1f97e9':0,'2031b2':4,'2e':0,'1cd83':0,'7b4ed0c':0,'838bee786f':0,'9229b35b0c':0,'be030202e3':0,'177418a96f4a6':0,'177418a96fb4b':0,'17742152c7ef2':0,'17742435e557f':0,'1f4de0c3b7b48':0,'1f4dfabfc063d':2,'30ac81':0,'3279dc':0,'327a33':0,'2e6450c0':0,'ddff0476bc7':0,'ddff057bd4a':0,'ddff579b8af':0,'ea5c65d4b08':4,'a72':0,'a441':0,'a462':0,'5226c85':0,'8fb7f02c':0,'9094afe9':0,'9ff9be48':0,'2c76cfe2b1':0,'bb7cd7844b9c':1,'bb7e77757e37':3,'120a':1,'594565':0,'5945a4':0,'57b4109':1,'299f4b24921e':0,'29cb21e2658d':0,'1a04ea9775750':2,'6a6':1,'104713':0,'15af5a':0,'15af99':0,'2e45b8d4':0,'c5752dda69':0,'c5849332d2':3}
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