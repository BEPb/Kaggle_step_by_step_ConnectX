from threading import Thread
from time import sleep
from kaggle_environments import make # connect the library kaggle_environments simulation virtual environment and its functions
import random  # connect the library to generate random numbers, letters, random selection of sequence elements
from random import choice  # connect the library to generate choose a random item from a sequence. Here seq can be a list, tuple, string, or any iterable like range.
import numpy as np  # connect the library general mathematical and numerical operations
import os  # connect the library operating system
import gym  # toolkit for developing and comparing reinforcement learning algorithms
from tqdm import tqdm  # connect the library show a smart progress


# with open("../02.step_02/06.fast_Nstep_lookahead_agent.py","r") as saved_agent:
#     print(saved_agent.read())
saved_agent = '../02.step_02/06.fast_Nstep_lookahead_agent.py'
print(saved_agent)

env = make("connectx")
env.render()


class QTable:
    def __init__(self, action_space):
        self.table = dict()
        self.action_space = action_space

    def add_item(self, state_key):
        self.table[state_key] = list(np.zeros(self.action_space.n))

    def __call__(self, state):
        board = state.board[:]  # Get a copy
        board.append(state.mark)
        state_key = np.array(board).astype(str)
        state_key = hex(int(''.join(state_key), 3))[2:]
        if state_key not in self.table.keys():
            self.add_item(state_key)

        return self.table[state_key]

# Environment parameters
cols = 7
rows = 6

action_space = gym.spaces.Discrete(cols)
observation_space = gym.spaces.Discrete(cols * rows)

# configure hyper-parameters
alpha = 0.1
gamma = 0.6
epsilon = 0.99
min_epsilon = 0.1

episodes = 1  # 15000
alpha_decay_step = 1000
alpha_decay_rate = 0.9
epsilon_decay_rate = 0.9999

q_table = QTable(action_space)
trainer = env.train([None, saved_agent])

all_epochs = []
all_total_rewards = []
all_avg_rewards = []
all_q_table_rows = []
all_epsilons = []



###############################################################

class CustomThread(Thread):
    def __init__(self, limit):
        Thread.__init__(self)
        self._limit = limit
    def run(self, alpha=0.1, epsilon = 0.99):
        for b in range(self._limit):
            for i in tqdm(range(episodes)):
                state = trainer.reset()
                epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)
                epochs, total_rewards = 0, 0
                done = False

                while not done:
                    if random.uniform(0, 1) < epsilon:
                        action = choice([c for c in range(action_space.n) if state.board[c] == 0])

                    else:
                        row = q_table(state)[:]
                        selected_items = []
                        for j in range(action_space.n):
                            if state.board[j] == 0:
                                selected_items.append(row[j])
                            else:
                                selected_items.append(-1e7)
                        action = int(np.argmax(selected_items))

                    next_state, reward, done, info = trainer.step(action)

                    # apply new rules
                    if done:
                        if reward == 1:
                            reward = 20
                        elif reward == 0:
                            reward = -20
                        else:
                            reward = 10

                    else:
                        reward = -0.05

                    old_value = q_table(state)[action]
                    next_max = np.argmax(q_table(next_state))

                    # update q value
                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    q_table(state)[action] = new_value

                    state = next_state
                    epochs += 1
                    total_rewards += reward

                all_epochs.append(epochs)
                all_total_rewards.append(total_rewards)
                avg_rewards = np.mean(all_total_rewards[max(0, i - 100): (i + 1)])
                all_avg_rewards.append(avg_rewards)
                all_q_table_rows.append(len(q_table.table))
                all_epsilons.append(epsilon)

                if (i + 1) % alpha_decay_step == 0:
                    alpha += alpha_decay_rate
            
            
            
cth = CustomThread(5)
cth.start()
cth.join()


# for i in range(5):
#     thread = threading.Thread(
#         target=write_genre,
#         args=[f"./threading/new_file{i}.txt"]
#     )
#     thread.start()
# threads = []
# threads.append(thread)
#
# for thread in threads:
#     thread.join()



############################################################

tmp_dict_q_table = q_table.table.copy()
dict_q_table = dict()

for k in tmp_dict_q_table:
    if np.count_nonzero(tmp_dict_q_table[k]) > 0:
        dict_q_table[k] = int(np.argmax(tmp_dict_q_table[k]))


def my_agent(observation, configuration):
    from random import choice

    q_table = dict_q_table
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


agent = """def my_agent(observation, configuration):
    from random import choice

    q_table = """ + str(dict_q_table).replace(" ", "") + """
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
    return action """

with open("submission.py", 'w') as f:
    f.write(agent)

print(len(q_table.table))
print("%s Kb" % round(os.stat('submission.py').st_size/1024))






















# from multiprocessing import Process
# import os
#
# def info(title):
#     print(title)
#     print('module name:', __name__)
#     print('parent process:', os.getppid())
#     print('process id:', os.getpid())
#
# def f(name):
#     info('function f')
#     print('hello', name)
#
# if __name__ == '__main__':
#     info('main line')
#     p = Process(target=f, args=('bob',))
#     p.start()
#     p.join()