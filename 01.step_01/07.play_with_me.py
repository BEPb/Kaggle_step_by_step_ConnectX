import numpy as np
import pandas as pd
import time
from IPython.display import clear_output
from kaggle_environments import make as make_game
from random import choice
from recordtype import recordtype

ExperienceStep = recordtype('ExperienceStep', [
    'game_id',
    'current_network_input',
    'action',
    'next_network_input',
    'last_episode_action',
    'episode_reward',
])

# Collect user input
def get_input(user, observation, configuration):
    ncol = configuration.columns
    time.sleep(0.1)
    input1 = 'Input from player {}: '.format(your_name)
    while True:
        try:
            print('Enter Value from 1 to 7')
            raw_input = input(input1)
            user_input = int(raw_input)
        except ValueError:
            try:
                print('Invalid input:', user_input)
                continue
            except UnboundLocalError:
                user_input = -1
                if raw_input == 'q':
                    break
                continue
        np_board = obs_to_board(observation, configuration)
        valid_actions = np.where(np_board[0] == 0)[0]
        if user_input <= 0 or user_input > ncol or (
                user_input - 1) not in valid_actions:
            print('invalid input:', user_input)
            print('Valid actions: {}'.format(valid_actions + 1))
        else:
            return user_input - 1


# Convert the 1D observation list to a 2D numpy array
def obs_to_board(observation, configuration):
    return np.array(observation.board).reshape(
        configuration.rows, configuration.columns)


def check_winner(observation):
    '''
    This function returns the value of the winner.
    INPUT:  observation
    OUTPUT: 1 for user Winner or 2 for Computer Winner
    '''
    line1 = observation.board[0:7]  # bottom row
    line2 = observation.board[7:14]
    line3 = observation.board[14:21]
    line4 = observation.board[21:28]
    line5 = observation.board[28:35]
    line6 = observation.board[35:42]

    board = [line1, line2, line3, line4, line5, line6]

    # Check rows for winner
    for row in range(6):
        for col in range(4):
            if (board[row][col] == board[row][col + 1] == board[row][col + 2] == (
                    board[row][col + 3])) and (board[row][col] != 0):
                return board[row][col]  # Return Number that match row

    # Check columns for winner
    for col in range(7):
        for row in range(3):
            if (board[row][col] == board[row + 1][col] == board[row + 2][col] == (
                    board[row + 3][col])) and (board[row][col] != 0):
                return board[row][col]  # Return Number that match column

    # Check diagonal (top-left to bottom-right) for winner
    for row in range(3):
        for col in range(4):
            if (board[row][col] == board[row + 1][col + 1] == board[
                row + 2][col + 2] == \
                board[row + 3][col + 3]) and (board[row][col] != 0):
                return board[row][col]  # Return Number that match diagonal

    # Check diagonal (bottom-left to top-right) for winner
    for row in range(5, 2, -1):
        for col in range(4):
            if (board[row][col] == board[row - 1][col + 1] == (
                    board[row - 2][col + 2]) == board[row - 3][col + 3]) and (
                    board[row][col] != 0):
                return board[row][col]  # Return Number that match diagonal

    # No winner: return None
    return None


# Custom class to reuse data of subsequent interations with the environment
# FIFO buffer. Experience buffer (also referred to as the replay buffer).
class ExperienceBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.episode_offset = 0
        self.data = []
        self.episode_ids = np.array([])

    def add(self, data):
        episode_ids = np.array([d.game_id for d in data])
        num_episodes = episode_ids[-1] + 1
        if num_episodes > self.buffer_size:
            # Keep most recent experience of the experience batch
            data = data[
                   np.where(episode_ids == (num_episodes - self.buffer_size))[0][0]:]
            self.data = data
            self.episode_ids = episode_ids
            self.episode_offset = 0
            return

        episode_ids = episode_ids + self.episode_offset
        self.data = data + self.data
        self.episode_ids = np.concatenate([episode_ids, self.episode_ids])

        unique_episode_ids = pd.unique(self.episode_ids)
        if unique_episode_ids.size > self.buffer_size:
            cutoff_index = np.where(self.episode_ids == unique_episode_ids[
                self.buffer_size])[0][0]
            self.data = self.data[:cutoff_index]
            self.episode_ids = self.episode_ids[:cutoff_index]
        self.episode_offset += num_episodes

    def get_all_data(self):
        return self.data

    def size(self):
        return len(self.data)

    def num_episodes(self):
        return np.unique(self.episode_ids).size


your_name = 'Andrej'  # @param {type:"string"}
play_against_random = True  # @param ["False", "True"] {type:"raw"}
plot_resolution = 400  # @param {type:"slider", min:200, max:500, step:1}


# Here we define an agent that picks a random non-empty column
def my_random_agent(observation, configuration):
    return int(choice([c for c in range(
        configuration.columns) if observation.board[c] == 0]))


def play_against_agent(opponent_agent):
    # Play as first position against the opposing agent.
    env = make_game("connectx", debug=False, configuration={"timeout": 10})
    trainer = env.train([None, opponent_agent])
    observation = trainer.reset()

    while not env.done:
        clear_output(wait=True)  # Comment if you want to keep track of every action
        print("{}'s color: Blue".format(your_name))
        out = env.render(mode="ansi", width=plot_resolution, height=plot_resolution, header=False, controls=False)
        print(out)
        # env.render(mode="ipython", width=plot_resolution, height=plot_resolution, header=False, controls=False)

        my_action = get_input(your_name, observation, env.configuration)
        if my_action is None:
            print("Exiting game after pressing q")
            return

        observation, reward, done, info = trainer.step(my_action)
        # print(observation, reward, done, info)
        if (check_winner(observation) == 1):
            print("You Won, Amazing! \nGAME OVER")

        elif (check_winner(observation) == 2):
            print("The opponent Won! \nGAME OVER")

    if (check_winner(observation) is None):
        print("That is a draw between you and the opponent")

    env.render(mode="ipython", width=plot_resolution, height=plot_resolution,
               header=False, controls=False)


if play_against_random:
    play_against_agent(my_random_agent)