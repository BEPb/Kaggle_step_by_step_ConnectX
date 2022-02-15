from kaggle_environments import evaluate, make, utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import os
import pickle
import seaborn
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


if torch.cuda.is_available():
    torch.cuda.current_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
map_location = torch.device('cpu')
print("running calculations on: ", device)




# Create ConnectX Environment
env = make("connectx", debug=True)
configuration = env.configuration
columns = configuration.columns
mid_action = int(np.floor(columns/2))
rows = configuration.rows
print(configuration)

# Model Definition and Utility Functions
class Qnet(nn.Module):
    def __init__(self, configuration):
        super(Qnet, self).__init__()
        self.columns = configuration.columns
        self.rows = configuration.rows
        # Number of Checkers "in a row" needed to win.
        self.inarow = configuration.inarow
        input_shape = (3, self.rows, self.columns)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        #         self.do1 = nn.Dropout2d()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        #         self.do2 = nn.Dropout2d()
        linear_input_size = self._get_conv_output(input_shape)
        self.lin1 = nn.Linear(linear_input_size, 64)
        self.relu_lin1 = nn.ReLU()
        self.Q_o = nn.Linear(64, self.columns)

    #         self.tanh = nn.Tanh()  # BUGGY

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        #         x = F.relu(self.do1(self.conv1(x)))
        #         x = F.relu(self.do2(self.conv2(x)))
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.relu_lin1(self.lin1(x.view(x.size(0), -1)))
        Q = self.Q_o(x)
        #         Q = self.tanh(Q)  # BUGGY
        return Q


def form_net_input(observation, configuration):
    """
    Reshape board and one-hot it.
    """
    print(observation)
    # The current serialized Board (rows x columns).
    board = observation.board
    # Which player the agent is playing as (1 or 2).
    mark = observation.mark
    columns = configuration.columns
    rows = configuration.rows
    opponent = 2 if mark == 1 else 1
    newboard = [1 if i == mark else 2 if i == opponent else 0 for i in board]
    #         x = torch.Tensor(newboard).view([1,1, rows, columns])
    newboard = torch.tensor(newboard).reshape([rows, columns])
    x = F.one_hot(newboard, 3).permute([2, 0, 1])
    x = x.view([1, 3, rows, columns]).float()
    return x.to(device)


def choose_action(observation, configuration, net, is_training=False, eps=None):
    """
    epsilon-greedy agent.
    """
    if is_training:
        if random.random() < eps:
            return random.choice([c for c in range(configuration.columns) if observation.board[c] == 0])
    Qs = net(form_net_input(observation, configuration)).cpu().detach().numpy().flatten()
    return int(np.argmax([q_col if observation.board[col] == 0 else -np.inf for (col, q_col) in enumerate(Qs)]))


def eps_calculator(step, eps_start, eps_end, eps_decay):
    """
    reduce epsilon with steps.
    """
    return eps_end + (eps_start - eps_end) * math.exp(-2 * math.exp(1) * step / eps_decay)


# Replay Memory
def flip(t, dim=0):
    """
    inputs:
    t - torch tensor
    dim - dimension to flip (currently only 1 dimension supported)
    outputs:
    t_flipped - input t with dimension dim flipped
    """
    dim_size = t.size()[dim]
    reverse_indices = torch.arange(dim_size - 1, -1, -1, device=t.device)
    return t.index_select(dim, reverse_indices)


class Transition:
    def __init__(self, *args):
        if args is ():
            self.episode = None
            self.cur_obs = None
            self.action = None
            self.reward = None
            self.next_obs = None
            self.done = None
            self.values = None
        else:
            episode, cur_obs, action, reward, next_obs, done = args
            self.episode = torch.tensor([episode], device=device)
            self.cur_obs = form_net_input(cur_obs, configuration)
            self.action = torch.tensor([action], device=device)
            reward_engineered = 1 if reward == 1 else -1 if reward == 0 else 0
            self.reward = torch.tensor([reward_engineered], device=device)
            self.next_obs = form_net_input(next_obs, configuration)
            self.done = torch.tensor([done], device=device)
            self.values = TensorDataset(self.episode, self.cur_obs, self.action,
                                        self.reward, self.next_obs, self.done)

    def mirror_copy(self):
        """
        Creates a mirrored transition, flipping current and next state and the action taken.
        """
        mirror_transition = Transition()
        mirror_transition.episode = self.episode
        mirror_transition.cur_obs = flip(self.cur_obs, 3)
        mirror_transition.action = configuration.columns - 1 - self.action
        mirror_transition.reward = self.reward
        mirror_transition.next_obs = flip(self.next_obs, 3)
        mirror_transition.done = self.done
        mirror_transition.values = TensorDataset(
            mirror_transition.episode, mirror_transition.cur_obs, mirror_transition.action,
            mirror_transition.reward, mirror_transition.next_obs, mirror_transition.done)
        return mirror_transition


class ReplayMemory:
    """
    Basic ReplayMemory class.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __getitem__(self, item):
        return self.memory[item]

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        transition = Transition(*args)
        self.memory[self.position] = transition.values
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def to_dataset(self):
        return ConcatDataset(self.memory)


class DualMemory(ReplayMemory):
    """
    Change push method to push an experience, generate its mirrored experience and push it as well.
    """

    def push(self, *args):
        """Saves a transition and its symmetrical transition"""
        # save original transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        transition = Transition(*args)
        self.memory[self.position] = transition.values
        self.position = (self.position + 1) % self.capacity
        # save mirrored transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition.mirror_copy().values
        self.position = (self.position + 1) % self.capacity

# Optimization step
def optimize_step():
    episode_training_loss = []
    # sample transitions from replay memory
    dataloader = DataLoader(memory.to_dataset(), BATCH_SIZE, shuffle=True, drop_last=False)
    for epoch in range(NUM_EPOCHS):
        epoch_loss = []
        for batch_i, batch in enumerate(dataloader):
            _, cur_obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = batch

            # Qs of current observation, and choose index of action taken
            Q_current_all = policy_net(cur_obs_batch)
            mask = F.one_hot(action_batch, columns).bool()
            Qs_to_step = Q_current_all.masked_select(mask).float()

            # Qs of next observation, and choose maximum LEGAL action (not full column)
            next_state_Qs = target_net(next_obs_batch).float()
            next_state_illegal_actions = next_obs_batch[:, 0, 0, :].eq(0)  # check if column not full
            next_state_Qs[next_state_illegal_actions] = -float('inf')
            max_next_state_Qs = next_state_Qs.max(dim=1).values

            # target to gradient descent (y from DQN paper).
            y = reward_batch.float()
            y[done_batch == False] += GAMMA * (
                - max_next_state_Qs[done_batch == False])  # the minus sign is the result of the zero-sum property
            if y.sum() == float('inf') or y.sum() == -float('inf'):
                raise Exception("Big Problem")

            # calculate loss
            cur_train_loss = loss(y, Qs_to_step)
            # save loss for plotting
            epoch_loss.append(cur_train_loss.flatten().mean().cpu().detach().numpy())
            # optimization
            optimizer.zero_grad()
            cur_train_loss.backward()
            optimizer.step()

        training_loss.append(np.mean(epoch_loss))
        episode_training_loss.append(np.mean(epoch_loss))
    training_loss_per_episode.append(np.mean(episode_training_loss))

# Self play and training
# learning parameters
BATCH_SIZE = 32
NUM_EPOCHS = 2
TRAIN_WAIT_EPISODES = 100
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 15000
MEM_SIZE = 30000
LEARNING_RATE = 5e-4
PLAY_COUNT = 200001
PRINT_CHECKPOINT = 100 * TRAIN_WAIT_EPISODES
MAX_TRAIN_TIME = 60 * 60 * 8
timeout_reached = False
load_net_params = True
net_load_path = 'net_params.pth'  # the folder name can be different obviously
load_experience = True  # buggy if switched between cuda and cpu
experience_load_path = 'replay_memory.pkl'  # the folder name can be different obviously

# model and agent initializtion
policy_net = Qnet(env.configuration)
if load_net_params and os.path.exists(net_load_path):
    print("Loading net parameters")
    policy_net.load_state_dict(torch.load(net_load_path, map_location=device))
policy_net.to(device)
policy_net.train()
target_net = Qnet(env.configuration).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
eps = EPS_START
is_training = True


def policy_net_agent(observation, configuration):
    return choose_action(observation, configuration, policy_net, is_training, eps)


# optimizer and loss
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
loss = nn.SmoothL1Loss()

# replay memory
if load_experience and os.path.exists(experience_load_path):
    print("Loading replay memory")
    in_mem_file = open(experience_load_path, 'rb')
    memory = pickle.load(in_mem_file)
    in_mem_file.close()
else:
    memory = DualMemory(MEM_SIZE)

step = 0

# holders for plots
eps_list = []
training_loss = []
training_loss_per_episode = []
first_move_qs_arr = []

start_time = time.time()
print("Starting training...")

for episode in range(PLAY_COUNT):
    env.reset()
    eps = eps_calculator(episode, EPS_START, EPS_END, EPS_DECAY)
    eps_list.append(eps)
    episode_length = 0
    while not env.done:
        # determine active player
        if env.state[0].status == 'ACTIVE':
            player_id = 0
        elif env.state[1].status == 'ACTIVE':
            player_id = 1

        # acquire state
        observation = env.state[player_id].observation

        # save initial Q values, and cut training if max time exceeded
        if episode_length == 0 and episode % TRAIN_WAIT_EPISODES == 0:
            # Save initial Q values
            first_move_qs = np.around(
                policy_net(form_net_input(observation, configuration)).cpu().detach().numpy().flatten(), 3)
            first_move_qs_arr.append(first_move_qs)
            # cut training after timeout
            if time.time() - start_time > MAX_TRAIN_TIME:
                timeout_reached = True

        # choose and perform action
        action = policy_net_agent(observation, configuration)
        env.step([int(action) if i == player_id else None for i in [0, 1]])
        episode_length += 1

        # acquire next state, reward and done
        next_opponent_observation = env.state[1 - player_id].observation  # notice we take the opponent id!
        reward = env.state[player_id].reward
        done = (env.state[player_id].status == 'DONE')

        # store transition in memory
        memory.push(episode, observation, action, reward, next_opponent_observation, done)

    # optimize policy net, snd then update target net
    if episode % TRAIN_WAIT_EPISODES == TRAIN_WAIT_EPISODES - 1:
        optimize_step()
        target_net.load_state_dict(policy_net.state_dict())

    # draw board and training time
    if episode % PRINT_CHECKPOINT == 0 or timeout_reached:
        print("Games Played: {}\nReplay memory size: {} transitions\nTime passed: {:.1f} seconds"
              .format(episode + 1, len(memory), time.time() - start_time))
        print("First move Q values:", first_move_qs)
        print("-" * 20)

    if timeout_reached:
        print("Timeout reached!")
        break

end_time = time.time()
print("Finished Training.\nTotal training time = {:.1f} seconds".format(end_time - start_time))

# pickle replay memory
out_mem_file = open("replay_memory.pkl", 'wb')
pickle.dump(memory, out_mem_file)
out_mem_file.close()

#
# # Plots
# # plots
# fig = plt.figure(figsize=(16,8))
# ax = fig.add_subplot(121)
# ax.plot(eps_list)
# ax.grid()
# ax.set_xlabel("episode")
# ax.set_ylabel("eps")
# ax.set_title("eps decay")
#
#
# ax2 = fig.add_subplot(122)
# ax2.plot(training_loss_per_episode)
# ax2.grid()
# ax2.set_xlabel("training episode")
# ax2.set_ylabel("loss")
# ax2.set_title("loss")
#
# first_move_qs_arr = np.array(first_move_qs_arr)
#
# fig2 = plt.figure(figsize=(16,8))
# ax3 = fig2.add_subplot(121)
# ax3.plot(first_move_qs_arr[:,mid_action])
# ax3.grid()
# ax3.axhline(color='red')
# ax3.set_xlabel("training episode")
# ax3.set_ylabel("q-value")
# ax3.set_title("Q-value of middle action on first move")
#
# ax4 = fig2.add_subplot(122)
# ax4.plot(first_move_qs_arr[:,mid_action] - np.concatenate([first_move_qs_arr[:, :mid_action], first_move_qs_arr[:, mid_action+1:]], axis=1).max(axis=1))
# ax4.grid()
# ax4.axhline(color='red')
# ax4.set_xlabel("training episode")
# ax4.set_ylabel("q-value difference")
# ax4.set_title("First move Q-value difference between mid action and max non-mid action ")
#
#
# bar_fig, ax5 = plt.subplots(figsize = (10,8))
#
#
# def update_qs_barplot(i):
#     title = "Q-values of first move after {} training sessions".format(i)
#     ax5.clear()
#     ax5.bar(np.arange(columns), first_move_qs_arr[i])
#     ax5.set_ylim(-0.5, 0.8)
#     ax5.grid()
#     ax5.axhline(color='red')
#     ax5.set_xlabel("action")
#     ax5.set_ylabel("Q-value")
#     ax5.set_title(title)
#
# anim = FuncAnimation(bar_fig, update_qs_barplot, frames=np.linspace(0, len(first_move_qs_arr)-1, 100, dtype='int'))
# anim.save('qs_barplot.gif', writer='imagemagick', fps=20)
#
# HTML(anim.to_jshtml())
#
# # Check for nans in net parameters
# is_ok = True
# for param_tensor in policy_net.state_dict():
#     if torch.isnan(policy_net.state_dict()[param_tensor]).sum() > 0:
#         print("Error: nan in network parameters")
#         is_ok = False
#         break
# if is_ok:
#     print("No nans in net parameters.")
#
# # Evaluate your agent
# policy_net.eval()
# is_training = False
#
# def mean_reward(rewards):
#     return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)
#
# # Run multiple episodes to estimate its performance.
# print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [policy_net_agent, "random"], num_episodes=50)))
# print("Random Agent vs My Agent:", mean_reward(evaluate("connectx", ["random", policy_net_agent], num_episodes=50)))
# print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [policy_net_agent, "negamax"], num_episodes=5)))
# print("Negamax Agent vs My Agent:", mean_reward(evaluate("connectx", ["negamax", policy_net_agent], num_episodes=5)))
#
# # Play your Agent
# # "None" represents which agent you'll manually play as (first or second player).
# env.play([policy_net_agent, None], width=500, height=450)
#
#
#
# # Create agent function
# def agent_function(observation, configuration):
#     import torch
#     import torch.nn as nn
#     import torch.nn.functional as F
#     import numpy as np
#     import random
#     import math
#     import base64
#     import io
#     import time
#
#     class Qnet(nn.Module):
#         def __init__(self, configuration):
#             super(Qnet, self).__init__()
#             self.columns = configuration.columns
#             self.rows = configuration.rows
#             # Number of Checkers "in a row" needed to win.
#             self.inarow = configuration.inarow
#             input_shape = (3, self.rows, self.columns)
#
#             self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
#             self.relu1 = nn.ReLU()
#             #         self.do1 = nn.Dropout2d()
#             self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#             self.relu2 = nn.ReLU()
#             #         self.do2 = nn.Dropout2d()
#             #         def conv2d_size_out(size, kernel_size = 3, stride = 1):
#             #             return (size - (kernel_size - 1) - 1) // stride  + 1
#             #         convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.columns)))
#             #         convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.rows)))
#             #         linear_input_size = convw * convh * 32
#             linear_input_size = self._get_conv_output(input_shape)
#             self.lin1 = nn.Linear(linear_input_size, 64)
#             self.relu_lin1 = nn.ReLU()
#             self.Q_o = nn.Linear(64, self.columns)
#
#         #         self.tanh = nn.Tanh()  # BUGGY
#
#         def _get_conv_output(self, shape):
#             bs = 1
#             input = torch.autograd.Variable(torch.rand(bs, *shape))
#             output_feat = self._forward_features(input)
#             n_size = output_feat.data.view(bs, -1).size(1)
#             return n_size
#
#         def _forward_features(self, x):
#             #         x = F.relu(self.do1(self.conv1(x)))
#             #         x = F.relu(self.do2(self.conv2(x)))
#             x = self.relu1(self.conv1(x))
#             x = self.relu2(self.conv2(x))
#             return x
#
#         def forward(self, x):
#             x = self._forward_features(x)
#             x = self.relu_lin1(self.lin1(x.view(x.size(0), -1)))
#             Q = self.Q_o(x)
#             #         Q = self.tanh(Q)  # BUGGY
#             return Q
#
#     def form_net_input(observation, configuration):
#         # The current serialized Board (rows x columns).
#         board = observation.board
#         # Which player the agent is playing as (1 or 2).
#         mark = observation.mark
#         columns = configuration.columns
#         rows = configuration.rows
#         opponent = 2 if mark == 1 else 1
#         newboard = [1 if i == mark else 2 if i == opponent else 0 for i in board]
#         #         x = torch.Tensor(newboard).view([1,1, rows, columns])
#         newboard = torch.tensor(newboard).reshape([rows, columns])
#         x = F.one_hot(newboard, 3).permute([2, 0, 1])
#         x = x.view([1, 3, rows, columns]).float()
#         return x.to(device)
#
#     def choose_action(observation, configuration, net):
#         Qs = net(form_net_input(observation, configuration)).cpu().detach().numpy().flatten()
#         return int(np.argmax([q_col if observation.board[col] == 0 else -np.inf for (col, q_col) in enumerate(Qs)]))
#
#     device = torch.device('cpu')
#     policy_net = Qnet(configuration)
#     encoded_weights = """
#     BASE64_PARAMS"""
#     decoded = base64.b64decode(encoded_weights)
#     buffer = io.BytesIO(decoded)
#     policy_net.load_state_dict(torch.load(buffer, map_location=device))
#     policy_net.eval()
#     return choose_action(observation, configuration, policy_net)
#
# # Write Submission File
# import inspect
# import os
#
# no_params_path = "submission_template.py"
#
#
# def append_object_to_file(function, file):
#     with open(file, "a" if os.path.exists(file) else "w") as f:
#         f.write(inspect.getsource(function))
#         print(function, "written to", file)
#
#
# def write_agent_to_file(function, file):
#     with open(file, "w") as f:
#         f.write(inspect.getsource(function))
#         print(function, "written to", file)
#
#
# write_agent_to_file(agent_function, no_params_path)
#
# # Write net parameters to submission file
#
# import base64
# import sys
#
# dict_path = "net_params.pth"
# torch.save(policy_net.state_dict(), dict_path)
#
# INPUT_PATH = dict_path
# OUTPUT_PATH = 'submission.py'
#
# with open(INPUT_PATH, 'rb') as f:
#     raw_bytes = f.read()
#     encoded_weights = base64.encodebytes(raw_bytes).decode()
#
# with open(no_params_path, 'r') as file:
#     data = file.read()
#
# data = data.replace('BASE64_PARAMS', encoded_weights)
#
# with open(OUTPUT_PATH, 'w') as f:
#     f.write(data)
#     print('written agent with net parameters to', OUTPUT_PATH)
#
# # Validate Submission
# # Note: Stdout replacement is a temporary workaround.
# import sys
# out = sys.stdout
# try:
#     submission = utils.read_file("/kaggle/working/submission.py")
#     agent = utils.get_last_callable(submission)
# finally:
#     sys.stdout = out
#
# env = make("connectx", debug=True)
# env.run([agent, agent])
# print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")
