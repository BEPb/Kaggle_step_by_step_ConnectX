"""
Python 3.9 программа для сравнения двух обученных агентов между собой с подсчетом процента выйгрыша
Название файла 99.analiz.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2022-02-16

воспользуемся одной из тестовых игр kaggle_environments, в частности, со средой «ConnectX»
результаты тестирования занесены в 00.analiz.md
"""
# Importing Dependencies
from kaggle_environments import make, evaluate, utils, agent  # connect the library kaggle_environments simulation virtual environment and its functions
from tqdm import tqdm  # connect the library show a smart progress
import random  # connect the library for working with random numbers
import numpy as np  # connect the library general mathematical and numerical operations

n_rounds = 10
agent1 = "../02.step_02/01.random.py"
# agent1 = "02.negamax.py"
# agent1 = "03.initital_agent.py"
# agent1 = "04.step_lookahead_agent.py"
# agent1 = "05.Nstep_lookahead_agent.py"
agent1 = "../02.step_02/06.fast_Nstep_lookahead_agent.py"


# agent2 = "Q_table_1episode.py"
# agent2 = "Q_table_10episode.py"
# agent2 = "Q_table_100episode.py"
# agent2 = "Q_table_1000episodes.py"
agent2 = "best_model_x.py"




# Use default Connect Four setup
config = {'rows': 6, 'columns': 7, 'inarow': 4}

# Agent 1 goes first (roughly) half the time
for n in tqdm(range(n_rounds-n_rounds//2)):
    outcomes = evaluate("connectx", [agent1, agent2], config, [], 1)

# Agent 2 goes first (roughly) half the time
for n in tqdm(range(n_rounds-n_rounds//2)):
    for [a,b] in evaluate("connectx", [agent2, agent1], config, [], 1):
        outcomes += [[b, a]]


print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))
print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))
print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))
