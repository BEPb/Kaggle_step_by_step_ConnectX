"""
Python 3.9 стартовая программа на Python по изучению обучения с подкреплением - Reinforcement Learning
Название файла 00. start.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2022-02-06

воспользуемся одной из тестовых игр OpenAI, в частности, со средой «MountainCar-v0»
"""
from kaggle_environments import make, evaluate

# Создать игровую среду
# Установите debug = True, чтобы увидеть ошибки, если ваш агент отказывается запускаться
# env = make("connectx", debug=True)
env = make("connectx", debug=False)

# Список доступных агентов по умолчанию
print(list(env.agents))