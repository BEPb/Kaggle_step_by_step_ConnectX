"""
Python 3.9 стартовая программа на Python по изучению обучения с подкреплением - Reinforcement Learning
Название файла 00. start.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2022-02-06

воспользуемся одной из тестовых игр kaggle_environments, в частности, со средой «ConnectX»
"""
from kaggle_environments import make, evaluate

# Создать игровую среду
env = make("connectx", debug=True)  # среда connectX
# env = make("tictactoe")  # среда крестики-нолики
# env = make("identity") # Для отладки, действие является наградой

'''
можно вызвать описание функции make
help(make)
make(environment, configuration={}, info={}, steps=[], logs=[[]], debug=False, state=None)
environment - имя среды строчная переменная
configuration - конфигурация, словарь параметров, см. ниже
info (dict, optional):
steps (list, optional):
debug - установите debug=True, чтобы увидеть ошибки, если ваш агент отказывается запускаться, по умолчанию debug=False
state (optional):
'''

'''
задаваемые параметры конфигурации:
EpisodeSteps     - Максимальное количество шагов в эпизоде.
agentTimeout     - Максимальное время выполнения (в секундах) для инициализации агента.
actTimeout       - Максимальное время выполнения (в секундах) для получения действия от агента.
runTimeout       - Максимальное время выполнения (в секундах) эпизода (не обязательно DONE).
'''

env = make("connectx", configuration={
  "columns": 19,  # этот параметр специфический только для игры ConnectX.
  "actTimeout": 10,
})
# print(dir(env))  # выведет все обрабатываемые параметры

env.reset  # сбросить среду


# Сброс
# Окружения сбрасываются по умолчанию после «сделать» (если не переданы начальные шаги), а также при вызове
# «выполнить». Сброс можно вызвать в любое время, чтобы очистить среду.
num_agents = 2
reset_state = env.reset(num_agents)

# Список доступных агентов по умолчанию
print(list(env.agents))