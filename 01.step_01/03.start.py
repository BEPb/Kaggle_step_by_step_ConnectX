from kaggle_environments import make
# установим свои парамтры игры connectX 10 строк и 8 колонок, нужно составить линию из 5 шаров
env = make("connectx", {"rows": 10, "columns": 8, "inarow": 5})

'''опишем собственного агента который будет всегда кидать фишку в 3-й колонке'''
def agent(observation, configuration):
  print(observation)  # {board: [...], mark: 1}
  print(configuration)  # {rows: 10, columns: 8, inarow: 5}
  return 3  # Действие: всегда ставьте отметку в 3-м столбце


'''варианты создания собственных агентов'''
# Агент принимает наблюдение и возвращает действие, т.е. он копирует действие первого агента
def agent1(obs):
  return [c for c in range(len(obs.board)) if obs.board[c] == 0][0]

# принимает значение одного из агента по умолчанию, который всегда ходит случайно
agent2 = "random"

# Загрузите агент из исходного кода.
agent3 = """
def act(obs):
  return [c for c in range(len(obs.board)) if obs.board[c] == 0][0]
"""

# Загрузите агент из файла.
agent4 = "C:\path\file.py"

# Возврат фиксированного действия, в нашем случае всегда ходить в колонку №3
agent5 = 3

# Возврат действия по URL-адресу.
agent6 = "http://localhost:8000/run/agent"



# Запустите эпизод, используя указанный выше агент, который играет против случайного агента (прописанного в программе
# игры по умолчанию)
env.run([agent, "random"])

# Распечатать схемы из спецификации.
print(env.specification.observation)  # вывести параметры виртуальной среды
print(env.specification.configuration)  # вывести параметры эпизода игры
print(env.specification.action)  # вывести входные данные от вашего агента