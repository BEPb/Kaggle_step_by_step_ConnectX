'''обработка ошибок
создадим огента который ничего не делает только возвращает текст о том что, что-то случилось

'''
from kaggle_environments import make


def agent():
  return "Something Bad"

env = make("tictactoe", debug=True)

env.run([agent, "random"])
# Prints: "Invalid Action: Something Bad"