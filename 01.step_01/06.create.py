from kaggle_environments import make

# Создать экземпляр среды.
env = make(
  # Спецификация или название для зарегистрированной спецификации.
  "connectx",

  # Переопределить конфигурацию по умолчанию и среду.
  configuration={"rows": 9, "columns": 10},

  # Инициализировать среду из предыдущего состояния (возобновление эпизода).
  steps=[],

  # Включить подробное ведение журнала.
  debug=True
)

steps = env.run(['random', 'random'])
print(steps)
print(env)