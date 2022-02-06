from kaggle_environments import make

env = make("connectx", debug=True)

'''Интерфейс Open AI Gym используется для помощи обучающим агентам. Ключевое слово None используется ниже для 
обозначения того, какого агента обучать (т. е. обучать в качестве первого или второго игрока connectx). '''
# Обучающий агент на первой позиции (игрок 1) против случайного агента по умолчанию.
trainer = env.train([None, "random"])
'''всегда в режиме тренировки один из агентов должен быть обозначен как 'None' в противном 
слуаче появится сообщение об ошибке: kaggle_environments.errors.InvalidArgument: One agent must be 
marked 'None' to train.'''

obs = trainer.reset()
for _ in range(100):
    env.render()
    action = 0  # Действие для обучаемого агента.
    obs, reward, done, info = trainer.step(action)
    if done:
        obs = trainer.reset()