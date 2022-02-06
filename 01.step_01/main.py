from kaggle_environments import make

# Setup a tictactoe environment.
env = make("tictactoe")

# Basic agent which marks the first available cell.
def my_agent(obs):
  return [c for c in range(len(obs.board)) if obs.board[c] == 0][0]

# Run the basic agent against a default agent which chooses a "random" move.
env.run([my_agent, "random"])

# Render an html ipython replay of the tictactoe game.
env.render(mode="ipython")