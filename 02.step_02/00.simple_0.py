# This agent always goes only to the zero column if it is non-empty.
def agent(obs):
  return [c for c in range(len(obs.board)) if obs.board[c] == 0][0]