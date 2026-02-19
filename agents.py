# agents.py
class NnMctsAgent:
    def __init__(self, mcts):
        self.mcts = mcts

    def get_action_probs(self, state, temp=1.0):
        return self.mcts.get_action_probs(state, temp=temp)

    def update_with_move(self, last_action):
        self.mcts.update_with_move(last_action)
