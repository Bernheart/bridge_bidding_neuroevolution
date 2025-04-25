from src.evolution.neural_net import NeuralNet


class EvoAgent:
    def __init__(self, model: NeuralNet):
        self.model = model

    # def act(self, state, legal_actions):
    #     """
    #     - `state`: numpy array (e.g., 32-length board vector)
    #     - `legal_actions`: list of legal action indices (0 to N-1)
    #     """
    #     action_scores = self.model.forward(state)
    #     legal_scores = [(a, action_scores[a]) for a in legal_actions]
    #     if not legal_scores:
    #         return None
    #     # Choose the legal action with highest score
    #     return max(legal_scores, key=lambda x: x[1])[0]

    def clone_and_mutate(self):
        new_model = self.model.clone()
        new_model.mutate()
        return EvoAgent(new_model)
