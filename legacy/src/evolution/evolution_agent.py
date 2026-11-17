from src.evolution.neural_net import NeuralNetTorch
import src.utils.globals as g


class EvoAgent:
    def __init__(self, model: NeuralNetTorch):
        self.model = model
        self._behavior_descriptor = None

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

    def clone_and_mutate(self, mutation_rate: g.MUTATION_RATE):
        new_model = self.model.clone()
        new_model.mutate(rate=mutation_rate)
        return EvoAgent(new_model)

    @property
    def behavior_descriptor(self):
        return self._behavior_descriptor

    def rl_update(self, trajectories, steps=5, lr=1e-3):
        """
        trajectories: list of bidding_mask sequences,
                      or (state, action, reward) tuples
        steps: how many gradient steps to take
        """
        for _ in range(steps):
            # compute policy gradient loss on this batch
            loss = compute_policy_loss(self.model, trajectories)
            # backpropagate through self.model.wn / bn
            # e.g. using simple finite‐diff or autograd if you switch to a DL framework
            self.model.apply_gradients(loss, lr)
