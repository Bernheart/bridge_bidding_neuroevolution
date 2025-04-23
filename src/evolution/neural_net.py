import numpy as np
import src.utils.globals as g


def masked_softmax(output, mask):
    masked = np.where(mask, output, -np.inf)
    exps = np.exp(masked - np.max(masked))
    exps = np.where(mask, exps, 0.0)  # Make sure masked out entries are zero
    return exps / np.sum(exps)


class NeuralNet:
    def __init__(self, input_size=g.INPUT_SIZE, hidden_size=g.HIDDEN_SIZE, output_size=g.OUTPUT_SIZE):
        # Weighs only; no gradients
        # W1 maps from input (dim=g.INPUT_SIZE) up to hidden (256)
        self.w1 = np.random.randn(hidden_size, input_size)   # shape: (256, INPUT_SIZE)
        self.b1 = np.zeros(hidden_size)                      # shape: (256,)

        # W2 maps from hidden (256) down to outputs
        self.w2 = np.random.randn(output_size, hidden_size)  # shape: (OUTPUT_SIZE, 256)
        self.b2 = np.zeros(output_size)                      # shape: (OUTPUT_SIZE,)

    def forward(self, x, availability_mask):
        z1 = np.dot(self.w1, np.array(x, dtype=np.float32)) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        probs = masked_softmax(z2, availability_mask)
        # probs is a 1D array summing to 1
        action = np.random.choice(len(probs), p=probs)
        one_hot = np.zeros_like(probs)
        one_hot[action] = 1
        return one_hot

    def clone(self):
        new_net = NeuralNet(self.w1.shape[0], self.w1.shape[1], self.w2.shape[1])
        new_net.w1 = self.w1.copy()
        new_net.w2 = self.w2.copy()
        new_net.b1 = self.b1.copy()
        new_net.b2 = self.b2.copy()
        return new_net

    def mutate(self, rate=g.MUTATION_RATE, sigma=g.MUTATION_SCALE):
        def mutate_array(arr):
            # 1) mask: which entries to mutate
            mutation_mask = np.random.rand(*arr.shape) < rate
            # 2) noise ~ N(0, sigma^2)
            noise = np.random.randn(*arr.shape) * sigma
            # 3) apply only where mask is True
            arr += mutation_mask * noise

        mutate_array(self.w1)
        mutate_array(self.w2)
        mutate_array(self.b1)
        mutate_array(self.b2)

    def crossover(self, other):
        def blend(a, b):
            mask = np.random.rand(*a.shape) < 0.5
            return np.where(mask, a, b)

        child = self.clone()
        child.w1 = blend(self.w1, other.w1)
        child.w2 = blend(self.w2, other.w2)
        child.b1 = blend(self.b1, other.b1)
        child.b2 = blend(self.b2, other.b2)
        return child
