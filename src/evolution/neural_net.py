import numpy as np
import src.utils.globals as g


def masked_softmax(output, mask):
    masked = np.where(mask, output, -np.inf)
    exps = np.exp(masked - np.max(masked))
    exps = np.where(mask, exps, 0.0)  # Make sure masked out entries are zero
    return exps / np.sum(exps)


class NeuralNet:
    def __init__(self, input_size=g.INPUT_SIZE, hidden_layers=g.HIDDEN_LAYERS, output_size=g.OUTPUT_SIZE):
        # Weighs only; no gradients
        # W1 maps from input (dim=g.INPUT_SIZE) up to hidden (256)
        self.w1 = np.random.randn(hidden_layers[0], input_size)   # shape: (256, INPUT_SIZE)
        self.b1 = np.zeros(hidden_layers[0])                      # shape: (256,)

        self.wn = []
        self.bn = []
        for i in range(len(hidden_layers)-1):
            self.wn.append(np.random.randn(hidden_layers[i+1], hidden_layers[i]))
            self.bn.append(np.zeros(hidden_layers[i+1]))

        # W2 maps from hidden (256) down to outputs
        self.w2 = np.random.randn(output_size, hidden_layers[-1])  # shape: (OUTPUT_SIZE, 256)
        self.b2 = np.zeros(output_size)                      # shape: (OUTPUT_SIZE,)

    def forward(self, x, availability_mask):
        z1 = np.dot(self.w1, np.array(x, dtype=np.float32)) + self.b1
        a1 = np.tanh(z1)
        for i in range(len(self.wn)):
            zn = np.dot(self.wn[i], a1) + self.bn[i]
            a1 = np.tanh(zn)
        z2 = np.dot(self.w2, a1) + self.b2
        probs = masked_softmax(z2, availability_mask)
        # probs is a 1D array summing to 1
        action = np.random.choice(len(probs), p=probs)
        one_hot = np.zeros_like(probs)
        one_hot[action] = 1
        return one_hot

    def clone(self):
        new_net = NeuralNet()
        new_net.w1 = self.w1.copy()
        new_net.w2 = self.w2.copy()
        for i in range(len(self.wn)):
            new_net.wn[i] = self.wn[i].copy()
            new_net.bn[i] = self.bn[i].copy()
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
        for i in range(len(self.wn)):
            mutate_array(self.wn[i])
            mutate_array(self.bn[i])
        mutate_array(self.b1)
        mutate_array(self.b2)

    def crossover(self, other):
        def blend(a, b):
            mask = np.random.rand(*a.shape) < 0.5
            return np.where(mask, a, b)

        child = self.clone()
        child.w1 = blend(self.w1, other.w1)
        child.b1 = blend(self.b1, other.b1)
        for i in range(len(self.wn)):
            child.wn[i] = blend(self.wn[i], other.wn[i])
            child.bn[i] = blend(self.bn[i], other.bn[i])
        child.w2 = blend(self.w2, other.w2)
        child.b2 = blend(self.b2, other.b2)
        return child
