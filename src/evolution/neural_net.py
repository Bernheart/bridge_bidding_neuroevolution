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
        self.wn = []
        self.bn = []

        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None

        self.wn.append(np.random.randn(hidden_layers[0], input_size))   # shape: (256, INPUT_SIZE)
        self.bn.append(np.zeros(hidden_layers[0]))                   # shape: (256,)

        for i in range(len(hidden_layers)-1):
            self.wn.append(np.random.randn(hidden_layers[i+1], hidden_layers[i]))
            self.bn.append(np.zeros(hidden_layers[i+1]))

        # W2 maps from hidden (256) down to outputs
        self.wn.append(np.random.randn(output_size, hidden_layers[-1]))  # shape: (OUTPUT_SIZE, 256)
        self.bn.append(np.zeros(output_size))                      # shape: (OUTPUT_SIZE,)

    def forward(self, x, availability_mask):
        z = np.dot(self.wn[0], np.array(x, dtype=np.float32)) + self.bn[0]
        a = np.tanh(z)
        for i in range(1, len(self.wn)-1):  # not the first and last
            z = np.dot(self.wn[i], a) + self.bn[i]
            a = np.tanh(z)
        z = np.dot(self.wn[-1], a) + self.bn[-1]
        probs = masked_softmax(z, availability_mask)
        # probs is a 1D array summing to 1
        action = np.random.choice(len(probs), p=probs)
        one_hot = np.zeros_like(probs)
        one_hot[action] = 1
        return one_hot

    def clone(self):
        new_net = NeuralNet()
        for i in range(len(self.wn)):
            new_net.wn[i] = self.wn[i].copy()
            new_net.bn[i] = self.bn[i].copy()
        return new_net

    def mutate(self, rate=g.MUTATION_RATE, sigma=g.MUTATION_SCALE):
        def mutate_array(arr):
            # 1) mask: which entries to mutate
            mutation_mask = np.random.rand(*arr.shape) < rate
            # 2) noise ~ N(0, sigma^2)
            noise = np.random.randn(*arr.shape) * sigma
            # 3) apply only where mask is True
            arr += mutation_mask * noise

        for i in range(len(self.wn)):
            mutate_array(self.wn[i])
            mutate_array(self.bn[i])

    def crossover(self, other):
        def blend(a, b):
            mask = np.random.rand(*a.shape) < 0.5
            return np.where(mask, a, b)

        child = self.clone()

        for i in range(len(self.wn)):
            child.wn[i] = blend(self.wn[i], other.wn[i])
            child.bn[i] = blend(self.bn[i], other.bn[i])

        return child

    def get_parameters(self):
        # print(self.wn)
        # for i in range(len(self.wn)):
        #     print(self.wn[i].shape)
        #     print(self.wn[i].tolist())
        #     print(self.wn[i])
        return {
            # 'weights': [self.w1.tolist()] + [w.tolist() for w in self.wn] + [self.w2.tolist()],
            # 'biases': [self.b1.tolist()] + [b.tolist() for b in self.bn] + [self.b2.tolist()],
            'weights': [self.w1.tolist()] + [self.w2.tolist()],
            'biases': [self.b1.tolist()] + [self.b2.tolist()],

        }

    def set_parameters(self, params):
        self.wn = [np.array(w) for w in params['weights']]
        self.bn = [np.array(b) for b in params['biases']]

