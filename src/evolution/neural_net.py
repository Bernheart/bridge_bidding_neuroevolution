import numpy as np
import src.utils.globals as g


def softmax_with_temp(z, mask, temp=1.0):
    masked = np.where(mask, z, -np.inf)
    exps = np.exp((masked - masked.max())/temp)
    exps = np.where(mask, exps, 0.0)
    return exps / exps.sum()


class NeuralNet:
    def __init__(self, input_size=g.INPUT_SIZE, hidden_layers=g.HIDDEN_LAYERS, output_size=g.OUTPUT_SIZE):
        # Prepare sizes: [input_size, *hidden_layers, output_size]
        layer_sizes = [input_size] + hidden_layers + [output_size]

        self.wn = []
        self.bn = []

        # Xavier‐uniform initialization for each pair of consecutive layers
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            W = np.random.uniform(-limit, limit, size=(fan_out, fan_in))
            b = np.zeros(fan_out)
            self.wn.append(W)
            self.bn.append(b)                     # shape: (OUTPUT_SIZE,)

    def forward(self, x, availability_mask):
        # print("INPUT VECTOR   :", x)
        # print("AVAILABILITY   :", availability_mask)
        z = np.dot(self.wn[0], np.array(x, dtype=np.float32)) + self.bn[0]
        a = np.where(z >= 0, z, 0.01 * z)
        for i in range(1, len(self.wn)-1):  # not the first and last
            z = np.dot(self.wn[i], a) + self.bn[i]
            a = np.where(z >= 0, z, 0.01 * z)
        z = np.dot(self.wn[-1], a) + self.bn[-1]
        probs = softmax_with_temp(z, availability_mask)
        # print("Probs:", np.round(probs, 3))
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

    def mutate(self, rate=-1, sigma=g.MUTATION_SCALE):
        from src.evolution.evolution import MUTATION_RATE
        rate = MUTATION_RATE

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
            'weights': [w.tolist() for w in self.wn],
            'biases': [b.tolist() for b in self.bn]
            # 'weights': [self.w1.tolist()] + [self.w2.tolist()],
            # 'biases': [self.b1.tolist()] + [self.b2.tolist()],
        }

    def set_parameters(self, params):
        self.wn = [np.array(w) for w in params['weights']]
        self.bn = [np.array(b) for b in params['biases']]

