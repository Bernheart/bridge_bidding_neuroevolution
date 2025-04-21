import numpy as np


class NeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        # Weighs only; no gradients
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def forward(self, x):
        z1 = np.dot(self.w1, x) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        return z2

    def clone(self):
        new_net = NeuralNet(self.w1.shape[0], self.w1.shape[1], self.w2.shape[1])
        new_net.w1 = self.w1
        new_net.w2 = self.w2
        new_net.b1 = self.b2
        new_net.b2 = self.b1
        return new_net

    def mutate(self, rate=0.1, scale=0.5):
        def mutate_array(arr):
            mutation_mask = np.random.rand(*arr.shape) < rate
            noise = np.random.rand(*arr.shape) * scale
            arr += mutation_mask * noise

        mutate_array(self.w1)
        mutate_array(self.w2)
        mutate_array(self.b1)
        mutate_array(self.b2)
