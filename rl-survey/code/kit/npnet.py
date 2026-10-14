"""A small multilayer perceptron with hand-written backprop, and Adam, in plain NumPy."""
import numpy as np


class MLP:
    """Fully connected layers: tanh between layers, a linear output. Inputs have shape (batch, n_in)."""

    def __init__(self, sizes, seed=0, out_gain=1.0):
        rng = np.random.default_rng(seed)
        self.params = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = out_gain if i == len(sizes) - 2 else 1.0
            self.params += [rng.normal(0.0, gain / np.sqrt(n_in), (n_in, n_out)), np.zeros(n_out)]

    def forward(self, x):
        self.cache = [x]  # the input of every layer, kept for backward
        n = len(self.params) // 2
        for i in range(n):
            x = x @ self.params[2 * i] + self.params[2 * i + 1]
            if i < n - 1:
                x = np.tanh(x)
                self.cache.append(x)
        return x

    def backward(self, dout):
        """Gradients of a loss for the last forward batch, given dLoss/dOutput (same shape as the output)."""
        grads = [None] * len(self.params)
        d = dout
        for i in reversed(range(len(self.params) // 2)):
            grads[2 * i] = self.cache[i].T @ d
            grads[2 * i + 1] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.params[2 * i].T) * (1.0 - self.cache[i] ** 2)  # tanh' = 1 - tanh^2
        return grads

    def copy_from(self, other):
        for p, q in zip(self.params, other.params):
            p[...] = q


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m += (1 - self.b1) * (g - m)
            v += (1 - self.b2) * (g * g - v)
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)  # in place: the network sees the new weights
