"""CartPole in plain NumPy, with the same physics as Gymnasium CartPole-v1."""
import numpy as np


class CartPole:
    gravity, masscart, masspole = 9.8, 1.0, 0.1
    length = 0.5                  # half the pole length, as in Gymnasium
    force_mag, tau = 10.0, 0.02   # newtons, seconds per step
    theta_limit = 12 * 2 * np.pi / 360
    x_limit = 2.4
    max_steps = 500               # CartPole-v1 time limit

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.t = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        total_mass = self.masspole + self.masscart
        pml = self.masspole * self.length
        cos, sin = np.cos(theta), np.sin(theta)
        temp = (force + pml * theta_dot ** 2 * sin) / total_mass
        theta_acc = (self.gravity * sin - cos * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * cos ** 2 / total_mass))
        x_acc = temp - pml * theta_acc * cos / total_mass
        # Euler integration: positions use the old velocities.
        x, x_dot = x + self.tau * x_dot, x_dot + self.tau * x_acc
        theta, theta_dot = theta + self.tau * theta_dot, theta_dot + self.tau * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.t += 1
        terminated = abs(x) > self.x_limit or abs(theta) > self.theta_limit
        truncated = self.t >= self.max_steps
        return self.state.copy(), 1.0, terminated, truncated
