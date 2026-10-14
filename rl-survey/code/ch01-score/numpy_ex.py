"""Chapter 1: the agent-environment loop on the track from the figure, in plain NumPy."""
import numpy as np


class Track:
    """Seven cells in a row. The robot starts in cell 3 and has 5 moves.
    Reaching the charger pays +1 and ends the episode; every other move pays 0."""

    def __init__(self, charger):
        self.charger = charger

    def reset(self):
        self.pos, self.moves = 3, 0
        return self.pos

    def step(self, action):  # 0 = left, 1 = right
        self.pos = min(6, max(0, self.pos + (1 if action == 1 else -1)))
        self.moves += 1
        reward = 1.0 if self.pos == self.charger else 0.0
        done = reward > 0 or self.moves == 5
        return self.pos, reward, done


def run_episode(env, policy):
    state, ret, done = env.reset(), 0.0, False
    while not done:
        action = policy(state)
        state, reward, done = env.step(action)
        ret += reward  # the return G: the sum of the rewards
    return ret


rng = np.random.default_rng(0)
policies = {
    'random': lambda s: int(rng.integers(2)),
    'always left': lambda s: 0,
    'always right': lambda s: 1,
}
for charger in (6, 0):
    env = Track(charger)
    print(f'charger in cell {charger}:')
    for name, policy in policies.items():
        returns = [run_episode(env, policy) for _ in range(10_000)]
        print(f'  {name:>12}: average return {np.mean(returns):.3f}')
