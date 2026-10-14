"""Chapter 2: epsilon-greedy and UCB on the 10-armed testbed of Sutton and Barto, section 2.3.
All 2000 tasks run at once: row i of Q holds task i's estimates."""
import numpy as np


def testbed(method, param, runs=2000, steps=1000, k=10, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 1, (runs, k))  # each task's hidden arm means
    best = means.argmax(axis=1)
    Q = np.zeros((runs, k))  # sample-average estimates
    N = np.zeros((runs, k))  # pull counts
    rows = np.arange(runs)
    reward, optimal = np.zeros(steps), np.zeros(steps)
    for t in range(steps):
        noise = 1e-9 * rng.random((runs, k))  # breaks ties at random
        if method == 'ucb':
            # S&B Eq. 2.10; an arm never pulled counts as the best choice.
            bound = Q + param * np.sqrt(np.log(t + 1) / np.maximum(N, 1))
            a = np.argmax(np.where(N == 0, 1e9, bound) + noise, axis=1)
        else:
            a = np.argmax(Q + noise, axis=1)
            explore = rng.random(runs) < param
            a[explore] = rng.integers(k, size=explore.sum())
        r = rng.normal(means[rows, a], 1)
        N[rows, a] += 1
        Q[rows, a] += (r - Q[rows, a]) / N[rows, a]  # incremental average, S&B Eq. 2.3
        reward[t], optimal[t] = r.mean(), (a == best).mean()
    return reward, optimal


for method, param in [('epsilon', 0.0), ('epsilon', 0.01), ('epsilon', 0.1), ('ucb', 2.0)]:
    reward, optimal = testbed(method, param)
    print(f'{method} {param:<4}: steps 901-1000 average reward {reward[-100:].mean():.2f}, '
          f'best arm chosen {100 * optimal[-100:].mean():.0f}% of the time')
