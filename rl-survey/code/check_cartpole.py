"""Check that code/kit/cartpole.py matches Gymnasium CartPole-v1 step for step."""
import sys
import numpy as np
import gymnasium as gym

sys.path.insert(0, __file__.rsplit('/', 1)[0] + '/kit')
from cartpole import CartPole

rng = np.random.default_rng(1)
gym_env = gym.make('CartPole-v1')
ours = CartPole()
episodes, steps, worst = 0, 0, 0.0
for ep in range(200):
    obs, _ = gym_env.reset(seed=ep)
    ours.reset()
    ours.state = np.array(gym_env.unwrapped.state, dtype=np.float64)
    while True:
        a = int(rng.integers(2))
        g_obs, g_r, g_term, g_trunc, _ = gym_env.step(a)
        o_obs, o_r, o_term, o_trunc = ours.step(a)
        worst = max(worst, float(np.max(np.abs(np.asarray(g_obs, dtype=np.float64) - o_obs))))
        assert (g_r, g_term, g_trunc) == (o_r, o_term, o_trunc), (ep, ours.t)
        steps += 1
        if g_term or g_trunc:
            break
    episodes += 1
# Gymnasium returns float32 observations, so allow float32 rounding.
assert worst < 1e-5, worst
print(f'{episodes} episodes, {steps} steps: same rewards and endings, max obs diff {worst:.1e}')
