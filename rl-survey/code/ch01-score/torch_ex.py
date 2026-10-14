"""Chapter 1: the same loop with a Gymnasium environment and an untrained PyTorch policy network."""
import gymnasium as gym
import torch
from torch import nn

torch.manual_seed(0)
env = gym.make('CartPole-v1')
# The policy maps the 4 numbers of the state to 2 action scores (logits), one per action.
policy = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 2))

returns = []
for episode in range(100):
    obs, _ = env.reset(seed=episode)
    ret, done = 0.0, False
    while not done:
        with torch.no_grad():
            logits = policy(torch.as_tensor(obs))
        action = torch.distributions.Categorical(logits=logits).sample().item()
        obs, reward, terminated, truncated, _ = env.step(action)
        ret += reward  # +1 for every step the pole stays up
        done = terminated or truncated
    returns.append(ret)
print(f'untrained policy: mean return {sum(returns) / len(returns):.1f} '
      f'over 100 episodes (best {max(returns):.0f}, the limit is 500)')
