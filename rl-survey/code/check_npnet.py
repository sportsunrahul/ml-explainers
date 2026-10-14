"""Check kit/npnet.py: backward() against finite differences, and Adam against torch.optim.Adam."""
import sys
import numpy as np
import torch

sys.path.insert(0, __file__.rsplit('/', 1)[0] + '/kit')
from npnet import MLP, Adam

rng = np.random.default_rng(0)
net = MLP([4, 16, 16, 3], seed=1)
x, target = rng.normal(size=(8, 4)), rng.normal(size=(8, 3))
loss = lambda: 0.5 * np.sum((net.forward(x) - target) ** 2)
loss()
grads = net.backward(net.forward(x) - target)
worst = 0.0
for p, g in zip(net.params, grads):
    for idx in [tuple(rng.integers(s) for s in p.shape) for _ in range(5)]:
        old = p[idx]
        p[idx] = old + 1e-6; up = loss()
        p[idx] = old - 1e-6; down = loss()
        p[idx] = old
        worst = max(worst, abs((up - down) / 2e-6 - g[idx]) / max(1.0, abs(g[idx])))
assert worst < 1e-6, worst
w0 = [p.copy() for p in net.params]
tw = [torch.tensor(p, requires_grad=True) for p in w0]
topt, opt = torch.optim.Adam(tw, lr=0.01), Adam(net.params, lr=0.01)
for _ in range(20):
    g = net.backward(net.forward(x) - target)
    opt.step(g)
    h = torch.tensor(x)
    for i in range(0, len(tw), 2):
        h = h @ tw[i] + tw[i + 1]
        if i < len(tw) - 2:
            h = torch.tanh(h)
    topt.zero_grad(); (0.5 * ((h - torch.tensor(target)) ** 2).sum()).backward(); topt.step()
diff = max(float(np.max(np.abs(p - t.detach().numpy()))) for p, t in zip(net.params, tw))
assert diff < 1e-6, diff
print(f'backward vs finite differences: worst relative error {worst:.1e}; Adam vs torch after 20 steps: max diff {diff:.1e}')
