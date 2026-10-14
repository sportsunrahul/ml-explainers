"""Chapter 2: the gradient bandit of Sutton and Barto (section 2.8) with autograd.
The loss -(R - baseline) * log pi(A) gives exactly the update of S&B Eq. 2.12,
so a bandit already contains a policy gradient (chapter 10)."""
import torch

torch.manual_seed(0)
runs, k, alpha, steps = 500, 10, 0.1, 1000


def gradient_bandit(use_baseline):
    means = torch.randn(runs, k) + 4.0  # S&B Figure 2.5 shifts the true means up by 4
    best = means.argmax(dim=1)
    H = torch.zeros(runs, k, requires_grad=True)  # preferences, one row per task
    opt = torch.optim.SGD([H], lr=alpha)
    rows = torch.arange(runs)
    baseline = torch.zeros(runs)
    chose_best = torch.zeros(steps)
    for t in range(1, steps + 1):
        pi = torch.softmax(H, dim=1)  # S&B Eq. 2.11
        a = torch.multinomial(pi.detach(), 1).squeeze(1)
        R = means[rows, a] + torch.randn(runs)
        if t == 1:
            baseline = R.clone()  # S&B: the baseline at the first step is the first reward
        b = baseline if use_baseline else torch.zeros(runs)
        # Eq. 2.12 by hand: every arm moves by -alpha (R - b) pi(a); the chosen arm also by +alpha (R - b).
        manual = H.detach() - alpha * (R - b)[:, None] * pi.detach()
        manual[rows, a] += alpha * (R - b)
        loss = -((R - b) * torch.log(pi[rows, a])).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert torch.allclose(H.detach(), manual, atol=1e-5)
        baseline = baseline + (R - baseline) / t  # average of the rewards so far, used at the next step
        chose_best[t - 1] = (a == best).float().mean()
    return chose_best


for use_baseline in (True, False):
    p = gradient_bandit(use_baseline)
    print(f'baseline {"on " if use_baseline else "off"}: best arm chosen {100 * p[-100:].mean():.0f}% '
          f'of the time in steps 901-1000; autograd matched Eq. 2.12 at all {steps} steps')
