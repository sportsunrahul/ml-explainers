# DPG: Deterministic Policy Gradient Algorithms (Silver et al., 2014)

Group F. Source: `sources/classic/silver2014-dpg.pdf` (9 PDF pages; no LaTeX source, so all TeX below is
transcribed from the PDF). Page numbers are PDF pages (the PDF prints none). The appendices that the text cites
(A: regularity conditions, B: proof of Theorem 1, C: proof of Theorem 2) are NOT in this file; they were
supplementary material.

## 1. Identity

- Title: Deterministic Policy Gradient Algorithms.
- Venue: Proceedings of the 31st International Conference on Machine Learning, Beijing, China, 2014. JMLR: W&CP
  volume 32 (p.1 footer). Not on arXiv.
- Authors: David Silver (first), Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller (last).
  DeepMind Technologies and University College London (p.1).
- Sections: 1 Introduction (p.1); 2 Background: 2.1 Preliminaries, 2.2 Stochastic Policy Gradient Theorem,
  2.3 Stochastic Actor-Critic Algorithms (p.2), 2.4 Off-Policy Actor-Critic (p.3); 3 Gradients of Deterministic
  Policies: 3.1 Action-Value Gradients (p.3), 3.2 Deterministic Policy Gradient Theorem (p.4), 3.3 Limit of the
  Stochastic Policy Gradient (p.4); 4 Deterministic Actor-Critic Algorithms: 4.1 On-Policy (p.4), 4.2 Off-Policy
  (p.4-5), 4.3 Compatible Function Approximation (p.5-6); 5 Experiments: 5.1 Continuous Bandit (p.6),
  5.2 Continuous Reinforcement Learning (p.7), 5.3 Octopus Arm (p.7-8); 6 Discussion and Related Work (p.8);
  7 Conclusion (p.8).

## 2. Core problem

Policy gradient methods for continuous actions sample a stochastic policy, and the stochastic policy gradient
integrates over both states and actions. That estimate needs many samples when the action has many dimensions.
The paper asks whether a deterministic policy a = mu_theta(s) has a policy gradient, and how such a policy can
still explore.

## 3. What prior methods did and why that hurt (per the source)

- Stochastic policy gradient (Eq. 2): "In the stochastic case, the policy gradient integrates over both state and
  action spaces, whereas in the deterministic case it only integrates over the state space. As a result, computing
  the stochastic policy gradient may require more samples, especially if the action space has many dimensions."
  (Sec. 1, p.1)
- "It was previously believed that the deterministic policy gradient did not exist, or could only be obtained when
  using a model (Peters, 2010)." (Sec. 1, p.1)
- Greedy policy improvement (mu^{k+1}(s) = argmax_a Q^{mu^k}(s,a)) "becomes problematic, requiring a global
  maximisation at every step" in continuous action spaces (Sec. 3.1, p.3).
- As a stochastic policy homes in on a good strategy it becomes more deterministic, and its gradient gets harder to
  estimate: "the variance of the stochastic policy gradient for a Gaussian policy N(mu, sigma^2) is proportional to
  1/sigma^2 (Zhao et al., 2012), which grows to infinity as the policy becomes deterministic." (Sec. 6, p.8)

## 4. The one key idea

Move the policy parameters along the gradient of the action-value with respect to the action, chained through the
policy: grad_theta J = E_s[ grad_theta mu_theta(s) grad_a Q^mu(s,a) |_{a = mu_theta(s)} ] (Theorem 1). This needs
no integral over actions. Exploration comes from learning off-policy: act with a stochastic behaviour policy, learn
the deterministic target policy (Sec. 1, p.1).

## 5. Mechanism, step by step

1. Intuition (Sec. 3.1, p.3): instead of a global argmax, move the policy in the direction of grad Q. Each visited
   state proposes a direction; average them under rho^mu (Eq. 6). The chain rule splits it into grad_a Q times
   grad_theta mu (Eq. 7). Convention: grad_theta mu_theta(s) is a Jacobian; column d is grad_theta [mu_theta(s)]_d.
2. Theorem 1 (Sec. 3.2, p.4) shows this is exactly the gradient of J(mu_theta); the state-distribution gradient is
   not needed.
3. On-policy deterministic actor-critic (Sec. 4.1, p.4), "primary purpose is didactic": Sarsa critic (Eqs. 11-12)
   plus actor step along grad_theta mu * grad_a Q^w (Eq. 13).
4. Off-policy deterministic actor-critic, OPDAC (Sec. 4.2, p.4-5): a stochastic behaviour policy beta(a|s)
   generates trajectories. The critic learns Q^w(s,a) ~ Q^mu(s,a) by Q-learning (Eqs. 16-17); the actor follows the
   off-policy deterministic gradient (Eq. 18). "because the deterministic policy gradient removes the integral over
   actions, we can avoid importance sampling in the actor; and by using Q-learning, we can avoid importance
   sampling in the critic." (p.5)
5. Compatible function approximation (Sec. 4.3, Theorem 3, p.5): a critic of the form
   Q^w(s,a) = (a - mu_theta(s))^T grad_theta mu_theta(s)^T w + V^v(s) leaves the gradient unbiased if w minimises
   the MSE between grad_a Q^w and grad_a Q^mu. COPDAC-Q (Eqs. 19-22, p.6) and COPDAC-GQ (gradient Q-learning critic,
   Eqs. 23-27, p.6). Natural gradient form: theta_{t+1} = theta_t + alpha_theta w_t (p.6).

Notation: delta_t = r_t + gamma Q^w(...) - Q^w(s_t, a_t) and the return r_t^gamma = sum_{k=t}^inf gamma^{k-t}
r(s_k, a_k) (Sec. 2.1, p.2). So DPG's r_t = r(s_t, a_t) is S&B's R_{t+1}.

## 6. Equations (transcribed from the PDF)

Eq. 2 (Sec. 2.2, p.2), stochastic policy gradient theorem (Sutton et al., 1999):
```tex
\nabla_\theta J(\pi_\theta) = \int_\mathcal{S} \rho^\pi(s) \int_\mathcal{A} \nabla_\theta \pi_\theta(a|s) Q^\pi(s,a) \,\mathrm{d}a \,\mathrm{d}s
  = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a) \right]   % (2)
```

Eqs. 6-7 (Sec. 3.1, p.3), policy improvement along the action-value gradient:
```tex
\theta^{k+1} = \theta^k + \alpha \mathbb{E}_{s \sim \rho^{\mu^k}}\left[ \nabla_\theta Q^{\mu^k}(s, \mu_\theta(s)) \right]   % (6)
\theta^{k+1} = \theta^k + \alpha \mathbb{E}_{s \sim \rho^{\mu^k}}\left[ \nabla_\theta \mu_\theta(s) \, \nabla_a Q^{\mu^k}(s,a)\big|_{a=\mu_\theta(s)} \right]   % (7)
```

Eq. 8 (Sec. 3.2, p.4), performance objective of a deterministic policy:
```tex
J(\mu_\theta) = \int_\mathcal{S} \rho^\mu(s) r(s, \mu_\theta(s)) \,\mathrm{d}s = \mathbb{E}_{s \sim \rho^\mu}\left[ r(s, \mu_\theta(s)) \right]   % (8)
```

Theorem 1 (Deterministic Policy Gradient Theorem), Eq. 9 (Sec. 3.2, p.4). Statement as printed: "Suppose that the
MDP satisfies conditions A.1 (see Appendix; these imply that grad_theta mu_theta(s) and grad_a Q^mu(s,a) exist and
that the deterministic policy gradient exists. Then,"
```tex
\nabla_\theta J(\mu_\theta) = \int_\mathcal{S} \rho^\mu(s) \nabla_\theta \mu_\theta(s) \, \nabla_a Q^\mu(s,a)\big|_{a=\mu_\theta(s)} \,\mathrm{d}s
  = \mathbb{E}_{s \sim \rho^\mu}\left[ \nabla_\theta \mu_\theta(s) \, \nabla_a Q^\mu(s,a)\big|_{a=\mu_\theta(s)} \right]   % (9)
```
Gloss: theta = policy parameters; mu_theta(s) = deterministic action; rho^mu = (improper) discounted state
distribution of mu; grad_theta mu_theta(s) = Jacobian of the action w.r.t. theta; grad_a Q^mu = slope of the true
action-value at the policy's action. "The proof follows a similar scheme to (Sutton et al., 1999) and is provided
in Appendix B." (p.4; the appendix is not in our file.)

Theorem 2 (limit of the stochastic policy gradient), Eq. 10 (Sec. 3.3, p.4). Statement as printed: "Consider a
stochastic policy pi_{mu_theta, sigma} such that pi_{mu_theta, sigma}(a|s) = nu_sigma(mu_theta(s), a), where sigma
is a parameter controlling the variance and nu_sigma satisfy conditions B.1 and the MDP satisfies conditions A.1
and A.2. Then,"
```tex
\lim_{\sigma \downarrow 0} \nabla_\theta J(\pi_{\mu_\theta, \sigma}) = \nabla_\theta J(\mu_\theta)   % (10)
```
"where on the l.h.s. the gradient is the standard stochastic policy gradient and on the r.h.s. the gradient is the
deterministic policy gradient." Section text: "for a wide class of stochastic policies, including many bump
functions, the deterministic policy gradient is indeed a special (limiting) case of the stochastic policy gradient"
(p.4). Abstract-side wording (p.1): "the deterministic policy gradient is the limiting case, as policy variance
tends to zero, of the stochastic policy gradient." Consequence stated: compatible function approximation, natural
gradients, actor-critic and episodic/batch methods all apply to deterministic policy gradients (p.4).

On-policy deterministic actor-critic with a Sarsa critic, Eqs. 11-13 (Sec. 4.1, p.4):
```tex
\delta_t = r_t + \gamma Q^w(s_{t+1}, a_{t+1}) - Q^w(s_t, a_t)   % (11)
w_{t+1} = w_t + \alpha_w \delta_t \nabla_w Q^w(s_t, a_t)   % (12)
\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu_\theta(s_t) \, \nabla_a Q^w(s_t, a_t)\big|_{a=\mu_\theta(s)}   % (13)
```

Off-policy objective, Eq. 14 (Sec. 4.2, p.4):
```tex
J_\beta(\mu_\theta) = \int_\mathcal{S} \rho^\beta(s) V^\mu(s) \,\mathrm{d}s = \int_\mathcal{S} \rho^\beta(s) Q^\mu(s, \mu_\theta(s)) \,\mathrm{d}s   % (14)
```

Off-policy deterministic policy gradient, Eq. 15 (Sec. 4.2, p.5). The first line is printed with
`\nabla_\theta \mu_\theta(a|s) Q^\mu(s,a)` (sic, the stochastic form's notation); the second line is the usable form:
```tex
\nabla_\theta J_\beta(\mu_\theta) \approx \int_\mathcal{S} \rho^\beta(s) \nabla_\theta \mu_\theta(a|s) Q^\mu(s,a) \,\mathrm{d}s
  = \mathbb{E}_{s \sim \rho^\beta}\left[ \nabla_\theta \mu_\theta(s) \, \nabla_a Q^\mu(s,a)\big|_{a=\mu_\theta(s)} \right]   % (15)
```
"Analogous to the stochastic case (see Equation 4), we have dropped a term that depends on
grad_theta Q^{mu_theta}(s,a); justification similar to Degris et al. (2012b) can be made in support of this
approximation." (p.5)

OPDAC, Eqs. 16-18 (Sec. 4.2, p.5), Q-learning critic:
```tex
\delta_t = r_t + \gamma Q^w(s_{t+1}, \mu_\theta(s_{t+1})) - Q^w(s_t, a_t)   % (16)
w_{t+1} = w_t + \alpha_w \delta_t \nabla_w Q^w(s_t, a_t)   % (17)
\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu_\theta(s_t) \, \nabla_a Q^w(s_t, a_t)\big|_{a=\mu_\theta(s)}   % (18)
```
Gloss: a_t comes from the behaviour policy beta; the bootstrap action is the target policy's mu_theta(s_{t+1})
(Q-learning style), so no importance ratio; alpha_w, alpha_theta = critic and actor step sizes; the actor step uses
the critic's action-slope at mu_theta(s), and the critic weights w are not changed by the actor step.

Theorem 3 (compatible function approximation, Sec. 4.3, p.5): Q^w(s,a) is compatible with mu_theta(s),
grad_theta J_beta(theta) = E[grad_theta mu_theta(s) grad_a Q^w(s,a)|_{a=mu_theta(s)}], if
(1) grad_a Q^w(s,a)|_{a=mu_theta(s)} = grad_theta mu_theta(s)^T w and (2) w minimises
MSE(theta, w) = E[eps(s;theta,w)^T eps(s;theta,w)], eps = grad_a Q^w - grad_a Q^mu at a = mu_theta(s).

COPDAC-Q, Eqs. 19-22 (Sec. 4.3, p.6):
```tex
\delta_t = r_t + \gamma Q^w(s_{t+1}, \mu_\theta(s_{t+1})) - Q^w(s_t, a_t)   % (19)
\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu_\theta(s_t) \left( \nabla_\theta \mu_\theta(s_t)^\top w_t \right)   % (20)
w_{t+1} = w_t + \alpha_w \delta_t \phi(s_t, a_t)   % (21)
v_{t+1} = v_t + \alpha_v \delta_t \phi(s_t)   % (22)
```
with features phi(s,a) = grad_theta mu_theta(s)(a - mu_theta(s)) (p.5). Cost: "the computational complexity of all
these updates is O(mn) per time-step" (m action dims, n policy parameters; p.6).

## 7. Experiment settings

| Setting | Value | Location |
|---|---|---|
| Continuous bandit cost | -r(a) = (a - a*)^T C (a - a*) | Sec. 5.1, p.6 |
| C | positive definite, eigenvalues chosen from {0.1, 1} | Sec. 5.1, p.6 |
| a* | [4, ..., 4]^T | Sec. 5.1, p.6 |
| Action dims m | 10, 25, 50 | Sec. 5.1, p.6 |
| SAC-B (stochastic actor-critic, bandit) | isotropic Gaussian pi_{theta,y} ~ N(theta, exp(y)); adapts mean and variance | Sec. 5.1, p.6 |
| COPDAC-B | target mu_theta = theta; behaviour beta ~ N(theta, sigma_beta^2), fixed width | Sec. 5.1, p.6 |
| Bandit critic | linear regression from compatible features to the costs, plus bias feature; recomputed from each batch of 2m steps; actor updated once per batch | Sec. 5.1, p.6 |
| Bandit evaluation | average cost per step of the mean (exploration not penalised for the on-policy algorithm); best step size / variance from a sweep; 5 runs | Sec. 5.1, p.6 |
| RL tasks | mountain car, pendulum, 2D puddle world; tile-coding features | Sec. 5.2, p.7 |
| RL episodes | truncated after 5000 steps | Sec. 5.2, p.7 |
| RL gamma | 0.99 (mountain car, pendulum); 0.999 (puddle world) | Sec. 5.2, p.7 |
| RL variance init | 1/2 the legal range; out-of-range actions capped | Sec. 5.2, p.7 |
| RL runs | best parameters, averaged over 30 runs | Sec. 5.2, p.7 |
| Octopus arm | 6 segments; 50 continuous state variables; 20 action variables | Sec. 5.3, p.7 |
| Octopus reward | proportional to change in distance to target; episode ends at hit (additional +50) or after 300 steps | Sec. 5.3, p.7 |
| Octopus nets | policy: sigmoidal MLP, 8 hidden units; V^v(s): MLP, 40 hidden units, linear output; A^w compatible | Sec. 5.3, p.8 |
| Octopus runs | 10 | Sec. 5.3 / Fig. 3, p.8 |

Naming trap: in this paper "SAC" means the stochastic actor-critic baseline (and "SAC-B" its bandit version), not
soft actor-critic.

## 8. Results (text only; no numbers read off plots)

- Bandit (Sec. 5.1, p.6; Fig. 1, p.7): "The results illustrate a significant performance advantage to the
  deterministic update, which grows larger with increasing dimensionality." With the same fixed variance for both,
  "COPDAC-B still outperforms SAC-B by a very wide margin that grows larger with increasing dimension."
- RL benchmarks (Sec. 5.2, p.7; Fig. 2): "COPDAC-Q slightly outperformed both SAC and OffPAC in all three domains."
- Octopus (Sec. 5.3, p.8; Fig. 3): "the octopus arm converged to a good solution in all cases" (10 runs).
- Conclusion (Sec. 7, p.8): "the deterministic actor-critic significantly outperformed its stochastic counterpart
  by several orders of magnitude in a bandit with 50 continuous action dimensions, and solved a challenging
  reinforcement learning problem with 20 continuous action dimensions and 50 state dimensions."

## 9. Limitations stated, and later fixes

- "These simple algorithms may have convergence issues in practice, due both to bias introduced by the function
  approximator, and also the instabilities caused by off-policy learning." (Sec. 4, p.4)
- A deterministic behaviour policy "will not ensure adequate exploration and may lead to suboptimal solutions"
  (Sec. 4.1, p.4); hence the off-policy algorithms.
- Eq. 15 drops a term (see above). Off-policy Q-learning with linear function approximation may diverge (p.6),
  hence COPDAC-GQ.
- A linear compatible critic "is not very useful for predicting action-values globally, since the action-value
  diverges to +-infinity for large actions" (p.5-6); it is a local critic.
- Later: DDPG (1509.02971v6, Sec. 5, p.6) says DPG used "tile-coding and linear function approximators" and "did not
  demonstrate scaling the approach to large, high-dimensional observation spaces"; DDPG adds replay, target networks
  and batch norm, and reports that learning "without a target network, as in the original work with DPG, is very
  poor in many environments" (Sec. 4, p.5).

## 10. Outline feedback

- The chapter's deterministic policy gradient can cite both DDPG Eq. 6 (as the outline says) and its origin, DPG
  Theorem 1, Eq. 9. The off-policy form the deep methods use is DPG Eq. 15; the update rules are OPDAC Eqs. 16-18.
- Theorem 2 / Eq. 10 is the clean bridge for the "Gaussian policy vs deterministic gradient" step: shrink sigma and
  the stochastic gradient tends to the deterministic one. The 1/sigma^2 variance remark (Sec. 6, p.8) explains why
  the deterministic estimator is the better one near convergence.
- The NumPy exercise ("Gaussian REINFORCE vs the deterministic gradient on the 1D toy") has a ready-made source
  toy: the Sec. 5.1 continuous bandit (quadratic cost, a* = 4 in every dimension, m = 10, 25, 50). A "Try this"
  that raises m reproduces the paper's finding qualitatively (the gap grows with dimension).
- If the page cites DPG experiments, rename its "SAC" to "stochastic actor-critic" to avoid a clash with soft
  actor-critic.
