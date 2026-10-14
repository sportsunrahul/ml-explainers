# SAC: Soft Actor-Critic, Off-Policy Maximum Entropy Deep RL with a Stochastic Actor (Haarnoja et al.)

Group F. Source: `sources/arxiv-1801.01290/` (abs.html, paper.pdf 14 pages, src/main_no_comments.tex, defs.tex,
preamble.tex). This is v2 of the arXiv record, the ICML 2018 paper. TeX below is verbatim from
`main_no_comments.tex` (`\resizebox{\columnwidth}{!}{$...$}` wrappers dropped); equation numbers are as printed
(checked on PDF page images). Page numbers are PDF pages (the ICML style prints none).

Custom macros (defs.tex): `\E{A}{B}` = `\operatorname{\mathbb{E}}_{A}\left[B\right]`; `\kl{A}{B}` =
`\mathrm{D_{KL}}\left(A\;\middle\|\;B\right)`; `\ent` = `\mathcal{H}`; `\voidarg` = `{\,\cdot\,}`; `\policy` = `\pi`;
`\st` = `{\mathbf{s}_t}`; `\at` = `{\mathbf{a}_t}`; `\stp` = `{\mathbf{s}_{t+1}}`; `\state` = `\mathbf{s}`;
`\action` = `\mathbf{a}`; `\urv` = `\mathbf{u}`; `\reward` = `r`; `\discount` = `\gamma`; `\pdyn` = `p`;
`\params` = `\theta` (Q); `\pparams` = `\phi` (policy); `\vparams` = `\psi` (value); `\vtargetparams` = `\bar\psi`;
`\V` = `V`; `\Q` = `Q`; `\gauss` = `\mathcal{N}`; `\sspace` = `\mathcal{S}`; `\aspace` = `\mathcal{A}`;
`\reals` = `\mathbb{R}`; `\opt` = `^*`; `\inv` = `^{-1}`.

## 1. Identity

- Title: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.
- arXiv 1801.01290, version read v2 (abs.html: v1 Thu 4 Jan 2018, v2 Wed 8 Aug 2018). Venue: ICML 2018 (abs.html
  comment "ICML 2018"; the arXiv PDF prints no proceedings footer).
- Authors: Tuomas Haarnoja (first), Aurick Zhou, Pieter Abbeel, Sergey Levine (last). Berkeley Artificial
  Intelligence Research, UC Berkeley. Code: github.com/haarnoja/sac (Sec. 5, p.6).
- Sections: 1 Introduction (p.1-2); 2 Related Work (p.2-3); 3 Preliminaries: 3.1 Notation, 3.2 Maximum Entropy
  Reinforcement Learning (p.3); 4 From Soft Policy Iteration to Soft Actor-Critic: 4.1 Derivation of Soft Policy
  Iteration (p.4), 4.2 Soft Actor-Critic (p.4-6); 5 Experiments (p.6): 5.1 Comparative Evaluation (p.7),
  5.2 Ablation Study (p.7-8); 6 Conclusion (p.8). Appendix A Maximum Entropy Objective (p.11); B Proofs (p.11-12);
  C Enforcing Action Bounds (p.12); D Hyperparameters (p.13); E Additional Baseline Results (p.14).

## 2. Core problem

Model-free deep RL needs many samples and is brittle to hyperparameters. On-policy methods (TRPO, PPO, A3C) need
new samples for each gradient step; the off-policy DDPG is sample-efficient but brittle; earlier off-policy maximum
entropy methods (soft Q-learning) need approximate inference in continuous action spaces.

## 3. What prior methods did and why that hurt (per the source)

- On-policy methods "require new samples to be collected for each gradient step. This quickly becomes extravagantly
  expensive" (Sec. 1, p.1).
- DDPG "provides for sample-efficient learning but is notoriously challenging to use due to its extreme brittleness
  and hyperparameter sensitivity" (Sec. 1, p.1); "the interplay between the deterministic actor network and the
  Q-function typically makes DDPG extremely difficult to stabilize" (Sec. 2, p.2).
- Soft Q-learning (Haarnoja et al. 2017) "is not a true actor-critic algorithm: the Q-function is estimating the
  optimal Q-function, and the actor does not directly affect the Q-function except through the data distribution";
  its convergence "hinges on how well this sampler approximates the true posterior" (Sec. 2, p.3).
- SVG(0) "optimizes the standard maximum expected return objective, and it does not make use of a separate value
  network, which we found to make training more stable" (Sec. 2, p.2).

## 4. The one key idea

Run policy iteration on a maximum entropy objective (reward plus alpha times policy entropy), off-policy: evaluate
the soft Q-function of the current policy from a replay buffer, then improve the stochastic policy by moving it
toward exp(Q)/Z (a KL projection), with the reparameterization trick so the policy gradient passes through Q.

## 5. Mechanism and Algorithm 1 (box transcribed, p.5)

1. Soft policy evaluation (Sec. 4.1, p.4): repeat the soft Bellman backup (Eq. 2) with the soft value (Eq. 3);
   Lemma 1 says this converges to the soft Q-value of pi (finite action space assumed).
2. Soft policy improvement (Eq. 4): project exp(Q^{pi_old})/Z onto the policy class Pi by KL; Lemma 2 says the new
   policy's soft Q is at least the old one's. Theorem 1 (Soft Policy Iteration): alternating the two converges to
   the optimal policy in Pi (|A| < infinity).
3. Practical SAC (Sec. 4.2, p.4-6): networks V_psi(s_t) (state value), Q_theta(s_t, a_t) (two of them), policy
   pi_phi(a_t|s_t), and a target value network V_psi-bar. "the policy as a Gaussian with mean and covariance given
   by neural networks" (p.4, given as an example).
4. Two Q-functions (Sec. 4.2, p.5), quoted: "Our algorithm also makes use of two Q-functions to mitigate positive
   bias in the policy improvement step that is known to degrade performance of value based methods (Hasselt, 2010;
   Fujimoto et al., 2018). In particular, we parameterize two Q-functions, with parameters theta_i, and train them
   independently to optimize J_Q(theta_i). We then use the minimum of the Q-functions for the value gradient in
   Equation 6 and policy gradient in Equation 13, as proposed by Fujimoto et al. (2018). Although our algorithm can
   learn challenging tasks, including a 21-dimensional Humanoid, using just a single Q-function, we found two
   Q-functions significantly speed up training, especially on harder tasks."
5. "In practice, we take a single environment step followed by one or several gradient steps" (p.5-6).

Algorithm 1 "Soft Actor-Critic":
```
Initialize parameter vectors psi, psi-bar, theta, phi.
for each iteration do
  for each environment step do
    a_t ~ pi_phi(a_t|s_t)
    s_{t+1} ~ p(s_{t+1}|s_t, a_t)
    D <- D U {(s_t, a_t, r(s_t, a_t), s_{t+1})}
  end for
  for each gradient step do
    psi <- psi - lambda_V grad-hat_psi J_V(psi)
    theta_i <- theta_i - lambda_Q grad-hat_theta_i J_Q(theta_i) for i in {1, 2}
    phi <- phi - lambda_pi grad-hat_phi J_pi(phi)
    psi-bar <- tau psi + (1 - tau) psi-bar
  end for
end for
```
Box TeX lines (verbatim):
```tex
\vparams \leftarrow \vparams - \lambda_V \hat \nabla_\vparams J_\V(\vparams)
\params_i \leftarrow \params_i - \lambda_Q \hat \nabla_{\params_i} J_\Q(\params_i) \text{ for } i\in\{1, 2\}
\pparams \leftarrow \pparams - \lambda_\policy \hat \nabla_\pparams J_\policy(\pparams)
\vtargetparams\leftarrow \tau \vparams + (1-\tau)\vtargetparams
```

Does this version use a separate state-value network? YES. Sec. 4.2 (p.4-5): "There is no need in principle to
include a separate function approximator for the state value, since it is related to the Q-function and policy
according to Equation 3. This quantity can be estimated from a single action sample from the current policy without
introducing a bias, but in practice, including a separate function approximator for the soft value can stabilize
training and is convenient to train simultaneously with the other networks." The only target network is the value
target V_psi-bar; there are no target Q-networks and no target actor.

Temperature: "For the rest of this paper, we will omit writing the temperature explicitly, as it can always be
subsumed into the reward by scaling it by alpha^-1" (Sec. 3.2, p.3). So alpha is absent from Eqs. 2-13; the per-task
reward scale (Table 2) plays the role of 1/alpha ("its natural interpretation as the inverse of the temperature",
Sec. 5.2, p.8).

Constants vs gradients: in Eq. 6 only V_psi is differentiated; Q_theta and log pi_phi enter as a fixed target, with
actions "sampled according to the current policy, instead of the replay buffer". In Eq. 9 the target uses the frozen
V_psi-bar. In Eqs. 12-13 the gradient flows through Q_theta into the action a_t = f_phi(eps_t; s_t); Algorithm 1
updates only phi with it. The partition function Z does not depend on phi and is dropped (p.5).

Reward notation: r(s_t, a_t) is the reward on the transition from s_t under a_t (S&B's R_{t+1}).

## 6. Equations (verbatim from LaTeX; printed numbers)

Eq. 1 (Sec. 3.2, p.3), maximum entropy objective:
```tex
J(\policy)  = \sum_{t=0}^{T} \E{(\st, \at) \sim \rho_\policy}{\reward(\st,\at) + \alpha\ent(\policy(\voidarg|\st))}.
```
Expanded: J(pi) = sum_{t=0}^T E_{(s_t,a_t) ~ rho_pi}[ r(s_t,a_t) + alpha H(pi(.|s_t)) ].
Role of alpha (p.3): "The temperature parameter alpha determines the relative importance of the entropy term against
the reward, and thus controls the stochasticity of the optimal policy. The maximum entropy objective differs from
the standard maximum expected reward objective used in conventional reinforcement learning, though the
conventional objective can be recovered in the limit as alpha -> 0." Benefits (p.3): "the policy is incentivized to
explore more widely, while giving up on clearly unpromising avenues. Second, the policy can capture multiple modes
of near-optimal behavior. In problem settings where multiple actions seem equally attractive, the policy will
commit equal probability mass to those actions."
Gloss: rho_pi = state-action marginal of the trajectory distribution under pi; H(pi(.|s_t)) = entropy of the action
distribution at s_t; alpha = temperature.

Eq. 2 (Sec. 4.1, p.4), soft Bellman backup operator:
```tex
\mathcal{T}^\policy Q(\st, \at) \triangleq  \reward(\st, \at) + \discount \E{\stp \sim \pdyn}{V(\stp)},
```
Eq. 3 (p.4), soft state value:
```tex
V(\st) = \E{\at\sim\policy}{\Q(\st, \at) - \log\policy(\at|\st)}
```
Eq. 4 (p.4), soft policy improvement:
```tex
\policy_\mathrm{new} = \arg\underset{\policy'\in \Pi}{\min}\kl{\policy'(\voidarg|\st)}{\frac{\exp\left(Q^{\policy_\mathrm{old}}(\st, \voidarg)\right)}{Z^{\policy_\mathrm{old}}(\st)}}.
```
Eq. 5 (Sec. 4.2, p.5), value loss:
```tex
J_V(\vparams) = \E{\st \sim \mathcal{D}}{\frac{1}{2}\left(\V_\vparams(\st) - \E{\at\sim\policy_\pparams}{Q_\params(\st, \at) - \log \policy_\pparams(\at|\st)}\right)^2}
```
Eq. 6 (p.5), its gradient estimate:
```tex
\hat \nabla_\vparams J_V(\vparams) = \nabla_\vparams \V_\vparams(\st) \left(\V_\vparams(\st) - Q_\params(\st, \at) + \log \policy_\pparams(\at|\st)\right),
```
Eq. 7 (p.5), soft Q loss (soft Bellman residual), with target Eq. 8:
```tex
J_\Q(\params) = \E{(\st, \at)\sim\mathcal{D}}{\frac{1}{2}\left(\Q_\params(\st, \at) - \hat \Q(\st, \at)\right)^2},
\hat \Q(\st, \at) = \reward(\st, \at) + \discount \E{\stp\sim\pdyn}{\V_\vtargetparams(\stp)},
```
Eq. 9 (p.5), Q gradient (printed with `\Q_\params(\at, \st)`, arguments swapped, sic):
```tex
\hat \nabla_\params J_Q(\params) =  \nabla_\params \Q_\params(\at, \st) \left(\Q_\params(\st, \at) - \reward(\st, \at) - \discount \V_\vtargetparams(\stp)\right)
```
"The update makes use of a target value network V_psi-bar, where psi-bar can be an exponentially moving average of
the value network weights ... Alternatively, we can update the target weights to match the current value function
weights periodically" (p.5).

Eq. 10 (p.5), policy loss (expected KL):
```tex
J_\policy(\pparams) = \E{\st\sim\mathcal{D}}{\kl{\policy_\pparams(\voidarg|\st)}{\frac{\exp\left(Q_\params(\st, \voidarg)\right)}{Z_\params(\st)}}}.
```
Reparameterization, Eq. 11 (p.5), "where epsilon_t is an input noise vector, sampled from some fixed distribution,
such as a spherical Gaussian":
```tex
\at = f_\pparams(\epsilon_t; \st),
```
Eq. 12 (p.5), reparameterized policy loss:
```tex
J_\policy(\pparams) = \E{\st\sim\mathcal{D},\epsilon_t\sim\gauss}{\log \policy_\pparams(f_\pparams(\epsilon_t;\st)|\st) - Q_\params(\st, f_\pparams(\epsilon_t;\st))},
```
Eq. 13 (p.5), gradient estimator:
```tex
\hat\nabla_\pparams J_\policy(\pparams) = \nabla_\pparams \log \policy_\pparams(\at|\st) + (\nabla_\at \log \policy_\pparams(\at|\st) - \nabla_\at Q(\st, \at))\nabla_\pparams f_\pparams(\epsilon_t;\st),
```
"This unbiased gradient estimator extends the DDPG style policy gradients (Lillicrap et al., 2015) to any tractable
stochastic policy." Why reparameterize (p.5): the target density is the Q-function, a differentiable network, so
the reparameterization trick gives "a lower variance estimator" than the likelihood ratio estimator.

Eq. 14 (App. A, p.11), discounted infinite-horizon objective:
```tex
J(\policy) = \sum_{t=0}^\infty \E{(\st,\at) \sim \rho_\policy}{ \sum_{l=t}^\infty \discount^{l-t} \E{\state_l\sim\pdyn,\action_l\sim\policy}{ \reward(\st,\at) + \alpha \ent(\policy(\voidarg|\st))|\st,\at}}.
```

Tanh squashing, Appendix C "Enforcing Action Bounds" (p.12). "We use an unbounded Gaussian as the action
distribution. However, in practice, the actions needs to be bounded to a finite interval. To that end, we apply an
invertible squashing function (tanh) to the Gaussian samples, and employ the change of variables formula to compute
the likelihoods of the bounded actions." With u in R^D, mu(u|s) its density, a = tanh(u) elementwise, support
(-1, 1):
```tex
\policy (\action|\state) &= \mu(\urv|\state)\left|\det \left(\frac{\mathrm{d}\action}{\mathrm{d}\urv} \right)\right|^{-1}.   % (20)
\log\policy (\action|\state) &= \log \mu(\urv|\state) - \sum_{i=1}^D\log\left(1 - \tanh^2(u_i)\right),   % (21)
```
"Since the Jacobian da/du = diag(1 - tanh^2(u)) is diagonal, the log-likelihood has a simple form"; u_i is the i-th
element of u. Note: here mu(u|s) is the Gaussian density, not a deterministic policy.

## 7. Config: Appendix D (p.13), every row

Table 1 "SAC Hyperparameters":

| Group | Parameter | Value |
|---|---|---|
| Shared | optimizer | Adam (Kingma and Ba, 2015) |
| Shared | learning rate | 3 * 10^-4 |
| Shared | discount (gamma) | 0.99 |
| Shared | replay buffer size | 10^6 |
| Shared | number of hidden layers (all networks) | 2 |
| Shared | number of hidden units per layer | 256 |
| Shared | number of samples per minibatch | 256 |
| Shared | nonlinearity | ReLU |
| SAC | target smoothing coefficient (tau) | 0.005 |
| SAC | target update interval | 1 |
| SAC | gradient steps | 1 |
| SAC (hard target update) | target smoothing coefficient (tau) | 1 |
| SAC (hard target update) | target update interval | 1000 |
| SAC (hard target update) | gradient steps (except humanoids) | 4 |
| SAC (hard target update) | gradient steps (humanoids) | 1 |

Table 2 "SAC Environment Specific Parameters":

| Environment | Action Dimensions | Reward Scale |
|---|---|---|
| Hopper-v1 | 3 | 5 |
| Walker2d-v1 | 6 | 5 |
| HalfCheetah-v1 | 6 | 5 |
| Ant-v1 | 8 | 5 |
| Humanoid-v1 | 17 | 20 |
| Humanoid (rllab) | 21 | 10 |

Measurement setup (Sec. 5 and 5.1, p.6-7): OpenAI gym tasks plus rllab Humanoid; baselines DDPG, PPO, SQL (with two
Q-functions), TD3 (author code); exploration noise off for DDPG and PPO at evaluation; SQL evaluated with its noise;
SAC evaluated with "the mean action". "We train five different instances of each algorithm with different random
seeds, with each performing one evaluation rollout every 1000 environment steps. The solid curves corresponds to
the mean and the shaded region to the minimum and maximum returns over the five trials."

## 8. Results (text only; the paper has no results table, and no numbers are read off its plots)

- Sec. 5.1 (p.7), Fig. 1 (p.6): "SAC performs comparably to the baseline methods on the easier tasks and
  outperforms them on the harder tasks with a large margin, both in terms of learning speed and the final
  performance. For example, DDPG fails to make any progress on Ant-v1, Humanoid-v1, and Humanoid (rllab)"; "SAC also
  learns considerably faster than PPO as a consequence of the large batch sizes PPO needs to learn stably on more
  high-dimensional and complex tasks"; SQL "is slower than SAC and has worse asymptotic performance."
- Stochastic vs deterministic (Sec. 5.2, p.7; Fig. 2, Humanoid (rllab), five seeds each): the deterministic variant
  "closely resembles DDPG, with the exception of having two Q-functions, using hard target updates, not having a
  separate target actor, and using fixed rather than learned exploration noise"; "Soft actor-critic performs much
  more consistently, while the deterministic variant exhibits very high variability across seeds".
- Evaluation (p.7, Fig. 3a on Ant-v1): using the mean action "can yield better performance"; training curves show
  the sum of rewards, not the entropy-augmented objective SAC optimizes.
- Reward scale (p.7-8, Fig. 3b): "particularly sensitive to the scaling of the reward signal, because it serves the
  role of the temperature"; small rewards give a nearly uniform policy that "fails to exploit the reward signal";
  large rewards give a nearly deterministic policy with "poor local minima due to lack of adequate exploration";
  "we found reward scale to be the only hyperparameter that requires tuning". (Printed "Larger reward magnitudes
  correspond to lower entries", sic: entropies.)
- Target update (p.8, Fig. 3c): "Large tau can lead to instabilities while small tau can make training slower.
  However, we found the range of suitable values of tau to be relatively wide and we used the same value (0.005)
  across all of the tasks." Hard-update variant copies weights "every 1000 gradient steps" and benefits from more
  than one gradient step per environment step (p.8; App. E, p.14: comparable to SAC except Humanoid (rllab), "on
  which SAC is the fastest").

## 9. Limitations stated, and later fixes

- Sensitivity to reward scale (the temperature), tuned per task (Table 2). Fixed later by learning the temperature
  under an entropy constraint (1812.05905v2, Sec. 5-6, Eqs. 17-18).
- The separate value network is an engineering choice ("can stabilize training"); 1812.05905v2 footnote 1 (p.6):
  "we introduced an additional function approximator for the value function, but later found it to be
  unnecessary."
- Theory (Lemmas 1-2, Theorem 1) assumes a finite action space: "The assumption |A| < infinity is required to
  guarantee that the entropy augmented reward is bounded" (App. B.1, p.11). The practical algorithm is an
  approximation with function approximators.

## 10. Outline feedback

- "SAC objective (SAC Eq. 1)" is right for this version (sum form, J(pi)). 1812.05905v2 also has an Eq. 1, but it is
  the argmax form pi* = argmax_pi sum_t E[...]. Cite 1801.01290v2 Eq. 1 for J(pi).
- viz-sacflow ("twin critics with target copies", learned temperature) matches 1812.05905v2, not this version.
  This version trains V_psi plus a target V_psi-bar, has no target Q-networks, and folds alpha into the reward scale.
  Say which version the walkthrough draws; 1812.05905v2 matches what CleanRL-style code implements.
- "log std" is not in either SAC paper; they say "a Gaussian with mean and covariance given by neural networks".
  Label log-std output as an implementation choice.
- The tanh step should show App. C Eq. 21 (this version) or Eq. 26 (1812.05905v2), with a worked number. Actions
  land in (-1, 1); the exercise must rescale to the environment's bounds (an implementation detail, not in the
  paper).
- viz-entropy: the quote "the policy will commit equal probability mass to those actions" describes the optimal
  maximum entropy policy (proportional to exp(Q/alpha), Eq. 4 here with alpha folded in). A single Gaussian cannot
  put mass on two separated peaks; under the KL projection of Eq. 4 it either sits on one peak or spreads over both.
  Draw the Boltzmann target exp(Q/alpha)/Z next to the fitted Gaussian so the slider shows both.
- Sec. 5.2's "deterministic variant" is a clean, sourced way to say why the entropy matters: same two Q-functions,
  no entropy, much higher seed-to-seed variability on Humanoid (rllab).
