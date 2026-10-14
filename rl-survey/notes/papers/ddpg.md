# DDPG: Continuous control with deep reinforcement learning (Lillicrap et al.)

Group F. Source: `sources/arxiv-1509.02971/` (abs.html, paper.pdf 14 pages, src/ddpg_iclr.tex). TeX below is
verbatim from `ddpg_iclr.tex`; equation numbers are as printed in the PDF. Page numbers are PDF pages (they match
the printed page numbers).

Custom macros: `\E` = `\DeclareMathOperator{\E}{\mathbb{E}}`; `\RR` = `I\!\!R` (the reals).

## 1. Identity

- Title: Continuous control with deep reinforcement learning.
- arXiv 1509.02971, version read v6 (abs.html: v1 Wed 9 Sep 2015, v6 Fri 5 Jul 2019). PDF stamp: "Published as a
  conference paper at ICLR 2016" (header of every page).
- Authors: Timothy P. Lillicrap (first; equal contribution with Jonathan J. Hunt), Jonathan J. Hunt, Alexander
  Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra (last). Google DeepMind, London.
- Sections: 1 Introduction (p.1-2); 2 Background (p.2-3); 3 Algorithm (p.3-4); 4 Results (p.4-6); 5 Related work
  (p.6-8); 6 Conclusion (p.8-9); supplementary: 7 Experiment Details (p.11); 8 Planning algorithm (p.11);
  9 Environment details (p.11-13).

## 2. Core problem

DQN learns from pixels but "can only handle discrete and low-dimensional action spaces": acting greedily needs
argmax_a Q, which for continuous actions "requires an iterative optimization process at every step" (Sec. 1, p.1).
Discretizing explodes: a 7 degree-of-freedom arm with a_i in {-k, 0, k} per joint gives 3^7 = 2187 actions (p.1).

## 3. What prior methods did and why that hurt (per the source)

- Naive discretization: curse of dimensionality, and it "throws away information about the structure of the action
  domain" (Sec. 1, p.1-2).
- DPG with neural function approximators: "a naive application of this actor-critic method with neural function
  approximators is unstable for challenging problems" (Sec. 1, p.2).
- NFQCA (Hafner and Riedmiller 2011) uses DPG's update rules with neural nets but "uses batch learning for stability,
  which is intractable for large networks"; a minibatch NFQCA without policy resets "is equivalent to the original
  DPG, which we compare to here" (Sec. 3, p.3).
- Large nonlinear value approximators were avoided "since theoretical performance guarantees are impossible, and
  practically learning tends to be unstable" (Sec. 2, p.3).

## 4. The one key idea

Take DPG's deterministic actor-critic and add DQN's two stabilizers (a replay buffer and target networks, here
"soft" target copies of both actor and critic), plus batch normalization and temporally correlated exploration
noise, then use one architecture and one hyperparameter set across all tasks.

## 5. Mechanism and Algorithm 1 (box transcribed, p.5)

Algorithm 1 "DDPG algorithm":
```
Randomly initialize critic network Q(s, a|theta^Q) and actor mu(s|theta^mu) with weights theta^Q and theta^mu.
Initialize target network Q' and mu' with weights theta^Q' <- theta^Q, theta^mu' <- theta^mu
Initialize replay buffer R
for episode = 1, M do
  Initialize a random process N for action exploration
  Receive initial observation state s_1
  for t = 1, T do
    Select action a_t = mu(s_t|theta^mu) + N_t according to the current policy and exploration noise
    Execute action a_t and observe reward r_t and observe new state s_{t+1}
    Store transition (s_t, a_t, r_t, s_{t+1}) in R
    Sample a random minibatch of N transitions (s_i, a_i, r_i, s_{i+1}) from R
    Set y_i = r_i + gamma Q'(s_{i+1}, mu'(s_{i+1}|theta^mu')|theta^Q')
    Update critic by minimizing the loss: L = (1/N) sum_i (y_i - Q(s_i, a_i|theta^Q))^2
    Update the actor policy using the sampled policy gradient:
      grad_{theta^mu} J ~= (1/N) sum_i grad_a Q(s, a|theta^Q)|_{s=s_i, a=mu(s_i)} grad_{theta^mu} mu(s|theta^mu)|_{s_i}
    Update the target networks:
      theta^Q' <- tau theta^Q + (1 - tau) theta^Q'
      theta^mu' <- tau theta^mu + (1 - tau) theta^mu'
  end for
end for
```
Box TeX lines (verbatim):
```tex
y_i = r_i + \gamma Q'(s_{i + 1}, \mu'(s_{i+1} | \theta^{\mu'}) | \theta^{Q'})
L = \frac{1}{N} \sum_i (y_i - Q(s_i, a_i | \theta^Q))^2
\nabla_{\theta^{\mu}} J \approx \frac{1}{N} \sum_i \nabla_{a} Q(s, a | \theta^Q)|_{s = s_i, a = \mu(s_i)} \nabla_{\theta^\mu} \mu(s | \theta^\mu)|_{s_i}
\theta^{Q'} \leftarrow \tau \theta^{Q} + (1 - \tau) \theta^{Q'}
\theta^{\mu'} \leftarrow \tau \theta^{\mu} + (1 - \tau) \theta^{\mu'}
```
- The printed box has NO terminal-state case. The LaTeX contains a commented-out `cases` block (non-terminal vs
  terminal `r_t`), so the authors removed it from print. TD3 Appendix D (1802.09477v3, p.14) states the rule our
  code should follow: y = r at a true terminal s' (not at the time limit).
- Constants vs gradients: y_i is computed with the target networks Q', mu' and is a fixed regression target for the
  critic ("While y_t is also dependent on theta^Q, this is typically ignored", Sec. 2, p.3). The actor step
  differentiates Q with respect to the action and chains into mu; the critic weights theta^Q are not changed by it.
- Replay: finite cache R; "When the replay buffer was full the oldest samples were discarded"; actor and critic are
  updated every timestep from a minibatch sampled uniformly (Sec. 3, p.4).
- Soft targets: "theta' <- tau theta + (1 - tau) theta' with tau << 1"; "having both a target mu' and Q' was required
  to have stable targets y_i in order to consistently train the critic without divergence" (Sec. 3, p.4).
- Batch norm (low-dimensional case): "on the state input and all layers of the mu network and all layers of the Q
  network prior to the action input" (Sec. 3, p.4). Running averages are used "during exploration or evaluation".
- Exploration: Eq. 7, an Ornstein-Uhlenbeck process "to generate temporally correlated exploration for exploration
  efficiency in physical control problems with inertia" (Sec. 3, p.4). The paper gives theta and sigma (Sec. 7) but
  prints no OU update equation and no time step dt.
- Notation trap: mu' names both the target actor (Sec. 3, p.4) and the exploration policy (Eq. 7).
- Reward notation: the reward after a_t is r_t = r(s_t, a_t) (S&B's R_{t+1}); return
  R_t = sum_{i=t}^T gamma^{(i-t)} r(s_i, a_i) (Sec. 2, p.2), i.e. S&B's G_t. The paper also prints
  "mu: S <- A" (sic, arrow reversed; Sec. 2, p.3).

## 6. Equations (verbatim from LaTeX; printed numbers)

Eq. 1 (Sec. 2, p.2), action-value:
```tex
Q^\pi(s_t, a_t) = \E_{r_{i \ge t}, s_{i > t} \sim E, a_{i > t} \sim \pi} \left[ R_t | s_t, a_t \right ]
```
Eq. 2 (p.3), Bellman equation, stochastic policy:
```tex
Q^{\pi}(s_t, a_t) = \E_{r_t, s_{t+1} \sim E} \left[ r(s_t, a_t) + \gamma \E_{a_{t + 1} \sim \pi} \left[ Q^{\pi}(s_{t+1}, a_{t + 1}) \right]  \right]
```
Eq. 3 (p.3), deterministic target policy, no inner expectation: "The expectation depends only on the environment.
This means that it is possible to learn Q^mu off-policy, using transitions which are generated from a different
stochastic behavior policy beta."
```tex
Q^{\mu}(s_t, a_t) = \E_{r_t, s_{t+1} \sim E} \left [ r(s_t, a_t) + \gamma Q^{\mu}(s_{t+1}, \mu(s_{t + 1})) \right]
```
Eq. 4 (p.3), critic loss:
```tex
L(\theta^{Q}) = \E_{s_t \sim \rho^\beta, a_t \sim \beta, r_t \sim E} \left[\left(Q(s_t, a_t | \theta^{Q}) - y_t\right)^2 \right]
```
Eq. 5 (p.3), target (as written in Background, before target networks are introduced):
```tex
y_t =  r(s_t, a_t) + \gamma Q(s_{t+1}, \mu(s_{t+1}) | \theta^Q).
```
Eq. 6 (Sec. 3, p.3), actor update, the deterministic policy gradient:
```tex
\begin{split}
\nabla_{\theta^\mu} J &\approx \E_{s_t \sim \rho^\beta}\left[\nabla_{\theta^{\mu}} Q(s, a | \theta^Q)|_{s = s_t, a = \mu(s_t | \theta^{\mu})} \right] \\
 & = \E_{s_t \sim \rho^\beta}\left[\nabla_{a} Q(s, a | \theta^{Q})|_{s = s_t, a = \mu(s_t)} \nabla_{\theta_\mu} \mu(s | \theta^{\mu})|_{s = s_t} \right]
\end{split}
```
Wording: "The actor is updated by following the applying the chain rule to the expected return from the start
distribution J with respect to the actor parameters" (sic); "Silver et al. (2014) proved that this is the policy
gradient, the gradient of the policy's performance" (p.3). Footnote 2 (p.3): "In practice, as in commonly done in
policy gradient implementations, we ignored the discount in the state-visitation distribution rho^beta."
Gloss: theta^mu = actor weights; theta^Q = critic weights; rho^beta = state distribution under the behaviour
(exploration) policy; grad_a Q = how the critic's value changes with the action at the actor's action;
grad_{theta^mu} mu = how the action changes with the actor weights.

Eq. 7 (Sec. 3, p.4), exploration policy:
```tex
\mu'(s_t) = \mu(s_t | \theta^\mu_t) + \mathcal{N}
```
Soft target update (in text, unnumbered, p.4): `\theta' \leftarrow \tau\theta + (1 - \tau) \theta'` with `\tau \ll 1`.

## 7. Config (Sec. 7 "Experiment Details", p.11, unless noted)

| Setting | Low-dimensional | Pixels |
|---|---|---|
| Optimizer | Adam | Adam |
| Learning rate, actor / critic | 10^-4 / 10^-3 | same |
| Critic L2 weight decay | 10^-2 | same |
| Discount gamma | 0.99 | same |
| Soft target tau | 0.001 | same |
| Hidden nonlinearity | ReLU ("rectified non-linearity") | same |
| Actor output | tanh, "to bound the actions" | same |
| Network | 2 hidden layers, 400 and 300 units (approx 130,000 parameters) | 3 conv layers, no pooling, 32 filters each; then 2 fully connected layers of 200 units (approx 430,000 parameters) |
| Where the action enters Q | "Actions were not included until the 2nd hidden layer of Q" | "not included until the fully-connected layers" |
| Final-layer init (actor and critic, weights and biases) | uniform [-3 x 10^-3, 3 x 10^-3] | printed "[3 x 10^-4, 3 x 10^-4]" (sic, the minus sign is missing; a typo) |
| Other layers init | uniform [-1/sqrt(f), 1/sqrt(f)], f = fan-in | same |
| Minibatch | 64 | 16 |
| Replay buffer | 10^6 | 10^6 |
| Exploration noise | Ornstein-Uhlenbeck, theta = 0.15, sigma = 0.2 | same |
| Batch normalization | state input, all mu layers, Q layers before the action input (Sec. 3, p.4) | not stated for pixels |
| Observation | joint angles, positions etc. | 3 action repeats, 9 feature maps (RGB of 3 renderings), 64x64 pixels, scaled to [0, 1] (Sec. 4, p.5) |

Final-layer init reason: "to ensure the initial outputs for the policy and value estimates were near zero". OU:
"models the velocity of a Brownian particle with friction, which results in temporally correlated values centered
around 0". Torcs: same network and hyperparameters, "altered the noise process for exploration because of the very
different time scales" (Sec. 4, p.6); reward = velocity along track, -1 for collisions, episode ends after 500
frames without progress (Sec. 9.1, p.11). Evaluation: "testing it without exploration noise" (Sec. 4, p.5).
Training length: "at most 2.5 million steps" (Table 1 caption, p.7).

## 8. Results worth remembering

- Abstract: "Using the same learning algorithm, network architecture and hyper-parameters, our algorithm robustly
  solves more than 20 simulated physics tasks".
- Table 1 (p.7): averaged over 5 replicas; scores normalized so a uniform-random policy scores 0 and iLQG (a planner
  with full access to the dynamics and its derivatives) scores 1; Torcs raw. Columns: R_av,lowd, R_best,lowd,
  R_av,pix, R_best,pix, R_av,cntrl, R_best,cntrl (cntrl = "the original DPG algorithm with a replay buffer and batch
  normalization"). Selected rows:

| environment | av lowd | best lowd | av pix | best pix | av cntrl | best cntrl |
|---|---|---|---|---|---|---|
| pendulum | 0.946 | 1.021 | 0.663 | 1.055 | 0.099 | 0.951 |
| cartpole | 0.844 | 1.115 | 0.482 | 1.138 | 0.244 | 0.755 |
| cheetah | 0.903 | 1.206 | 0.457 | 0.792 | -0.008 | 0.425 |
| hardCheetah | 1.311 | 1.990 | 1.204 | 1.431 | -0.031 | 1.411 |
| hopper | 0.676 | 0.936 | 0.112 | 0.924 | 0.078 | 0.917 |
| walker2d | 0.705 | 1.573 | 0.944 | 1.476 | 0.393 | 1.397 |
| torcs (raw) | -393.385 | 1840.036 | -401.911 | 1876.284 | -911.034 | 1961.600 |

- Ablation (Sec. 4, p.5; Fig. 2 caption, p.6): target network and batch norm are both necessary to perform well
  across all tasks; "learning without a target network, as in the original work with DPG, is very poor in many
  environments"; caption: "Target networks are crucial." (Curves only; no numbers taken.)
- Value accuracy (Sec. 4, p.6; Fig. 3, p.7): "in simple tasks DDPG estimates returns accurately without systematic
  biases. For harder tasks the Q estimates are worse, but DDPG is still able learn good policies."
- Pixels (Sec. 4, p.5): "in some simpler tasks, learning policies from pixels is just as fast as learning using the
  low-dimensional state descriptor."
- Sample count (Sec. 6, p.8-9): "Nearly all of the problems we looked at were solved within 2.5 million steps of
  experience (and usually far fewer), a factor of 20 fewer steps than DQN requires for good Atari solutions."

## 9. Limitations stated, and later fixes

- "DDPG requires a large number of training episodes to find solutions" (Sec. 6, p.9).
- "the use of non-linear function approximators nullifies any convergence guarantees" (Sec. 6, p.8).
- Harder tasks: Q estimates are worse (p.6). Torcs: "other replicas failed to learn a sensible policy" (p.6).
- Later (sources in this group): TD3 (1802.09477v3, Sec. 4.1 and Fig. 1, p.3) measures a clear overestimation bias
  in DDPG's critic on Hopper-v1 and Walker2d-v1 and adds clipped double Q, delayed updates and target smoothing;
  TD3 also drops OU noise ("noise drawn from the Ornstein-Uhlenbeck process offered no performance benefits",
  Sec. 6.1, p.7). SAC (1801.01290v2, Sec. 1, p.1) calls DDPG "notoriously challenging to use due to its extreme
  brittleness and hyperparameter sensitivity" and reports that it "fails to make any progress on Ant-v1,
  Humanoid-v1, and Humanoid (rllab)" (Sec. 5.1, p.7).

## 10. Outline feedback

- "DDPG Eq. 6 as numbered" is correct: Eq. 6 is the two-line actor gradient; the second line is the chain-rule form
  to show. Footnote 2 (discount ignored in rho^beta) is worth one sentence.
- The step "DDPG = DQN's tricks + a deterministic actor + exploration noise" omits batch normalization, which DDPG
  lists as its third ingredient (Sec. 1, p.2; Sec. 3, p.4). TD3's re-tuned DDPG drops it (no normalized
  observations, TD3 Table 3). Mention it once, so readers of the original paper are not surprised.
- Soft targets update BOTH actor and critic copies every step with tau = 0.001; DQN copied weights periodically.
  This contrast fits the "DQN's tricks" step.
- Name the mu' notation clash (target actor vs exploration policy) if the page shows Eq. 7.
- The OU process has no printed equation or dt here; any discretization in our code is our choice; label it.
- Our code must add a terminal mask that the printed Algorithm 1 lacks (cite TD3 App. D for the rule).
