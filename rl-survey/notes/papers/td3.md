# TD3: Addressing Function Approximation Error in Actor-Critic Methods (Fujimoto et al.)

Group F. Source: `sources/arxiv-1802.09477/` (abs.html, paper.pdf 15 pages, src/example_paper.tex). TeX below is
verbatim from `example_paper.tex`; equation numbers are as printed (checked on PDF page images). Page numbers are
PDF pages (the ICML style prints none).

Custom macros: `\y` = `\gamma`; `\e` = `\epsilon`; `\s` = `\sigma`; `\E` = `\mathbb{E}`; `\N` = `\mathcal{N}`;
`\lb`/`\rb` = `\left [`/`\right ]`; `\lp`/`\rp` = `\left (`/`\right )`; `\B` = `\mathcal{B}`;
`\clip` = `\DeclareMathOperator*{\clip}{clip}`; `\argmin` = `\DeclareMathOperator*{\argmin}{argmin}`.

## 1. Identity

- Title: Addressing Function Approximation Error in Actor-Critic Methods. Algorithm name: Twin Delayed Deep
  Deterministic policy gradient (TD3).
- arXiv 1802.09477, version read v3 (abs.html: v1 Mon 26 Feb 2018, v2 7 Jun 2018, v3 Mon 22 Oct 2018). Venue:
  "Proceedings of the 35th International Conference on Machine Learning, Stockholm, Sweden, PMLR 80, 2018" (p.1
  footer); abs.html comment "Accepted at ICML 2018".
- Authors: Scott Fujimoto (first), Herke van Hoof, David Meger (last). McGill University; University of Amsterdam.
  Code: https://github.com/sfujim/TD3 (p.2).
- Sections: 1 Introduction (p.1-2); 2 Related Work (p.2); 3 Background (p.2-3); 4 Overestimation Bias (p.3):
  4.1 Overestimation Bias in Actor-Critic (p.3-4), 4.2 Clipped Double Q-Learning for Actor-Critic (p.4);
  5 Addressing Variance (p.4-5): 5.1 Accumulating Error (p.5), 5.2 Target Networks and Delayed Policy Updates
  (p.5-6), 5.3 Target Policy Smoothing Regularization (p.6); 6 Experiments (p.6): 6.1 Evaluation (p.7-8),
  6.2 Ablation Studies (p.8); 7 Conclusion (p.8). Appendix A Proof of Convergence of Clipped Double Q-Learning
  (p.11-12); B Overestimation Bias in Deterministic Policy Gradients (p.12); C DDPG Network and Hyper-parameter
  Comparison (p.13); D Additional Implementation Details (p.14); E Soft Actor-Critic Implementation Details (p.14);
  F Additional Learning Curves (p.15).

## 2. Core problem

Function approximation error makes value estimates biased upward (overestimation) and noisy, and TD learning
accumulates the error. The paper shows this overestimation also happens in actor-critic methods with a
deterministic policy gradient (DDPG), and that Double DQN's fix does not work there.

## 3. What prior methods did and why that hurt (per the source)

- Q-learning's greedy target maximizes a noisy estimate: "E_eps[max_a' (Q(s',a') + eps)] >= max_a' Q(s',a')"
  (Thrun and Schwartz 1993), so "even initially zero-mean error can cause value updates to result in a consistent
  overestimation bias, which is then propagated through the Bellman equation" (Sec. 4, p.3).
- Double DQN carried to actor-critic (Eq. 8): "with the slow-changing policy in actor-critic, the current and target
  networks were too similar to make an independent estimation, and offered little improvement" (Sec. 4.2, p.4).
- Double Q-learning carried to actor-critic (Eq. 9, two actors, two critics): "more effective, it does not entirely
  eliminate the overestimation" (Sec. 4.2, p.4); the critics are not independent (shared replay buffer, each uses
  the other in its target).
- DDPG's critic: Fig. 1 (p.3) shows "A very clear overestimation bias occurs from the learning procedure" (p.4).

## 4. The one key idea

Treat the critic's error as the enemy. Take the minimum of two critics in the TD target (overestimation cannot grow
through the target), update the actor and the targets less often than the critics (actor steps use a lower-variance
critic), and add clipped noise to the target action (similar actions get similar values).

## 5. Mechanism and Algorithm 1 (box transcribed, p.6)

Algorithm 1 "TD3":
```
Initialize critic networks Q_theta1, Q_theta2, and actor network pi_phi with random parameters theta1, theta2, phi
Initialize target networks theta'1 <- theta1, theta'2 <- theta2, phi' <- phi
Initialize replay buffer B
for t = 1 to T do
  Select action with exploration noise a ~ pi_phi(s) + eps,
  eps ~ N(0, sigma) and observe reward r and new state s'
  Store transition tuple (s, a, r, s') in B

  Sample mini-batch of N transitions (s, a, r, s') from B
  a~ <- pi_phi'(s') + eps,  eps ~ clip(N(0, sigma~), -c, c)
  y <- r + gamma min_{i=1,2} Q_theta'i(s', a~)
  Update critics theta_i <- argmin_theta_i N^-1 sum (y - Q_theta_i(s,a))^2
  if t mod d then
    Update phi by the deterministic policy gradient:
    grad_phi J(phi) = N^-1 sum grad_a Q_theta1(s, a)|_{a=pi_phi(s)} grad_phi pi_phi(s)
    Update target networks:
    theta'_i <- tau theta_i + (1 - tau) theta'_i
    phi' <- tau phi + (1 - tau) phi'
  end if
end for
```
Box TeX lines (verbatim):
```tex
\tilde a \leftarrow \pi_{\phi'}(s') + \e, \quad \e \sim \clip(\N(0, \tilde \s), -c, c)
y \leftarrow r + \y \min_{i=1,2} Q_{\theta'_i}(s', \tilde a)
\theta_i \leftarrow \argmin_{\theta_i} N^{-1} \sum (y - Q_{\theta_i}(s,a))^2
\nabla_{\phi} J(\phi) = N^{-1} \sum \nabla_{a} Q_{\theta_1}(s, a) |_{a=\pi_{\phi}(s)} \nabla_{\phi} \pi_\phi(s)
\theta'_i \leftarrow \tau \theta_i + (1 - \tau) \theta'_i
\phi' \leftarrow \tau \phi + (1 - \tau) \phi'
```
Notes on the box:
- Two noise scales: sigma for exploration, sigma~ for target smoothing (Sec. 6.1 gives 0.1 and 0.2).
- The box prints `\IF{$t$ mod $d$}` ("if t mod d then"), which read literally fires when t mod d != 0. The text says
  "Every d iterations, the policy is updated with respect to Q_theta1" (Sec. 6, p.6). Our code should update when
  t mod d == 0.
- Both critics regress to the same y. The actor follows Q_theta1 only; the min appears only in the target.
- Constants vs gradients: y uses the target critics theta'_i and target actor phi' plus fixed noise, so it is a
  constant for the critic regression. The actor gradient passes through Q_theta1 into pi_phi; the critic weights
  are not changed by the actor step.
- Terminal and action-range rules (App. D, p.14): y = r "if terminal s' and t < max horizon", else
  r + gamma Q_theta'(s', pi_phi'(s')); the smoothed target action is clipped to the valid action range:
  `y = r + \y Q_{\theta'}(s', \clip (\pi_{\phi'}(s') + \e, \text{min action}, \text{max action})), \e \sim \clip(\N(0, \s), -c, c).`
- Reward notation: transitions (s, a, r, s'); r is the reward after a (S&B's R_{t+1}); return
  R_t = sum_{i=t}^T gamma^{i-t} r(s_i, a_i) (Sec. 3, p.2). Eq. 12 uses r_t for the same reward.

## 6. Equations (verbatim from LaTeX; printed numbers)

Eq. 1 (Sec. 3, p.2), deterministic policy gradient (DPG):
```tex
\nabla_{\phi} J(\phi) = \E_{s \sim p_\pi} \lb \nabla_a Q^\pi(s,a)|_{a=\pi(s)} \nabla_{\phi} \pi_\phi(s) \rb.
```
Eq. 3 (p.3), target-network TD target:
```tex
y = r + \y Q_{\theta'}(s',a'), \quad a' \sim \pi_{\phi'}(s'),
```
Overestimation argument, Sec. 4.1 (p.3). Eq. 4 defines one actor step on the approximate critic and a hypothetical
step on the true value (Z_1, Z_2 normalize the gradient: Z^-1 ||E[.]|| = 1):
```tex
\begin{aligned}
\phi_\textnormal{approx} &= \phi + \frac{\alpha}{Z_1} \E_{s \sim p_\pi} \lb \nabla_\phi \pi_\phi(s) \nabla_a Q_\theta(s,a)|_{a=\pi_\phi(s)} \rb\\
\phi_\textnormal{true} &= \phi + \frac{\alpha}{Z_2} \E_{s \sim p_\pi} \lb \nabla_\phi \pi_\phi(s) \nabla_a Q^\pi(s,a)|_{a=\pi_\phi(s)} \rb,
\end{aligned}
```
For a small enough step, Eq. 5 (approximate value: pi_approx >= pi_true) and Eq. 6 (true value: pi_true >=
pi_approx) hold. Key statement (p.3): "If in expectation the value estimate is at least as large as the true value
with respect to phi_true, E[Q_theta(s, pi_true(s))] >= E[Q^pi(s, pi_true(s))], then Equations (5) and (6) imply that
if alpha < min(eps_1, eps_2), then the value estimate will be overestimated:"
```tex
\E \lb Q_\theta(s, \pi_\textnormal{approx}(s)) \rb \geq \E \lb Q^\pi(s, \pi_\textnormal{approx}(s)) \rb.   % (7)
```
Why it matters (p.3): "Although this overestimation may be minimal with each update, the presence of error raises
two concerns. Firstly, the overestimation may develop into a more significant bias over many updates if left
unchecked. Secondly, an inaccurate value estimate may lead to poor policy updates. This is particularly problematic
because a feedback loop is created, in which suboptimal actions might be highly rated by the suboptimal critic,
reinforcing the suboptimal action in the next policy update." Appendix B (p.12) covers unnormalized gradients
("guaranteed to occur under a slightly stronger condition").

Eq. 8 (Sec. 4.2, p.4), actor-critic Double DQN target (current policy, target critic):
```tex
y = r + \y Q_{\theta'}(s', \pi_\phi(s')).
```
Eq. 9 (p.4), actor-critic Double Q-learning (two actors, two critics):
```tex
\begin{aligned}
y_1 &= r + \y Q_{\theta'_2}(s',\pi_{\phi_1}(s')) \\
y_2 &= r + \y Q_{\theta'_1}(s',\pi_{\phi_2}(s')). \\
\end{aligned}
```
Eq. 10 (Sec. 4.2, p.4), Clipped Double Q-learning target, verbatim:
```tex
y_1 = r + \y \min_{i=1,2} Q_{\theta'_i}(s', \pi_{\phi_1}(s')).
```
Wording (p.4): "we propose to simply upper-bound the less biased value estimate Q_theta2 by the biased estimate
Q_theta1. This results in taking the minimum between the two estimates"; "the value target cannot introduce any
additional overestimation over using the standard Q-learning target. While this update rule may induce an
underestimation bias, this is far preferable to overestimation bias, as unlike overestimated actions, the value of
underestimated actions will not be explicitly propagated through the policy update." Implementation: "a single
actor optimized with respect to Q_theta1. We then use the same target y_2 = y_1 for Q_theta2." Secondary benefit:
the min "should provide higher value to states with lower variance estimation error" (p.4).
Gloss: r = reward; gamma = discount; theta'_1, theta'_2 = target critics; pi_phi1(s') = the (current) actor's
action at the next state; the min over the two target critics is the bootstrap value.

Eq. 11 (Sec. 5.1, p.5), residual TD error; Eq. 12 (p.5), the estimate tracks return minus discounted future errors:
```tex
Q_\theta(s, a) = r + \y \E[Q_\theta(s', a')] - \delta(s,a).   % (11)
Q_\theta(s_t, a_t) = \E_{s_i \sim p_\pi, a_i \sim \pi} \lb \sum_{i=t}^T \y^{i-t} (r_i - \delta_i) \rb.   % (12, last line)
```
Delayed policy updates (Sec. 5.2, p.5-6): "the policy network should be updated at a lower frequency than the value
network, to first minimize error before introducing a policy update"; "The modification is to only update the
policy and target networks after a fixed number of updates d to the critic." Fig. 3 (p.5): Hopper-v1 with
tau = 1, 0.1, 0.01; with a fixed policy all rates converge similarly; "when the policy is trained with the current
value estimate, the use of fast-updating target networks results in highly divergent behavior."

Target policy smoothing (Sec. 5.3, p.6): "A concern with deterministic policies is they can overfit to narrow peaks
in the value estimate." Eq. 13 (printed without gamma, sic) and its sampled form Eq. 14:
```tex
y = r + \E_{\e} \lb Q_{\theta'}(s',  \pi_{\phi'}(s') + \e) \rb,   % (13)
\begin{aligned}
y &= r + \y Q_{\theta'}(s', \pi_{\phi'}(s') + \e), \\
& \e \sim \clip(\N(0, \s), -c, c),
\end{aligned}   % (14)
```
"the added noise is clipped to keep the target close to the original action"; "reminiscent of Expected SARSA";
"The value estimate learned is with respect to a noisy policy defined by the parameter sigma."

Eq. 15 (Sec. 6, p.6), the full TD3 target (both critics regress to it):
```tex
\begin{aligned}
y &= r + \y \min_{i=1,2} Q_{\theta'_i}(s', \pi_{\phi'}(s') + \e), \\
& \e \sim \clip(\N(0, \s), -c, c). \\
\end{aligned}
```
Expanded: y = r + gamma * min(Q'_1(s', a~), Q'_2(s', a~)), a~ = pi_phi'(s') + clip(N(0, sigma), -c, c).

## 7. Config (Sec. 6.1, p.7, and Table 3, p.13)

| Setting | TD3 ("Ours") | DDPG (OpenAI baselines), Table 3 |
|---|---|---|
| Actor / critic net | 400 and 300 hidden, ReLU, actor output tanh | (DDPG arch, App. C) |
| Critic input | state and action at the first layer ("Unlike the original DDPG") | action at the 2nd layer (App. C listing) |
| Optimizer | Adam | Adam |
| Critic learning rate | 10^-3 | 10^-3 |
| Actor learning rate | 10^-3 | 10^-4 |
| Critic regularization | None | 10^-2 * ||theta||^2 |
| Target update rate tau | 5 * 10^-3 (text: tau = 0.005) | 10^-3 |
| Batch size | 100 | 64 |
| Iterations per time step | 1 | 1 |
| Discount | 0.99 | 0.99 |
| Reward scaling | 1.0 | 1.0 |
| Normalized observations | False | True |
| Gradient clipping | False | False |
| Exploration | N(0, 0.1) | OU, theta = 0.15, mu = 0, sigma = 0.2 |
| Target policy noise | N(0, 0.2), clipped to (-0.5, 0.5) | n/a |
| Policy delay d | 2 | n/a |
| Replay | "containing the entire history of the agent", sampled uniformly | |
| Warm-up | purely exploratory policy for the first 10000 steps (HalfCheetah-v1, Ant-v1) and 1000 steps (others) | |

On d (p.7): "While a larger d would result in a larger benefit with respect to accumulating errors, for fair
comparison, the critics are only trained once per time step, and training the actor for too few iterations would
cripple learning." On noise: "we used uncorrelated noise for exploration as we found noise drawn from the
Ornstein-Uhlenbeck process offered no performance benefits." The paper writes the noise as N(0, sigma) with sigma
the printed number; it does not say whether noise is scaled by the action bound.
Appendix C (p.13) prints "Our Critic Architecture" as (state dim + action dim, 400) ReLU (action dim + 400, 300)
RelU (300, 1), and the actor ending (300, 1) tanh (as printed; the second critic layer width and the actor output
width look copied from the DDPG listing).

Measurement setup (p.7-8): seven MuJoCo tasks via OpenAI Gym, original v1 tasks "with no modifications to the
environment or reward": HalfCheetah-v1, Hopper-v1, Walker2d-v1, Ant-v1, Reacher-v1, InvertedPendulum-v1,
InvertedDoublePendulum-v1. Each run 1 million time steps; evaluation every 5000 steps = average reward over 10
episodes without exploration noise; 10 random seeds (simulator and network initialization). Baselines: PPO, ACKTR,
TRPO from OpenAI baselines; SAC from the author's GitHub; "Our DDPG" = all architecture and hyperparameter changes
without the three fixes. Overestimation measurement (Fig. 1, p.3-4): average value estimate over 10000 states vs
the average discounted return over 1000 episodes from states sampled from the replay buffer.

## 8. Results

Table 1 (p.7): "Max Average Return over 10 trials of 1 million time steps"; +- is one standard deviation.

| Environment | TD3 | DDPG | Our DDPG | PPO | TRPO | ACKTR | SAC |
|---|---|---|---|---|---|---|---|
| HalfCheetah | 9636.95 +- 859.065 | 3305.60 | 8577.29 | 1795.43 | -15.57 | 1450.46 | 2347.19 |
| Hopper | 3564.07 +- 114.74 | 2020.46 | 1860.02 | 2164.70 | 2471.30 | 2428.39 | 2996.66 |
| Walker2d | 4682.82 +- 539.64 | 1843.85 | 3098.11 | 3317.69 | 2321.47 | 1216.70 | 1283.67 |
| Ant | 4372.44 +- 1000.33 | 1005.30 | 888.77 | 1083.20 | -75.85 | 1821.94 | 655.35 |
| Reacher | -3.60 +- 0.56 | -6.51 | -4.01 | -6.18 | -111.43 | -4.26 | -4.44 |
| InvPendulum | 1000.00 +- 0.00 | 1000.00 | 1000.00 | 1000.00 | 985.40 | 1000.00 | 1000.00 |
| InvDoublePendulum | 9337.47 +- 14.96 | 9355.52 | 8369.95 | 8977.94 | 205.85 | 9081.92 | 8487.15 |

Text: "TD3 matches or outperforms all other algorithms in both final performance and learning speed across all
tasks" (Sec. 6.1, p.8). Note the DDPG entry 9355.52 on InvDoublePendulum is above TD3's 9337.47 (both bolded).

Table 2 (p.8), ablation: "Average return over the last 10 evaluations over 10 trials of 1 million time steps"
(DP = delayed policy updates, TPS = target policy smoothing, CDQ = clipped double Q, AHE = architecture,
hyper-parameters and exploration):

| Method | HCheetah | Hopper | Walker2d | Ant |
|---|---|---|---|---|
| TD3 | 9532.99 | 3304.75 | 4565.24 | 4185.06 |
| DDPG | 3162.50 | 1731.94 | 1520.90 | 816.35 |
| AHE | 8401.02 | 1061.77 | 2362.13 | 564.07 |
| AHE + DP | 7588.64 | 1465.11 | 2459.53 | 896.13 |
| AHE + TPS | 9023.40 | 907.56 | 2961.36 | 872.17 |
| AHE + CDQ | 6470.20 | 1134.14 | 3979.21 | 3818.71 |
| TD3 - DP | 9590.65 | 2407.42 | 4695.50 | 3754.26 |
| TD3 - TPS | 8987.69 | 2392.59 | 4033.67 | 4155.24 |
| TD3 - CDQ | 9792.80 | 1837.32 | 2579.39 | 849.75 |
| DQ-AC | 9433.87 | 1773.71 | 3100.45 | 2445.97 |
| DDQN-AC | 10306.90 | 2155.75 | 3116.81 | 1092.18 |

Text (Sec. 6.2, p.8): "While the addition of only a single component causes insignificant improvement in most
cases, the addition of combinations performs at a much higher level. The full algorithm outperforms every other
combination in most tasks. Although the actor is trained for only half the number of iterations, the inclusion of
delayed policy update generally improves performance, while reducing training time." DQ-AC and DDQN-AC give
"insignificant improvements over TD3 - CDQ, with an exception in the Ant-v1 environment".

## 9. Limitations stated, and later fixes

- Clipped double Q "may induce an underestimation bias" (p.4); accepted because underestimates are not propagated
  through the policy update.
- The overestimation proof assumes normalized gradients and a small step (alpha < min(eps_1, eps_2)); App. B gives a
  stricter condition for unnormalized gradients. The convergence proof (App. A, Theorem 1) is for the finite-MDP,
  tabular version only.
- d trade-off (p.7, quoted above).
- The SAC baseline is an older SAC (App. E, p.14): "a Gaussian mixture policy with 4 Gaussian distributions" (single
  Gaussian on Reacher-v1), rewards multiplied by 3 for Walker2d-v1 and Ant-v1, trained "for only 1 iteration per
  time step, rather than the 4 iterations used by the results reported by the authors"; "the most recent variant
  includes our Clipped Double Q-learning in the value update and produces competitive results to TD3 on most tasks."
- Later in this group: SAC (1801.01290v2, Sec. 4.2, p.5) adopts the min of two Q-functions "as proposed by Fujimoto
  et al. (2018)".

## 10. Outline feedback

- "TD3 Eq. 10 as numbered" is correct for the clipped double-Q target, but Eq. 10 bootstraps with the current
  actor pi_phi1(s') and has no smoothing noise. The full TD3 target, used by Algorithm 1, is Eq. 15 (target actor
  pi_phi' plus clipped noise). The worked TD3 target and the self-check should use Eq. 15, or show both and say
  which is which.
- viz-dpg: in TD3 the actor climbs Q_theta1 only; the min lives in the critics' training target. A toy where the
  actor climbs min(Q1, Q2) is SAC's use (SAC takes the min in its policy gradient). Also, with a one-step toy
  (gamma = 0 or no bootstrap), the TD3 target is just r and the min has no effect. Either label the toy's "twin
  critics (min)" as the SAC-style use, or make the toy bootstrap so the min acts through the target.
- "TD3's three fixes" matches the paper (Secs. 4.2, 5.2, 5.3). Table 2 also shows the AHE changes alone lift
  HalfCheetah from 3162.50 (DDPG) to 8401.02; worth one line so the fixes are not over-credited.
- Name the box quirk ("if t mod d") in the code comments; follow the text (every d iterations).
- If the page compares TD3 and SAC numbers, do not use TD3 Table 1's SAC column as "SAC": App. E says it is an older
  SAC run with 1 gradient step per time step.
