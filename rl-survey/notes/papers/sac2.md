# SAC v2: Soft Actor-Critic Algorithms and Applications (Haarnoja et al.)

Group F. Source: `sources/arxiv-1812.05905/` (abs.html, paper.pdf 17 pages, src/root.tex, body.tex, appendix.tex,
defs.tex, preamble.tex). TeX below is verbatim from `body.tex` / `appendix.tex`; equation numbers are as printed
(checked on PDF page images). Page numbers are PDF pages (they match the printed page numbers).

Custom macros: defs.tex is identical to 1801.01290's (see sac.md for the table: `\E{A}{B}`, `\kl`, `\ent`,
`\voidarg`, `\st`, `\at`, `\stp`, `\atp`, `\policy`, `\params` = theta, `\pparams` = phi, `\reward`, `\discount`,
`\pdyn`, `\gauss`, `\opt` = `^*`, `\urv`, `\state`, `\action`, `\sT` = `{\mathbf{s}_T}`, `\aT` = `{\mathbf{a}_T}`,
`\entropy` = `\mathcal{H}`). Note `\bar\ent` renders as H-bar (the target entropy).

## 1. Identity

- Title: Soft Actor-Critic Algorithms and Applications.
- arXiv 1812.05905, version read v2 (abs.html: v1 Thu 13 Dec 2018, v2 Tue 29 Jan 2019). No venue stated (abs.html
  comment: "arXiv admin note: substantial text overlap with arXiv:1801.01290"). The outline cites "1812.05905" with
  no version; v2 is what we read.
- Authors (11): Tuomas Haarnoja (first), Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan,
  Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, Sergey Levine (last). UC Berkeley and Google Brain.
  Code: github.com/rail-berkeley/softlearning/ (Sec. 7.1, p.9).
- Sections: 1 Introduction (p.1-2); 2 Related Work (p.2-3); 3 Preliminaries: 3.1 Notation, 3.2 Maximum Entropy RL
  (p.3-4); 4 From Soft Policy Iteration to Soft Actor-Critic: 4.1 Soft Policy Iteration (p.4-5), 4.2 Soft
  Actor-Critic (p.5-6); 5 Automating Entropy Adjustment for Maximum Entropy RL (p.6-7); 6 Practical Algorithm (p.8);
  7 Experiments (p.8): 7.1 Simulated Benchmarks (p.9-10), 7.2 Quadrupedal Locomotion in the Real World (p.10),
  7.3 Dexterous Hand Manipulation (p.11); 8 Conclusion (p.11-12). Appendix A Infinite Horizon Discounted Maximum
  Entropy Objective (p.15); B Proofs (p.15-16); C Enforcing Action Bounds (p.16); D Hyperparameters (p.17).

## 2. Core problem

SAC as published in 2018 is brittle to the temperature: "Unlike in conventional reinforcement learning, where the
optimal policy is independent of scaling of the reward function, in maximum entropy reinforcement learning the
scaling factor has to be compensated by the choice a of suitable temperature, and a sub-optimal temperature can
drastically degrade performance" (Sec. 1, p.2).

## 3. What prior methods did and why that hurt (per the source)

- SAC 2018 "can suffer from brittleness to the temperature hyperparameter" (Sec. 1, p.2). Its per-task tuning is
  the reward-scale table in 1801.01290v2 (App. D, Table 2, p.13: 5, 5, 5, 5, 20, 10).
- A fixed entropy target per state is also poor: "Simply forcing the entropy to a fixed value is a poor solution,
  since the policy should be free to explore more in regions where the optimal action is uncertain, but remain more
  deterministic in states with a clear distinction between good and bad actions." (Sec. 5, p.6)
- The reward magnitude "differs not only across tasks, but it also depends on the policy, which improves over time
  during training", so a good temperature also drifts (Sec. 5, p.6).

## 4. The one key idea

Constrain the average entropy instead of fixing the temperature: maximize return subject to expected entropy >= a
target H-bar. The temperature alpha becomes the dual variable of that constraint and is learned by gradient descent
on J(alpha) (Eq. 18), with target entropy -dim(A). The final algorithm also drops the separate value network and
uses target Q-networks.

## 5. Mechanism and Algorithm 1 (box transcribed, p.8)

1. Soft policy iteration as in 2018, now with alpha written explicitly (Eqs. 2-4).
2. Soft Q-function and policy only: "We will consider a parameterized soft Q-function Q_theta(s_t, a_t) and a
   tractable policy pi_phi(a_t|s_t)" (Sec. 4.2, p.5). The value function is "implicitly parameterized through the
   soft Q-function parameters via Equation 3" (p.6).
3. Does the final algorithm drop the value network? YES. Footnote 1 (p.6), quoted: "In (Haarnoja et al., 2018c) we
   introduced an additional function approximator for the value function, but later found it to be unnecessary."
   (LaTeX: `In \cite{haarnoja2018soft} we introduced an additional function approximator for the value function,
   but later found it to be unnecessary.`)
4. Target soft Q-function: "a target soft Q-function with parameters theta-bar that are obtained as an exponentially
   moving average of the soft Q-function weights" (p.6).
5. Two soft Q-functions (Sec. 6, p.8): "We then use the minimum of the the soft Q-functions for the stochastic
   gradient in Equation 6 and policy gradient in Equation 10, as proposed by Fujimoto et al. (2018)." (sic "the the")
6. Temperature (Sec. 6, p.8): "we also learn alpha by minimizing the dual objective in Equation 17. This can be done
   by approximating dual gradient descent (Boyd & Vandenberghe, 2004)"; one truncated gradient step is used; "While
   such assumptions do not apply to the case of nonlinear function approximators such as neural networks, we found
   this approach to still work in practice."

Algorithm 1 "Soft Actor-Critic" (comments as printed):
```
Input: theta_1, theta_2, phi                                  > Initial parameters
theta-bar_1 <- theta_1, theta-bar_2 <- theta_2                 > Initialize target network weights
D <- empty set                                                 > Initialize an empty replay pool
for each iteration do
  for each environment step do
    a_t ~ pi_phi(a_t|s_t)                                      > Sample action from the policy
    s_{t+1} ~ p(s_{t+1}|s_t, a_t)                              > Sample transition from the environment
    D <- D U {(s_t, a_t, r(s_t, a_t), s_{t+1})}                > Store the transition in the replay pool
  end for
  for each gradient step do
    theta_i <- theta_i - lambda_Q grad-hat_theta_i J_Q(theta_i) for i in {1, 2}   > Update the Q-function parameters
    phi <- phi - lambda_pi grad-hat_phi J_pi(phi)                                > Update policy weights
    alpha <- alpha - lambda grad-hat_alpha J(alpha)                              > Adjust temperature
    theta-bar_i <- tau theta_i + (1 - tau) theta-bar_i for i in {1, 2}           > Update target network weights
  end for
end for
Output: theta_1, theta_2, phi                                 > Optimized parameters
```
Box TeX lines (verbatim):
```tex
\params_i \leftarrow \params_i - \lambda_Q \hat \nabla_{\params_i} J_\Q(\params_i) \text{ for } i\in\{1, 2\}
\pparams \leftarrow \pparams - \lambda_\policy \hat \nabla_\pparams J_\policy(\pparams)
\alpha \leftarrow \alpha - \lambda \hat \nabla_\alpha J(\alpha)
\bar\params_i\leftarrow \tau \params_i + (1-\tau)\bar\params_i \text{ for } i\in\{1,2\}
```
Networks in the final algorithm: two soft Q-networks (theta_1, theta_2), their two targets (theta-bar_1,
theta-bar_2), one policy (phi), and the scalar alpha. No value network, no target policy.

Constants vs gradients: in Eq. 6 the target uses the frozen Q_theta-bar and a fresh a_{t+1} from the current policy;
only Q_theta is differentiated. In Eqs. 9-10 the gradient passes through Q_theta into a_t = f_phi(eps_t; s_t); only
phi is updated by J_pi. In Eq. 18 only alpha is differentiated; log pi_t enters as data. The paper updates alpha
directly and does not say how alpha >= 0 (required of the dual variable, Eq. 13) is kept.

Reward notation: r(s_t, a_t) on the transition from s_t (S&B's R_{t+1}).

## 6. Equations (verbatim from LaTeX; printed numbers)

Eq. 1 (Sec. 3.2, p.4), objective written as the optimal policy (differs in form from 1801.01290v2's Eq. 1):
```tex
\policy\opt = \arg\max_{\policy} \sum_{t} \E{(\st, \at) \sim \rho_\policy}{\reward(\st,\at) + \alpha\ent(\policy(\voidarg|\st))},
```
"where alpha is the temperature parameter that determines the relative importance of the entropy term versus the
reward, and thus controls the stochasticity of the optimal policy" (p.4).

Eq. 2 (p.4) is the same soft backup as 1801.01290v2 Eq. 2. Eq. 3 (p.5), soft value, now with alpha:
```tex
V(\st) = \E{\at\sim\policy}{\Q(\st, \at) - \alpha\log\policy(\at|\st)}
```
Eq. 4 (p.5), soft policy improvement, now exp(Q/alpha):
```tex
\policy_\mathrm{new} = \arg\underset{\policy'\in \Pi}{\min}\kl{\policy'(\voidarg|\st)}{\frac{\exp\left(\frac{1}{\alpha}Q^{\policy_\mathrm{old}}(\st, \voidarg)\right)}{Z^{\policy_\mathrm{old}}(\st)}}.
```
Eq. 5 (Sec. 4.2, p.6), soft Q loss (no V network; V_theta-bar via Eq. 3):
```tex
J_\Q(\params) = \E{(\st, \at)\sim\mathcal{D}}{\frac{1}{2}\left(\Q_\params(\st, \at) - \left(\reward(\st, \at) + \discount \E{\stp\sim\pdyn}{V_{\bar\params}(\stp)}\right)\right)^2},
```
Eq. 6 (p.6), stochastic gradient (printed `\Q_\params(\at, \st)`, arguments swapped, sic):
```tex
\hat \nabla_\params J_Q(\params) =  \nabla_\params \Q_\params(\at, \st) \left(\Q_\params(\st, \at) - \left(r(\st,\at) + \gamma \left(Q_{\bar\params}(\stp, \atp) - \alpha \log\left(\pi_\pparams(\atp|\stp\right)\right)\right)\right).
```
Expanded target: y = r(s_t,a_t) + gamma (Q_theta-bar(s_{t+1}, a_{t+1}) - alpha log pi_phi(a_{t+1}|s_{t+1})),
a_{t+1} ~ pi_phi; with two Q-functions, the min of the two is used here (Sec. 6).

Eq. 7 (p.6), policy loss (printed `Q_\params(\st, \st)`, sic, for Q_theta(s_t, a_t)):
```tex
J_\policy(\pparams) = \E{\st\sim\mathcal{D}}{\E{\at\sim\policy_\pparams}{\alpha \log\left(\policy_\pparams(\at|\st)\right) - Q_\params(\st, \st)}}
```
Eq. 8 (p.6) reparameterization, Eq. 9 (p.6) reparameterized loss, Eq. 10 (p.6) gradient:
```tex
\at = f_\pparams(\epsilon_t; \st),   % (8)
J_\policy(\pparams) = \E{\st\sim\mathcal{D},\epsilon_t\sim\gauss}{\alpha \log \policy_\pparams(f_\pparams(\epsilon_t;\st)|\st) - Q_\params(\st, f_\pparams(\epsilon_t;\st))},   % (9)
\hat\nabla_\pparams J_\policy(\pparams) = \nabla_\pparams \alpha \log\left( \policy_\pparams(\at|\st)\right) + (\nabla_\at \alpha \log \left(\policy_\pparams(\at|\st)\right) - \nabla_\at Q(\st, \at))\nabla_\pparams f_\pparams(\epsilon_t;\st),   % (10)
```

Eq. 11 (Sec. 5, p.7), entropy-constrained RL:
```tex
\max_{\pi_{0:T}} \E{\rho_\pi}{\sum_{t=0}^T r(\st,\at)} \text{ s.t. } \E{(\st, \at)\sim\rho_\pi}{-\log(\pi_t(\at|\st))} \geq \entropy\ \ \forall t
```
"where H is a desired minimum expected entropy. Note that, for fully observed MDPs, the policy that optimizes the
expected return is deterministic, so we expect this constraint to usually be tight and do not need to impose an
upper bound on the entropy." (p.7)

Eq. 13 (p.7), dual at the last step (alpha_T is the dual variable; strong duality claimed):
```tex
\max_{\pi_T} \E{(\st, \at)\sim\rho_\pi}{r(\state_T, \action_T)} = \min_{\alpha_T \geq 0} \max_{\pi_T} \E{}{r(\state_T, \action_T) - \alpha_T \log \policy(\action_T|\state_T)} -\alpha_T \entropy,
```
Eq. 17 (p.7), optimal temperature:
```tex
\alpha_t\opt = \arg \min_{\alpha_t} \E{\at\sim \policy_t\opt}{- \alpha_t\log\pi_t\opt(\at|\st; \alpha_t) - \alpha_t \bar\ent}.
```
Eq. 18 (Sec. 6, p.8), the temperature objective actually used, verbatim:
```tex
J(\alpha)  = \E{\at\sim \policy_t}{ - \alpha\log\pi_t(\at|\st) - \alpha \bar\ent}.
```
Expanded: J(alpha) = E_{a_t ~ pi_t}[ -alpha log pi_t(a_t|s_t) - alpha H-bar ].
Gloss: alpha = temperature (dual variable); pi_t = current policy; -log pi_t(a_t|s_t) = a one-sample entropy
estimate; H-bar = target entropy. Our reading (derived, not stated in the paper): dJ/dalpha = E[-log pi] - H-bar,
so a gradient step lowers alpha when the policy's entropy is above the target and raises it when below.

Tanh squashing, Appendix C (p.16), same text as 1801.01290v2 App. C, renumbered:
```tex
\policy (\action|\state) &= \mu(\urv|\state)\left|\det \left(\frac{\mathrm{d}\action}{\mathrm{d}\urv} \right)\right|^{-1}.   % (25)
\log\policy (\action|\state) &= \log \mu(\urv|\state) - \sum_{i=1}^D\log\left(1 - \tanh^2(u_i)\right),   % (26)
```

## 7. Config: Appendix D, Table 1 "SAC Hyperparameters" (p.17), every row

| Parameter | Value |
|---|---|
| optimizer | Adam (Kingma and Ba, 2015) |
| learning rate | 3 * 10^-4 |
| discount (gamma) | 0.99 |
| replay buffer size | 10^6 |
| number of hidden layers (all networks) | 2 |
| number of hidden units per layer | 256 |
| number of samples per minibatch | 256 |
| entropy target | -dim(A) (e.g., -6 for HalfCheetah-v1) |
| nonlinearity | ReLU |
| target smoothing coefficient (tau) | 0.005 |
| target update interval | 1 |
| gradient steps | 1 |

No reward-scale table in this version (the temperature is learned). The table does not list an initial alpha or a
separate learning rate lambda for alpha.

Measurement setup (Sec. 7.1, p.9): same benchmark protocol as 2018: five seeds per algorithm, one evaluation
rollout every 1000 environment steps, mean curve with min/max shading; SAC with fixed per-task temperature (orange)
and with learned temperature (blue).

## 8. Results

- Sec. 7.1 (p.9-10), Fig. 1 (curves only): SAC "performs comparably to the baseline methods on the easier tasks and
  outperforms them on the harder tasks with a large margin"; "DDPG fails to make any progress on Ant-v1,
  Humanoid-v1, and Humanoid (rllab)"; "The results also indicate that the automatic temperature tuning scheme works
  well across all the environments, and thus effectively eliminates the need for tuning the temperature." (p.10)
- Minitaur quadruped (Sec. 7.2, p.10): "eight direct-drive actuators"; state built from "the current and past five
  observations and actions"; policy and value networks with "two hidden-layers and 256 neurons per layer"; "Our
  method successfully learns to walk from 160k environment steps, or approximately 400 episodes with maximum length
  of 500 steps, which amount to approximately 2 hours of real-world training time." Trained on flat ground, it
  walked on slopes, over wooden blocks and down stairs "without any additional learning".
- Dexterous hand (Sec. 7.3, p.11): 3-finger "dclaw" hand with 9 DoFs turning a valve, from raw RGB images (32x32
  pixels, Fig. 3 caption, p.12); image network: "two convolutional (four 3x3 filters) and max pool (3x3) layers,
  followed by two fully connected layers (256 units)". "Learning this task directly from raw RGB images requires
  300k environment interaction steps, which is the equivalent of 20 hours of training"; from the valve angle,
  "learning takes 3 hours, which is substantially faster than what has been reported earlier on the same task using
  PPO (7.4 hours)".
- Conclusion (Sec. 8, p.12): SAC "matches or exceeds the performance of state-of-the-art model-free deep RL methods,
  including the off-policy TD3 algorithm and the on-policy PPO algorithm without any environment specific
  hyperparameter tuning."

## 9. Limitations stated, and later fixes

- The dual-gradient argument needs convexity; "such assumptions do not apply to the case of nonlinear function
  approximators such as neural networks, we found this approach to still work in practice" (Sec. 6, p.8).
- The derivation is for the finite-horizon case, then approximated "for stationary policies by dropping the time
  dependencies from the policy, soft Q-function, and the temperature" (Sec. 5, p.7).
- The entropy constraint is a lower bound only; an upper bound is judged unnecessary (p.7).
- Real-world runs still needed manual resets after falls (Sec. 7.2, p.10).

## 10. Outline feedback

- The outline cites "1812.05905 (learned temperature)": add the version, v2. The temperature objective is Eq. 18
  (Sec. 6, p.8); the target entropy is -dim(A) (Table 1, p.17), e.g. -6 for HalfCheetah-v1. Both are worth printing
  in the SAC step ("learned temperature").
- viz-sacflow should draw this version: two Q-networks with two target copies, one policy, a scalar alpha, no value
  network, no target policy. Cite footnote 1 (p.6) for the dropped value network.
- Mark the two printing typos if the page shows Eqs. 6-7 (`Q(a_t, s_t)` and `Q(s_t, s_t)`); write the corrected
  forms and say they are corrected.
- The page's exercise "SAC on Pendulum-v1" will use this version's recipe (twin Q, targets, learned alpha). Pendulum
  is not in this paper's experiments, so any number the exercise prints is ours, not the paper's.
