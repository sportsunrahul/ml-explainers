# Dreamer: Dream to Control (Hafner et al. 2020)

## 1. Identity
- Title: Dream to Control: Learning Behaviors by Latent Imagination. arXiv 1912.01603, version read v3
  (2020-03-17); v1 2019-12-03. Published at ICLR 2020 (PDF header).
- Authors: Danijar Hafner (first), Mohammad Norouzi (last; 4 authors). PDF 20 pages.
- Sections: 1 Introduction, 2 Control with World Models, 3 Learning Behaviors by Latent Imagination,
  4 Learning Latent Dynamics, 5 Related Work, 6 Experiments, 7 Conclusion, App. A Hyper Parameters (p14).

## 2. Core problem
Given a learned latent world model, derive long-horizon behaviors efficiently. Rewards beyond a finite
imagination horizon must count, and the model's differentiability should be used.

## 3. What prior methods did and why that hurt (per the source, Sec. 1)
- Imagined rewards were maximized with a parametric policy or by online planning; "considering only rewards within a
  fixed imagination horizon results in shortsighted behaviors".
- Prior work "commonly resorts to derivative-free optimization for robustness to model errors, rather than
  leveraging analytic gradients offered by neural network dynamics".

## 4. Key idea
Learn an actor and a critic purely inside the latent model: imagine trajectories from real model states, estimate
values with lambda-returns that bootstrap from a learned value model (so rewards beyond the horizon count), and
update the actor by backpropagating analytic value gradients through the imagined latent dynamics.

## 5. Mechanism
Three processes, interleaved (Sec. 2): learn the latent dynamics model from the dataset; learn action and value
models from predicted latent trajectories; execute the action model in the world to grow the dataset.

Algorithm 1 (p3), transcribed:
- Initialize dataset D with S random seed episodes. Initialize neural network parameters theta, phi, psi randomly.
- while not converged:
  - for update step c = 1..C:
    - Dynamics learning: draw B data sequences {(a_t, o_t, r_t)}_{t=k}^{k+L} ~ D. Compute model states
      s_t ~ p_theta(s_t | s_{t-1}, a_{t-1}, o_t). Update theta using representation learning.
    - Behavior learning: imagine trajectories {(s_tau, a_tau)}_{tau=t}^{t+H} from each s_t. Predict rewards
      E(q_theta(r_tau | s_tau)) and values v_psi(s_tau). Compute value estimates V_lambda(s_tau) via Eq. 6.
      Update phi <- phi + alpha grad_phi sum_{tau=t}^{t+H} V_lambda(s_tau).
      Update psi <- psi - alpha grad_psi sum_{tau=t}^{t+H} 1/2 || v_psi(s_tau) - V_lambda(s_tau) ||^2.
  - Environment interaction: o_1 <- env.reset(). for time step t = 1..T: compute s_t ~ p_theta(s_t | s_{t-1},
    a_{t-1}, o_t) from history; compute a_t ~ q_phi(a_t | s_t) with the action model; add exploration noise to
    action; r_t, o_{t+1} <- env.step(a_t).
  - Add experience to dataset D <- D U {(o_t, a_t, r_t)_{t=1}^T}.

What the world model predicts (Sec. 2 Eq. 1; Sec. 4 Eq. 9): a representation model encodes observations and
actions into states; a transition model predicts the next state without seeing the observation; a reward model
predicts rewards from states. With the reconstruction objective an observation model q(o_t | s_t) is added, but
"the observation model is only used to provide a learning signal" (Sec. 4). Imagination never decodes images.

Gradient bookkeeping (Sec. 3, p5): value targets V_lambda are stop-gradient for the value loss ("around which we
stop the gradient as typical"); the actor gets analytic gradients through rewards, values, imagined states and
reparameterized actions; "The world model is fixed while learning behaviors."

## 6. Equations
- Macros (commands.tex): `\p(x)` prints p(x), `\q(x)` prints q(x), `\pr[q_\phi](x)` prints q_phi(x) (each with | as
  \mid); `\E{sub}{arg}` prints E_sub(arg); `\defined` is `\doteq`; `\Normal(x)` prints Normal(x).
- Eq. 1 (Sec. 2, p2), latent dynamics (labels set with \makebox):
  `Representation model: && &\p(s_t|s_{t-1},a_{t-1},o_t) \\ Transition model: && &\q(s_t|s_{t-1},a_{t-1}) \\ Reward model: && &\q(r_t|s_t).`
  Source: "We use p for distributions that generate samples in the real environment and q for their approximations
  that enable latent imagination."
- Eq. 2 (Sec. 3, p4):
  `Action model: && &a_\tau \sim\ \pr[q_\phi](a_\tau|s_\tau) \\ Value model: && &v_\psi(s_\tau) \approx \E{q(\cdot|s_\tau)}{\textstyle\sum_{\tau=t}^{t+H}\gamma^{\tau-t}r_\tau}.`
- Eq. 3 (p4): `a_\tau=\operatorname{tanh}\!\big(\mu_\phi(s_\tau)+\sigma_\phi(s_\tau)\,\epsilon\big), \quad\epsilon\sim\Normal(0,\I)`
  (reparameterized tanh-Gaussian action).
- Eq. 4-6 (p4, one alignat block): `&\mathrm{V}_{\mathrm{R}}(s_\tau)&&\defined\E[\bigg]{q_\theta,q_\phi}{\sum_{n=\tau}^{t+H} r_n}`;
  `&\mathrm{V}_{\mathrm{N}}^k(s_\tau)&&\defined\E[\bigg]{q_\theta,q_\phi}{\sum_{n=\tau}^{h-1} \gamma^{n-\tau}r_n +\gamma^{h-\tau} v_\psi(s_h)} \quad\text{with}\quad h=\min(\tau+k,t+H)`;
  `&\mathrm{V}_{\mathrm{\lambda}}(s_\tau)&&\defined (1-\lambda) \sum_{n=1}^{H-1} \lambda^{n-1} \mathrm{V}_{\mathrm{N}}^n(s_\tau) + \lambda^{H-1}\mathrm{V}_{\mathrm{N}}^H(s_\tau)`.
  Source: "Dreamer uses V_lambda, an exponentially-weighted average of the estimates for different k to balance bias
  and variance." Gloss: V_R sums imagined rewards to the horizon only; V_N^k sums k rewards then bootstraps from the
  value model; V_lambda mixes all k (the lambda-return of S&B ch. 12, on imagined data).
- Eq. 7 (actor, p5): `\max_\phi\E[\bigg]{q_\theta,q_\phi}{\sum_{\tau=t}^{t+H} \mathrm{V}_{\mathrm{\lambda}}(s_\tau)}`.
- Eq. 8 (critic, p5): `\min_\psi\E[\bigg]{q_\theta,q_\phi}{\sum_{\tau=t}^{t+H} \frac{1}{2}\Big\|v_\psi(s_\tau)-\mathrm{V}_{\mathrm{\lambda}}(s_\tau))\Big\|^2}`
  (the stray ")" is in the source).
- Notation: the POMDP in Sec. 2 draws o_t, r_t ~ p(o_t, r_t | o_{<t}, a_{<t}) (reward indexed with the next state,
  S&B style), but Algorithm 1 writes r_t, o_{t+1} <- env.step(a_t) (r_t for S&B's R_{t+1}).

## 7. Config (App. A, p14)
| Setting | Value |
|---|---|
| Imagination horizon H | 15 (discrete-action tasks: 10) |
| gamma, lambda for V_lambda | 0.99, 0.95 |
| Batch | 50 sequences of length 50 |
| Adam learning rates (world model, value, action) | 6 x 10^-4, 8 x 10^-5, 8 x 10^-5 |
| Gradient norm clip | 100 |
| KL scale beta, free nats | 1 (discrete: 0.1), 3 |
| Networks | encoder/decoder from World Models, RSSM from PlaNet, other functions 3 dense layers of 300 units, ELU |
| Latent distributions | 30-dimensional diagonal Gaussians |
| Seed episodes S | 5 random episodes |
| Train/collect | 100 training steps per 1 collected episode |
| Exploration | Normal(0, 0.3) noise on the mode action |
| Action repeat | 2 for all environments |
| Not needed (per source) | latent overshooting, entropy bonus, target networks for the value model |

## 8. Results
- Benchmark: "20 visual control tasks of the DeepMind Control Suite" (Sec. 6, p8); images 64x64x3, actions 1 to 12
  dimensions, rewards 0 to 1, episodes 1000 steps.
- Headline (Sec. 6 "Performance", p9): "With an average score of 823 across tasks after 5 x 10^6 environment steps,
  Dreamer exceeds the performance of the strong model-free D4PG agent that achieves an average of 786 within 10^8
  environment steps." Figure 6 caption (p6) adds PlaNet at 332; "Results are averages over 5 seeds."
- Compute: about 3 hours per 10^6 environment steps on one V100, vs 11 hours for PlaNet and 24 hours for D4PG
  (Sec. 6, p8).
- Long horizons (Sec. 6, p9): the value model makes Dreamer robust to the imagination horizon; it "outperforms the
  alternatives on 16 of 20 tasks, with 4 ties". The same sentence says "all 19 tasks", an inconsistency in the source.
- Representation learning (Sec. 6): pixel reconstruction beats contrastive estimation on most tasks; reward
  prediction alone "was not sufficient".

## 9. Limitations (stated) and later fixes
- Conclusion: "Future research on representation learning can likely scale latent imagination to environments of
  higher visual complexity." Reward-only representations failed in their experiments (Sec. 6).
- DreamerV3 (2301.04104) says "The DreamerV1 algorithm was limited to continuous control" (Methods) and adds fixed
  hyperparameters across 8 domains.

## 10. Outline feedback
- viz-mbfamily, Dreamer panel: model = RSSM latent transition + reward (+ image decoder used only as a training
  signal); trained in imagination = actor and critic, H = 15 steps, actor gradient flows through the model. That is
  the contrast with World Models (CMA-ES, no gradient through M) and MuZero (search, no actor network).
- The V_lambda target is the lambda-return of S&B ch. 12; a one-line link to the earlier chapter is cheap.
