# DreamerV3: Mastering Diverse Domains through World Models (Hafner et al.)

## 1. Identity
- Title: Mastering Diverse Domains through World Models. arXiv 2301.04104, version read v2 (2024-04-17); v1
  2023-01-10. PDF 40 pages; unnumbered sections (Introduction p2, Learning algorithm p3, Results p8, Related work,
  Discussion p11, Methods p17, supplementary tables to p40).
- Authors: Danijar Hafner (first), Timothy Lillicrap (last; 4 authors).

## 2. Core problem
RL algorithms need per-domain hyperparameter tuning. Build one algorithm, with one configuration, that learns
across very different domains (continuous and discrete actions, images and vectors, dense and sparse rewards).

## 3. What prior methods did and why that hurt (per the source)
- "configuring them for new application domains requires significant human expertise and experimentation"
  (abstract). Specialized algorithms per domain; PPO is robust but outperformed.
- Minecraft diamonds: "previous approaches resorted to using human expert data and domain-specific curricula"
  (Introduction, p3). VPT used "720 GPUs for 9 days" with contractor data (Related work, p11).

## 4. Key idea
Same world-model actor-critic family as Dreamer, made robust by "normalization, balancing, and transformations"
(symlog predictions, free bits, return normalization with a floor, categorical critic over exponential bins), so
one fixed configuration works across domains.

## 5. Mechanism (only what the bridge needs)
- World model = RSSM (Eq. 1, p4): sequence model h_t = f(h_{t-1}, z_{t-1}, a_{t-1}); encoder z_t ~ q(z_t | h_t, x_t);
  dynamics predictor z^_t ~ p(z^_t | h_t); reward predictor r^_t ~ p(r^_t | h_t, z_t); continue predictor
  c^_t ~ p(c^_t | h_t, z_t); decoder x^_t ~ p(x^_t | h_t, z_t). Latents are vectors of softmax (categorical)
  distributions with straight-through gradients.
- World model loss Eq. 2 (p4): beta_pred = 1, beta_dyn = 1, beta_rep = 0.1; free bits clip both KL terms below 1 nat.
- Actor and critic learn "purely from abstract trajectories of representations predicted by the world model"
  (Critic learning, p5); acting samples from the actor "without lookahead planning".
- Critic: distributional, trained on bootstrapped lambda-returns R^lambda_t = r_t + gamma c_t ((1 - lambda) v_t +
  lambda R^lambda_{t+1}), R^lambda_T = v_T (Eq. 5, p5).
- Actor (Eq. 6, p6): "We use the Reinforce estimator for both discrete and continuous actions"; loss
  -sum_t sg((R^lambda_t - v(s_t)) / max(1, S)) log pi(a_t | s_t) + eta H[pi(a_t | s_t)], with eta = 3 x 10^-4 and S
  an EMA of the 5th-95th percentile range of returns. This differs from DreamerV1, which backpropagated value
  gradients through the dynamics.

## 6. Equations
None needed for the bridge (outline: "Equations: none new"). Printed numbers if cited: Eq. 1 world model, Eq. 2
world-model loss, Eq. 4 actor/critic, Eq. 5 lambda-return and critic loss, Eq. 6 actor loss.

## 7. Config (Table 4, p21, "The same values are used across all benchmarks")
| Setting | Value |
|---|---|
| Imagination horizon H | 15 (Table 4); text says "prediction horizon T = 16" (Critic learning, p5) |
| Discount | gamma = 0.997 (text, p5); Table 4 prints "Discount horizon 1/(1-gamma) = 333" |
| Return lambda | 0.95 |
| Batch size / batch length | 16 / 64 |
| Loss weights | beta_pred 1, beta_dyn 1, beta_rep 0.1 |
| Entropy scale eta | 3 x 10^-4 |
| Model sizes studied | 6 sizes, 12M to 400M parameters (Results, Scaling properties) |
| Hardware | one Nvidia A100 GPU per agent (Results, p8) |
| Seeds | 5 per benchmark; 1 for ProcGen; 10 for BSuite; 10 for Minecraft (Methods, p19) |

## 8. Results
- Fixed hyperparameters: "We evaluate the generality of Dreamer across 8 domains---with over 150 tasks---under fixed
  hyperparameters." (Results, p8). Abstract: "outperforms specialized methods across over 150 diverse tasks, with a
  single configuration."
- The 8 domains (Results, p8-9): Atari (57 games, 200M frames), ProcGen (16 games, 50M frames), DMLab (30 tasks,
  100M frames), Atari100k (26 games, 400K frames), Proprio Control (18 tasks, 500K steps), Visual Control (20 tasks,
  1M steps), BSuite (23 environments, 468 configurations), Minecraft.
- Minecraft diamond, abstract (p1) verbatim: "Applied out of the box, Dreamer is the first algorithm to collect
  diamonds in Minecraft from scratch without human data or curricula."
- Results, Minecraft (p10): "All the Dreamer agents we trained on Minecraft discover diamonds in 100M environment
  steps." Episodes last up to 36000 steps (30 minutes); the player must discover "a sequence of 12 items from sparse
  rewards". Related work (p11): Dreamer used "1 GPU for 9 days, without human data".
- Atari: "Dreamer outperforms the powerful MuZero algorithm while using only a fraction of the computational
  resources" (Results, p8-9).

## 9. Limitations (stated) and later fixes
- Discussion (p11) names future directions only: learning from internet videos, one world model across domains.
- Sparse-reward caution relevant to GRPO (Actor learning, p6): "Normalizing rewards or returns by standard deviation
  can fail under sparse rewards where their standard deviation is near zero, drastically amplifying rewards
  regardless of their size."

## 10. Outline feedback
- For viz-mbfamily, cite DreamerV3 for "one configuration, over 150 tasks, 8 domains, diamonds in Minecraft", but
  draw the Dreamer mechanism from 1912.01603. The V3 actor uses REINFORCE, not gradients through the model, so a
  combined "Dreamer" panel must not mix the two.
- H = 15 (Table 4) vs T = 16 (text) is a source inconsistency; print H = 15 with "Table 4" if a horizon is shown.
