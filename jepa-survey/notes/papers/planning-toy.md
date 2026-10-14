# Planner toy spec (ch-planning, 3D latent planner)

Purpose: the browser toy runs a real CEM loop on a toy world (floor grid of states, draggable goal, horizon
slider, iterations button, readout of best imagined distance). This spec gives the minimal loop that is
faithful to V-JEPA 2-AC and DINO-WM, then the exact settings each paper used, so the caption can say which
numbers are the papers' and which are the toy's.

Sources: DINO-WM 2411.04983v2 Sec 3.2 + App A.5.1 (p15) + A.8 (Table 10, p16) + README; V-JEPA 2
2506.09985v1 src/main.tex (Sec "Inferring Actions by Planning", Eq `eq:energy`, Fig `fig:planning` caption,
App `app:robo_exp`); PLDM 2502.14819v4 Sec 3.3 + Tables 13/15/17; NWM 2412.03572v2 Sec 3.3 + App 7 (p12) +
App 8.2 (p14) + README.

## The minimal CEM loop (DINO-WM / V-JEPA 2-AC form)

```
inputs:  z0 = enc(o0)            current embedding (frozen encoder)
         zg = enc(og)            goal embedding (same encoder)
         p(z, a)                 one-step predictor, applied autoregressively
         T                       planning horizon (steps)
         K                       population size (samples per iteration)
         E                       elite count (top-k)
         I                       number of iterations
         dist(z, zg)             squared L2 (DINO-WM) or L1 (V-JEPA 2-AC)

init:    mu[t] = 0, sigma[t] = 1 for t in 0..T-1       (one Gaussian per action coordinate per step)
repeat I times:
    for k in 1..K:
        a_k[0..T-1] ~ Normal(mu, sigma^2)                (sample an action sequence)
        z = z0
        for t in 0..T-1:
            z = p(z, a_k[t])                             (imagine forward)
        cost_k = dist(z, zg)                             (score the final imagined state only)
    elites = the E sequences with the lowest cost
    mu, sigma = mean and std of the elites, per step and coordinate   (refit)
output:  the action sequence mu (V-JEPA 2-AC returns the Gaussian means)
execute: the first action (V-JEPA 2-AC) or the first k actions (DINO-WM); observe; replan (MPC)
```

What is fixed by the papers: Gaussian initialisation (both), score = distance of the final predicted embedding
to the goal embedding (both), refit on the top-k elites (both), replanning after executing the first action(s)
(both). What the toy must label as its own: the toy world's "predictor" (a hand-written dynamics on the floor
grid, not a learned net), the grid size, and any sample count smaller than the papers' for frame rate.

## Settings per paper

| Setting | DINO-WM (2411.04983v2) | V-JEPA 2-AC (2506.09985v1) | PLDM (2502.14819v4) | NWM (2412.03572v2) |
|---|---|---|---|---|
| Optimiser | CEM inside MPC (A.5.1) | CEM, receding horizon (Sec 3.4 / Fig planning) | MPPI, not CEM (Sec 3.3) | CEM, single iteration (App 7) |
| Samples per iteration | 100 (A.8 timing config) | 800 (App robo_exp) | 500 (Tables 13, 15, 17) | 120 (App 7) |
| Iterations | 10 (A.8); README `opt_steps=30` code-only | 10 refinement steps | n/a (MPPI reweights once per plan) | 1 |
| Elites | not stated | top 10 | n/a (softmax weights, lambda 0.005 Two-Rooms / 0.0025 others) | not stated in paper; README `--topk 5` code-only |
| Horizon | not stated in paper; README `goal_H=5` code-only | 1 ("tasks are relatively greedy") | 16 (predictor horizon H) | 8 steps x 0.25 s = 2 s |
| Init distribution | Gaussian (mean/cov not stated) | zero mean, unit variance per action coordinate per step | MPPI noise sigma 5 | mu_dy = mu_phi = 0, sigma^2 0.1; mu_dx/sigma^2_dx per dataset (RECON -0.1 / 0.02) |
| Action constraint | none stated | L1-ball radius 0.075 per action (about 13 cm end-effector) | norm <= 2.45 (env) | straight-line (one u, one phi) |
| Cost | squared L2, final state only | L1, final state only (Eq energy) | sum over t = 0..H of distance to goal, averaged over K = 5 ensemble, plus beta x discounted ensemble variance (gamma 0.9) | negative LPIPS between VAE-decoded final frame and goal image; averaged over 3 stochastic rollouts; plus indicator penalties |
| Embedding scored | DINOv2 patch features (196 x 384) | V-JEPA 2 predictor output (H x W x D feature map) | learned latent (e.g. 18 x 26 x 26 PointMaze) | pixels (decoded) |
| Execute per plan | first k actions (k hyperparameter) | first action | first action (replan every step; PointMaze every 4) | whole 2 s trajectory |
| Planning cost in time | 53.0 s per plan (CEM 100 x 10, A6000, Table 10) | not stated for V-JEPA 2-AC; Cosmos baseline with 80 samples x 10 steps x horizon 1 takes 4 min per action on an RTX 4090 | 16.0 s per 200-step episode (Table 5) | 30.3 s per trajectory (Table 8) |

Equations to show next to the toy:
- DINO-WM: `\mathcal{C} = \left\| \hat{z}_T - z_g \right\|^2` with `\hat{z}_t = p(\hat{z}_{t-1}, a_{t-1})`, `\hat{z}_0 = \text{enc}(o_0)`, `z_g = \text{enc}(o_g)` (Sec 3.2, p4).
- V-JEPA 2-AC (src/main.tex, `eq:energy`):
  `\mathcal{E}(\hat{a}_{1:T};\ z_k, s_k, z_g) \coloneqq \lVert P(\hat{a}_{1:T}; s_k, z_k) - z_g \rVert_1,`
  where P is the autoregressive rollout of the predictor from (end-effector state s_k, frame embedding z_k).

V-JEPA 2-AC CEM description, verbatim from the fig:planning caption: "we sample the action coordinates at
each point in the planning horizon from a sequence of Gaussian distributions initialized with zero mean and
unit variance. The population statistics of the top-k actions trajectories are used to update the mean and
variance of the Gaussian distributions. This process is repeated for several iterations before finally
returning the mean of the sequence of Gaussians as the selected action trajectory."

## Toy defaults (ours, for the page)
- Use the V-JEPA 2-AC loop shape (zero-mean unit-variance init, top-k refit, return the mean) with DINO-WM's
  squared-L2 cost so the readout is "best imagined distance" in the same units as the floor grid.
- Defaults that keep 60 fps in the browser: K = 64 samples, E = 8 elites, I up to 10, T from 1 to 8. Caption
  must say these are reduced from the papers' 800 / 10 / 10 / 1 (V-JEPA 2-AC) and 100 / 10 (DINO-WM).
- Horizon slider: the papers disagree (1 for V-JEPA 2-AC, 16 for PLDM); the slider makes that trade-off
  visible: longer T = larger search space (V-JEPA 2 Sec 7 limitation), shorter T = greedy.
- "Frozen features" label on the encoder box (DINO-WM, NWM tokeniser) vs "end-to-end" (PLDM) for ch-lewm.
