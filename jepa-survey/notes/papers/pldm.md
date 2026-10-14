# PLDM: Learning from Reward-Free Offline Data (2502.14819v4)

Lens: analysis (benchmark study) plus architecture for the PLDM model. Sources read: abs.html, src/main.tex
(single file, 1901 lines), paper.pdf (30 pages, for page and table numbers). Macros: `\JEPA` = `PLDM`;
`\change{x}` renders as `x` (camera-ready diff marker). `include/preamble/math_commands.tex` is absent from the
bundle, so `\gM \gS \gA \gP \gZ \gD \sR` are undefined here; standard definitions are calligraphic M, S, A, P,
Z, D and blackboard R (our reading). Appendix equations are `gather*` (unnumbered).

## 1. Identity
- Title: Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models
- arXiv 2502.14819, version read v4. v1 2025-02-20; v2 2025-07-17; v3 2025-08-10; v4 2025-10-29 (abs.html).
- Authors: Vlad Sobal*, Wancong Zhang*, Kyunghyun Cho, Randall Balestriero, Tim G. J. Rudner, Yann LeCun
  (* equal, order by coin flip). NYU, Genentech, Brown, Toronto, Meta FAIR.
- Venue: not on abs.html; main.tex loads `include/neurips/neurips_2025` with `[final]` -> NeurIPS 2025 (our reading).
- Repo: github.com/vladisai/PLDM. Site: latent-planning.github.io.

## 2. Core problem
Given only reward-free offline trajectories of varying quality, which paradigm generalises better: model-free
RL (goal-conditioned or zero-shot) or planning with a learned dynamics model? The paper builds 2 navigation
environments, 23 datasets, and stress-tests 6 methods on data quality, trajectory length, dataset size, unseen
layouts and a new task (Sec 1, p1-2).

## 3. What prior methods did and why that hurt (per the paper)
- Offline RL needs reward labels and trains one task at a time (Sec 1).
- Reward-free GCRL (GCIQL, HIQL, CRL, GCBC) and zero-shot HILP learn policies by dynamic programming over
  sampled (state, goal) pairs; when trajectories are short or random, far goals are out of distribution and
  TD learning fails to bridge them (Sec 4.4, 4.5). HILP's latent distances break on unseen layouts (App I, p25).
- Prior data studies (ExORL, OGBench) compare exploratory vs task data without isolating which data property
  matters (Sec 2).
- Reconstruction-based latents (DreamerV3, pixel reconstruction) plan poorly: 24.0 and 26.2 success vs 97.4 for
  PLDM on good Two-Rooms data; TD-MPC2 without rewards collapses to 0.0 (Table 6, p24).

## 4. The one key idea
Use the offline data only to learn a JEPA latent dynamics model (encoder + action-conditioned predictor, trained
with latent prediction plus VICReg-style anti-collapse terms, no EMA, no pretrained encoder), then plan in that
latent space at test time. The data's policy quality does not matter for learning dynamics, and the planner
transfers to new layouts and new objectives by changing only the cost.

Main argument in one sentence (abstract): "planning with a latent dynamics model proves to be a strong approach
for handling suboptimal offline data and adapting to diverse environments."

## 5. Mechanism, step by step
1. Encoder h_theta maps observation s_t to z_t (Eq 1). Two-Rooms: Impala Small CNN on a 2 x 64 x 64 image
   (channel 0 agent, channel 1 walls). Diverse PointMaze: 5-layer conv net 3x64x64 -> 16x26x26, with agent
   velocity expanded to 2 planes and concatenated -> z in R^{18x26x26}. Ant-U-Maze: 2-layer MLP on (x, y) ->
   256-dim, concatenated with raw proprioception (App D.1.3-D.1.5, p20-21).
2. Predictor f^k_theta rolls the latent forward with the action (Eq 2), K ensemble members. Two-Rooms: 2-layer
   GRU, 512 hidden, initial hidden state = h_theta(s_0), input = 2-D action; LayerNorm on encoder and predictor
   outputs. PointMaze: 3-layer conv predictor on [z; Expander2D(a)] (20 -> 32 -> 32 -> 18 channels). Ant:
   3-layer MLP 291 -> 256 -> 256 -> 283.
3. Training unrolls H = 16 steps with teacher forcing from z_0 only (predictions chain on predictions: Eq 2
   uses \hat z_{t-1}) and minimises L_sim (Eq 3) plus alpha L_var + beta L_cov + delta L_time-sim + omega L_IDM
   (App D.1.1). Variance is taken across the batch per time step and feature; the authors say they modify
   VICReg to "apply variance objective across the time dimension". Everything is trained end to end; there is
   no target network and no stop-gradient stated.
4. Parameter counts: Two-Rooms total 2218672 (encoder 1426096, predictor 793600); PointMaze 53666 (33296 +
   20370); Ant 1080615 (9120 + 1072007) (App D.1).
5. Planning (Eq 4-7): encode s_0 and s_g; roll out each ensemble member; cost = mean over K of the sum over
   t = 0..H of ||h(s_g) - f^k(\hat z^k_t, a_t)|| plus beta times a discounted ensemble-variance penalty;
   optimise with MPPI (500 samples, noise sigma 5, lambda 0.005 Two-Rooms / 0.0025 others, Tables 13, 15, 17);
   replan every step (i = 1; PointMaze uses 4). New task = new cost: flip the sign to flee a chaser (Sec 4.6).

## 6. Equations
Eq 1-2 (p4), the world model, "we specify the PLDM world model as":
```
\begin{align}
    \mathrm{Encoder:} &\quad \hat{z_0} = z_0 = h_{\theta}(s_0)
    \\
    \mathrm{Predictors:} & \quad \hat{z}_{t}^k = f_{\theta}^k(\hat{z}_{t-1}^k, a_{t-1}), \forall k \in \{1, \ldots, K\}
\end{align}
```
- h_theta: encoder; f^k_theta: k-th predictor of an ensemble of K; \hat z^k_t: its prediction at step t;
  a_{t-1}: action; the rollout starts from the encoded first frame only.

Eq 3 (p5), similarity objective, "minimizing the distance between predicted and encoded latents summed over all timesteps":
```
\mathcal{L}_{\mathrm{sim}}= \sum_{k=1}^K \sum_{t=0}^H \frac{1}{N}\sum_{b=0}^N\|\hat{Z}^k_{t,b} - Z_{t,b} \|^2_2
```
- Z, \hat Z^k in R^{H x N x D}: encoded and predicted latents; H prediction horizon (<= T), N batch, D feature dim.

Eq 4-7 (p5), planning objective (gather):
```
\begin{gather}
    \forall k \in \{1, \ldots, K\}: \
    \hat{z}_{0}^k = z_{0}^k = h_{\theta}(s_0), \
    \hat{z}_{t}^k = f_{\theta}^k(\hat{z}_{t-1}^k, a_{t-1}) \\
    C_{\text{goal}}(\mathbf{a}, s_0, s_g) = \frac{1}{K} \sum_{k=1}^{K} \sum_{t=0}^{H} \|h_\theta(s_g) - f_{\theta}^k(\hat{z}_{t}^k, a_{t}) \| \label{eq:cost} \\
    C_{\text{uncertainty}}(\mathbf{a}, s_0, s_g) = \sum_{t=0}^H \gamma^t \sum_{j=1}^d \mathrm{Var}(\{ f_{\theta_k}(s_{t}^k, a_t)_j \}_{k=1}^K) \\
    \mathbf{a}^* = \arg\min_{\mathbf{a}} \{ C_{\text{goal}}(\mathbf{a}, s_0, s_g) + \beta C_{\text{uncertainty}}(\mathbf{a}, s_0, s_g) \}
\end{gather}
```
- C_goal (Eq 5): distance from the goal embedding at every step of the horizon, averaged over the ensemble
  (norm unspecified in the source; L_sim uses squared L2). C_uncertainty (Eq 6): per-feature variance across
  the K predictors, discounted by gamma^t; "penalizes the model from choosing state-action transitions that
  deviate from the training distribution". beta: weight (0.0001 Two-Rooms, 1 Ant; gamma 0.9). Eq 7: the planner.

App D.1.1 (p20, unnumbered gather*), anti-collapse terms:
```
\mathcal{L}_{\mathrm{var}} = \frac{1}{HD} \sum^H_{t=0} \sum^D_{j=0} \mathrm{max}(0, \gamma - \sqrt{\mathrm{Var}(Z_{t,:,j}) + \epsilon} ) \\
C(Z_t) = \frac{1}{N-1}(Z_t-\bar{Z_t})^\top(Z_t-\bar{Z_t}),  \ \bar{Z} =  \frac{1}{N} \sum^N_{b=1} Z_{t,b} \\
\mathcal{L}_{\mathrm{cov}} = \frac{1}{H} \sum^{H}_{t=0} \frac{1}{D} \sum_{i \neq j} [C(Z_t)]^2_{i,j} \\
\mathcal{L}_{\mathrm{IDM}} = \sum^H_{t=0} \frac{1}{N} \sum^N_{b=0} \| a_{t,b} - \mathrm{MLP}(Z_{(t,b)}, Z_{(t+1,b)}) \|^2_2
```
```
\mathcal{L}_{\mathrm{time-sim}}= \sum_{t=0}^{H-1} \frac{1}{N}\sum_{b=0}^N\|Z_{t,b} - Z_{t+1,b} \|^2_2
```
```
\mathcal{L_{\mathrm{JEPA}}} = \mathcal{L}_{\mathrm{sim}} + \alpha \mathcal{L}_{\mathrm{var}} + \beta \mathcal{L}_{\mathrm{cov}} + \delta \mathcal{L}_{\mathrm{time-sim}} + \omega \mathcal{L}_{\mathrm{IDM}}
```
- L_var: hinge pushing each feature's std across the batch above gamma (this gamma is the VICReg margin, a
  different symbol reuse from the planner discount). L_cov: off-diagonal covariance penalty. L_IDM: inverse
  dynamics, predict a_t from (Z_t, Z_{t+1}). L_time-sim: adjacent latents stay close. Weights alpha, beta,
  delta, omega are per dataset (Tables 14, 16, 17; e.g. Two-Rooms seq 91: alpha 4.0, beta 6.9, delta 0.75,
  omega 0). Note beta is reused for L_cov weight and for the planner uncertainty weight.

## 7. Measurement setup
| Item | Value | Loc |
|---|---|---|
| Methods | GCIQL, HIQL, HILP, CRL, GCBC (OGBench / HILP code) vs PLDM | Sec 3.2, App D |
| Two-Rooms | point mass, obs 2 x 64 x 64, action 2-D displacement, norm <= 2.45, 200-step episodes; default T = 91, 3M transitions (~30,000 traj) | Sec 4.1, 4.2, Fig 3 |
| Data knobs | episode length 91/64/32/16 (3M total); dataset size down to 634 transitions; fraction of non-random data; no door-crossing (else ~35% cross) | Sec 4.3-4.5, Table 14 |
| Diverse PointMaze | Mujoco PointMaze, 4 x 4 grid, random layouts; obs 64x64x3 RGB + velocity; 2-D acceleration; 1M transitions over 5/10/20/40 layouts; test on 40 held-out layouts (1 trial each), or groups of 5 layouts by min edit distance (25 trials per group) | Sec 4.8, App C.2, Table 3 |
| Ant-U-Maze | 29-dim state, 8-dim action, 5M transitions from a noisy directional expert; traj len 25/50/100/250/500; 10 trials; start and goal ~200 steps apart | Sec 4.7, App C.3 |
| Chase task | same dynamics, cost sign flipped; success = keep distance >= 1.4 px for 100 steps | Sec 4.6 |
| Seeds | 3 (Fig 4, Table 2); 5 pooled and 10 for selected settings in the t-tests | App H |
| PLDM training | Adam, cosine schedule, batch 64 (Two-Rooms, Ant) / 128 (PointMaze), horizon 16, ensemble K = 5 (Two-Rooms, Ant) / 1 (PointMaze) | Tables 13, 15, 17 |
| PLDM planning | MPPI, 500 samples, sigma 5, lambda 0.005 / 0.0025, replan every 1 step (PointMaze 4) | Tables 13, 15, 17; App D.1.4 |
| Compute | single GPU, up to 1 day per run on an Nvidia V100; whole project 500 to 2000 GPU days | App "Discussion of Computational Costs", p30 |
| Falsifier | PLDM degrading with layout edit distance like the model-free methods, or model-free methods matching PLDM on held-out layouts, would break the headline claim | Fig 2 right, p3 |

## 8. Results worth remembering
- Table 1 (p4), the roadmap: PLDM is the only method rated good or medium in all six properties; model-free
  methods are all rated poor on "transfer to new environment layouts".
- Table 2 (p5), Two-Rooms success rate (+- standard error, 3 seeds), good data / no door-crossing data:
  CRL 89.3 +- 0.7 / 14.7 +- 4.1; GCBC 86.0 +- 2.0 / 8.4 +- 1.2; GCIQL 98.0 +- 0.9 / 99.6 +- 0.4;
  HILP 100.0 +- 0.0 / 100.0 +- 0.0; HIQL 96.4 +- 1.3 / 26.3 +- 5.6; PLDM 97.8 +- 0.7 / 34.4 +- 2.7.
  So PLDM is not the stitching winner; HILP and GCIQL are.
- Fig 4 (p6), curves only: PLDM and GCIQL reach "almost 80% success rate even with a few thousand transitions"
  (caption); PLDM, GCIQL, HILP most robust as the share of good data goes to 0; GCRL except GCIQL fails on short
  episodes.
- Fig 2 right (p3), curves only: success on held-out layouts vs number of training layouts (5, 10, 20, 40) and
  vs edit distance for 5-layout training; "PLDM generalizes best -- even when trained on just 5 maps -- while
  other methods fail" (Sec 4.8). No numeric values in text; Fig 8 (p17) shows trajectories.
- Table 4 (p19), single fixed maze: PLDM 0.990 +- 0.001; GCIQL, HIQL, HILP 1.000; CRL 0.980; GCBC 0.970.
- Sec 4.7 text + Fig 6 (p8): on Ant-U-Maze, PLDM, HIQL and HILP reach 100% success with short training
  trajectories where other methods fail.
- Sec 4.6 + Fig 5b (p8): on the chase task PLDM beats HILP (curves only).
- Uncaptioned ablation table, App D.1.2 (p20): full model 98.0 +- 1.5 (Two-Rooms) / 98.7 +- 2.8 (Diverse
  Maze); remove variance term 13.4 / 11.4; remove covariance 29.2 / 7.8; remove time-sim 71.0 / 95.6;
  remove IDM 98.0 / 75.5. Collapse terms are load-bearing.
- Table 5 (p23), seconds per 200-step episode: PLDM replan every 1: 16.0; every 4: 4.8 (0.95 normalised
  success); every 16: 2.6 (0.90); GCIQL 3.6; HIQL 4.0. Sec 3.3: PLDM ~4x slower than model-free by default.
- Table 6 (p24), good Two-Rooms data: DreamerV3 24.0 +- 6.9; Reconstruction 26.2 +- 13.9; TD-MPC2 0.0;
  TD-MPC2 + IDM 35.0; PLDM 97.4 +- 1.3.

## 9. Limitations the paper states
(App "Limitations", p30) Navigation only, no manipulation or partial observability; PLDM about 4x slower at
inference. Future work (Sec 5): manipulation and partial observability, accumulating prediction error over
long horizons, faster planning by gradients through the model or amortised planning. Per the outline, the
speed point is taken up by LeWorldModel (ch-lewm); not verified here.

## 10. Outline feedback
- Ch 10 says "CEM" planning for the three designs; PLDM uses MPPI (500 samples, sigma 5), not CEM, and its
  cost sums distance over every step of a 16-step horizon plus an ensemble-variance penalty. The toy should
  label its loop as DINO-WM / V-JEPA 2-AC style and note PLDM's variant.
- "vs model-free RL under distribution shift": the shifts are (a) unseen maze layouts measured by edit
  distance, (b) short trajectories needing stitching, (c) random-policy data, (d) small datasets, (e) missing
  door crossings, (f) a new task (chase). PLDM wins on (a), (d), (f); HILP and GCIQL win on (b) and (e).
- Headline numbers for a claim card: Table 2 (97.8 good data; 34.4 no-door vs HILP 100.0) and the Table 6
  reconstruction contrast (DreamerV3 24.0 vs PLDM 97.4). Layout-shift results exist only as curves.
- For ch-lewm labels: PLDM is "end-to-end" (encoder + predictor trained from pixels with VICReg-style terms,
  no EMA, no pretrained encoder), and it is tiny (2.2M params on Two-Rooms, 54K on PointMaze).
- PLDM encoder input is 64 x 64, not 224; the planner visual should not imply DINOv2-scale inputs for PLDM.
