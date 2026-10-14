# DINO-WM (2411.04983v2)

Lens: architecture. Sources read: abs.html, src/{abstract,introduction,method,experiments,appendix,main}.tex,
paper.pdf (21 pages, for table and page numbers), README.md. Macro: `\method` = `\textsc{DINO-WM}`.
Note: the LaTeX reuses labels `table:plan_results` and `table:unseen_envs` for two tables each, so all
locations below use the PDF table numbers.

## 1. Identity
- Title: DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning
- arXiv 2411.04983, version read v2. v1 submitted 2024-11-07, v2 2025-02-01 (abs.html).
- Authors: Gaoyue Zhou, Hengkai Pan, Yann LeCun, Lerrel Pinto (Courant Institute NYU; LeCun also Meta AI).
- Venue: not on abs.html. main.tex loads `\usepackage[accepted]{icml2025}` -> ICML 2025 (our reading).
- Repo: github.com/gaoyuezhou/dino_wm. Site: dino-wm.github.io. Data + checkpoints (PointMaze, PushT, Wall) on OSF.

## 2. Core problem
Learn a task-agnostic world model from pre-collected offline trajectories (no rewards, no demos), then at
test time reach an arbitrary goal given only an RGB goal image, by optimising an action sequence through
the model. Setting is a POMDP (O, A, p); planning is visual goal reaching (Sec 3, p3).

## 3. What prior methods did and why that hurt (per the paper)
- Online RL world models (DreamerV3, TD-MPC2, IRIS) learn latents tied to rewards, discounts or termination,
  so the model is task-specific and only accurate on the data cover of the policy being optimised (Sec 1, Sec 3.1).
- Offline world models need extra auxiliary signal: expert demos, keypoints, pretrained inverse models or dense
  rewards (Sec 1).
- Latent-space models usually tie the latent to pixel reconstruction, so the features may not carry task
  information; generative video models (AVDC) are costly and physically implausible over time (Sec 2, Sec 4.6).
- Single-vector encoders (R3M, ResNet, DINO CLS) lose spatial detail needed for manipulation (Sec 4.4).

## 4. The one key idea
Freeze a pretrained DINOv2 encoder, use its patch features as the latent state, and train only a small ViT
predictor to map (past patch features, actions) to the next patch features with a latent consistency loss.
No reconstruction anywhere in the loop; the decoder is optional and separately trained. Plan with CEM/MPC on
the squared distance between predicted and goal patch features.

## 5. Mechanism, step by step
1. Observation model: frozen DINOv2 encodes each frame o_t to z_t in R^{N x E}. Config: features of shape
   (14 x 14, 384) from input resized to 196 x 196 (A.9, p17), so N = 196 patches, E = 384. The variant name
   is never given; 384-dim with 14x14 patches at 196 px is ViT-S/14 (our reading).
2. Action encoder phi: an MLP maps the raw action to a K-dim vector (K = action emb dim 10, Table 12, p17).
   This vector is concatenated to every patch vector z^i_t, i = 1..N. Proprioception, when available, is
   concatenated to the observation latents the same way (Sec 3.1.2).
3. Transition model p_theta: a ViT with its tokenisation layer removed ("decoder-only transformer"),
   depth 6, 16 heads, MLP dim 2048, about 19M parameters (A.9). Attention is causal at the frame level: every
   patch of z_t attends to all patches of z_{t-H..t-1}; it does not attend token-by-token inside the frame as
   IRIS does. Context length H = 3 (PointMaze, Reacher, PushT, PushObj) or 1 (Wall, Rope, Granular), Table 11.
4. Training: slice trajectories into segments of H + 1 frames (frameskip 5, or 1 for Rope/Granular) and use
   teacher forcing; the loss is Eq 1 on each of the H predicted frames. Gradients reach p_theta (lr 5e-5) and
   phi (lr 5e-4). DINOv2 gets none. AdamW, 100 epochs, batch 32 (Table 12).
5. Optional decoder q_theta (stack of transposed convolutions) trained with Eq 2 on its own; its loss is not
   backpropagated into the predictor (doing so drops PushT success from 0.92 to 0.80, Table 7, p14).
6. Planning: encode o_0 and o_g; CEM samples N action sequences of length T from a Gaussian, rolls each out
   through p autoregressively in latent space, scores by the cost C below, refits mean and covariance on the
   top K, repeats; then the first k actions are executed and the loop replans (MPC) (A.5.1, p15). Gradient
   descent on the same cost is possible but loses (A.5.2, Table 8).

## 6. Equations
Eq 1 (p4), latent consistency loss, "compute a latent consistency loss on each of the H predicted frames":
```
\mathcal{L}_{pred} = \left\| p_{\theta}\left( \text{enc}_{\theta}(o_{t-H:t}), \phi(a_{t-H:t}) \right) - \text{enc}_{\theta} \left( o_{t+1} \right) \right\|^2
```
- p_theta: the ViT predictor. enc_theta: frozen DINOv2 (theta is used loosely for all module parameters).
- o_{t-H:t}: the H+1 context frames; a_{t-H:t}: their actions; phi: the action MLP.
- enc_theta(o_{t+1}): the target, DINOv2 patch features of the true next frame. Squared L2 over all N x E entries.

Eq 2 (p4), decoder loss, trained independently of the predictor:
```
\mathcal{L}_{rec} = \left\| q_{\theta}(z_t) - o_t \right\|^2, \quad \text{where} \quad z_t = \text{enc}_{\theta}(o_t)
```
- q_theta: transposed-conv decoder; o_t: pixels. Only interpretability; never used for planning.

Planning cost (unnumbered display, Sec 3.2, p4; repeated as Eq 4 and Eq 7 in A.5, p15), "MSE between the
current latent state and the goal's latent state":
```
\mathcal{C} = \left\| \hat{z}_T - z_g \right\|^2, \quad \text{where} \quad
\begin{aligned}
\hat{z}_t &= p(\hat{z}_{t-1}, a_{t-1}), \\
\hat{z}_0 &= \text{enc}(o_0), \\
z_g &= \text{enc}(o_g).
\end{aligned}
```
- Only the final imagined state z_T is scored; intermediate states are not.

Eq 5 (p15), rollout inside CEM: `\hat{z}_t = p(\hat{z}_{t-1}, a_{t-1}), \quad t = 1, \ldots, T.`
Eq 8 (p15), gradient-descent variant: `a_{t} \leftarrow a_{t} - \eta \frac{\partial \mathcal{C}}{\partial a_{t}}, \quad t = 0, \ldots, T-1,` with eta the learning rate.

Transition model, as stated (align*, Sec 3.1, p3): `z_{t+1} \sim p_\theta(z_{t+1} \mid z_{t-H:t}, a_{t-H:t})`.

## 7. Config / setup table
| Item | Value | Loc |
|---|---|---|
| Observation | RGB (224, 224) | Sec 4.1, p5; Table 12 "Image size 224" |
| Encoder input | resized to 196 x 196 -> (14 x 14, 384) features | A.9 text, p17 |
| Encoder | DINOv2, frozen in training and test | Sec 3.1.1 |
| Predictor | ViT, depth 6, 16 heads, MLP dim 2048, ~19M params, frame-level causal mask | A.9, p17; Sec 3.1.2 |
| Action embedding | MLP -> 10 dims, concatenated to each patch | Sec 3.1.2; Table 12 |
| History H | 3 (Maze, Reach, PushT, PushObj); 1 (Wall, WallRandom, Rope, Granular) | Table 11, p17 |
| Frameskip | 5 (all except Rope, Granular = 1) | Table 11 |
| Trainable | predictor (lr 5e-5), action encoder (lr 5e-4); decoder (lr 3e-4) on its own loss | Table 12 |
| Optimiser | AdamW, 100 epochs, batch 32 | Table 12 |
| Datasets | Maze 2000 traj x 100; Reach 3000 x 100; PushT 18500 x 100-300; PushObj 20000 x 100; Wall 1920 x 50; WallRandom 10240 x 50; Rope 1000 x 5; Granular 1000 x 5 | Table 11 |
| Planner | CEM inside MPC; cost = squared L2 to goal patch features | Sec 3.2; A.5.1 |
| CEM settings (paper) | 100 samples per iteration, 10 optimisation steps (used for timing) | A.8 text + Table 10, p16 |
| CEM settings (code-only) | `planner=cem goal_H=5 planner.opt_steps=30 n_evals=5` | README "Plan with a DINO-WM" |
| Top-K elites | not stated in paper or README | - |
| Training compute | not stated | - |
| Inference HW | NVIDIA A6000: 0.014 s per step (batch 32); sim rollout 3.0 s; planning 53.0 s (CEM 100x10) | Table 10, p16 |
| Eval protocol | 50 start/goal pairs for Maze, Wall, Reach, PushT (success rate); 10 for Rope, Granular (Chamfer distance) | Sec 4.3, p5 |

## 8. Results worth remembering
Table 1 (p5), planning results, MPC with CEM, trained offline with no reward:
| Model | Maze SR | Wall SR | Reach SR | PushT SR | Rope CD | Granular CD |
|---|---|---|---|---|---|---|
| IRIS | 0.74 | 0.04 | 0.18 | 0.32 | 1.11 | 0.37 |
| DreamerV3 | 1.00 | 1.00 | 0.64 | 0.30 | 2.49 | 1.05 |
| TD-MPC2 | 0.00 | 0.00 | 0.00 | 0.00 | 2.52 | 1.21 |
| DINO-WM | 0.98 | 0.96 | 0.92 | 0.90 | 0.41 | 0.26 |

Table 2 (p5), same predictor, different frozen encoders ("frozen features" argument):
| Encoder | Maze | Wall | Reach | PushT | Rope | Granular |
|---|---|---|---|---|---|---|
| R3M | 0.94 | 0.34 | 0.40 | 0.42 | 1.13 | 0.95 |
| ResNet (ImageNet) | 0.98 | 0.12 | 0.06 | 0.20 | 1.08 | 0.90 |
| DINO CLS | 0.96 | 0.58 | 0.60 | 0.44 | 0.84 | 0.79 |
| DINO patch (ours) | 0.98 | 0.96 | 0.92 | 0.90 | 0.41 | 0.26 |
Paper's reading: single-vector encoders lose spatial detail; patch features keep it (Sec 4.4).

Table 3 (p7), unseen configurations: WallRandom SR 0.82 (DreamerV3 0.76, DINO CLS 0.64, R3M 0.40, ResNet 0.40,
IRIS 0.06); PushObj SR 0.34 (DreamerV3 0.18, DINO CLS 0.18); GranularRandom CD 0.63 (IRIS 0.86, ResNet 0.98).
Table 4 (p7), LPIPS of decoded predictions: PushT 0.007 vs DINO CLS 0.039, AVDC 0.046; Wall 0.0016; Rope 0.009; Granular 0.035.
Intro bullets (p2): LPIPS improves on prior SOTA by 56%; goal-reaching on hardest tasks improves by 45% on average.
Table 5 (p13), PushT data scaling: n=200 SR 0.08; 1000: 0.48; 5000: 0.72; 10000: 0.88; 18500: 0.92.
Table 6 (p13), causal mask on PushT: h=1 0.76/0.76; h=2 0.88 with mask vs 0.36 without; h=3 0.92 vs 0.08.
Table 7 (p14): decoder loss into predictor 0.80 vs 0.92 without.
Table 8 (p15), planner ablation (open-loop CEM / open-loop GD / MPC): PointMaze 0.8 / 0.22 / 0.98;
Push-T 0.86 / 0.28 / 0.90; Wall 0.74 / NA / 0.96; Rope NA / NA / 0.41; Granular NA / NA / 0.26.
Note Table 1 reports the MPC row; Table 5 row n=18500 reports 0.92 for PushT where Table 1 says 0.90 (different
runs; do not reconcile).

## 9. Limitations the paper states
(Sec 5, p8) 1) Needs offline data with sufficient state-action coverage. 2) Needs ground-truth actions, so cannot
use raw internet video. 3) Plans in raw action space; a hierarchy with high-level planning is left for future work.
Per the outline: PLDM (2502.14819) drops the pretrained encoder and trains encoder + predictor with anti-collapse
terms; LeWorldModel (2603.19312, ch-lewm) trains from pixels with no pretrained encoder; H-JEPA (2610.06805)
adds the hierarchy. (Not verified in this extraction; from outline.md only.)

## 10. Outline feedback
- Ch 10 says "CEM/gradient planning". The paper's results use CEM inside MPC; gradient descent is an appendix
  ablation that loses badly (Table 8: Maze 0.22 vs 0.8 open-loop CEM). Say "CEM + MPC" and mention GD only as a
  failed alternative.
- "zero-shot goal reaching on PushT, point maze, etc.": the six suites are Maze, Wall, Reach, PushT, Rope,
  Granular; Rope and Granular are scored by Chamfer distance (lower is better), not success rate. Claim cards
  must not present 0.41 / 0.26 as success rates.
- The latent planner toy should score only the final imagined state (squared L2 to the goal patch features), to
  match DINO-WM; see planning-toy.md.
- Predictor size for the page: ~19M (ViT depth 6). Encoder: DINOv2, 384-dim patches, 14 x 14 grid.
- For the ch-lewm "frozen features vs end-to-end" labels: DINO-WM is frozen DINOv2 + trained ~19M predictor.
- Top-K elite count and horizon for CEM are not in the paper; README gives `goal_H=5`, `opt_steps=30` (code-only).
