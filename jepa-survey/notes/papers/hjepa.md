# H-JEPA (2610.06805v1)

Lens: architecture. Sources read: abs.html, src/main_release.tex (single file; numeric table cells are LaTeX
macros defined at lines 79-497 of the source, each macro carrying a comment with its data path), main_release.bbl,
paper.pdf (40 pages; figure/table/page numbers below are PDF numbers). No README in the source bundle.
Several macros (AntDepthSR*, AntCapSR*, AntLadder*) are defined but never referenced in the text; they are not
cited here.

## 1. Identity
- Title: H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning
- arXiv 2610.06805, version read v1, submitted 2026-10-05 (abs.html).
- Authors: Wancong Zhang, Basile Terver, Michael Rabbat, Yann LeCun, Randall Balestriero.
- Venue: not stated; source uses iclr2027_conference.sty (ICLR 2027 submission style; our reading).
- Repo: github.com/kevinghst/H-JEPA. Site: h-jepa.com.

## 2. Core problem
Task-agnostic JEPA world models (DINO-WM, V-JEPA 2-AC, PLDM, LeWM) "predict and plan within a single latent
space, often at a single timescale". Two costs: long-horizon rollouts compound error and grow the action search;
and one latent must serve both fine dynamics and goal matching, so a goal that is more abstract than the state
(a location, not a pose) gets a poor cost (Sec 1).

## 3. What prior methods did and why that hurt (per the paper)
- Flat JEPA world models: one latent, one stride; the level-1 goal cost on AntMaze is "nearly constant away from
  the anchor, providing little signal to guide a gradient planner until it is close" (Sec 4.2, Fig 7).
- Hierarchical latent video models (Clockwork VAE, VTA, STREAMER): reconstruct pixels, not used for control.
- Hierarchical model-based RL (Director, Puppeteer, THICK): reward-driven, task-specific.
- HWM (2604.03208, Zhang et al.): task-agnostic JEPA that plans hierarchically over several horizons, but all
  predictors share one latent space: "It decomposes the horizon but cannot match objectives in more abstract
  spaces" (Sec 1).

## 4. The one key idea
Stack action-conditioned JEPAs. Level 1 is LeWM. Each level above encodes the latent of the level below at a
coarser stride into its own latent, pools the intervening actions into a macro-action, and predicts its own next
latent. Train all levels jointly with (per level) LeWM's two-term loss; gradients from upper losses flow down
through lower encoders. Plan top-down: the top level plans to the goal, its first predicted state becomes the
subgoal for the level below, measured in the upper latent space.

## 5. Mechanism, step by step
1. Level 1 = LeWM (Sec 3 "Setup"): ViT-Tiny encoder, CLS -> MLP+BN projector -> z^(1)_t; causal transformer
   predictor with AdaLN action conditioning. A level-1 step spans 5 environment steps with the 5 actions
   concatenated (App C.1). On AntMaze a 27-dim proprio state is embedded (128-dim MLP) and concatenated to a
   256-dim pixel projection -> 384-dim latent (App C.1; Table 4).
2. Level l > 1 (Eq 1): state encoder E^(l), a two-layer MLP (hidden 384), applied to a window of w_l lower
   latents starting at lower time t * s_l; action encoder A^(l), a one-layer transformer that pools the s_l
   lower-level action embeddings between consecutive upper states into a 4-dim macro-action. All experiments use
   stride s_l = 2 and window w_l = 1, so upper encoders are pointwise and each upper step spans twice the
   environment time of the level below: level L step = 5 * 2^(L-1) env steps (our arithmetic).
3. Per-level predictor F^(l): causal transformer (dims per env in Table 4) fed a context of c_l = 3 latents and
   their macro-actions; teacher-forced one-step targets (Eq 2).
4. Per-level loss (Eq 3): L_pred^(l) + lambda_l SIGReg(Z^(l)), Z flattened over batch and time (the SIGReg test
   is grouped (T, N, D): one normality test per timestep over the batch, App K). lambda uniform across levels
   within a model (0.18 FourRoom, 0.09 elsewhere), M = 1024.
5. Training: "We minimize the sum of L_pred + lambda_l SIGReg over levels ... with M = 1024 random projections and
   no stop-gradient, moving-average targets, or pretrained weights" (App C.1). "All levels are trained end-to-end,
   with gradients from upper-level objectives propagating through lower-level encoders" (Fig 2 caption). So
   E^(1) receives gradient from every level's prediction and SIGReg losses; E^(l) from levels >= l; F^(l) from
   its own L_pred only (our reading of the graph). Each training sample is one clip giving T = 4 states at the
   top level; lower levels take a random 4-state crop of their own stream, so every level sees 3 teacher-forced
   transitions per clip (Table 3). AdamW, wd 1e-3, 1% linear warm-up + cosine, grad clip 1.0, bf16, peak LR
   5e-5 (level 1) and 1e-5 (upper levels), batch 128 clips, 3 seeds (42, 43, 44). H200 GPUs; no hour count.
6. DROID variant (Sec 4.3; App C.2): adds an inverse-dynamics loss per level (Eq 8) because with scene changes
   per episode the predictive objective "admits a slow-feature solution in which the encoder keeps the scene
   identity ... and discards the moving agent". Level-1 encoder ViT-S/16 at 256 px, predictors 384 wide / 12
   deep, s_2 = 3, lambda 0.08 / 0.16, IDM weights 100 / 50, action-embedding SIGReg 0.005 at level 2, batch 256,
   100 epochs, LR 5e-4 both levels.
7. Planning (Sec 2.2; App E): encode current and goal observation at every level. Top level: gradient descent
   (AdamW, at most 90 iterations, early stop) over S candidate macro-action sequences, horizon H_L, cost = squared
   L2 between final predicted latent and goal latent in the top space (Eq 4). Each lower level l: roll out its
   own predictor for H_l = K_{l+1} s_{l+1} + w_{l+1} - 1 steps, lift the predictions into the upper space with
   E^(l+1) (Eq 5), and minimise the squared L2 to the upper level's optimised predictions (Eq 6), terminal
   subgoal weight 1, intermediate subgoals weight beta. Closed loop uses K = 1 (first upper prediction is the only
   subgoal; with s = 2, w = 1 the lower horizon is 2). Level 1 returns primitive actions. Execute two level-1
   blocks (10 env steps), re-encode, replan from scratch. At planning time every level conditions on the latest
   state only (c_l = 1). Horizons shrink across replanning calls (App E "Replanning and horizons").
8. HWM control baseline (Sec 4.1): identical to H-JEPA except upper encoders are identity maps, so all levels
   predict and plan in the level-1 latent.

## 6. Equations
Eq 1 (label eq:level-state, Sec 2.1), level-l state and action embeddings; "The state encoder E^(l) pools the
w_l-step lower-level window into one abstract state. The action encoder A^(l) instead aggregates s_l lower-level
action embeddings":
```
z^{(\ell)}_t = E^{(\ell)}\!\left(z^{(\ell-1)}_{t\cdot s_\ell\,:\,t\cdot s_\ell+w_\ell}\right),
\qquad
a^{(\ell)}_t = A^{(\ell)}\!\left(a^{(\ell-1)}_{t\cdot s_\ell+w_\ell-1\,:\,(t+1)\cdot s_\ell+w_\ell-1}\right).
```
- a:b denotes indices a..b-1. s_l stride (how far each window advances), w_l window (how far it spans); both = 2 / 1
  in all experiments. The action slice starts at the last frame of the state window and runs s_l steps, so
  a^(l)_t covers the whole transition z^(l)_t -> z^(l)_{t+1}.

Eq 2 (label eq:pred-loss), "One-step training uses the teacher-forced latent prediction loss":
```
\mathcal{L}^{(\ell)}_{\mathrm{pred}}
= \frac{1}{c_\ell}
\sum_{\tau=1}^{c_\ell}
\left\|
F^{(\ell)}(z^{(\ell)}_{<\tau}, a^{(\ell)}_{<\tau}) - z^{(\ell)}_{\tau+1}
\right\|_2^2 .
```
- c_l context length (3 in training); F^(l) sees the true latents up to tau (teacher forcing) and predicts
  z_{tau+1}; squared L2 in the level-l latent (192 or 384 dims). Averaged over the c_l positions.

Eq 3 (label eq:level-obj):
```
\mathcal{L}^{(\ell)}
= \mathcal{L}^{(\ell)}_{\mathrm{pred}}
+ \lambda_\ell \mathrm{SIGReg}\!\left(Z^{(\ell)}\right).
```
- Z^(l) in R^{n x D}: n level-l embeddings flattened over batch and time. Total objective = sum over levels.

SIGReg (Eq 9, App K): `\mathrm{SIGReg}(Z^{(\ell)}) = \frac{1}{M}\sum_{m=1}^{M} \mathcal{T}_{\mathrm{EP}}(h_m)`,
h_m = Z u_m, u_m random unit directions; quadrature "on the half-domain [0,3] with 17 trapezoidal knots".

Eq 4 (label eq:top-plan), top-level planning:
```
a^{(L),*}_{0:H_L-1}
= \arg\min_{a^{(L)}_{0:H_L-1}}
\left\|\hat{z}^{(L)}_{H_L}-g^{(L)}\right\|_2^2.
```
- g^(L) = goal observation encoded through all L encoders; hat z from Roll^(L) (autoregressive F^(L)).

Eq 5 (label eq:lift), lifting the lower rollout into the upper space:
```
\tilde{z}^{(\ell+1)}_i
= E^{(\ell+1)}\!\left(
\hat{z}^{(\ell)}_{i s_{\ell+1}\,:\,i s_{\ell+1}+w_{\ell+1}}\right),
\qquad i=1,\ldots,K_{\ell+1}.
```
Eq 6 (label eq:lower-plan), lower-level planning; "beta >= 0 weights each intermediate subgoal relative to the
terminal target. Closed-loop planning uses K_{l+1} = 1":
```
a^{(\ell),*}_{0:H_\ell-1}
= \arg\min_{a^{(\ell)}_{0:H_\ell-1}}
\left[
\left\|\tilde{z}^{(\ell+1)}_{K_{\ell+1}}-\hat{z}^{(\ell+1),*}_{K_{\ell+1}}\right\|_2^2
+ \beta\sum_{i=1}^{K_{\ell+1}-1}
\left\|\tilde{z}^{(\ell+1)}_i-\hat{z}^{(\ell+1),*}_i\right\|_2^2
\right],
```
- hat z^(l+1),* are the upper level's optimised predictions (the subgoals); the cost lives in the upper latent.
  On DROID beta = 0.5 (Table 8).

Eq 7 (label eq:skill), DROID Frechet fidelity: `\text{fidelity} = \frac{d_F(\mathbf{0}, \tilde{p}) - d_F(p, \tilde{p})}{d_F(\mathbf{0}, \tilde{p})}`
(0% = arm does not move, 100% = expert path).

Eq 8 (label eq:idm-loss), inverse dynamics, DROID only:
```
\mathcal{L}^{(\ell)}_{\mathrm{idm}}
= \frac{1}{T-1}\sum_{t=1}^{T-1}
\frac{1}{d_a}\left\|
I^{(\ell)}\!\left(z^{(\ell)}_t,\, z^{(\ell)}_{t+1}\right)
- \operatorname{sg}\!\left[a^{(\ell)}_t\right]
\right\|_2^2 ,
```
- I^(l) IDM head; sg stop-gradient on the (pooled, at l > 1) action target so gradient reaches the encoder only
  through the two states. This is the one stop-gradient in the paper, and it is on an action target, not a latent.

## 7. Config / setup table
Simulated environments (Table 4, p22; App C.1):
| Item | FourRoom | AntMaze | Push-T | Cube |
|---|---|---|---|---|
| Frame / patch | 224 / 14 | 64 / 4 | 224 / 14 | 224 / 14 |
| Action / proprio dim | 2 / - | 8 / 27 | 2 / - | 5 / - |
| Latent d (all levels) | 192 | 384 | 192 | 192 |
| Upper predictor layers / heads | 4 / 8 | 5 / 12 | 6 / 16 | 6 / 16 |
| Upper predictor width / FF | 192 / 1024 | 224 / 1536 | 192 / 2048 | 192 / 2048 |
| SIGReg lambda, LeWM / H-JEPA | 0.72 / 0.18 | 0.09 / 0.09 | 0.09 / 0.09 | 0.09 / 0.09 |
| H-JEPA epochs | 7 | 5 | 9 | 8 |
| Data | 1.0M transitions | 10.05M (explore + stitch) | 18,685 expert ep (2.34M frames) | 9,800 ep x 200 |
Shared: stride 2, window 1, level-1 step = 5 env steps, c_l = 3, batch 128 clips, T = 4 top-level states per
clip, AdamW wd 1e-3, LR 5e-5 / 1e-5, grad clip 1.0, bf16, H200 GPUs. App C.1 text gives the level-1 predictor as
"4 layers, 8 heads, width 192, feed-forward width 1024" with upper predictors "the same structure"; this
matches only the FourRoom column of Table 4 (our reading: the text describes FourRoom).
Clip length and data per depth (Table 3, p20): level-1 frames per clip 4 / 7 / 13 / 25 and minimum clip length 16
/ 31 / 61 / 121 env steps for LeWM / 2 / 3 / 4 levels. Push-T clips per epoch fall to 86% / 58% / 14% of LeWM's;
FourRoom and AntMaze keep 96 / 88 / 73%; Cube 92 / 75 / 42%.
Planning (Table 7, p26): gradient planner (AdamW <= 90 iters); episode budgets 210 / 350 / 90 / 30 env steps;
LeWM horizon H = 23 / 23 / 15 / 4; two-level H = (2, 12) / (4, 9) / (2, 8) / (2, 2); three-level (2, 2, 6) /
(4, 2, 5) / (2, 2, 4) / (2, 2, 1); candidate counts S selected under a 100-TFLOP budget. Evaluation: 50 tasks per
env; FourRoom and Push-T goals 75 env steps apart, AntMaze grid distance 3 (82 steps avg), Cube 20-step grasp
segments. Mean +- SE over 3 paired seeds.
DROID (Table 5, p23; Table 8, p26): 74,530 episodes re-encoded at 256 px, 3 exterior cameras one at a time;
offline, open-loop plans scored by Frechet fidelity on a 16-clip set; H = 36 (level 1), 12 (level 2); S = 32
flat, (16, 16) hierarchical.

## 8. Results worth remembering
Fig 6 bottom (p6), success % vs depth, 50 tasks, mean +- SE (values from the source macros, lines 267-294):
| Env | LeWM (flat) | H-JEPA 2 | H-JEPA 3 | H-JEPA 4 | HWM 2 | HWM 3 | HWM 4 |
|---|---|---|---|---|---|---|---|
| AntMaze | 18.0 +- 3.5 | 39.3 +- 3.7 | 73.3 +- 3.5 | 63.3 +- 7.7 | 34.0 +- 1.2 | 38.7 +- 3.5 | 20.0 +- 2.3 |
| FourRoom | 40.7 +- 6.8 | 81.3 +- 6.7 | 96.0 +- 1.1 | 80.0 +- 4.0 | 75.3 +- 3.3 | 99.3 +- 0.7 | 92.7 +- 2.7 |
| Cube | 32.0 +- 6.1 | 47.3 +- 2.7 | 60.0 +- 5.0 | 55.3 +- 4.4 | 46.7 +- 2.9 | 53.3 +- 5.2 | 39.3 +- 6.4 |
| Push-T | 40.0 +- 2.3 | 45.3 +- 1.3 | 17.3 +- 3.5 | 0.7 +- 0.7 | 42.0 +- 6.4 | 12.0 +- 2.0 | 0.0 +- 0.0 |
Abstract: "on Visual AntMaze, a three-level hierarchy raises success from 18% to 73% using less planner compute".
Text: three-level H-JEPA "reaches nearly twice HWM's success" on AntMaze (73.3 vs 38.7). Push-T three-level
failure is attributed to data (58% of LeWM's transitions per epoch), not the method.
Important: this "LeWM" is H-JEPA's own retrain under a gradient planner with 75-step goals; its Push-T 40.0% is
not comparable to the 96% CEM / 25-step result in the LeWM paper.

Fig 6 top (p6): success vs planner TFLOPs per episode; on FourRoom, AntMaze and Cube "each additional level up
to three shifts the Pareto frontier up and to the left".

Table 1 (p7), AntMaze, change only the cost space (flat level-1 planning): 2-level model native L1 23.3 +- 3.3 ->
L2 projection 31.3 +- 2.7; 3-level 16.7 -> L3 proj 22.0; 4-level 10.0 -> L4 proj 26.7; hierarchical reference
39.3 / 73.3 / 63.3. Table 9 (p29): FourRoom 3-level native 43.3 -> L3 proj 71.3 (hier 96.0); Cube and Push-T show
no clear gain from projected costs.
Table 10 (p29), cost monotonicity (-Spearman vs time): AntMaze level-1 goal cost 0.48; H-JEPA level-2 0.68;
level-1 subgoal tracking 0.94. FourRoom 0.56 / 0.85 / 1.00. Cube 0.65 / 1.00 / 0.99. Push-T 0.75 / 0.77 / 0.89.
Timescale-separation evidence (Sec 3; Fig 4, p5; Fig 5; App F.2): MLP-probe NMSE of AntMaze body state and
FourRoom distractor position rises with level while agent position stays recoverable; Humanoid, AntMaze and
FourRoom have entity-frequency gaps (ratio of spectral centroids of fastest to slowest state entity) of
3.1-12.7x and show selective abstraction; Push-T, Cube and DROID have gaps of 1.2-1.6x and "show little
selective abstraction" (gaps below 2x shaded in Fig 5). Source macros: AntFreqGap 8.3, DistrFreqGap 3.1,
HumanFreqGap 12.7, PushtFreqGap 1.2, CubeFreqGap 1.6, DroidFreqGap 1.4.
Fig 9b (p9), DROID Frechet fidelity (%): LeWM without IDM ~0 (slow-feature collapse); LeWM+IDM 34; HWM 35;
H-JEPA 40 (bar labels). Fig 17 (p34): without SIGReg the effective rank collapses to "between 1 and 14 out of
384" at every IDM weight and planning falls below the do-nothing floor; with SIGReg but no IDM, rank is high but
planning still near the floor; IDM weight 100 best. Latent variance between episodes: 75% on DROID vs 28% on
Cube (Sec 4.3).
Training dynamics (App C.1, Fig 10): "Neither term degenerates at any level, so end-to-end training keeps all
levels non-collapsed at the deepest setting we train"; higher levels converge to larger prediction loss and
larger SIGReg value; "A prediction loss falling toward zero is the signature of collapse".
App G (Fig 13), AntMaze two-level, stagewise: explore at level 1 + stitch at level 2 reaches 67 vs end-to-end
union/union 54; reverse pairing 2.

## 9. Limitations the paper states
- Conclusion: gains shown in simulation and offline on DROID; "A next step is to test whether these gains
  translate to closed-loop control on a physical robot." Open: action-free video, language-aligned upper latents,
  variable-duration segments instead of fixed strides.
- Sec 4.1: deeper hierarchies need longer clips; on short-episode data (Push-T) three or more levels starve.
- Sec 3: selective abstraction appears only when the data has a frequency gap (above ~2x); manipulation
  datasets show little.
- Sec 4.3 / App J.5: SIGReg sees only the marginal law, so on diverse scenes the encoder keeps scene identity and
  drops the agent; an IDM term is needed, which needs action labels. Only one camera is encoded at a time; the
  wrist camera is unused.
- Four levels never beat three in any environment (Fig 6 bottom).

## 10. Outline feedback
- The outline's H-JEPA line is right: "hierarchy of action-conditioned JEPAs, each level its own latent and
  horizon; top-down planning; when factors evolve at separated timescales higher levels drop fast detail".
  Add the condition: the paper measures this as an entity-frequency gap and says it does not happen below ~2x
  (Push-T, Cube, DROID).
- Stacked-planes figure facts: stride 2, window 1, level-1 step = 5 env steps, so a level-l step spans
  5 * 2^(l-1) env steps; in closed loop the subgoal is the FIRST predicted upper state (K = 1), and the lower
  horizon is K * s + w - 1 = 2 steps; the subgoal distance is measured in the UPPER latent (lift the lower
  rollout through E^(l+1)), not the lower one. "What does the top level forget on purpose": AntMaze leg pose,
  FourRoom distractor position; it keeps agent xy.
- Headline numbers for the page: AntMaze 18 -> 73 (abstract wording; macros 18.0 and 73.3); FourRoom 40.7 ->
  96.0; Cube 32.0 -> 60.0; Push-T 40.0 -> 45.3 at two levels. Do not put LeWM-paper success rates (96 on Push-T)
  next to H-JEPA-paper rates; different planner, horizon and task set.
- H-JEPA plans by gradient descent through the predictor (AdamW), LeWM by CEM. If ch-lewm's loss build-up
  says "same two-term loss at every level", that is exact for the simulated envs; DROID adds a third IDM term.
- The ch-lewm intuition "with SIGReg you no longer need a pretrained encoder or EMA" holds in H-JEPA's
  simulated runs (no stop-grad, no EMA, no pretrained weights, up to four levels non-collapsed) but needs the
  DROID caveat (slow-feature collapse without IDM).
- HWM (2604.03208) is the control that isolates "separate latent per level" from "temporal decomposition";
  worth one sentence: HWM already beats flat LeWM everywhere, H-JEPA adds most on AntMaze.
