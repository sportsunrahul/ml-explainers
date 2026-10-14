# LeWorldModel / LeWM (2603.19312v3)

Lens: architecture. Sources read: abs.html, README.md, src/{neurips_2026,sections/1-intro,2-related_work,
3-method,4-exp,appendix}.tex, paper.pdf (28 pages; figure and table numbers below are PDF numbers).
Macros from math_commands.tex: `\vo \va \vz` = bold o, a, z; `\mZ` = bold Z; `\gL \gC` = calligraphic L, C;
`\enc`, `\pred` = roman "enc", "pred"; `\R` = reals. Planning results exist only as bar labels in figures,
not as a table; locations below say which panel and bar.

## 1. Identity
- Title: LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels
- arXiv 2603.19312, version read v3. v1 2026-03-13, v2 2026-03-24, v3 2026-06-03 (abs.html).
- Authors: Lucas Maes*, Quentin Le Lidec*, Damien Scieur, Yann LeCun, Randall Balestriero (* equal;
  Mila/UdeM, NYU, Samsung SAIL, Brown).
- Venue: not stated. Source uses `\usepackage[main, preprint]{neurips_2026}` (NeurIPS 2026 preprint style; our reading).
- Repo: github.com/lucas-maes/le-wm. Site: le-wm.github.io. Checkpoints + data: huggingface.co/collections/quentinll/lewm.

## 2. Core problem
Learn a latent world model (encoder + action-conditioned predictor) end to end from raw pixels, offline and
reward-free, without collapse, then plan by MPC in the latent space. Prior end-to-end JEPAs need many loss terms
and are unstable; prior stable ones freeze a pretrained encoder (Sec 1; Fig 2, p2).

## 3. What prior methods did and why that hurt (per the paper)
- I-JEPA / V-JEPA line: EMA target encoder + stop-gradient; "the theoretical understanding of EMA and SG remains
  limited, as they do not in general correspond to the minimization of a well-defined objective" (Sec 2).
- Foundation-based (DINO-WM, V-JEPA 2-AC): frozen pretrained encoder avoids collapse "but limits the expressivity
  of representation to the pretrained encoder used" (Sec 2); 196 patch tokens make planning slow (Fig 3).
- End-to-end PLDM: VICReg-style objective with seven terms and six tunable weights, "at the cost of known training
  instabilities and scalability limitations"; grid search is O(n^6) (Sec 4.3; App C.2 Table 2).
- Task-specific (Dreamer, TD-MPC): need rewards or privileged state (Fig 2 caption).

## 4. The one key idea
Two loss terms only: MSE next-embedding prediction (teacher forced) plus SIGReg (from LeJEPA), which pushes the
batch of embeddings at each timestep toward an isotropic Gaussian. No EMA, no stop-gradient, no pretrained
weights, no reconstruction; every parameter gets a gradient from the same two-term loss. One effective
hyperparameter (lambda).

## 5. Mechanism, step by step
1. Data: offline trajectories (o_{1:T}, a_{1:T}), frame-skip 5 (5 consecutive actions grouped into one action
   block), sub-trajectories of 4 frames + 4 action blocks, batch 128, frames 224 x 224 (App D).
2. Encoder: ViT-Tiny (HF implementation), patch 14, 12 layers, 3 heads, width 192, ~5M params. Take the last-layer
   [CLS] token, then a 1-layer MLP projector with BatchNorm. The projector exists because the final ViT
   LayerNorm "prevents our anti-collapse objective from being optimized effectively" (Sec 3.1). Output z_t is
   192-dim (the decoder paragraph says "single latent embedding (192 dim)"). Per frame: 224/14 = 16, so 256
   patch tokens + CLS inside the ViT; only CLS leaves (our reading).
3. Action conditioning: AdaLN (adaptive LayerNorm, DiT style) at every predictor layer, AdaLN parameters
   zero-initialised "so that action conditioning impacts the predictor training progressively" (Sec 3.1).
4. Predictor: transformer, 6 layers, 16 heads, dropout 0.1, ~10M params ("ViT-S backbone", App D), learned
   positional embeddings, temporal causal mask over a history of N frame embeddings; followed by its own
   MLP+BN projector. History N = 3 for PushT and OGBench-Cube, 1 for TwoRoom (App D). Total ~15M.
5. Loss: L_pred = MSE between predicted and encoded next embedding at every position (teacher forcing, so the
   predictor always sees true z_t, never its own output). SIGReg is applied step-wise: Z transposed to
   (T, B, d), one normality test per timestep over the batch, then averaged (Alg 1, App D). Total:
   L_pred + lambda * SIGReg, lambda = 0.1, M = 1024 projections.
6. Gradient flow: "Gradients are propagated through all components of the loss, and all parameters are
   optimized jointly in an end-to-end manner" (Sec 3.1). Encoder gets gradient from both terms (as target z_{t+1}
   and as input z_t); predictor from L_pred only. No stop-gradient anywhere.
7. Planning: encode o_1 and o_g; CEM over action sequences of horizon H = 5 (= 25 env steps), 300 samples per
   iteration, 30 elites, initial variance 1, 30 iterations on PushT and 10 elsewhere; cost = squared L2 between
   the final predicted latent and the goal latent; MPC executes the whole 5-step plan, then replans (App D;
   App B). "This configuration follows the setup used in [DINO-WM]."
8. Diagnostic decoder (never in the training loss): transformer decoder from the 192-dim CLS to 224 x 224 via
   196 learned query tokens (App D).

## 6. Equations
Model (tag LeWM, Sec 3.1):
```
\text{Encoder:} \quad & \vz_t = \enc(\vo_t) \\
\text{Predictor:} \quad & \hat{\vz}_{t+1} = \pred(\vz_t, \va_t)
```

Eq 1 (label eq:l2, Sec 3.1), "The prediction loss (teacher-forcing) computes the error between the predicted
embedding of consecutive time-steps":
```
\mathcal{L}_{\rm pred} \triangleq  \|\hat{\vz}_{t+1} - {\vz}_{t+1}\|^2_2,
    \quad\quad
    \hat{\vz}_{t+1} = {\rm pred}_\phi(\vz_t, \va_t).
```
- hat z_{t+1}: predictor output from the true z_t (teacher forced) and action block a_t; z_{t+1}: encoder output
  on the true next frame; squared L2 over the d = 192 dims. Averaged over batch and the 3 transitions per clip.

Eq 2 (Sec 3.1), SIGReg; "SIGReg encourages the latent embeddings to match an isotropic Gaussian target distribution":
```
{\rm SIGReg}(\mZ) \triangleq  \frac{1}{M} \sum_{m=1}^M T(\vh^{(m)}).
```
with projections `\vh^{(m)} = \mZ \vu^{(m)}`, `\vu^{(m)} \in \mathbb{S}^{d-1}` random unit directions,
Z in R^{N x B x d} (history length N, batch B, dim d). M = 1024.

Epps-Pulley statistic (tag EP, App A):
```
T^{(m)} = \int_{-\infty}^{\infty} w(t) \left|\phi_N(t; \vh^{(m)}) - \phi_0(t)\right|^2 dt,
```
- phi_N(t; h) = (1/N) sum_n exp(i t h_n): empirical characteristic function of the 1-D projected sample;
  phi_0: characteristic function of N(0,1); w(t) = exp(-t^2 / (2 lambda^2)) a Gaussian weight (the lambda here is
  the EP kernel width, not the loss weight; the paper reuses the symbol). Integral by trapezoid with "T nodes
  uniformly distributed in [0.2, 4]" (App A). H-JEPA (2610.06805, App K) says the released LeWM code uses the
  half-domain [0, 3] with 17 knots; note the discrepancy, do not reconcile.
- Cramer-Wold (tag Cramer-Wold, App A): `{\rm SIGReg}(\mZ)\rightarrow 0 \iff \mathbb{P}_\mZ\rightarrow N(0,\mI)`.

Eq 3 (Sec 3.1), total:
```
\gL_{\rm LeWM} \triangleq \gL_{\rm pred} + \lambda\,{\rm SIGReg}(\mZ).
```
- lambda = 0.1 default. "The method introduces only two training hyperparameters: the number of random
  projections M used in SIGReg and the regularization weight lambda ... making lambda the only effective
  hyperparameter to tune."

Planning rollout (unnumbered, Sec 3.2): `\hat{\vz}_{t+1} = \pred(\hat{\vz}_t, \va_t), \quad \hat{\vz}_1 = \enc(\vo_1)`.
Cost (Eq 4, Sec 3.2): `\gC(\hat{\vz}_H) = \| \hat{\vz}_H - \vz_g \|_2^2, \quad \vz_g = \enc(\vo_g)` (only the
final latent is scored). Eq 5 (eq:argmin_plan): `\va^*_{1:H} = \arg\min_{\va_{1:H}} \gC(\hat{\vz}_H)`.

Algorithm 1 (App D), verbatim, including the odd single-argument mse_loss call:
```
def LeWorldModel(obs,actions,lambd=0.1):
    emb = encoder(obs) # (B, T, D)
    next_emb = predictor(emb,actions) #(B, T, D)
    # next-embedding prediction loss
    pred_loss = F.mse_loss(emb[:, 1:] - next_emb[:, :-1])
    # step-wise sigreg (anti-collapse)
    sigreg_loss = mean(SIGReg(emb.transpose(0, 1))
    return pred_loss + lambd * sigreg_loss
```

Exact stability wording (Fig 1 caption): "LeWM does not rely on any training heuristics, such as stop-gradient,
exponential moving averages, or pre-trained representations." (Sec 3.1): "We do not employ stop-gradient,
exponential moving averages, or additional stabilization heuristics."
Exact "six to one" wording (abstract): "This reduces tunable loss hyperparameters from six to one compared to the
only existing end-to-end alternative." The alternative is PLDM: seven loss terms, six tunable weights
(alpha, beta, gamma, zeta, nu, mu), Table 2 (App C.2) best values 18.0, 12, 0.2, 0.7, 0.0, 0.0.
Exact compute wording (abstract): "With ~15M parameters trainable on a single GPU in a few hours". GPU type and
hours are not stated anywhere in the paper.

## 7. Config / setup table
| Item | Value | Loc |
|---|---|---|
| Input | 224 x 224 RGB frames; frame-skip 5; clips of 4 frames + 4 action blocks; batch 128 | App D |
| Encoder | ViT-Tiny, patch 14, 12 layers, 3 heads, width 192, ~5M; CLS -> 1-layer MLP + BatchNorm | Sec 3.1; App D |
| Latent dim | 192 | Sec 5.1 "Decoding Latent Space"; App D |
| Predictor | transformer 6 layers, 16 heads, dropout 0.1, ~10M ("ViT-S"), causal mask, learned pos emb, AdaLN actions (zero init), MLP+BN projector | Sec 3.1; App D |
| History N | 3 (PushT, Cube), 1 (TwoRoom); Reacher not stated | App D |
| Total params | ~15M | abstract; Sec 1 bullet 2 "15M-parameter model" |
| Loss | L_pred (MSE, teacher forced) + lambda SIGReg; lambda 0.1; M 1024 | Sec 3.1 |
| Trainable | everything (encoder, projectors, predictor, AdaLN); no EMA, no stop-grad, no pretrained weights | Sec 3.1 |
| Optimizer / LR / schedule | not stated in paper (code-only: Hydra configs under config/train/) | README |
| Epochs | 10 per environment | App E |
| Datasets | TwoRoom 10,000 ep x ~92 steps (noisy heuristic); PushT 20,000 expert ep x ~196 (DINO-WM data); OGBench-Cube 10,000 x 200 (scripted); Reacher 10,000 x 200 (SAC) | App E |
| Hardware | "a single GPU in a few hours"; type and hours not stated | abstract |
| Planner | CEM: 300 samples, 30 elites, init var 1, 30 iters (PushT) / 10 (others); H = 5 steps = 25 env steps; MPC executes all 5 then replans | App D; App B |
| Eval | budget 50 env steps; goal sampled 25 steps ahead on the same dataset trajectory; 50 trajectories (Table 5 caption) | App F.1; App G |
| Baselines | DINO-WM (no proprio unless "+prop"), PLDM (7 terms, grid 256 configs), GCBC, GCIQL, GCIVL (DINOv2 patch inputs) | Sec 4.1; App C |

## 8. Results worth remembering
Fig 6 (p7), success rate (%), bar labels, left to right in each panel:
| Env | LeWM | DINO-WM+prop | PLDM | DINO-WM | GCBC | GCIQL | GCIVL | Random |
|---|---|---|---|---|---|---|---|---|
| Two-Room | 87 | 100 | 97 | 100 | 100 | 100 | 100 | 0 |
| Reacher | 86 | - | 78 | 79 | - | - | - | 10 |
| Push-T | 96 | 92 | 78 | 74 | 75 | 20 | 33 | 2 |
| OGBench-Cube | 74 | - | 65 | 86 | 84 | 64 | 56 | 48 |
Text (Sec 4.2): "18% higher success rate on PushT" than PLDM (96 - 78). LeWM loses on Two-Room (paper blames
SIGReg's Gaussian prior vs low intrinsic dimension) and on Cube (DINO-WM 86 vs 74).

Fig 3 (p3): full planning time averaged over 50 runs: DINO-WM 47s, LeWM 0.98s (bar labels). Text: "48x faster
planning time, with the full planning completing in under one second" (Sec 4.2); caption: "~200x fewer tokens
than DINO-WM ... up to ~50x faster" (1 CLS token vs 196 patches; our reading). Fixed-FLOPs panels: Push-T
DINO-WM 13 vs LeWM 90; OGB-Cube DINO-WM 48 vs LeWM 74. Planning hardware not stated.

Table 5 (p24), Push-T SR over 3 training seeds: DINO-WM 92.0 +- 1.63; PLDM 78.0 +- 5.0; LeWM 96.0 +- 2.83.
Note: Fig 6 shows DINO-WM 74 and DINO-WM+prop 92 on Push-T; Table 5's 92.0 matches the +prop bar. Do not reconcile.
Table 6 (p24), predictor size, Push-T: tiny 80.67 +- 6.54; small 96.0 +- 2.83; base 86.7 +- 3.06.
Table 7 (p25): with decoder (reconstruction) loss 86.0 +- 7.54 vs 96.0 without.
Table 8 (p25): ResNet-18 encoder 94.0 +- 3.27 vs ViT 96.0.
Table 9 (p26), predictor dropout: 0.0 -> 78 +- 6.54; 0.1 -> 96.0; 0.2 -> 85.33; 0.5 -> 66.67.
Table 10 (p26), solver, LeWM / PLDM: CEM 96.0 / 78.0; SGD 26 / 4.67; RMSProp 67.33 / 49.33; Adam 84 / 80.
Fig 16 (p25), lambda sweep: SR above 80% for lambda in [0.01, 0.2], peak near 0.09, sharp drop at 0.5. No
lambda = 0 point is plotted.
Fig 15 (p24): embedding dim threshold "around 184"; number of projections and integration knots: no effect.
Table 1 (p8), Push-T probing, linear MSE / r: agent location LeWM 0.052 / 0.974, PLDM 0.090 / 0.955, DINO-WM
1.888 / 0.977; block angle DINO-WM 0.050 / 0.979, LeWM 0.187 / 0.902, PLDM 0.446 / 0.745.
Fig 8 (p9): surprise spike on teleport perturbation significant in all three envs (paired t-test p < 0.01);
colour change not significant.
Stability evidence as stated: Sec 3.1 "if alone, the loss in Eq. 1 leads to representation collapse, yielding a
trivial solution in which the encoder maps all inputs to a constant representation"; Fig 18 vs 19 (p27-28):
LeWM's two curves "smooth and monotonic", PLDM's seven "noisy and non-monotonic" (Sec 4.3). The paper gives no
number for a run without SIGReg; the quantitative no-SIGReg evidence is in H-JEPA (see hjepa.md, Fig 17).

## 9. Limitations the paper states
(Conclusion, p9) 1) "Planning remains restricted to short horizons, motivating hierarchical world modeling for
long-horizon reasoning" -> addressed by H-JEPA 2610.06805 (hjepa.md). 2) Needs offline data with sufficient
coverage; "low data diversity weakens SIGReg in simple, low-dimensional environments" (Two-Room). 3) Depends on
action labels; inverse dynamics suggested. Also Sec 4.2: Two-Room result is "a potential limitation of the
SIGReg regularization in very low-complexity environments".

## 10. Outline feedback
- "results vs DINO-WM-style baselines": LeWM beats DINO-WM only on Push-T (96 vs 74; 92 with proprio) and Reacher
  (86 vs 79); it loses on Two-Room (87 vs 100) and Cube (74 vs 86). Say where it wins; the honest headline is
  "competitive at ~15M params and 48x faster planning", which is also the paper's own framing.
- Planning speed: use "47s vs 0.98s, 48x" with "Fig 3" as location; hardware is not stated.
- "training cost": the only statement is "a single GPU in a few hours"; no GPU type, no hour count. Do not
  put a number on the page.
- "no EMA / no pretrained encoder with SIGReg" needs the H-JEPA caveat: on DROID (scene changes per episode),
  LeWM with SIGReg alone collapses to slow features (scene identity) and needs an inverse-dynamics term.
- LeWM plans with CEM; H-JEPA plans with gradient descent (AdamW). The chapter should not imply one planner.
- Light architecture block: everything trainable; L_pred grades every predicted position (teacher forced, 3
  transitions per 4-frame clip); SIGReg grades the batch distribution per timestep, not individual samples.
- Light block shapes: 224 px / patch 14 -> 16 x 16 = 256 patches + CLS -> 192-dim CLS -> MLP+BN -> z_t (192);
  predictor sees N <= 3 latents + action blocks (5 x action dim) via AdaLN.
- README (code-only): checkpoints for pusht, cube, tworooms, reacher on HF (quentinll/lewm-*); baseline suite
  (PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop) on Google Drive; model = JEPA(encoder ViT via
  stable-pretraining vit_hf, ARPredictor, Embedder action encoder, two MLP projectors with BatchNorm1d);
  train with `python train.py data=pusht`; no numeric defaults appear in the README itself. The README links the
  v1 PDF.
