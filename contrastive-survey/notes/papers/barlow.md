# Barlow Twins (spine)

## 1. Identity
- Title: Barlow Twins: Self-Supervised Learning via Redundancy Reduction
- arXiv 2103.03230, read v3 (2021-06-14). v1 2021-03-04.
- Authors: Jure Zbontar ... Stephane Deny (accent dropped for ASCII; abs.html spells it with an accented e).
- Venue: ICML 2021 (abs.html comment "to appear at ICML 2021").
- Repo: github.com/facebookresearch/barlowtwins (abstract footnote).
- Macro: `\AlgoName` = Barlow Twins (0_main.tex line 25).

## 2. Core problem
Joint-embedding SSL has trivial constant solutions. Existing methods avoid them with implementation tricks: many negatives,
or asymmetry (predictor, stop-gradient, momentum encoder). Barlow Twins wants a loss that avoids collapse by construction.

## 3. Prior methods and why that hurt (per the paper)
- InfoNCE methods need many negatives / large batches; SimCLR loses about 4 p.p. at batch 256 (Sec 4, "Robustness to Batch Size").
- InfoNCE's contrastive term is a non-parametric entropy estimator, "prone to the curse of dimensionality" (Sec 5.1).
- BYOL / SimSiam rely on asymmetry whose role is not fully understood (abstract, Sec 4 "Breaking Symmetry").

## 4. Key idea
Make the D x D cross-correlation matrix between the two views' embeddings (computed over the batch) equal to the identity:
diagonal = 1 (invariance), off-diagonal = 0 (redundancy reduction, i.e. decorrelated features).

## 5. Mechanism
1. Two augmented views Y^A, Y^B of a batch (BYOL augmentations: crop, resize 224, flip, color jitter, grayscale, blur, solarization; blur/solarize probs differ per view) (Sec 2.2).
2. Same network f (no asymmetry): ResNet-50 (2048) -> projector 3 linear layers of 8192, BN+ReLU after first two -> Z^A, Z^B in R^{N x 8192}.
3. Normalize each of the D columns over the batch (mean 0, std 1) (Algorithm 1).
4. C = Z_A_norm^T Z_B_norm / N, a D x D matrix (Algorithm 1).
5. Loss: sum_i (1 - C_ii)^2 + lambda sum_{i != j} C_ij^2, lambda = 0.005.
6. Gradient flows to both branches (no stop-grad).

Which positions are graded: entries of a feature-by-feature matrix, not sample pairs. Diagonal (i,i) compares feature i of view A with feature i of view B across the batch (pulled to 1).
Off-diagonal (i,j), i != j, compares feature i of view A with feature j of view B (pushed to 0). No sample-vs-sample negatives.

## 6. Equations
Eq. (1) `eq:lossBarlow`:
```
\mathcal{L_{BT}} \triangleq  \underbrace{\sum_i  (1-\mathcal{C}_{ii})^2}_\text{invariance term}  + ~~\lambda \underbrace{\sum_{i}\sum_{j \neq i} {\mathcal{C}_{ij}}^2}_\text{redundancy reduction term}
```
"lambda is a positive constant trading off the importance of the first and second terms of the loss."
C_ii: correlation of feature i between the two views; C_ij: cross-correlation of feature i (view A) and feature j (view B); lambda: off-diagonal weight.
Paper: invariance term "makes the embedding invariant to the distortions applied"; redundancy term "decorrelates the different vector components".

Eq. (2) `eq:crosscorr`:
```
\mathcal{C}_{ij} \triangleq \frac{
\sum_b z^A_{b,i} z^B_{b,j}}
{\sqrt{\sum_b {(z^A_{b,i})}^2} \sqrt{\sum_b {(z^B_{b,j})}^2}}
```
"b indexes batch samples and i,j index the vector dimension of the networks' outputs." Z^A, Z^B "assumed to be mean-centered along the batch dimension".
C is square, size = embedding dim, values in [-1, 1].

Pseudocode (Algorithm 1) equivalently: z_norm = (z - z.mean(0)) / z.std(0); c = mm(z_a_norm.T, z_b_norm) / N; loss = sum((c - I)^2) with off-diagonals times lambda.

## 7. Config
| item | value | loc |
|---|---|---|
| encoder | ResNet-50, 2048 output units | Sec 2.2 Architecture |
| projector | 3 linear layers, each 8192; BN + ReLU after first two | Sec 2.2 |
| lambda | 5e-3 (TeX `5\cdot10^{-3}`), found by search | Sec 2.2 Optimization |
| optimizer | LARS, protocol of BYOL | Sec 2.2 |
| epochs | 1000 | Sec 2.2 |
| batch | 2048 ("works well with batches as small as 256") | Sec 2.2 |
| lr | 0.2 weights, 0.0048 biases/BN, times batch/256; 10-epoch warmup; cosine, reduced by factor 1000 | Sec 2.2 |
| weight decay | 1.5e-6 (biases/BN excluded) | Sec 2.2 |
| hardware | 32 V100 GPUs, about 124 hours (BYOL reimpl., batch 4096: 113 hours) | Sec 2.2 |
| ablation setting | 300 epochs, linear top-1 71.4 / top-5 90.2 | Sec 4 intro |

## 8. Results
- 73.2 ImageNet linear top-1 (top-5 91.0), ResNet-50, 1000 ep, batch 2048, 2 crops 224, no multi-crop (Table 1, row Barlow Twins (ours)).
- Table 1 rows as printed (ResNet-50 linear): Supervised 76.5; MoCo 60.6; PIRL 63.6; SimCLR 69.3 (top-5 89.0); MoCo v2 71.1 (90.1); SimSiam 71.3; SwAV (w/o multi-crop) 71.8; BYOL 74.3 (91.6); SwAV 75.3.
- Table 2 semi-sup 1% / 10% top-1: 55.0 / 69.7 (best at 1%).
- Batch size (Fig 2): BT best 71.7 at batch 1024 (caption); "performance almost unaffected for a batch as small as 256"; SimCLR "drops about 4 p.p. for batch size 256". Batches tested: 128 to 4096. BYOL best 72.5 at 4096, SimCLR best 67.1 at 4096 (300 ep) (caption). These runs are 300-epoch ablations.
- Projector dim (Fig 4): BT "keeps improving with all output dimensionality tested"; x-axis BT points 32 to 16384; lambda fixed. SimCLR and BYOL saturate. Exact plotted values not printed.
- Loss ablations (Table 3, 300 ep): only invariance 57.3; only redundancy 0.1; feature-dim normalization 69.8; no BN in MLP 71.2; no BN + no normalization 53.4; cross-entropy with temperature 63.3.
- Asymmetry (Table 6): stop-grad 70.5; predictor 70.2; both 61.3 (baseline 71.4).
- Lambda sweep (Fig 5): x values 0.002, 0.005, 0.007, 0.010, 0.015, 0.020; plot range about 70.8 to 71.7; "not very sensitive". Values read from plot only.

## 9. Limitations stated
- Not robust to removing augmentations (like SimCLR, unlike BYOL) (Fig 3, Sec 4).
- Large projector costs memory; dimension of the cross-correlation grows quadratically (our reading; the paper says "benefits from very high-dimensional embeddings").
- Later: VICReg (2105.04906) splits the loss into variance, invariance and covariance terms, and "does not require any normalization" (VICReg src iclr2022_final.tex line 193; also intro contribution list line 171).

## 10. Outline feedback
- Outline says "Barlow Twins Eq. 1-2": correct (Eq. 1 = loss, Eq. 2 = cross-correlation).
- `viz-barlow`: 8 x 8 toy is fine; say the real matrix is 8192 x 8192. Lambda slider should span 0.002 to 0.02 (paper sweep) with default 0.005; note that in an 8-dim toy the off-diagonal count is 56, not ~67M, so the paper's lambda will look tiny; scale or explain.
- The "batch size" message: say "almost unaffected down to 256" (Fig 2), not "works at any batch".
- Barlow's Table 1 "SwAV (w/o multi-crop) 71.8" is SimSiam's reproduction (SimSiam Table 4, 800 ep), not a number SwAV printed. SwAV's own no-multi-crop number is 70.1 at 400 ep.
