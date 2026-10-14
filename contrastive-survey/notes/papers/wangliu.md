# Wang & Liu 2021 -- understanding the behaviour of contrastive loss

## 1. Identity
- Title: "Understanding the Behaviour of Contrastive Loss".
- arXiv 2012.09740, read v2 (2021-03-20). v1 published 2020-12-15 (abs.html).
- Authors: Feng Wang, Huaping Liu (Tsinghua). Venue: CVPR 2021 (abs.html comment "Accepted to CVPR2021").
- Repo: none given in the source.

## 2. Core problem
How does temperature tau change what the contrastive loss learns? The paper links tau to how hard the loss pushes the nearest negatives, and to a trade-off between uniformity and keeping similar images close.

## 3. Prior methods and why that hurt (per paper)
Instance discrimination pushes all other instances away, "ignoring the underlying relations between samples". With small tau the nearest negatives, often the same class, get the biggest push, which "may break the underlying semantic structure" (abstract).

## 4. Key idea
The softmax contrastive loss is hardness-aware: the gradient on each negative is proportional to exp(s_ij/tau), so tau sets how concentrated the push is on the nearest negatives. Small tau -> more uniform but less tolerant; large tau -> tolerant but less uniform.

## 5. Mechanism
1. Write loss with similarities s_ij = f(x_i)^T g(x_j) (g = f, memory bank or queue).
2. Gradient (Eq. 4): positive gets -(1/tau) sum_k P_ik; each negative gets (1/tau) P_ij.
3. Relative penalty r_i(s_ij) (Eq. 5) is a Boltzmann distribution over negatives; its entropy decreases strictly as tau decreases (proof in supp).
4. Limits: tau -> 0+ gives a margin-0 triplet loss on the hardest negative (Eq. 6); tau -> inf gives the "simple" linear loss (Eq. 7).
5. Explicit hard negative sampling (Eq. 8-9): keep only negatives above the upper alpha-quantile of similarity.
6. Measure uniformity (Eq. 10, from Wang & Isola) and tolerance (Eq. 11) across a tau sweep.

## 6. Equations (exact TeX from src/CVPR2021_8484_FINAL.tex; numbers verified in PDF)
E1. Eq. (1), `equation: conloss`, Sec. 3:
```
\mathcal{L}(x_i) = - {\rm log}  \left[ \frac{ {\rm exp} (s_{i,i}/\tau)}{\sum_{k \neq i} {\rm exp} (s_{i,k}/\tau) + {\rm exp} (s_{i,i}/\tau)} \right]
```
"where $s_{i,j} = f(x_i)^Tg(x_j)$". Gloss: s_ii positive similarity (two views of x_i); s_ik for k != i negatives.

E2. Eq. (2): `P_{i,j} = \frac{ {\rm exp} (s_{i,j}/\tau) }{ \sum_{k \neq i} {\rm exp} (s_{i,k}/\tau) + {\rm exp}(s_{i,i}/\tau) }` -- "the probability of x_i being recognized as x_j".

E3. Eq. (3), `eq: lsimple`: `\mathcal{L}_{simple}(x_i) = -s_{i,i}+ \lambda \sum_{i \neq j}s_{i,j}`.

E4. Eq. (4), `form:gradient`, Sec. 3.1 "Gradients Analysis" -- THE GRADIENT THE PAGE NEEDS:
```
\frac{\partial \mathcal{L}(x_i)}{\partial s_{i,i}} = - \frac{1}{\tau} \sum_{k \neq i} P_{i,k}, \quad \frac{\partial \mathcal{L}(x_i)}{\partial s_{i,j}} = \frac{1}{\tau} P_{i,j}
```
Paper: "The gradients with respect to negative samples is proportional to the exponential term exp(s_ij/tau), indicating that the contrastive loss is a hardness-aware loss function"; "The magnitude of gradient with respect to positive sample is equal to the sum of gradients with respect to all negative samples". This is the printed gradient-of-InfoNCE wrt similarities (no derivation shown; one line).

E5. Eq. (5), Sec. 3.2 "The Role of temperature":
```
r_{i}(s_{i,j}) = \frac{{\rm exp}(s_{i,j}/\tau)}{\sum_{k \neq i} {\rm exp}(s_{i,k}/\tau)}, \quad i \neq j
```
"which obeys the Boltzman distribution". r_i = |dL/ds_ij| / |dL/ds_ii|, the relative penalty (share of the push) on negative j. Fig. 3: r vs s with s sampled uniform in [-1,1] for several tau.

E6. Eq. (6): tau -> 0+ limit `= \lim\limits_{\tau \to 0^+} \frac{1}{\tau}max[s_{max}-s_{i,i}, 0]` -- "a triplet loss with the margin of 0, which only focuses on the nearest negative sample".
E7. Eq. (7): tau -> +inf limit `-\frac{N-1}{N\tau} s_{i,i} + \frac{1}{N\tau} \sum_{k \neq i} s_{i,k} + {\rm log}N` -- equals L_simple.

E8. Eq. (8)-(9), Sec. 3.3 "Explicit Hard Negative Sampling". Upper alpha quantile s_alpha^(i); r_i = 0 for s_ij < s_alpha^(i). Hard loss Eq. (9), `eq: hardloss`:
```
\mathcal{L}_{{\rm hard}}(x_i) = - {\rm log}  \frac{ {\rm exp} (s_{i,i}/\tau)} {\sum_{s_{i,k} \geqslant s_{\alpha}^{(i)} } {\rm exp} (s_{i,k}/\tau) + {\rm exp} (s_{i,i}/\tau)}
```
Paper notes LocalAggregation (Zhuang et al.) used this mining before.

E9. Eq. (10) uniformity = Wang & Isola L_uniform; Eq. (11) tolerance:
```
T = \mathop{\mathbb{E}}\limits_{x,y \sim p_{data}} \left[(f(x)^Tf(y)) \cdot I_{l(x)=l(y)}\right]
```
Mean similarity of same-class pairs (uses labels; evaluation only).

Uniformity-tolerance dilemma (Sec. 4, end of 4.2, verbatim): "The contrastive loss meets a uniformity-tolerance dilemma. On the one hand, we hope to decrease the temperature $\tau$ to increase the uniformity of the embedding distribution, on the other hand, we hope to increase the temperature to make the embedding space tolerant to the similar samples."

## 7. Config (Sec. 5.1)
- Data: CIFAR10, CIFAR100, SVHN, ImageNet100.
- Backbone: ResNet-18 with 3x3 first conv (32x32 input) for CIFAR/SVHN; ResNet-50 for ImageNet100.
- Aug: 32x32 random resized crop, color jitter, h-flip, grayscale (follows Wu 2018); ImageNet100 adds Gaussian blur.
- Negatives: memory bank of EMA features (Wu 2018). ImageNet100 hard loss: 4095 negatives.
- Hard loss alpha (quantile): 0.0819 (CIFAR10), 0.0819 (CIFAR100), 0.0315 (SVHN), 0.034 (ImageNet100).
- SGD momentum 0.9; weight decay 5e-4 (CIFAR/SVHN), 1e-4 (ImageNet100); lr 0.03; batch 128; 200 epochs, lr x0.1 at 160 and 190.
- Linear eval: frozen convs, 100 epochs, lr 30.0, x0.2 at 40, 60, 80, wd 0. "1-crop top-1".
- Feature dim and uniformity t: not stated (our reading).

## 8. Results
Main Table 1 (p.7): linear top-1, uniformity (reported as -L_uniformity, larger = more uniform), tolerance. Columns: Contrastive tau = 0.07 / 0.3 / 0.7 / 1.0 | Simple | HardContrastive tau = 0.07 / 0.3 / 0.7 / 1.0 | HardSimple.
- CIFAR10 acc: 79.75 / 83.27 / 82.69 / 82.21 | 74.83 | 79.2 / 83.63 / 84.19 / 84.19 | 84.84
- CIFAR10 unif: 3.86 / 3.60 / 3.17 / 2.96 | 1.68 | 3.88 / 3.89 / 3.87 / 3.86 | 3.85
- CIFAR10 tol: 0.04 / 0.178 / 0.333 / 0.372 | 0.61 | 0.034 / 0.0267 / 0.030 / 0.030 | 0.030
- CIFAR100 acc: 51.82 / 56.44 / 50.99 / 48.33 | 39.31 | 50.77 / 56.55 / 57.54 / 56.77 | 55.71
- CIFAR100 unif: 3.86 / 3.60 / 3.18 / 2.96 | 2.12 | 3.87 / 3.88 / 3.87 / 3.86 | 3.86
- CIFAR100 tol: 0.10 / 0.269 / 0.331 / 0.343 | 0.39 | 0.088 / 0.124 / 0.158 / 0.172 | 0.174
- SVHN acc: 92.55 / 95.47 / 94.17 / 92.07 | 70.83 | 91.82 / 94.79 / 95.02 / 95.26 | 94.99
- SVHN unif: 3.88 / 3.65 / 3.27 / 3.05 | 1.50 | 3.89 / 3.91 / 3.90 / 3.88 | 3.85
- SVHN tol: 0.032 / 0.137 / 0.186 / 0.197 | 0.074 | 0.025 / 0.021 / 0.021 / 0.023 | 0.026
- ImageNet100 acc: 71.53 / 75.10 / 69.03 / 63.57 | 48.09 | 68.33 / 74.21 / 74.70 / 74.28 | 74.31
- ImageNet100 unif: 3.917 / 3.693 / 3.323 / 3.08 | 1.742 | 3.929 / 3.932 / 3.927 / 3.923 | 3.917
- ImageNet100 tol: 0.093 / 0.380 / 0.427 / 0.456 | 0.528 | 0.067 / 0.096 / 0.121 / 0.134 | 0.157

Supp Table 1 (full tau sweep, ordinary contrastive, linear top-1; tau = 0.05, 0.07, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0):
- CIFAR10: 76.10, 79.75, 81.82, 83.78, 83.27, 83.22, 82.54, 82.97, 82.69, 82.67, 81.97, 82.21
- CIFAR100: 49.80, 51.82, 52.46, 56.05, 56.44, 55.47, 54.17, 53.05, 50.99, 50.08, 50.21, 48.33
- SVHN: 88.96, 92.55, 94.21, 95.46, 95.47, 95.36, 94.66, 94.47, 94.17, "93,22" (typo in source), 92.66, 92.07
- ImageNet100: 63.91, 71.53, 74.59, 75.41, 75.10, 72.98, 71.10, 70.47, 69.03, 67.91, 65.93, 65.49
Supp Table 2 (hard contrastive, same tau grid):
- CIFAR10: 76.22, 79.20, 80.44, 83.28, 83.63, 84.14, 84.31, 84.60, 84.19, 84.60, 84.45, 84.19
- CIFAR100: 49.21, 50.77, 51.91, 55.61, 56.55, 55.66, 57.37, 57.17, 57.54, 57.15, 56.53, 56.77
- SVHN: 89.07, 91.82, 93.37, 94.58, 94.79, 94.62, 94.93, 95.06, 95.02, 95.03, 95.08, 95.26
- ImageNet100: 61.54, 68.33, 72.10, 74.53, 74.21, 75.04, 74.84, 74.46, 74.70, 74.28, 74.74, 73.78

Inconsistencies (do not reconcile): ImageNet100 contrastive tau=1.0 is 63.57 in main Table 1 but 65.49 in supp Table 1; ImageNet100 hard tau=1.0 is 74.28 in main but 73.78 in supp (74.28 is supp's tau=0.8). Main Table 1 caption names only CIFAR10, CIFAR100, SVHN although it has ImageNet100 rows.

Text: best tau is 0.2 or 0.3 (Sec. 5.3). Simple vs contrastive: "74.83 vs 83.27 on CIFAR10, 39.31 vs 56.44 on CIFAR100, 70.83 vs 95.47 on SVHN, 48.09 vs 75.10 on ImageNet100" (Sec. 5.5). Fig. 2: t-SNE at tau 0.07 vs 0.2 on CIFAR10.

## 9. Limitations stated
The dilemma comes from instance discrimination ignoring semantic relations; the paper suggests modelling instance relations as future work. Hard loss only alleviates it. Later: Chuang (debiasing) and Robinson (hard + debiased) address the false-negative side; SupCon uses labels.

## 10. Outline feedback
- Good: Eq. (4) is the printed gradient wrt similarities; Eq. (5) is the share-of-push r_i the viz-negatives bars should show. Cite "Wang & Liu Eq. (4)-(5)".
- Outline says small tau "pushes apart semantically similar images too" -- matches the abstract.
- The tau sweep spans 0.05 to 1.0; viz-sphere tau control should use this range. Peak at tau 0.2-0.3.
- Uniformity numbers are -L_uniform; say so on the page. The t used is not stated.
- Negatives come from a memory bank, not SimCLR in-batch.
