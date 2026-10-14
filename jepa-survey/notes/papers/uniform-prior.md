# The Hidden Uniform Cluster Prior in Self-Supervised Learning (2210.07277v1)

## 1. Identity
- Title: The Hidden Uniform Cluster Prior in Self-Supervised Learning
- arXiv: 2210.07277v1 (version read). Published 2022-10-13 (abs.html: citation_date 2022/10/13, "Submitted on 13 Oct 2022").
- Authors: Mahmoud Assran (first) ... Nicolas Ballas (last); 9 authors (Assran, Balestriero, Duval, Bordes, Misra, Bojanowski, Vincent, Rabbat, Ballas). Meta AI.
- Venue: not stated on the abs page. The LaTeX uses the ICLR 2023 template (`ICLRtemplate/iclr2023_conference`).
- Repo: none of its own. Pretraining used the public MSN, VICReg, MAE and data2vec code bases and the VISSL evaluation code (Appendix D).
- Source read: `src/main.tex` (all), `paper.pdf` for page numbers.
- Custom macros in the quoted TeX: `\mZ` = `{\bm{Z}}`, `\mG`, `\mI`, `\mP`, `\mW` = bold matrices; `\vz`, `\vx`, `\vp`, `\vpi` = bold vectors; `\sX` = `{\mathbb{X}}` (all from `ICLRtemplate/math_commands.tex`, `\def` lines e.g. 241, 307, 398); `\Cov` = `\mathrm{Cov}` (main.tex line 492); `\defeq` = `\coloneqq` (line 38).

## 2. Core problem
Joint-embedding SSL methods must stop the encoder from outputting a constant vector. The usual fix is a "volume maximization" regulariser (negatives, decorrelation, or high-entropy clustering). The paper asks what that regulariser quietly assumes about the data, and shows the answer is a uniform cluster prior: a bias toward features that split the data into equal-size clusters. That prior is fine on class-balanced ImageNet and harmful on class-imbalanced (long-tailed) data.

## 3. What prior methods did and why that hurt (per the paper)
- SimCLR, VICReg, SwAV, MSN all prevent collapse with mini-batch statistics (contrast negatives, whiten the covariance, or spread cluster assignments). The paper shows each is a K-means variant (Section 3), and K-means assumes isotropic clusters of roughly equal size (Figure 1: on imbalanced 2-D data K-means picks the wrong split).
- Consequence: "these joint-embedding methods will penalize features that do not uniformly cluster the data, even if such features correlate well with class information" (Section 1).
- Prior empirical reports (Tian et al. 2021, SEER) saw drops on long-tailed pretraining data without an explanation.

## 4. The one key idea
The anti-collapse term is a hidden prior. Write each method's loss as K-means (explicit or implicit) and the uniform-cluster assumption is exposed; test it by controlling the class mix inside each mini-batch; fix it by swapping the uniform prior for a power-law prior (PMSN).

## 5. Mechanism, step by step
1. K-means (eq (1)) equals a pairwise within-cluster distance sum (eq (2)); the centroids can be implicit (Proposition 1).
2. VICReg's simplified loss (eq (3)) has the same pairwise term over positive pairs, so with gamma >> alpha it is implicit K-means on the embeddings plus an orthogonality regulariser on the centroids (Proposition 2). Under Garrido et al. 2022's duality the same applies to SimCLR.
3. MSN's loss (eq (6)) is a soft-assignment (GMM) K-means with the mean-entropy term equal to a KL to the uniform prior (eq (5), Proposition 3).
4. SwAV's Sinkhorn-Knopp constraint (eq (8)) is K-means with hard equal-size clusters (eq (7), Proposition 4).
5. Empirical test (Section 4): pretrain on ImageNet-1K with either class-balanced mini-batches (960 classes per batch, equal images per class) or class-imbalanced mini-batches (K = 8 classes for SimCLR, K = 2 for VICReg, MSN, MAE, data2vec), same total batch size and same marginal sampling probability per image (Appendix G). Linear-probe on 7 downstream tasks. Table 1 reports balanced minus imbalanced.
6. Proposal (Section 5, eq (9)): PMSN replaces MSN's negative entropy with KL(mean posterior || user prior); prior = true class frequencies (Table 2) or a power-law with exponent tau (Table 3).
7. Checks: toy CIFAR10 + MNIST-digit dataset with a power-law digit distribution (Figure 3); iNaturalist18 pretraining (long-tailed, about 430K images, over 8142 species) vs ImageNet-1K; RCDM visualisations of prototypes (Figures 2, 4); inverse square-root sampling as an alternative (Table 4, does not help).

## 6. Equations (TeX verbatim from `src/main.tex`)

(1) `\label{eq:kmeans}`, Section 3.1, p3. Explicit K-means.
```
    \min_{\{\sX_k\}_{k=1}^K}
    \sum_{k=1}^K \sum_{\vx \in \sX_k} \|\vx-\mu_k\|^2_2,
```
Gloss: X_k = the k-th disjoint group of the N points, mu_k = its mean, K = number of clusters.

(2) `\label{eq2}`, p3. Radial = pairwise form. Paper: "K-means does not require explicit computation of the cluster centroids".
```
    \sum_{k=1}^K \sum_{\vx \in\sX_k} \lVert\vx-\mu_k\rVert^2_2 & = \sum_{k=1}^K \frac{1}{2\lvert\sX_k\rvert} \sum_{\vx, \vx^\prime \in \sX_k} \lVert\vx-\vx'\rVert^2_2. \label{eq2}
```
Proposition 1 (p3), verbatim: "The explicit K-means problem, defined by learning a set of K centroids mu_1, ..., mu_K, admits the same global optimum as the implicit K-means problem, defined by learning a cluster membership matrix P in {0,1}^{N x K}, such that P 1_K = 1_N."

(3) `\label{eq:VICR2}`, Section 3.2, p3. Simplified VICReg (variance + covariance merged into one whitening term).
```
    \mathcal{L}=&\alpha \|\Cov(\mZ) - \mI \|_F^2 +\frac{\gamma}{N} \sum_{i,j=1}^{N}(\mG)_{i,j}\|\vz_{i}-\vz_{j}\|_2^2,\label{eq:VICR2}
```
Gloss: Z = N x D matrix of embeddings z_n = f_theta(x_n); Cov(Z) = its covariance; G_{i,j} = 1 when z_i, z_j are two views of the same image; alpha, gamma > 0 weights. The second term is the right-hand side of (2) with positive-view pairs as the clusters.
Proposition 2 (p4), verbatim: "VICReg with hyper-parameters gamma >> alpha recovers the K-means loss from (1) on the embeddings, with an additional regularizer enforcing orthogonality of the centroids."

(4) `\label{eq:GMM}`, p4. GMM objective with cluster prior pi.
```
    \min_{\{\mu_k,\Sigma_k\}_{k=1}^K}\sum_{\vx \in \sX}\sum_{k=1}^{K}
     \frac{[p(\vx)]_k}{2}\lVert\vx-\mu_k\rVert^2_{\Sigma^{-1}}+N\sum_{k=1}^{K}\log\det(\Sigma_k)+
     \sum_{\vx \in \sX}D_{\text{KL}}(p(\vx)\ \lVert\ \vpi),\label{eq:GMM}
```
Gloss: [p(x)]_k = posterior of x in cluster k (a softmax of W^T x + log pi - ...); pi = cluster prior; Sigma_k = cluster covariance. The KL term is where the prior lives.

(5) `\label{eq:GMM_uni}`, p4. Same with uniform prior, isotropic covariance, l2-normalised vectors.
```
    \min_{\mu_1,\ldots,\mu_K}\sum_{\vx \in \sX}\sum_{k=1}^{K}
     \frac{[\text{softmax}(\mW^T\vx)]_k}{2}\|\vx-\mu_k\|^2_{2}-\sum_{x\in \sX} H\left(\text{softmax}(\mW^T\vx)\right).
```
(6) `\label{eq:msn}`, p4. MSN loss; p_n = softmax(W^T x_n / sigma), p_n^+ for the positive view, p-bar = mean posterior over the batch, lambda > 0.
```
    \frac{1}{N} \sum^N_{n=1} H(\vp_{n}^+, \vp_n) - \lambda H({\overline{\vp}}),
```
Proposition 3 (p4), verbatim: "MSN recovers the GMM loss from (5) with the variation that the l2 distance is replaced with the cross-entropy distance of the posterior."

(7) `\label{eq:constrained}`, p4. Size-constrained K-means; N_k = N/K is the strict uniform prior.
```
    \min_{\{\sX_k\}_{k=1}^K \text{s.t.} |\sX_k|=N_k}
    \sum_{k=1}^K \sum_{\vx \in \sX_k} \|\vx-\mu_k\|^2_2.\label{eq:constrained}
```
(8) `\label{eq:swav}`, p5. SwAV loss with the Sinkhorn-Knopp equipartition constraint; P = [p_1^+, ..., p_n^+].
```
    \frac{1}{N} \sum^N_{n=1} H(\vp_{n}^+, \vp_n)
    \quad \text{subject to}
    \quad \mP \mathbf{1}_N = \frac{N}{K} \mathbf{1}_K;
    \quad \mP^\top \mathbf{1}_K = \mathbf{1}_N,
```
Proposition 4 (p5), verbatim: "SwAV recovers the constrained K-means loss from (5) with constraint enforcement through Sinkhorn-Knopp normalization."

(9) `\label{eq:msn_prior}`, Section 5, p7. PMSN: the proposal. Footnote: "The negative entropy in (6) is simply the KL-divergence to the uniform distribution plus a constant."
```
     \frac{1}{N} \sum^N_{i=1} H(\vp_{i}^+, \vp_i) + \lambda D_{\text{KL}}( \overline{\vp}\ \lVert\ \vp_{\text{\sc pl}(\tau)}),
```
Gloss: p_PL(tau) = power-law prior with [p_PL(tau)]_k proportional to (1/k)^tau, tau > 0; the class-prior variant uses [p_class]_k = D_k / D (D_k images in class k).

## 7. Measurement setup (analysis lens)
Section 4 (mini-batch class mix on ImageNet-1K; Appendix D):
| Method | Arch | Batch | Epochs | Balanced K | Imbalanced K |
|---|---|---|---|---|---|
| SimCLR | ResNet-50 (VISSL) | 4096 | 300 | 960 | 8 |
| VICReg | ResNet-50 | 1024 | 300 | 960 | 2 |
| MSN | ViT-B/16, 1024 prototypes | 1024 | 300 | 960 | 2 |
| MAE | ViT-L/16, mask 0.75 | 1024 | 800 | 960 | 2 |
| data2vec | ViT-B/16, target layers [6..11] | 2048 | 800 | 960 | 2 |
- Sampling: pick K classes, then equal images per class; same total batch size; same per-image marginal probability (Appendix G).
- Metric: linear probe top-1 (VISSL defaults; best of two representations and two heads) on ImageNet, iNat18, CIFAR100, Places205, Clevr/Count, Clevr/Dist, KITTI; ImageNet 1% low-shot in Table 11. Table 1 reports (balanced minus imbalanced).
Section 5 (PMSN):
- Toy: grayscale CIFAR10 with an MNIST digit in the top-left corner, digit frequencies power-law with exponent 0.5; ViT-Tiny/4, 10 prototypes, batch 1024, 300 epochs; compare uniform prior vs power-law prior with exponent 0.5 (Figure 3, nearest neighbours).
- iNat18: ViT-S/16, 8142 prototypes, batch 4096, 300 epochs, entropy coefficient 5.0, masking 0.15; priors: uniform, true class prior, power-law tau = 0.25. ImageNet-1K control: ViT-B/16.
- Same linear-probe protocol as Section 4.

## 8. Results worth remembering
- Table 1 (p6), balanced minus imbalanced, columns ImageNet / iNat18 / CIFAR100 / Places205 / Clevr/Count / Clevr/Dist / KITTI:
  - SimCLR: -11.2 / -8.0 / -10.2 / -5.2 / -4.3 / +0.9 / +1.2
  - MSN: -17.7 / -15.1 / -13.2 / -4.6 / -6.4 / +1.9 / -1.6
  - VICReg: -17.7 / -17.3 / -12.0 / -6.0 / -3.0 / +0.7 / -1.1
  - data2vec: -0.8 / +0.3 / -1.6 / +0.0 / -2.1 / -1.5 / -0.1
  - MAE: -0.1 / +1.4 / +2.5 / +0.1 / -0.8 / +0.0 / +0.0
  Headline (Section 4): "drops by as much as 17.7 top-1 on IN1K"; depth tasks "remain relatively stable (and even marginally improve)"; "Only methods with explicit volume maximization terms exhibit sensitivity to the mini-batch class distribution."
  Reviewer notes: (a) Table 1's "CIFAR100" column equals the appendix tables' "CIFAR100 1%" column (Tables 5-9: SimCLR -10.2, MSN -13.2, VICReg -12.0, data2vec -0.5 vs -1.6 in Table 1, MAE +2.5), while full CIFAR100 deltas are -6.4 / -12.9 / -8.9 / -1.7 / +0.4. (b) SimCLR ImageNet is -11.2 in Table 1 but 66.9 -> 55.8 = -11.1 in Table 10. Cite as written; do not reconcile.
- Table 10 (p25), ImageNet linear, balanced -> imbalanced: SimCLR 66.9 -> 55.8; MSN 77.1 -> 59.4; VICReg 69.1 -> 51.4; data2vec 41.5 -> 40.6; MAE 65.9 -> 65.8.
- Table 11 (p25), ImageNet 1% low-shot: MSN 66.2 -> 28.0 (-38.2); VICReg 48.6 -> 18.1 (-30.5); data2vec 27.4 -> 31.4 (+4.0); MAE 35.1 -> 34.8 (-0.3).
- Table 2 (p8), iNat18 pretraining, ViT-S/16, columns iNat18 / CIFAR100 / Place205 / Clevr/Count / Clevr/Dist / KITTI: uniform 29.1 / 59.4 / 36.9 / 69.4 / 56.8 / 68.2; class-prior 32.1 / 60.3 / 38.3 / 71.5 / 59.1 / 70.9; delta +3.0 / +0.9 / +1.4 / +2.1 / +2.3 / +2.7 (improves on all tasks).
- Table 3 (p9), power-law prior tau = 0.25: iNat18 pretraining (ViT-S/16) power-law 30.1 / 60.1 / 37.7 / 71.1 / 58.9 / 68.2, delta +1.0 / +0.7 / +0.8 / +1.7 / +2.1 / +0.0; ImageNet pretraining (ViT-B/16) uniform 41.9 / 81.7 / 54.0 / 74.3 / 57.8 / 73.3 vs power-law 22.0 / 63.8 / 38.3 / 71.4 / 65.8 / 66.6, delta -19.9 / -17.8 / -15.7 / -2.9 / +8.0 / -6.7. Reading: match the prior to the data; a power-law prior hurts on balanced ImageNet.
- Table 4 (p22): inverse square-root frequency sampling on iNat18 does not help (iNat18 23.4 vs 29.1 uniform).
- Figure 2 (p7) and Figure 4 (p9): RCDM samples from prototypes; balanced / power-law prototypes look like classes, imbalanced / uniform prototypes look like shape, pose, texture.
- Figure 3 (p8): with the uniform prior the MNIST digit is discarded (neighbours have different digits); with the power-law prior it is encoded.
- What would falsify it: a volume-maximisation method (VICReg, MSN, SimCLR) whose linear-probe accuracy is unchanged under K = 2 class-imbalanced mini-batches with the same marginal sampling; or data2vec/MAE dropping by the same amount (which would point to sampling, not the prior); or PMSN with the matched prior failing to beat the uniform prior on iNat18 while beating it on balanced ImageNet.

## 9. Limitations the paper states
- SimCLR is covered only "with limited assumptions" (via the VICReg-SimCLR duality of Garrido et al. 2022); DINO is not analysed (source comment: "we don't have DINO").
- Power-law results "do not claim to have completely solved the issue of class-imbalanced pretraining"; the true class prior is better but usually unknown (Section 5.2).
- Only MSN is extended (PMSN); the other methods' priors are diagnosed, not fixed.
- No later paper in this source set addresses these directly. The chapter's contrast: 2407.03475 analyses the stop-gradient family that this paper found insensitive (data2vec).

## 10. Outline feedback (ch-collapse)
1. Scope: the outline says "joint-embedding methods carry a hidden uniform-cluster prior". The paper's claim is about joint-embedding methods WITH volume-maximisation regularisers (SimCLR, VICReg, SwAV, MSN). data2vec (self-distillation, EMA teacher, no batch statistics) and MAE are unaffected in Table 1 (-0.8 and -0.1 on ImageNet). State the scope and the data2vec row explicitly; it is the chapter's own thesis in one table: the "term in the loss" family carries the prior, the "asymmetry" family does not.
2. The paper's mechanism is NOT about collapse through EMA/stop-grad; it is about what the explicit anti-collapse term assumes. Place this card as the "cost of the loss-term route" after the asymmetry cards, not as a third theory of the asymmetry.
3. Suggested claim-card numbers: Table 1 ImageNet row (SimCLR -11.2, MSN -17.7, VICReg -17.7, data2vec -0.8, MAE -0.1) and Table 2's +3.0 iNat18 gain from matching the prior. Avoid the CIFAR100 column (see reviewer note in Section 8).
4. The toy should not implement this paper; a two-point version of Figure 1 (K-means on an imbalanced 2-D blob, uniform vs true prior) is the right static visual if the chapter wants one.
