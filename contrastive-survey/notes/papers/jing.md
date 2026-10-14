# Jing et al. 2022: dimensional collapse (spine, analysis lens)

## 1. Identity
- Title: Understanding Dimensional Collapse in Contrastive Self-supervised Learning
- arXiv 2110.09348, read v3 (2022-04-23). v1 2021-10-18.
- Authors: Li Jing ... Yuandong Tian. Venue: ICLR 2022 (abs.html comment; PDF header).
- Repo: none given in sources.

## 2. Core problem
Contrastive methods avoid complete collapse (all embeddings to one point) through negatives.
Do they use all embedding dimensions? The paper shows they do not: embeddings span a lower-dimensional subspace ("dimensional collapse"), and explains why.

## 3. Prior work and gap (per the paper)
- Dimensional collapse was observed in non-contrastive methods (Hua et al. 2021; Tian et al. 2021) (Sec 1).
- Intuition says negatives' repulsion should prevent it in contrastive methods; the paper shows it still happens (Sec 1, Fig 7).
- Tian et al. 2020 Theorem 4 covers one negative, infinite batch; Lemma 2 here covers many negatives, finite N, varying alpha_ij (Sec 4.2).

## 4. Claims
1. Contrastive SSL (SimCLR) shows dimensional collapse in the embedding space (Sec 3, Fig 2).
2. Mechanism 1, strong augmentation: in a single linear layer, if augmentation variance exceeds data variance along a direction, the weight collapses there (Sec 4, Theorem 1, Corollary 1).
3. Mechanism 2, implicit regularization: with weak augmentation, a 2-layer linear net still collapses because adjacent weights align and small singular values grow slower (Sec 5, Theorems 2-3, Corollary 2).
4. DirectCLR: drop the projector, apply InfoNCE to a fixed sub-vector r[0:d0] of the representation; beats SimCLR with a 1-layer linear projector (Sec 6, Table 1).

## 5. Measurement setup
- Spectrum (Sec 3, Eq. 1, Fig 2): train SimCLR with a 2-layer MLP projector, ImageNet, 100 epochs. Collect embeddings on the ImageNet validation set; d = 128.
  Compute the covariance C (Eq. 1), SVD C = U S V^T, plot log(sigma^k) in sorted order. "A number of singular values drop to zero, indicating collapsed dimensions."
  Fig 2 y-axis is log of singular values (down to about -25); the cliff is near rank index 90-100 of 128 (read from plot; not printed).
- Representation spectrum (Sec 6.1, Fig 7b): ResNet-50 output, 2048-d; SimCLR without a projector collapses in representation space; with a projector it does not.
- Toy, mechanism 1 (Fig 3): single linear layer, W 16x16; input isotropic Gaussian (data covariance I); augmentation additive Gaussian with covariance block_diagonal(0, k*I), block 8x8. k legend: 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2. Larger k -> the 8 augmented directions' singular values vanish.
- Toy, mechanism 2 (Figs 5-6): 2-layer linear, W1, W2 16x16, no bias, z not normalized, plain SGD (no momentum, no weight decay). Alignment matrix A = V2^T U1 -> I (Fig 5). Lowest singular values of W1, W2 and embedding covariance stay much smaller (Fig 6).
- ImageNet DirectCLR (App, Table 1): ResNet-50, LARS, 100 epochs, batch 4096 on 32 GPUs, lr 4.8, 10-epoch warmup, cosine; BYOL-style augmentations incl. solarization.
- Falsifier: a contrastive model whose embedding covariance has full rank with no near-zero singular values under the same measurement; or a linear toy where X has negative eigenvalues but W keeps all singular values.

## 6. Equations (src/main.tex)
Eq. (1) `eq:cov-embedding-layer`:
```
C = \frac{1}{N}\sum_{i=1}^N (\textbf{z}_i-\bar{\textbf{z}})(\textbf{z}_i-\bar{\textbf{z}})^T
```
z_i: embedding of validation image i; z-bar: mean embedding; N: number of images. Its sorted log singular values are the "spectrum".

Eq. (2) `eqn:infoNCE` (the loss analysed; squared-distance form, not SimCLR's cosine form):
```
    L = -\sum_{i=1}^N\log\frac{\exp(-|\textbf{z}_i -\textbf{z}_i'|^2/2)}{\sum_{j\neq i}\exp(-|\textbf{z}_i -\textbf{z}_j|^2/2) + \exp(-|\textbf{z}_i -\textbf{z}_i'|^2/2)}
```
z_i, z_i': the two views of sample i (positive); z_j: other samples in the batch (negatives). For unit vectors, -|z_i - z_i'|^2/2 can be replaced by z_i^T z_i'.

Lemma 1 (Eq. 3): `\dot{W} = -G`, G = sum_i (g_{z_i} x_i^T + g_{z_i'} x_i'^T).
Eq. (5) `eqn:G2X`: `G= - WX`.
Eq. (6) `eqn:X`:
```
X := - \sum_i\left(\sum_{j\neq i} \alpha_{ij}(\textbf{x}_i'-\textbf{x}_j) + \sum_{j\neq i}\alpha_{ji}(\textbf{x}_i-\textbf{x}_j)\right)\textbf{x}_i^T - \sum_i(1-\alpha_{ii})(\textbf{x}_i'-\textbf{x}_i){\textbf{x}_i'}^T
```
Lemma 2 (Sec 4.2): `X = \hat{\Sigma}_0 - \hat{\Sigma}_1`, with Sigma_0 = sum_{i,j} alpha_ij (x_i - x_j)(x_i - x_j)^T "a weighted data distribution covariance matrix" and
Sigma_1 = sum_i (1 - alpha_ii)(x_i' - x_i)(x_i' - x_i)^T "a weighted augmentation distribution covariance matrix". "the amplitude of augmentation determines whether X is a positive definite matrix."

Theorem 1 (Sec 4.2): "With fixed matrix X (defined in Eqn 6) and strong augmentation such that X has negative eigenvalues, the weight matrix W has vanishing singular values."
Corollary 1 (Dimensional Collapse Caused by Strong Augmentation): "With strong augmentation, the embedding space covariance matrix becomes low-rank."
Lemma 3 (Eq. 8): `\dot{W_1} = -W_2^TG, \quad\quad \dot{W_2} = -GW_1^T`; Eq. (9): `G = -W_2 W_1 X`.
Theorem 2 (Weight matrices align, Sec 5.3): "If for all t, W_2(t)W_1(t) != 0, X(t) is positive-definite and W_1(+inf), W_2(+inf) have distinctive singular values, then the alignment matrix A = V_2^T U_1 -> I."
Theorem 3 (Sec 5.3, `eqn:sigma-evolve`, PDF Eqs. 10-11):
```
\dot{\sigma}_{1}^k =\sigma_{1}^k  (\sigma_{2}^k)^2 ({\textbf{v}_{1}^k}^TX\textbf{v}_{1}^k) \\
\dot{\sigma}_{2}^k =\sigma_{2}^k(\sigma_{1}^k)^2 ({\textbf{v}_{1}^k}^TX\textbf{v}_{1}^k)
```
sigma_1^k, sigma_2^k: k-th singular values of W1, W2; v_1^k: k-th right singular vector of W1. Growth rate is proportional to the singular value itself, so "the smallest group of singular values grow significantly slower".
Corollary 2 (Dimensional Collapse Caused by Implicit Regularization): "With small augmentation and over-parametrized linear networks, the embedding space covariance matrix becomes low-rank."
Propositions 1-2 (Sec 6.1): "A linear projector weight matrix only needs to be diagonal." / "... only needs to be low-rank."

DirectCLR loss (Sec 6.2, unnumbered): z = r[0:d0], z-hat = z/|z|,
```
L = \sum_i\log\frac{\exp(\hat{\textbf{z}}_i \cdot\hat{\textbf{z}}_i')}{\sum_j\exp(\hat{\textbf{z}}_i \cdot\hat{\textbf{z}}_j)}
```
(sign as printed; no leading minus in the source.)

## 7. Results
- Table 1 (ResNet-50, 100 ep, linear probe): SimCLR 2-layer nonlinear projector 66.5; SimCLR 1-layer linear projector 61.1; SimCLR no projector 51.5; DirectCLR no projector 62.7.
- Table 2 ablation: no projector 51.5; orthogonal projector 52.2; trainable 61.1; trainable diagonal 60.2; fixed low-rank 62.3; fixed low-rank diagonal 62.7.
- App F: linear probe on sub-vector z only 47.9; random d0 dropout instead of fixed sub-vector 43.0.
- d0: Fig 12 sweeps d0 from near 0 to 2048; peak near 360-ish (read from plot). The chosen d0 is not printed in the PDF. A commented-out table in src/main.tex lists d0=360 -> 62.104 (best) and d0=2048 -> 51.652; this 62.104 does not match Table 1's 62.7. Do not cite d0 on the page without "about".

## 9. Limitations stated
- Theory limited to linear networks; "strong augmentation" in nonlinear nets depends on higher-order augmentation statistics (Sec 4.2 end).
- "our theory is not able to fully explain why a nonlinear projector is able to prevent dimensional collapse"; DirectCLR relies on the last ResNet block acting like a nonlinear projector (Sec 6.2 disclaimer).

## 10. Outline feedback
- Outline: "Jing: embeddings fill only a low-dimensional subspace; contrastive methods suffer the second one too." Correct. Add: the measured collapse is in the 128-d embedding (projector output); the 2048-d representation stays healthy when a projector is used (Fig 7b). This explains why the projector is thrown away, a link back to ch-simclr.
- `viz-dimcollapse` button `btn-dim-dimensional`: per Theorem 1, the trigger is augmentation variance exceeding data variance along a direction (X gets a negative eigenvalue), not "strong augmentation" in general. Toy can follow Fig 3: 16-d, 8 augmented directions, k from 0.1 to 2. In 3D: augment one axis with k > data variance and show its singular value shrink.
- Mechanism 2 (implicit regularization) needs depth >= 2; mention it in one sentence; it is the reason weak augmentation does not save you.
- The spectrum measure for the equation slot is Eq. 1 (covariance of embeddings) plus "sorted log singular values".
