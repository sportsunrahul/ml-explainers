# VICReg (branch)

- Title: VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. arXiv 2105.04906, read v3 (2022-01-28); v1 2021-05-11.
- Authors: Adrien Bardes ... Yann LeCun. Venue: ICLR 2022 (abs.html comment).
- Key idea: three explicit terms on each branch's batch of embeddings: keep each dimension's std above gamma=1 (variance), decorrelate dimensions (covariance, from Barlow Twins), and pull the two views together with plain MSE (invariance). No normalization, no asymmetry, no negatives.
- Defining equation (`eq:loss`, src/iclr2022_final.tex):
```
\ell(Z, Z^{\prime}) = \ \lambda s(Z, Z^{\prime}) + \mu [ v(Z) + v(Z^{\prime}) ] + \nu [ c(Z) + c(Z^{\prime}) ],
```
  v(Z) = (1/d) sum_j max(0, gamma - S(z^j, eps)) with S = sqrt(Var + eps) (`eq:var_loss`); c(Z) = (1/d) sum_{i != j} [C(Z)]_{ij}^2 (`eq:cov_loss`); s = (1/n) sum_i ||z_i - z'_i||^2 (`eq:invariance`).
- Config (Sec 4.2 Implementation details): lambda = mu = 25, nu = 1, eps = 0.0001; ResNet-50 (2048) -> expander 3 layers of 8192; LARS, 1000 ep, batch 2048, base lr 0.2.
- Key number: 73.2 ImageNet linear top-1 (top-5 91.1), ResNet-50, 1000 ep, batch 2048, no multi-crop (Table 1, row VICReg (ours)).
- Table 1 prints the same baseline numbers as Barlow Twins, plus OBoW 73.8 and InfoMin Aug 73.0, CMC 66.2, CPC v2 63.8, Barlow Twins 73.2.
