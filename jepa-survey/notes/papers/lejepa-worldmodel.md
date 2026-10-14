# When Does LeJEPA Learn a World Model?

## 1. Identity
- arXiv 2605.26379, version read: v1 (only version). Submitted 2026-05-25 (abs.html).
- Authors: David Klindt (Cold Spring Harbor Laboratory), Yann LeCun (NYU), Randall Balestriero (Brown). Three authors.
- Venue: none stated (NeurIPS 2026 preprint template). PDF 48 pages. Code, Lean proofs and demo: github.com/klindtlab/lejepa-identifiability (not downloaded; meta.json repo null).
- Sources: src/main.tex, src/appendix.tex, paper.pdf. No custom math macros beyond standard amsmath.
- Lens: theory. One claim card.

## 2. Core problem
A representation that scrambles the world's true degrees of freedom cannot support planning or compositional generalization. The question: when is a learned representation a faithful map of the world's latent variables, i.e. linearly identifiable? No identifiability result existed for JEPAs because prior methods prevent collapse implicitly, leaving the embedding distribution unspecified (Sec 1).

## 3. What prior methods did and why that hurt (per the paper)
- Contrastive (negatives), non-contrastive (stop-gradient teachers, VICReg covariance), self-distillation (DINO): collapse prevention is implicit, so nothing pins the embedding distribution and no identifiability theorem can be stated (Sec 1, Sec 2).
- Nonlinear ICA is unidentifiable without extra structure; prior results need non-stationarity, temporal dependence, auxiliary variables, augmentations, sparsity, interventions or supervision, and usually a diffeomorphic encoder. Here the encoder is any measurable map (Sec 2).
- LeJEPA changes the picture by regularizing the embedding distribution to an isotropic Gaussian explicitly (SIGReg), which makes the distribution part of the hypothesis.

## 4. The one key idea
With Gaussian latents under an Ornstein-Uhlenbeck transition, the transition operator's eigenfunctions are Hermite polynomials with eigenvalue rho^degree. Alignment maximizes correlation across positive pairs, and every nonlinear (higher-degree) component strictly loses correlation, so the optimum is linear; Gaussianity then forces the linear map to be orthogonal. Conversely, a linear first eigenfunction forces a linear score function, hence a Gaussian latent: the Gaussian is the unique such distribution.

## 5. Mechanism, step by step
1. World: latents z in R^n, observations x = g(z) with g unknown and nonlinear; positive pairs (g(z), g(z')). Assumptions box (Sec 3.1): (i) independence of latent components and of their transitions, (ii) stationarity p(z) = p(z'), (iii) additive noise z'_i = m_i(z_i) + eta_i.
2. Gaussian world (Sec 3.1.1): z ~ N(0, I_n); the only additive-noise transition preserving the marginal is OU, eq (1), with correlation rho in (0, 1).
3. Learner (Sec 3.2): h = f o g; objective eq (2): minimize E||h(z') - h(z)||^2 subject to h(z) ~ N(0, I_n) (SIGReg assumed to have succeeded). Whitening fixes E||h(z)||^2 = n, so eq (3): L(h) = 2n - 2 sum_i E[h_i(z') h_i(z)]; minimizing distance = maximizing correlation.
4. Spectral decomposition (Sec 4): E[h_i(z') h_i(z)] = sum_d w_d rho^d <= rho, eq (4), equality iff h_i is linear (w_1 = 1).
5. Theorem 1 (p5): L(h) >= 2(1 - rho) n with equality iff h(z) = Qz, Q orthogonal; then h(z') | h(z) ~ N(rho h(z), (1 - rho^2) I).
6. Theorem 2 (p5): in any world satisfying the Assumptions, if every whitened minimizer is linear then z is Gaussian. Proof: affine eigenfunction forces (log p)' linear, so log p is quadratic.
7. Theorem 3 (p6): approximate alignment gap delta and whitening error epsilon give E||h(z) - Qz||^2 <= D + (epsilon + D)^2, D = delta / (2 rho (1 - rho)), eq (5).
8. Theorem 4 (p6): with h(z) = Qz, any finite-horizon control problem with O(n)-invariant costs has the same value and the same optimal plan in learned and true coordinates, eq (6).
9. All proofs verified in Lean 4 (App G, p31): Lean 4 v4.28.0, Mathlib v4.28.0, 8,032 build targets, zero sorry, with standard results (Hermite polynomials, Mazur-Ulam, weighted AM-GM, pushforward of costs) introduced as axioms (Table 4, p38).

## 6. Equations (TeX verbatim)
(1) `\label{eq:dgp}`, p3:
```
z' = \rho\, z + \sqrt{1-\rho^2}\;\eta, \qquad \eta \sim \mathcal{N}(0, I_n),\quad \eta \perp z,
```
OU transition; Var(z') = I, Cov(z, z') = rho I. "The Gaussian is the unique distribution for which a channel of this form preserves the marginal, a consequence of the Gaussian being a fixed point of convolution up to rescaling."

(2) Sec 3.2, p4:
```
\min_h \mathcal{L}(h) = \underbrace{\mathbb{E}\!\left[\|h(z') - h(z)\|^2\right]}_{\textbf{Alignment}} \qquad \text{s.t.} \qquad \underbrace{h(z) \sim \mathcal{N}(0, I_n)}_{\textbf{Gaussianity}}
```
(3) `\label{eq:loss-expand}`:
```
\mathcal{L}(h) = 2n - 2\sum_{i=1}^n \mathbb{E}\!\left[h_i(z')\, h_i(z)\right].
```
(4) `\label{eq:corr-decomp}`:
```
\mathbb{E}\!\left[h_i(z')\, h_i(z)\right] = w_1 \cdot \rho + w_2 \cdot \rho^2 + w_3 \cdot \rho^3 + \cdots \;\leq\; \rho,
```
w_d = variance fraction of h_i in Hermite degree d, sum to 1.

Theorem 1 (p5), `thm:main`, statement verbatim: "Consider the Gaussian world (Sec. 3.1.1). Let h: R^n -> R^n be any measurable map with h(z) ~ N(0, I_n). Then L(h) >= 2(1-rho) n, with equality if and only if h(z) = Qz for some orthogonal Q in O(n). At any such optimum, h(z') | h(z) ~ N(rho h(z), (1-rho^2) I_n)." TeX of the key inequality: `\mathcal{L}(h) \geq 2(1-\rho)\,n`.

Theorem 2 (p5), `thm:converse`: "Consider any world satisfying Assumptions. Suppose every minimizer of (3) with Cov(h(z)) = I_n is linear, h(z) = Qz. Then z is Gaussian."

(5) Theorem 3 `\label{eq:stability}`, p6:
```
\mathbb{E}\!\left[\|h(z) - Qz\|^2\right] \;\leq\; D \;+\; (\varepsilon + D)^2
```
with `D = \delta/(2\rho(1-\rho))`, delta = alignment gap, epsilon = ||Cov(h(z)) - I_n||_F. "In practice the first dominates ... alignment is hard, whitening essentially free."

(6) Theorem 4 `\label{eq:orth-inv-app}`: `\ell(Rz, a) = \ell(z, a), \qquad \ell_T(Rz) = \ell_T(z), \qquad \forall\, R \in O(n).`

Training loss used in experiments (App H.3, `eq:loss-combined`): `\mathcal{L} = \lambda\,\mathcal{L}_{\mathrm{SIG}} + (1 - \lambda)\,\mathcal{L}_{\mathrm{inv}}`, same convention as LeJEPA.

Main theorem in one sentence: if the world's latents are independent standard Gaussians evolving by an OU transition with correlation rho, then any measurable encoder whose outputs are N(0, I) and which minimizes the alignment loss equals the true latents up to an orthogonal rotation, and (converse) within independent, stationary, additive-noise worlds the Gaussian is the only latent distribution for which this holds.

Why Gaussian is unique (Sec 5.2): the slowest feature of an additive-noise process is the first Sturm-Liouville eigenfunction, which is monotonic for any latent distribution (identifiability up to a monotone map); demanding it be affine forces the score (log p)' to be linear, i.e. Gaussian. Also Sec 5.2 notes Theorem 1 uses only whitened covariance of h(z), not full Gaussianity.

## 7. Measurement setup
- 2D forward check (Sec 6.1, Fig 3, Fig 1): z ~ N(0, I_2); four mixings g: norm-dependent rotation (spiral, measure-preserving), sinusoidal shear, parabolic shear, RealNVP coupling; 4-layer MLP, hidden 256, GELU; loss lambda L_SIG + (1 - lambda) L_inv; positive pairs via OU eq (1); AdamW lr 3e-3, batch 256, 20,000 steps, online data; rho = 0.95, lambda = 1e-3 (App H.4 table). Metric: linear R^2 from h to z (and z to h), orthogonality error of the fitted map.
- Grid (App H.6, Fig 6 p39): lambda in {1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1}, rho in {0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99}, 5 seeds. lambda = 0.5 collapses (R^2 ~ 0); best R^2 > 0.97 at lambda in {1e-3, 5e-3}, rho in {0.9, 0.95}.
- Scaling (Table 1, p7): N in {2, ..., 1024} latent dims, RealNVP mixing, matched inverse-NVP encoder, 5 seeds, lambda = 1e-6, rho = 0.95; SIGReg vs VICReg vs InfoNCE (kernel width 1). R^2(h -> z): SIGReg 0.999998 at N = 2 to 0.999561 at N = 1024; VICReg 0.999582 at 1024; InfoNCE 0.950961 at N = 2, 0.720241 at 1024. Raw mixing R^2(x -> z) 0.781 (N = 2) to 0.763 (N = 1024).
- Converse: generalized-normal sweep over shape alpha (alpha = 1 Laplace, 2 Gaussian, infinity uniform); recovery peaks at alpha = 2 for all three objectives (Fig 4b, App H.7); SIGReg plateau wider than VICReg for heavy tails.
- Pixel-based RL (Table 2, p8; App H.11): DMC Reacher, 2D latent (two joint angles), CNN encoder, 3 seeds; (i) OU-sampled angles vs (ii) joint-angle pairs at stride delta from 10k RL episodes (LeWorldModel policy). OU: R^2 rises with rho to 0.95 at rho = 0.99. Trajectory: total R^2 never above 0.5 (best 0.50 at delta = 8); anisotropic rho_0 != rho_1. BatchNorm was needed: without it about 36% of runs collapsed.
- Bound check (Fig 4a): for every run, epsilon, delta, bound D + (epsilon + D)^2 and actual min_Q E||h(z) - Qz||^2; points lie below the diagonal except two near-zero outliers.
- Planning (Fig 4c-d, Fig 5): straight-line goal reaching in latent space, K = 30 random start-goal pairs, control cost (ideal 1); Gaussian encoder indistinguishable from oracle, trajectory encoder biased upward; cost decreases with R^2.

## 8. Results worth remembering
- Lower bound 2(1 - rho) n, attained only by orthogonal maps (Theorem 1).
- R^2 > 0.999 up to 1024 latent dims for SIGReg and VICReg (Table 1).
- OU R^2 = 0.95 vs trajectory R^2 <= 0.5 on Reacher (Table 2).
- lambda = 0.5 collapses; lambda <= 1e-2 suffices (Fig 6).
- Lean 4: five results machine-checked, 8,032 build targets, zero sorry (App G).

## 9. Limitations the paper states (Sec 7)
- Whether real latents are Gaussian is unknowable from observations; central-limit argument offered for aggregate latents.
- Encoder output dimension must equal latent dimension (m = n); m < n (superposition) and m > n open.
- Population-level statement; no sample-size or training-dynamics analysis; a few empirical bound violations attributed to finite-sample noise.
- Only the state side of a world model; action-conditioned transitions still to be learned (Sec 8).
- SIGReg is assumed to have succeeded exactly in Theorems 1-2; Theorem 3 covers the approximate case.

## 10. Outline feedback
- The outline's one-liner "LeJEPA linearly identifies the world's latents; Gaussian is the unique distribution for which this holds" matches Theorems 1-2. Add the assumptions in the card: independent latents, stationary marginal, additive-noise transitions, encoder output dim = latent dim, and OU correlation rho in (0, 1).
- The identifiability is up to an orthogonal matrix Q (rotation/reflection), not up to any linear map; say "up to rotation".
- The empirical check to cite: Table 1 (R^2 > 0.999 to 1024 dims) and Table 2 (OU 0.95 vs trajectory <= 0.5), plus Fig 4b (peak at Gaussian).
- What would falsify: a non-Gaussian latent distribution in the assumption class with linear recovery as good as the Gaussian (Fig 4b shows a sharp peak at alpha = 2); or a trained encoder violating the Theorem 3 bound beyond finite-sample noise (Fig 4a); or OU-sampled Reacher failing to reach high R^2.
- The world-model paper uses the same lambda L_SIG + (1 - lambda) L_inv convention as LeJEPA; the outline's "prediction + lambda * SIGReg" is wrong for both.
