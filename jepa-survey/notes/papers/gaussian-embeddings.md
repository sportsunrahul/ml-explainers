# Gaussian Embeddings: How JEPAs Secretly Learn Your Data Density

## 1. Identity
- arXiv 2510.05949, version read: v1 (only version). Submitted 2025-10-07 (abs.html).
- Authors: Randall Balestriero (Meta-FAIR and Brown), Nicolas Ballas, Mike Rabbat (Meta-FAIR), Yann LeCun (Meta-FAIR and NYU). Four authors.
- Venue: none stated (NeurIPS 2025 preprint template). PDF 10 pages. No repo (meta.json repo null); Listing 1 in the appendix gives the full implementation.
- Sources: src/main.tex (everything after line 730 is commented-out drafts), paper.pdf. Macros from src/math_commands.tex: `\vx={\bm{x}}`, `\mI={\bm{I}}`, `\vtheta={\bm{\theta}}`, `\vu={\bm{u}}`.
- Lens: theory. One claim card.

## 2. Core problem
JEPAs combine a latent prediction term and an anti-collapse term. The anti-collapse term is treated as a mere safeguard. The paper asks whether an encoder can make f(X) follow a prescribed density (Gaussian) without learning anything about the input density p_X. Its answer is no: the anti-collapse term forces the encoder to estimate p_X.

## 3. What prior views did and why that hurt (per the paper)
- Sec 1: all state-of-the-art SSL methods push f_theta(X) toward maximum entropy; the Gaussian is the max-entropy distribution under a covariance constraint, so JEPAs produce "Gaussian Embeddings". JEPAs are "seen as disconnected from generative models whose goal is to estimate p_X", and "the absence of generative modeling is a praised benefit". This hides that a density estimator is already inside every trained JEPA.
- Three JEPA forms listed: moment matching (VICReg, W-MSE), non-parametric (SimCLR, MoCo, CLIP), implicit teacher-student (DINO, I-JEPA).

## 4. The one key idea
For f(X) to be N(0, I/K), the change-of-variables formula ties the output density to p_X divided by the product of the Jacobian's singular values. Since the output density is constant on the sphere (Lemma 1), the Jacobian volume must track p_X. So log p_X(x) is, up to a constant, the sum of log singular values of J_f(x): the JEPA-SCORE.

## 5. Mechanism, step by step
1. Lemma 1 (p2, `thm:uniform`): X = Z / sqrt(K) with Z a K-dim standard Gaussian concentrates on the unit sphere and converges to uniform on it as K grows (norm^2 is chi-squared with mean 1, variance 2/K).
2. Eq (1): density of f(X) at f(x) is an integral over the level set {u : f(u) = f(x)} of p_X / prod_k sigma_k(J_f), Hausdorff measure; f need not be bijective.
3. Lemma 2 (p3, `thm:general_density`): for f(X) ~ N(0, I/K) at large K, f must learn p_X "up to mean-preserving rescaling within each level-set".
4. Eq (2) JEPA loss = predictive invariance (dist(Pred(Enc(x^(1))), Enc(x^(2)))) + diversity(Enc(x_n)). The invariance term makes the level sets of Enc coincide with the view-generating transformations; the proof of Theorem 1 argues the predictor does not change the encoder's level set (invariance must hold for every mask^(1) realization).
5. Eq (3) data model p_X = p_mu (x) p_T: samples are transformations T of generators mu; in practice the generators are the un-augmented training images.
6. Theorem 1 (p4) eq (4): at optimality p_mu(mu) is proportional to the inverse of E_{p_T}[1 / prod_k sigma_k(J_f(mu, T))].
7. Eq (5): JEPA-SCORE(x) = sum_k log sigma_k(J_f(x)), single-sample Monte Carlo estimate of (4) in log scale. Listing 1 (p7): `jacobian` of `model(x).sum(0)`, flatten, `svdvals`, clip at eps = 1e-6, log, sum.

## 6. Equations (TeX verbatim)
(1) `\label{eq:general}`, p2:
```
p_{f(X)}(f(\vx))=\int_{\{\vu \in \mathbb{R}^{D} | f(\vu)=f(\vx)\}} \frac{p_{X}(x)}{\prod_{k=1}^{\rank(J_{f}(\vx))}\sigma_k(J_{f}(\vx))} \mathrm{d} \mathcal{H}^{r}(\vx),
```
(index typos kept: integrand uses x and \vx while integrating over \vu). H^r = r-dimensional Hausdorff measure, r = dimension of the level set; sigma_k = singular values of the Jacobian J_f(x) in R^{K x D}.

(2) `\label{eq:JEPA}`, p3:
```
\mathcal{L}\triangleq&  \sum_{n=1}^{N} \mathbb{E}_{(\vx_n^{(1)},\vx_n^{(2)})\sim \mathcal{G}(\vx_n)}\left[{\rm dist}\left({\rm Pred}\left({\rm Enc}\left(\vx_n^{(1)}\right)\right),{\rm Enc}\left(\vx_n^{(2)}\right)\right)\right]&&(\text{predictive invariance})\nonumber\\
&+{\rm diversity}\left(\left({\rm Enc}\left(\vx_{n}\right)\right)_{n \in [N]}\right),&&(\text{anti-collapse}),
```
G = stochastic view generator; dist e.g. L2.

(3) `\label{eq:data}`: `p_{X} \triangleq p_{\mu}\otimes p_{T}.`

(4) Theorem 1 `\label{eq:JEPA_density}`, p4:
```
p_{\mu}(\mu) \propto \mathbb{E}_{p_T}\left[\frac{1}{\prod_{k=1}^{\rank(J_{f}(\vx))}\sigma_k(J_{f}(\mu,T))}\right]^{-1}.
```
"At optimality, JEPA embeddings estimate the data density as per" (4). Gloss: mu = generator (clean sample), T = transformation, expectation over augmentations.

(5) `\label{eq:JEPA_SCORE}`, p4:
```
\text{\bf JEPA-SCORE}(\vx)\triangleq  \sum_{k=1}^{\rank(J_{f}(\vx))}\log\left(\sigma_k(J_{f}(\vx)\right),
```
"which exactly recovers p_mu as long as the JEPA loss is minimized."

Density statement in the paper's words (Lemma 2): "In order for f(X) to be distributed as N(0, I/K) for large K, f must learn the data density p_X up to mean-preserving rescaling within each level-set."

## 7. Measurement setup (how the claim is validated)
- Synthetic (Fig 2 top-right, p3): Gaussian-mixture data; Pearson correlation between JEPA-SCORE and true log p(x), rows input dim {64, 128, 256, 512}, columns samples {512, 1024, 2048, 4096}. Values: dim 64: 0.64, 0.69, 0.84, 0.90; dim 128: 0.75, 0.85, 0.90, 0.94; dim 256: 0.82, 0.83, 0.69, 0.76; dim 512: 0.72, 0.75, 0.84, 0.88. Range 0.64 to 0.94; non-monotonic in samples for dim 256. Training recipe for this toy is not stated beyond "producing Gaussian embeddings".
- Langevin sampling (Fig 2 bottom): using JEPA-SCORE as a score, Langevin dynamics recovers a 2D data distribution; qualitative.
- Pretrained models (Fig 1 p1, Figs 5-6 pp8-9): DINOv2, I-JEPA (IN-1k), I-JEPA (IN-22k), MetaCLIP; ImageNet classes 21, 141, 147; 5 least and 5 most likely of 1,000 images per class. Claimed: same kinds of images (and the same images) at the extrema across models; for birds, high-score images show flying birds, low-score seated.
- Out-of-domain histograms (Fig 3 p4): JEPA-SCORE for 5,000 samples each of ImageNet-1k, ImageNet-A, ImageNet-R, MNIST, Galaxy; MNIST and Galaxy score lower, more so for models trained on more data ("all models against IJEPA-1k").
- Listing 1: eps = 1e-6, "not sensitive to the choice of eps".
- Held fixed: nothing is retrained; scores come from released checkpoints. No quantitative metric on real models (no AUROC for outlier detection, no curation benchmark).

## 8. Results worth remembering
- Pearson up to 0.94 (dim 128, 4096 samples) and as low as 0.64 (dim 64, 512 samples), Fig 2 table.
- Qualitative agreement across four heterogeneous models (DINOv2, I-JEPA 1k/22k, MetaCLIP) on which ImageNet images are typical (Fig 1).
- Unseen domains get lower JEPA-SCORE (Fig 3).

## 9. Limitations the paper states or that follow from its setup
- "At optimality" only: the score "exactly recovers p_mu as long as the JEPA loss is minimized"; no bound for approximate optimality.
- Level-set ambiguity: with a non-injective f, p_X is determined only up to rescaling inside each level set (Lemma 2 proof).
- Large-K asymptotics (Lemma 1); conclusion calls the large-scale results "qualitative" and "only a first step".
- Cost: a full Jacobian (K x D) plus SVD per image.
- Not tested on LeJEPA itself (predates it by five weeks); LeJEPA targets N(0, I), this paper's normalization is N(0, I/K): same shape up to scale.

## 10. Outline feedback
- The outline's companion line "the anti-collapse term estimates data density: JEPA-SCORE" is accurate. Cite Lemma 2 (p3) for the statement, Theorem 1 eq (4) and eq (5) (p4) for the estimator.
- What would falsify it: on the synthetic GMM, Pearson between JEPA-SCORE and log p(x) near 0; or MNIST/Galaxy not scoring below ImageNet on ImageNet-trained models; or Langevin samples from JEPA-SCORE not matching the 2D data. Also, if models trained with different anti-collapse terms disagreed on the typical/atypical ImageNet images, the "model-agnostic" claim would fail.
- For the claim card, the numbers to show: 0.64-0.94 Pearson, 4 models, 5,000 samples per dataset.
