# recon-uninformative: Balestriero & LeCun, Learning by Reconstruction Produces Uninformative Features For Perception

Lens: analysis (claim card). Source of truth: `sources/recon-uninformative/src/{abstract,body,proofs}.tex`,
`main.tex`; `paper.pdf` for figure/equation/theorem numbers and pages.

## 1. Identity
- Title: Learning by Reconstruction Produces Uninformative Features For Perception
- arXiv: 2402.11337v1 (only version). Published (abs.html, v1): 2024-02-17.
- Authors: Randall Balestriero (first, "Independent"), Yann LeCun (last, NYU); 2 authors.
- Venue: ICML 2024 style file (`\usepackage[accepted]{icml2024}`, "Copyright 2024 by the author(s)");
  acceptance not stated in the arXiv version.
- Repo: none (meta.json repo: null).
- Macros used in quoted TeX (`src/math_commands.tex`): `\vx` = `{\bm{x}}`; `\mX`, `\mY`, `\mV`, `\mW`,
  `\mZ`, `\mP`, `\mD`, `\mH`, `\mA`, `\mG`, `\mS`, `\mI` = bold capitals (`{\bm{X}}` etc.);
  `\rmat{n}{m}` = `\mathcal{M}_{n,m}(\mathbb{R})`; `\argmin` = `\operatorname*{arg\,min}`.

## 2. Core problem
Why does reconstruction-based learning (AEs, denoising AEs, MAE) produce convincing reconstructions but
representations that need long training and fine-tuning to be competitive for perception? The paper argues
reconstruction allocates capacity to the top-variance subspace of pixel space, which is the least useful
subspace for classification, and that the useful subspace is learned last.

## 3. What prior methods did and why that hurt (per the paper)
- Reconstruction methods (DAE/VAE/MAE, autoregressive) learn a latent sufficient to recover the input;
  the loss of choice remains MSE in pixel space (Sec 2, p.2-3).
- Per the paper's characterisation of MAE (Sec 2, "Known limitations", p.3): "evaluation performance of
  the learned representation does not plateau even after 1600 epochs on Imagenet" and the gap with and
  without fine-tuning goes "from 70% to 84% on top-1 Imagenet classification". These are the paper's
  statements about MAE, not measurements in this paper.
- Three reasons named in Sec 1 (p.2): R1 Misaligned (top-variance features least informative), R2
  Ill-conditioned (perception features learned last), R3 Ill-posed (same reconstruction error, very
  different perception accuracy).

## 4. The one key idea
In the linear regime, the reconstruction-optimal encoder spans the top-K eigenvectors of X X^T (PCA), and
that subspace is provably learned first; perception needs the bottom-variance subspace, so the two
objectives are misaligned unless K is as large as D. A closed-form `alignment(k)` quantifies this, and the
same machinery shows masking noise can raise alignment while additive Gaussian noise cannot change the
supervised solution at all.

## 5. Mechanism, step by step
1. Set up a joint loss: supervised head W on top of a linear encoder V, plus lambda times a linear
   reconstruction loss through decoder Z (Eq (3)).
2. Solve it in closed form (Theorem 1, Eqs (4)-(6)); lambda -> 0 gives OLS, lambda -> infinity gives PCA
   (Corollary 1.1).
3. Perfect alignment iff the top-K eigenspaces of X^T X and Y^T Y intersect in dimension K (Proposition 1);
   "In practice ... never fulfilled" (p.4).
4. Define the continuous `alignment(k)` (Eq (7)); it is 0 iff misaligned, 1 iff aligned, increasing in k
   (Corollary 1.2). Evaluate it on 7 datasets (Fig 2).
5. Argue the linear analysis transfers: for any high-capacity encoder, the nonlinear loss (Eq (8)) is
   equivalent to the linear one at initialisation for any decoder, and always when the decoder is linear
   (Theorem 2).
6. Empirically project images onto the top or bottom PCA subspace, train a ResNet9 classifier on the
   filtered images, compare test accuracy (Fig 1 top, Fig 4).
7. Empirically track which eigen-directions a deep AE fits first during training (Fig 1 bottom, Fig 3).
8. Train paired resnet34 AEs on Imagenette with and without an added supervised head: same train and test
   reconstruction loss, very different probe accuracy (Figs 5, 6) -> R3.
9. Extend the closed form to denoising (Eq (9), Theorem 3). Corollary 3.1: additive isotropic Gaussian
   noise leaves W*^T V*^T unchanged for every sigma, "regardless of the supervised task". Fig 7: random
   masking with patch sizes (1,1), (2,2), (4,4) and drop probability 0% to 99% changes alignment, with
   dataset-dependent sign.

## 6. Equations (TeX verbatim from body.tex)
Eq (1), `eq:AE`, learning by reconstruction:
```
\min_{\theta} \mathbb{E}_{\vx \sim p_{\vx}}\left[d\left(g_{\theta}\left(f_{\theta}(\vx)\right), \vx\right)\right],
```
f_theta encoder R^D -> R^K, g_theta decoder R^K -> R^D, d a distance; linear case = PCA (p.2).

Eq (2), `eq:DAE`, denoising variant:
```
\min_{\theta} \mathbb{E}_{\vx \sim p_{\vx}}\left[\mathbb{E}_{\vx' \sim p_{\vx'|\vx}}\left[d\left(g_{\theta}\left(f_{\theta}(\vx')\right), \vx\right)\right]\right],
```
p_{x'|x} = noise transformation, e.g. x' ~ N(x, eps I) (p.3).

Eq (3), `eq:bilinear`, joint linear loss (Sec 3.1, p.3):
```
\mathcal{L}(\mV\hspace{-0.07cm},\mW\hspace{-0.07cm},\mZ) \hspace{-0.07cm}= &\| \mW^\top\mV^\top\hspace{-0.04cm}\mX \hspace{-0.07cm}- \hspace{-0.07cm}\mY\|_F^2
    + \lambda\hspace{-0.02cm}\|\mZ^\top\mV^\top\hspace{-0.07cm}\mX \hspace{-0.07cm}-\hspace{-0.07cm} \mX \|_F^2,
```
V in M_{K,D} encoder, Z in M_{D,K} decoder, W in M_{C,K} predictor head, X in M_{D,N} data, Y targets,
lambda >= 0 trade-off; "the latent representation, VX, is shared between the two losses".

Theorem 1 (Eqs (4)-(6), `eq:V`,`eq:W`,`eq:Z`), minimiser of Eq (3):
```
\mV^*  &\text{ spans } \mP_{\mX\mX^\top}\mD^{-\frac{1}{2}}_{\mX\mX^\top}(\mP_{\mH})_{.,1:K},\\
\mW^* =& \left({\mV^*}^\top\mX\mX^\top\mV^*\right)^{-1}{\mV^*}^\top\mX\mY^\top,\\
\mZ^* =& \left({\mV^*}^\top\mX\mX^\top\mV^*\right)^{-1}{\mV^*}^\top\mX\mX^\top,
```
with `\mH \triangleq \mD^{-\frac{1}{2}}_{\mX\mX^\top}\mP^\top_{\mX\mX^\top}\mA\mP_{\mX\mX^\top} \mD^{-\frac{1}{2}}_{\mX\mX^\top}`,
`\mA \triangleq \mX\left(\mY^\top\mY + \lambda \mX^\top\mX\right)\mX^\top`, and M = P_M D_M P_M^T the
eigendecomposition. Gloss: V* is the top-K eigenspace of a lambda-weighted mix of the label Gram matrix and
the data Gram matrix, whitened by the data; W*, Z* are least-squares heads on top of V*X.

Proposition 1 (p.4): "The supervised and reconstruction tasks are aligned (the optimal solutions do not
depend on $\lambda$) iff the intersection of the top-$K$ eigenspaces of $\mX^\top\mX$ and $\mY^\top\mY$ is
of dimension $K$."

Eq (7), `eq:alignment`, continuous alignment (Sec 3.1, p.4):
```
\text{alignment}(k) \triangleq \frac{\| \mY^\top\mY(\mP_{\mX\mX^\top})_{1:k}\|_F^2}{\|\mY^\top\mY\mP_{\mX\mX^\top}\|_F^2},
```
Paper's wording: "the (scaled) minimum supervised training error that can be achieved given the
representation (V^T X) minimizing the reconstruction loss, which is measured by how much of the matrix
Y^T Y can be reconstructed from the top-k subspace of X^T X". Corollary 1.2: increases with k; 0 iff
misaligned; 1 iff aligned. Gloss: numerator = energy of the label Gram matrix captured by the top-k data
eigenvectors; denominator = its total energy.

Eq (8), `eq:nonlinear`, nonlinear encoder/decoder with linear head (p.5):
```
\mathcal{L}(\mW\hspace{-0.04cm},\theta,\gamma) \hspace{-0.04cm}= &\| \mW^\top\hspace{-0.08cm} f_{\theta}(\mX)\hspace{-0.04cm} - \hspace{-0.06cm}\mY\|_F^2\hspace{-0.04cm}+\hspace{-0.04cm} \lambda\|g_{\gamma}\hspace{-0.04cm}\left(f_{\theta}(\mX)\hspace{-0.02cm}\right) \hspace{-0.04cm}- \hspace{-0.06cm}\mX \|_F^2.
```
Theorem 2 (p.5): "For any high-capacity encoder $f_{\theta}$, studying \cref{eq:bilinear} and
\cref{eq:nonlinear} is equivalent at initialization for any decoder, and is always equivalent when the
decoder is linear."

Eq (9), `eq:optimal_V`, denoising alignment (Sec 4.2, p.7):
```
\text{alignment}(k) \triangleq \min_{\mW}\| \mW^\top{\mV^*}^\top\mX - \mY\|_F^2,\nonumber\\
{\mV^*} = \argmin_{\mV}\min_{\mZ} \mathbb{E}_{\mX'|\mX}\left[ \| \mZ^\top\mV^\top\mX'- \mX\|_F^2\right],
```
Theorem 3 (p.8): `\mV^* \text{ spans } \mP_{\mG}\mD^{-\frac{1}{2}}_{\mG}(\mP_{\mH})_{.,1:K}` with
`\mH \triangleq \mD^{-\frac{1}{2}}_{\mG}\mP^\top_{\mG}\mS\mX^\top\mX\mS^\top\mP_{\mG} \mD^{-\frac{1}{2}}_{\mG}`,
`G \triangleq \mathbb{E}_{\mX'|\mX}[\mX'{\mX'}^\top]`, `\mS \triangleq \mathbb{E}_{\mX'|\mX}[\mX']`.
Corollary 3.1 (p.9): "additive Gaussian noise has no impact in the supervised task performance as
${\mW^*}^\top{\mV^*(\sigma)}^\top={\mW^*}^\top{\mV^*(0)}^\top,\forall \sigma \geq 0$, regardless of the
supervised task."
Note if quoting Corollary 1.1: its text says "The solution from Theorem 2 recovers the OLS solution"; the
source `\cref{thm:linear_decoder}` resolves to Theorem 2 but the sense is Theorem 1 (typo in the paper).

## 7. Measurement setup
| Experiment | Data | Model | Metric | Held fixed | Location |
|---|---|---|---|---|---|
| Closed-form alignment(k) sweep | 7 datasets: ArabicCharacters, ArabicDigits, SVHN, FashionMNIST, CIFAR10, CIFAR100, TinyImagenet | linear theory (Eq (7)) | task alignment (%) vs K and vs K/D | - | Fig 2, p.4 |
| Top vs bottom PCA subspace classification | TinyImagenet (Fig 1); ArabicCharacters, ArabicDigits, SVHN, CIFAR10, CIFAR100 (Fig 4) | ResNet9, trained and tested on projected images; with and without data augmentation | top-1 test accuracy vs explained variance kept | projection applied to train and test before training | Fig 1 top (p.1); Fig 4 (p.6) |
| Which subspace is learned first | CIFAR10 | deep AE: MLP depth 3/5, deconv depth 3/5 | reconstruction loss per eigen-direction, over training | - | Fig 1 bottom (p.1); Fig 3 (p.5) |
| Same reconstruction loss, different accuracy (R3) | Imagenette (Imagenet-10) | resnet34 AE, with vs without supervised head; several embedding dims | linear and nonlinear probe test accuracy; train and test rec loss | paired runs identical except the supervised signal | Fig 5 (p.7); Fig 6 (p.8) |
| Denoising alignment | ArabicCharacters, ArabicDigits, FashionMNIST, SVHN, CIFAR100 | linear closed form (Theorem 3) | relative alignment difference vs K | masking patch (1,1), (2,2), (4,4); drop prob 0% to 99% | Fig 7 (p.9) |
Baselines: there is no MAE or SSL model trained in the paper; comparisons are top vs bottom subspace,
reconstruction-only vs reconstruction+supervised, masking vs Gaussian noise.

## 8. Results worth remembering
- Abstract: TinyImagenet classification on images projected to the top subspace "explaining 90% of the
  pixel variance can be solved with 45% test accuracy. Using the bottom subspace instead, accounting for
  only 20% of the pixel variance, reaches 55% test accuracy." (Fig 1 top; y-axis ticks 0.9 to 58.7.)
- Fig 1 middle panels: example images at "bottom 25% variance" vs "top 75% variance"; Fig 9 (appendix,
  p.19) shows the same on Imagenet images.
- Sec 1 R3 (p.2): on Imagenet-10 "for a given reconstruction error the top-1 accuracy ... can vary from 50%
  to almost 90%" (Fig 5: linear test acc. roughly 40-50 for reconstruction-only vs roughly 88-90 with the
  supervised signal; read off the plot, not stated as numbers).
- Fig 2 findings (p.4): background-free images are aligned "even for small latent dimension, as low as 20%
  of the input dimension"; TinyImagenet alignment "only increases linearly with the latent space dimension
  K, requiring K = D"; CIFAR100 is less aligned than CIFAR10.
- Fig 4 (p.6): for every filtering level, keeping the bottom subspace beats keeping the top subspace, except
  for background-free datasets; holds with and without augmentation.
- Fig 6 (p.8): the AE with the supervised signal (orange) and without (blue) have indistinguishable train
  and test reconstruction loss and reconstructions, but very different linear and nonlinear test accuracy
  (plot: roughly 88 vs 60 linear; read off the plot).
- Corollary 3.1: Gaussian noise cannot help any supervised task (closed form).
- Fig 7: masking helps on poorly aligned datasets (CIFAR100), hurts on already-aligned ones (ArabicDigits)
  except with (1,1) patches; the benefit shrinks as K grows.
- Conclusion (p.9): the paper relates its finding to "MAE's performances going from about 50% to 74% on
  Imagenet top-1 accuracy when masking is employed" (the paper's statement; not measured here).

## 9. Limitations the paper states
- Theory is linear; Theorem 2 extends it only to initialisation (any decoder) or to linear decoders.
- Tuning the noise strategy "without knowledge of the perception task seems challenging" (abstract); the
  best mask patch size "var[ies] with the dataset" (p.8).
- Largest dataset is TinyImagenet (theory) / Imagenette (deep AE). MAE itself is never trained or probed.
- Fixed by later sources: none of the given sources re-tests this claim; I-JEPA / V-JEPA (sources/ijepa,
  sources/vjepa) move the loss out of pixel space, which is the design response. Not verified in this note.

## 10. Outline feedback (ch-ssl claim card)
1. Scope the central claim precisely. Paper shows: (a) the top-variance pixel subspace, which
   reconstruction fits first, is the least informative for classification (Fig 1 top, Fig 4); (b) that
   subspace is learned first and the useful one "exponentially slower" (Fig 1 bottom, Fig 3, Theorem 2);
   (c) same reconstruction loss can hide very different perception quality (Figs 5, 6); (d) masking is a
   valid guidance, Gaussian noise provably is not (Theorem 3, Corollary 3.1, Fig 7). It does not show that
   a converged MAE's features are uninformative; the MAE numbers quoted (70 -> 84 fine-tuning gap, 1600
   epochs, 50 -> 74 with masking) are the paper's characterisation of MAE, cite them as such.
2. Metric for the card: `alignment(k)` (Eq (7)) and ResNet9 top-1 on PCA-projected images (Fig 1, Fig 4).
   Headline numbers: 90% variance -> 45% acc; 20% variance -> 55% acc (abstract); Imagenet-10 50% to
   almost 90% at equal reconstruction loss (Sec 1); TinyImagenet needs K = D for alignment (Fig 2).
3. What would falsify it: the top-variance projection beating the bottom projection at matched explained
   variance on natural-image datasets; alignment(k) close to 1 at small k/D on TinyImagenet; a deep AE
   fitting bottom eigen-directions before top ones; W*^T V*^T changing with sigma under Gaussian noise.
4. The chapter's "slider adds pixel noise to the target: pixel loss grows, embedding loss does not" is a toy
   illustration; do not cite 2402.11337 for it. Corollary 3.1 is about input noise in a denoising AE, not
   target noise, and says the supervised solution is unchanged, not that a loss grows.
5. The paper's three camps (Sec 1): reconstruction, score matching, SSL (invariance + anti-collapse). The
   outline's three families (pixels, two views, JEPA) are a different cut; fine, but do not attribute the
   outline's taxonomy to this paper.
