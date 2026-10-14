# Diffusion Models Beat GANs on Image Synthesis (ADM / ADM-G)

Dhariwal and Nichol, arXiv 2105.05233v4. Source: `sources/beat-gans/tex/main.tex` (single file). Code: https://github.com/openai/guided-diffusion (stated in abstract).

Short names used in the paper: ADM = "ablated diffusion model" (the improved architecture), ADM-G = ADM + classifier guidance, ADM-U = upsampling stack.

## Core problem

In mid 2021 GANs (BigGAN-deep, StyleGAN2) held state-of-the-art FID on hard datasets like ImageNet and LSUN. Diffusion models (DDPM, Improved DDPM) covered the data distribution better (higher recall) and trained stably, but their samples still looked worse by FID at high resolution, and there was no knob to trade diversity for fidelity the way BigGAN's truncation trick does.

## What prior methods did and why that hurt

- DDPM / Improved DDPM (IDDPM): UNet with a single-head global attention layer at 16x16 only, timestep embedding added into each residual block. On ImageNet 256 IDDPM needed a two-resolution upsampling stack and still reached only FID 12.26 vs BigGAN-deep 6.95 (Table 5).
- BigGAN-deep: great fidelity (precision 0.87 at 256) but low recall (0.28): it drops modes. GANs are also hard to train ("collapsing without carefully selected hyperparameters and regularizers", Sec 1).
- Diffusion models had no truncation analogue. Naive low-temperature sampling (scaling the score or the noise) gives blurry samples (App. G "Reduced Temperature Sampling"; also restated by the CFG paper Sec 3).

## The one key idea

Two ideas, one per axis:
1. Architecture: a systematically ablated, wider UNet with more attention (more heads, more resolutions), BigGAN-style up/down residual blocks, and adaptive group norm. This alone wins unconditional LSUN and ImageNet 64.
2. Classifier guidance: train a classifier on noisy images x_t, and at each sampling step nudge the Gaussian mean along the classifier's gradient toward class y, scaled by s. Larger s = sharper p(y|x)^s = higher fidelity, lower diversity. This is the missing "truncation knob" for diffusion.

## Delta from predecessors

- vs Improved DDPM: keeps IDDPM's learned variance (Eq. 1) and hybrid loss L_simple + lambda L_vlb, and uses DDIM when sampling with fewer than 50 steps. Changes the network: attention at 32,16,8 (was 16 only), 64 channels per head (was 1 head), BigGAN residual up/downsampling, AdaGN, width over depth. Adds classifier guidance. ImageNet 64 FID 2.92 -> 2.61 (ADM) -> 2.07 (ADM, dropout); ImageNet 256 12.26 (IDDPM, two-stage) -> 10.94 (ADM, single model, unguided) -> 4.59 (ADM-G) (Table 5).
- vs BigGAN-deep: better FID at every ImageNet resolution tested (128/256/512) with better recall, but at the cost of many forward passes per sample (250, or 25 with DDIM) instead of one. Classifier guidance is "strictly better than BigGAN-deep when trading off FID for Inception Score"; on precision/recall it is better only up to a precision threshold, beyond which BigGAN truncation reaches higher precision (Sec 4.3, Fig 5).

## Mechanism step by step

1. Train a class-conditional (or unconditional) diffusion model epsilon_theta(x_t, t) with learned variance Sigma_theta (IDDPM parameterization, Eq. 1), using the improved UNet. Timestep and class embeddings enter every residual block via AdaGN.
2. Separately train a classifier p_phi(y | x_t, t) on noised images from the same noise schedule. Architecture = the UNet's downsampling trunk + attention pool at 8x8; random crops to reduce overfitting (Sec 4.3).
3. Sampling (ancestral, Algorithm 1): at each step compute mu, Sigma from the diffusion model, compute g = grad_x log p_phi(y | x_t), and sample x_{t-1} from N(mu + s Sigma g, Sigma).
4. Sampling (DDIM, Algorithm 2): replace eps with eps_hat = eps_theta(x_t) - sqrt(1 - alpha_bar_t) grad log p_phi(y|x_t) and run the normal deterministic DDIM update. (Note: Algorithm 2 as printed has no explicit s in the eps_hat line even though s is an input; the scales in Table 14 are applied in practice.)
5. Why s > 1 is needed: with s = 1 on an unconditional model, the classifier gave the intended class ~50% probability but samples did not look like that class; scaling up fixed it (Sec 4.3, Fig 3: corgi FID 33.0 at s=1 vs 12.0 at s=10).
6. Why it is principled: s * grad log p(y|x) = grad log (1/Z) p(y|x)^s, i.e. guidance samples from a sharpened classifier distribution (Sec 4.3).
7. Optional: combine with an upsampling stack (ADM-U); guide only the low-resolution base model (Table 6).

## Equations (TeX copied verbatim from `tex/main.tex`)

Macro notes for rendering: `\grad` -> `\nabla` (main.tex line 59, NewDocumentCommand wrapping `\nabla`). `\kld{a}{b}` -> `D_{KL}(a \| b)` (line 50). Equation numbers from the compiled PDF.

### Eq. 1 (Sec 2.1 "Improvements") - learned variance inherited from IDDPM
```
\Sigma_{\theta}(x_t,t) &= \exp(v \log \beta_t + (1-v) \log \tilde{\beta}_t)
```
- Sigma_theta(x_t, t): variance of the Gaussian used to take one denoising step.
- v: a per-dimension number output by the network that interpolates (in log space) between two bounds.
- beta_t: the forward-noise variance at step t (upper bound on reverse variance).
- tilde beta_t: the posterior variance (lower bound), defined in the appendix (Eq. 19).

### Eq. 2 (Sec 4.1 "Conditional Reverse Noising Process") - conditioning by Bayes
```
p_{\theta,\phi}(x_t|x_{t+1},y) = Z p_{\theta}(x_t|x_{t+1})p_{\phi}(y|x_t)
```
- p_theta(x_t | x_{t+1}): the unconditional denoising step.
- p_phi(y | x_t): the noisy-image classifier's probability of label y.
- Z: normalizing constant.
- Plain words: to denoise toward class y, weight each candidate next image by how strongly the classifier believes it is y.

### Eq. 10 (Sec 4.1) - the approximation is a shifted Gaussian
```
&= \log p(z) + C_4, z \sim \mathcal{N}(\mu + \Sigma g, \Sigma)
```
(last line of the align ending in `\label{eq:condsamp}`; preceding lines are Eqs. 7-9.) With `g = \grad_{x_t} \log p_{\phi}(y|x_t)|_{x_t = \mu}` (defined in text after Eq. 6).
- mu, Sigma: mean and variance of the unguided denoising step.
- g: classifier gradient evaluated at the mean: "which direction in pixel space makes the image look more like class y".
- Sigma g: the shift, scaled by the step's own variance (small late in sampling, when Sigma is small).
- Assumption: log p_phi(y|x_t) has low curvature relative to Sigma^{-1} (first-order Taylor expansion, Eqs. 5-6).

### Algorithm 1 (Sec 4.1) - classifier guided diffusion sampling
```
\STATE Input: class label $y$, gradient scale $s$
\STATE $x_T \gets \text{sample from } \mathcal{N}(0, \mathbf{I})$
\FORALL{$t$ from $T$ to 1}
    \STATE $\mu, \Sigma \gets \mu_{\theta}(x_t), \Sigma_{\theta}(x_t)$
    \STATE $x_{t-1} \gets \text{sample from } \mathcal{N}(\mu + s \Sigma \grad_{x_t} \log p_{\phi}(y|x_t), \Sigma)$
\ENDFOR
\RETURN $x_0$
```
- s: gradient scale, the fidelity/diversity knob. s = 0 is unguided.
- x_T: pure Gaussian noise; x_0: final image.

### Eq. 14 (Sec 4.2 "Conditional Sampling for DDIM") - guidance as a modified noise prediction
```
\hat{\epsilon}(x_t) \coloneqq \epsilon_{\theta}(x_t) - \sqrt{1-\bar{\alpha}_t} \grad_{x_t} \log p_{\phi}(y|x_t)
```
- epsilon_theta(x_t): the model's predicted noise; relates to the score by grad log p_theta(x_t) = -eps_theta / sqrt(1 - alpha_bar_t) (Eq. 11).
- alpha_bar_t: cumulative signal fraction at step t; sqrt(1 - alpha_bar_t) is the noise std.
- hat epsilon: the noise prediction for the joint p(x_t) p(y|x_t). Survey bridge: CFG's Eq. 6 is exactly this form with the classifier gradient replaced by a difference of two noise predictions.

### AdaGN (Sec 3.1, unnumbered inline)
```
$\text{AdaGN}(h, y) = y_s  \text{ GroupNorm}(h) + y_b$
```
- h: activations in a residual block after its first convolution.
- y = [y_s, y_b]: scale and bias from a linear projection of the timestep and class embedding.

### Sharpening identity (Sec 4.3, unnumbered inline)
```
$s \cdot \grad_x \log p(y|x) = \grad_x \log \frac{1}{Z} p(y|x)^s$
```
- Plain words: multiplying the gradient by s is the same as raising the classifier's probabilities to the power s, which concentrates mass on the classifier's modes.

## Key hyperparameters

- 1000 diffusion steps for training; ImageNet sampled with 250 steps (IDDPM uniform stride) or 25 DDIM steps; LSUN with 1000 steps (Table 5 caption, App. I text).
- Final architecture (Sec 3.1 end): variable width, 2 residual blocks per resolution, 64 channels per head, attention at 32/16/8, BigGAN up/downsampling, AdaGN. Rescaling residuals by 1/sqrt(2) was tried and hurt (Table 1).
- Model sizes (Table 11): 296M (IN64), 422M (IN128), 554M (IN256), 559M (IN512); 256 base channels except IN64 (192); batch 256 (IN64: 2048); lr 1e-4 (IN64: 3e-4); iterations 4360K (IN128), 1980K (IN256), 1940K (IN512). Linear noise schedule except IN64 (cosine).
- Classifiers (Table 12): 43M-65M params, 300K-500K iterations. Tiny next to the diffusion model.
- Guidance scales used (Table 14): 250 steps: IN64 1.0, IN128 0.5, IN256 1.0, IN512 4.0. 25 DDIM steps: IN128 1.25, IN256 2.5, IN512 9.0.
- Adam/AdamW beta1 0.9 beta2 0.999, fp16 with loss scaling, EMA 0.9999, V100s (App. I).

## Headline results

1. ImageNet FID (Table 5), ADM-G vs BigGAN-deep: 128x128 2.97 vs 6.02; 256x256 4.59 vs 6.95; 512x512 7.72 vs 8.43. Recall is higher for ADM-G at every resolution (e.g. 256: 0.52 vs 0.28).
2. 25 DDIM steps (Table 5, "ADM-G (25 steps)" rows): FID 5.98 (128), 5.44 (256), 8.41 (512): roughly matches BigGAN-deep with 25 forward passes.
3. Guidance effect at 256x256 (Table 4): conditional unguided FID 10.94 / IS 100.98 / P 0.69 / R 0.63; + guidance s=1.0: FID 4.59 / IS 186.70 / P 0.82 / R 0.52; s=10.0: FID 9.11 / IS 283.92 / P 0.88 / R 0.32. Unconditional + s=10 reaches FID 12.00 (from 26.21 unguided).
4. Trade-off statement (Sec 4.3): "scaling the gradients beyond 1.0 smoothly trades off recall (a measure of diversity) for higher precision and IS (measures of fidelity). Since FID and sFID depend on both diversity and fidelity, their best values are obtained at an intermediate point." The smooth sweep is Figure 4 (ImageNet 128, figure only; no numeric table of the sweep exists in the source).
5. Guidance + upsampling (Table 6): 3.94 on ImageNet 256 and 3.85 on ImageNet 512 (ADM-G base + ADM-U, 250+250 steps).
6. Unconditional LSUN (Table 5): bedrooms FID 1.90 (vs StyleGAN 2.35), horses 2.57 (vs StyleGAN2 3.84), cats 5.57 (vs StyleGAN2 7.25).

Architecture ablations (Table 1, ImageNet 128, FID deltas vs baseline 15.33 at 700K): 4 heads -0.54, attention 32,16,8 -0.72, BigGAN up/down -1.20, rescale resblock +0.16 (worse), all combined -3.14. AdaGN 13.06 vs Addition+GroupNorm 15.08 (Table 3).

## Limitations and later fixes

- Stated (Sec 7): still slower than GANs at sampling because of multiple forward passes; points to distillation of DDIM (Luhman and Luhman) as future work.
- Requires a separately trained noise-aware classifier; cannot plug in an off-the-shelf ImageNet classifier (this is exactly what the CFG paper removes).
- Guidance lowers recall: diversity is sacrificed.
- Beyond a precision threshold, BigGAN truncation still reaches higher precision (Sec 4.3, Fig 5).
- Later fixes (external, not verified from this source): classifier-free guidance (Ho and Salimans 2022) removes the classifier; latent diffusion (Rombach et al.) cuts compute; progressive distillation / consistency models attack step count.

## Surprising or worth flagging

- The unguided *conditional* model at 256 (FID 10.94) is worse than BigGAN-deep; the guidance is what wins the headline.
- Best FID at 128 used s = 0.5, i.e. guidance below 1 (Table 14).
- Algorithm 2 prints no s in the eps_hat line.
- Guided IN512 uses s = 4.0 (250 steps) and 9.0 (25 DDIM steps): much larger than at lower resolutions.
