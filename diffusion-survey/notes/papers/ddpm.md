# DDPM: Denoising Diffusion Probabilistic Models

Ho, Jain, Abbeel (UC Berkeley). arXiv 2006.11239v2. NeurIPS 2020 (`neurips_2020` style, `[final]` option, main.tex line 15).
Source: `sources/ddpm/tex/main.tex` (all line numbers below refer to this file).
Official code: https://github.com/hojonathanho/diffusion (abstract, line 117).

## Core problem

Diffusion probabilistic models (Sohl-Dickstein et al. 2015) were "straightforward to define and efficient to train", but "there has been no demonstration that they are capable of generating high quality samples" (line 127). DDPM asks: can this model class produce state-of-the-art images?

## What prior methods did and why that hurt

- Sohl-Dickstein 2015: same Markov-chain framework, but trained on the plain variational bound with learned means and learned covariances, MLP / multi-scale-conv networks, and reported only log-likelihood bounds on small or toy datasets. Samples were not competitive (see sohl-dickstein.md).
- NCSN (Song and Ermon 2019), score matching + annealed Langevin dynamics: good samples (CIFAR-10 FID 25.32, Table 1), but per Appendix "Discussion on related work" (lines 669-678): (1) RefineNet with t fed only in norm layers or output; (2) no sqrt(1-beta_t) rescaling so variance grows with noise; (3) the final noise level does not fully destroy signal, so prior and aggregate posterior of x_T mismatch; (4) sampler coefficients (step size, noise scale) set by hand post hoc, so training does not directly optimize the sampler.
- GANs: best FIDs (StyleGAN2+ADA 3.26 unconditional) but no likelihood; not discussed as a flaw in the paper beyond being a comparison point.

## The one key idea

Parameterize the reverse-step mean so the network predicts the noise epsilon that was added (epsilon-prediction). With that parameterization, each term of the variational bound becomes a weighted denoising-score-matching loss, and the sampler becomes a Langevin-like update whose step sizes are derived from beta_t. Then drop the weights (L_simple), which trades likelihood for much better samples.

## Delta from predecessors

- vs Sohl-Dickstein: forward variances beta_t fixed to a linear schedule (not learned); reverse variance fixed to sigma_t^2 in {beta_t, tilde beta_t} (not learned), which are exactly the upper/lower entropy bounds from Sohl-Dickstein (line 195); network predicts epsilon instead of the mean directly; unweighted loss L_simple instead of the true bound; U-Net (PixelCNN++ backbone) with self-attention and sinusoidal time embedding instead of MLP/multi-scale conv with time "bump" readout; discrete decoder for x_0 so the bound is a lossless codelength on 8-bit data.
- vs NCSN: the four points above, and diffusion models "admit straightforward log likelihood evaluation, and the training procedure explicitly trains the Langevin dynamics sampler using variational inference" (line 544). DDPM also shows the equivalence: a certain weighted denoising score matching is variational inference for a Langevin-like sampler (line 545).

## Mechanism, step by step

1. Scale pixels from {0..255} linearly to [-1, 1] (line 271).
2. Forward process: T=1000 steps, each shrinks the image by sqrt(1-beta_t) and adds Gaussian noise of variance beta_t (Eq. 2). Because every step is Gaussian, you can jump straight to any x_t from x_0 in one shot (Eq. 4).
3. Training (Algorithm 1): pick a real image x_0, a random t uniform in {1..T}, a random noise eps; build x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps; ask the network eps_theta(x_t, t) to guess eps; take a gradient step on squared error (Eq. 14).
4. Why this is principled: the true bound compares the model's reverse step with the tractable forward posterior q(x_{t-1}|x_t,x_0) (Eqs. 5-7). Matching means reduces to matching noise (Eqs. 11-12) up to a per-t weight; L_simple drops that weight, which down-weights easy small-t terms (lines 298-299).
5. Sampling (Algorithm 2): start from pure noise x_T ~ N(0, I); for t = T..1 subtract the predicted noise (scaled), divide by sqrt(alpha_t), add fresh noise sigma_t z (none at the last step). 1000 network evaluations per sample.
6. At the end, display mu_theta(x_1, 1) without noise (line 285).

## Key equations (verbatim TeX from `sources/ddpm/tex/main.tex`)

Macros used (lines 46-81): `\bx`=`\mathbf{x}`, `\bI`=`\mathbf{I}`, `\bzero`=`\mathbf{0}`, `\bz`=`\mathbf{z}`, `\bmu`=`\boldsymbol{\mu}`, `\bepsilon`=`\boldsymbol{\epsilon}`, `\bSigma`=`\boldsymbol{\Sigma}`, `\defeq`=`\coloneqq`, `\Eb{#1}{#2}`=`\E_{#1}\!\left[#2\right]` with `\E`=`\mathbb{E}`, `\kl{#1}{#2}`=`D_{\mathrm{KL}}\!\left(#1 ~ \| ~ #2\right)`, `\grad`=`\nabla`.

### E1. Forward process step, Eq. (2), line 144

Verbatim:
```
q(\bx_{1:T} | \bx_0) &\defeq \prod_{t=1}^T q(\bx_t | \bx_{t-1} ), \qquad 
q(\bx_t|\bx_{t-1}) \defeq \mathcal{N}(\bx_t;\sqrt{1-\beta_t}\bx_{t-1},\beta_t \bI) \label{eq:forwardprocess}
```
Renderable: `q(\mathbf{x}_t|\mathbf{x}_{t-1}) \coloneqq \mathcal{N}(\mathbf{x}_t;\sqrt{1-\beta_t}\mathbf{x}_{t-1},\beta_t \mathbf{I})`

Gloss: x_{t-1} is the image one step less noisy; x_t the next, noisier one. beta_t is a small number (the noise variance added at step t). sqrt(1-beta_t) shrinks the previous image slightly so total variance stays about 1. I is the identity: every pixel gets independent noise. N(x; m, v) means "x is Gaussian with mean m and variance v". The product over t says the whole noising chain is just these steps applied in sequence.

### E2. Jump to any step in closed form, Eq. (4), lines 164-167

Verbatim (definitions line 164, equation line 166):
```
using the notation $\alpha_t \defeq 1-\beta_t$ and $\bar\alpha_t \defeq \prod_{s=1}^t \alpha_s$, we have
q(\bx_t|\bx_0) = \mathcal{N}(\bx_t; \sqrt{\bar\alpha_t}\bx_0, (1-\bar\alpha_t)\bI) \label{eq:q_marginal_arbitrary_t}
```
Reparameterized form used in the text (line 205, unnumbered): `\bx_t(\bx_0, \bepsilon) = \sqrt{\bar\alpha_t}\bx_0 + \sqrt{1-\bar\alpha_t}\bepsilon` for `\bepsilon \sim \mathcal{N}(\bzero, \bI)`.

Gloss: alpha_t = 1 - beta_t is the fraction of variance kept at step t. alpha_bar_t is the product of all alphas so far: how much of the original signal's variance survives to step t (starts near 1, decays toward 0). x_0 is the clean image. So x_t is "sqrt(alpha_bar_t) times the clean image plus sqrt(1 - alpha_bar_t) times fresh standard noise eps". This is what makes training cheap: no need to simulate t steps.

### E3. Forward-process posterior, Eqs. (6)-(7), lines 175-177

Verbatim:
```
q(\bx_{t-1}|\bx_t,\bx_0) &=  \mathcal{N}(\bx_{t-1}; \tilde\bmu_t(\bx_t, \bx_0), \tilde\beta_t \bI), \\
\text{where}\quad \tilde\bmu_t(\bx_t, \bx_0) &\defeq \frac{\sqrt{\bar\alpha_{t-1}}\beta_t }{1-\bar\alpha_t}\bx_0 + \frac{\sqrt{\alpha_t}(1- \bar\alpha_{t-1})}{1-\bar\alpha_t} \bx_t \quad \text{and} \quad
\tilde\beta_t \defeq \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t  \label{eq:q_posterior_mean_var}
```
Gloss: if you knew both the noisy x_t and the clean x_0, the previous step x_{t-1} is exactly Gaussian. Its mean tilde mu_t is a weighted blend of x_0 and x_t (weights depend only on the schedule). Its variance tilde beta_t is a bit smaller than beta_t. The model's reverse step p_theta(x_{t-1}|x_t) is trained to match this (the L_{t-1} KL terms of Eq. 5), but without access to x_0.

### E4. Epsilon parameterization of the reverse mean, Eq. (11), line 216

Verbatim:
```
\bmu_\theta(\bx_t, t) = \tilde\bmu_t\!\left(\bx_t,  \frac{1}{\sqrt{\bar\alpha_t}} (\bx_t - \sqrt{1-\bar\alpha_t}\bepsilon_\theta(\bx_t)) \right) = \frac{1}{\sqrt{\alpha_t}}\left( \bx_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \bepsilon_\theta(\bx_t, t) \right) \label{eq:mu_func_approx_langevin}
```
Gloss: eps_theta(x_t, t) is the neural network: given a noisy image and the step number, it guesses the noise. Plugging "x_0 estimated by removing the guessed noise" into the posterior mean formula gives this simple expression. Note the source drops the `, t` argument in the middle expression (`\bepsilon_\theta(\bx_t)`); it is the same network.

### E5. Per-step bound term = weighted noise regression, Eq. (12), line 225

Verbatim:
```
\Eb{\bx_0, \bepsilon}{ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar\alpha_t)}  \left\| \bepsilon - \bepsilon_\theta(\sqrt{\bar\alpha_t} \bx_0 + \sqrt{1-\bar\alpha_t}\bepsilon, t) \right\|^2} \label{eq:vb_term_langevin_eps}
```
Gloss: the exact variational-bound term for step t is a squared error between true noise and predicted noise, multiplied by a weight that depends on t (sigma_t^2 is the fixed reverse variance). The paper says this "resembles denoising score matching over multiple noise scales indexed by t" (line 227). Score/Langevin link: eps_theta acts as a (scaled, negated) learned gradient of the log data density, and Algorithm 2 "resembles Langevin dynamics with eps_theta as a learned gradient of the data density" (line 222). Explicitly, eps_theta(x_t,t) is proportional to -sqrt(1-alpha_bar_t) times the score of q(x_t); that scaling is standard but NOT written in this paper.

### E6. L_simple, Eq. (14), line 293

Verbatim:
```
 L_\mathrm{simple}(\theta) \defeq \Eb{t, \bx_0, \bepsilon}{ \left\| \bepsilon - \bepsilon_\theta(\sqrt{\bar\alpha_t} \bx_0 + \sqrt{1-\bar\alpha_t}\bepsilon, t) \right\|^2} \label{eq:training_objective_simple}
```
Gloss: same as E5 but with the weight thrown away; t uniform in 1..T. "Add noise, predict the noise, penalize squared error." theta = network weights. This is the entire training loop (Algorithm 1).

### E7. Sampling step, Algorithm 2 line 4, line 260 (not a numbered equation)

Verbatim:
```
\State $\bx_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( \bx_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}} \bepsilon_\theta(\bx_t, t) \right) + \sigma_t \bz$
```
with (line 259) `$\bz \sim \mathcal{N}(\bzero, \bI)$ if $t > 1$, else $\bz = \bzero$`, loop `\For{$t=T, \dotsc, 1$}` from `$\bx_T \sim \mathcal{N}(\bzero, \bI)$`.
Note: Algorithm 2 writes `1-\alpha_t`; the inline text form (line 220) writes `\beta_t`. Identical since alpha_t = 1 - beta_t.
Gloss: start from pure noise; at each step remove a scaled amount of the predicted noise, rescale up by 1/sqrt(alpha_t), then add back a little fresh noise sigma_t z (sigma_t^2 = beta_t or tilde beta_t, both worked similarly, line 195). No noise at the last step.

## Key hyperparameters

- T = 1000 for all experiments, "so that the number of neural network evaluations needed during sampling matches previous work" (line 304); not swept (line 656).
- beta_t linear from beta_1 = 1e-4 to beta_T = 0.02 (line 304, line 656); chosen from constant/linear/quadratic candidates, all with L_T ~ 0. L_T ~ 1e-5 bits/dim (line 304).
- sigma_t^2 = beta_t or tilde beta_t (fixed, not learned); the CIFAR model filename says "fixedlarge" suggesting beta_t for the reported model (inferred from filenames only, unverified).
- Network: U-Net, PixelCNN++ backbone (Wide ResNet), group norm, 2 residual blocks per resolution, self-attention at 16x16, sinusoidal time embedding added in each residual block (lines 306, 649). 32x32 models: 4 resolutions; 256x256: 6.
- Params: CIFAR-10 35.7M; LSUN/CelebA-HQ 114M; large LSUN Bedroom ~256M (line 649).
- Adam, lr 2e-4 (2e-5 for 256x256), batch 128 CIFAR / 64 larger, EMA 0.9999, dropout 0.1 on CIFAR (0 elsewhere), random horizontal flips (lines 657-661).
- Hardware/speed: TPU v3-8; CIFAR 21 steps/s at batch 128, 800k steps = 10.6 h; sampling 256 CIFAR images takes 17 s; 256^2 models sample 128 images in 300 s (line 651-652).

## Headline results

1. CIFAR-10 unconditional: IS 9.46 +/- 0.11, FID 3.17 (vs training set), NLL <= 3.75 bits/dim test (3.72 train). Table 1, row "Ours (L_simple)" (line 340). FID vs test set = 5.24 (line 369).
2. Training on the true bound instead: IS 7.67, FID 13.51, NLL <= 3.70 (3.69). Table 1, row "Ours (L, fixed isotropic Sigma)" (line 339). Better likelihood, worse samples.
3. Ablation (Table 2, lines 345-364): mu-prediction works only with the true bound (FID 13.22); learned diagonal Sigma hurts (mu-prediction trains but reaches only FID 23.69; eps-prediction with learned Sigma was unstable, as was mu-prediction with unweighted MSE); eps + L_simple best (FID 3.17).
4. LSUN 256x256 FID: Bedroom 6.36 (4.90 large model), Church 7.89, Cat 19.75 (Appendix Table "FID scores for LSUN", lines 590-591). "Sample quality similar to ProgressiveGAN" (abstract).
5. Lossy-compression view: rate 1.78 bits/dim, distortion 1.97 bits/dim = RMSE 0.95 on 0-255 scale; "More than half of the lossless codelength describes imperceptible distortions" (Sec. 4.3, line 416).
6. Progressive generation: large-scale features appear first, fine details last (line 498); rate-distortion table (lines 598-618).

Plot-ready data in the source: `sources/ddpm/tex/plot_data/cifar10_eps-fixedlarge-mse_step790000.csv` (columns incl. timestep, rmse_scaled, rate; rate-distortion vs time, Fig. 5) and `sources/ddpm/tex/plot_data/cifar10_sample_quality.csv` (timestep, is_mean, fid; progressive sample quality). The rate-distortion table (lines 604-615) gives 10 rows, e.g. at reverse step 1000: rate 1.77581, RMSE 0.95136; at 500: 0.00716, 38.03236.

## Limitations and who addressed them

- Log-likelihood not competitive (3.75 vs Sparse Transformer 2.80, Table 1; line 131). Addressed by Improved DDPM (learned variances, cosine schedule, L_hybrid / importance-sampled L_vlb; CIFAR 2.94) and later VDM (Kingma et al. 2021).
- Slow sampling: 1000 sequential network evaluations. Addressed by Improved DDPM (strided sampling with learned variances, ~50-100 steps) and DDIM (Song et al. 2020, deterministic non-Markovian sampler), later DPM-Solver, distillation.
- Learned reverse variances were unstable (Table 2). Addressed by Improved DDPM's log-space interpolation between beta_t and tilde beta_t.
- Linear schedule destroys information too fast at low resolution. Addressed by Improved DDPM's cosine schedule.
- Unconditional only; no guidance. Addressed by Dhariwal and Nichol 2021 (classifier guidance) and Ho and Salimans (classifier-free guidance).
- Pixel-space cost at high resolution. Addressed by latent diffusion (Rombach et al. 2022).
- Unified continuous-time view (SDE) came from Song et al. 2021.
