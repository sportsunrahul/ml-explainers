# Understanding: diffusion models, 2015-2024

Per-paper detail (verbatim TeX, equation numbers, glosses, macros) lives in `notes/papers/<slug>.md`.
Every number the page shows lives in `notes/claims.json` (checked by `notes/factcheck-*.md`).
This file is the through-line: what each paper changed and why.

## Core problem
Sample from an unknown data distribution p_data given only examples. Earlier families each paid a
price: GANs (adversarial game: unstable, mode dropping), VAEs (blurry, loose bound), autoregressive
models (sequential, slow), normalizing flows (invertible architectures only).

## The key idea (Sohl-Dickstein et al. 2015, 1503.03585v8)
Destroying structure is easy and needs no learning: a Markov chain of small Gaussian steps turns any
data distribution into an isotropic Gaussian. If steps are small, each reverse step is approximately
Gaussian too (same functional form, Kolmogorov forward/backward), so the reverse chain can be
learned step by step, trained with a variational bound. Demoed on a 2D swiss roll, MNIST, CIFAR-10.
Sample quality lagged; the idea waited five years.

## DDPM (2006.11239v2): the recipe that worked
- Forward: q(x_t|x_{t-1}) = N(sqrt(1-beta_t) x_{t-1}, beta_t I) (Eq. 2); linear beta 1e-4 -> 0.02,
  T = 1000. Closed form q(x_t|x_0) = N(sqrt(abar_t) x_0, (1-abar_t) I) (Eq. 4), abar_t = prod(1-beta_s).
- Posterior q(x_{t-1}|x_t,x_0) is Gaussian with known mean/variance (Eqs. 6-7).
- Reverse mean parametrized through a noise predictor eps_theta (Eq. 11); the per-step bound term
  becomes a weighted MSE on the noise (Eq. 12), which resembles denoising score matching over
  noise scales. Dropping the weights gives L_simple (Eq. 14): E ||eps - eps_theta(x_t, t)||^2.
- Sampling = Algorithm 2 (Langevin-like: subtract scaled predicted noise, add sigma_t z).
- Results: CIFAR-10 FID 3.17, IS 9.46 (Table 1); good samples but NLL worse than likelihood models
  (3.75 bits/dim for L_simple). Samples form coarse-to-fine (plot_data/cifar10_sample_quality.csv).
- Pain point: 1000 sequential network calls per sample.

## Score view: NCSN (1907.05600v3) -> Score SDE (2011.13456v2)
- Score = grad_x log p(x): an arrow field toward higher density; needs no normalizing constant.
- NCSN: score estimation fails where data is absent (low density, data on a manifold). Fix: perturb
  data at many noise levels (geometric sigma sequence), learn one noise-conditional score network
  with denoising score matching, sample by annealed Langevin dynamics (large sigma -> small).
- Link: eps_theta = -sqrt(1-abar_t) * score. DDPM and NCSN learn the same thing.
- Score SDE: take the number of noise levels to infinity. Forward SDE dx = f dt + g dw; reverse-time
  SDE uses the score; NCSN = variance-exploding (VE) SDE, DDPM = variance-preserving (VP) SDE.
  A deterministic probability-flow ODE shares every marginal -> exact likelihoods, encoding,
  any ODE solver. Predictor-corrector samplers. CIFAR-10 FID 2.20, and 1024x1024 samples.

## Faster sampling: DDIM (2010.02502v4) -> EDM (2206.00364v2)
- DDIM: a family of non-Markovian forward processes with the SAME marginals, so a DDPM-trained
  model works unchanged. Update = predicted x0 + direction pointing to x_t + optional noise (eta).
  eta = 0 is deterministic; can skip to a subsequence of timesteps. 10x-50x faster (abstract);
  consistent latents allow interpolation. Its alpha_t = DDPM's abar_t (notation trap).
  Equivalent to an Euler discretization of the (VE-form) probability-flow ODE.
- EDM: all prior models are one ODE in a sigma parametrization with different design choices
  (Table 1). Choose them well: sigma-schedule with rho = 7 concentrating steps at low noise,
  Heun 2nd order sampler (35 NFE on CIFAR-10), preconditioning c_skip/c_out/c_in/c_noise so the
  network always sees unit-variance inputs/targets, log-normal training-noise distribution.
  CIFAR-10 FID 1.97 uncond / 1.79 cond at 35 NFE; ImageNet-64 1.36 (retrained, 511 NFE).

## Better training + control: Improved DDPM (2102.09672v1), ADM (2105.05233v4), CFG (2207.12598v1)
- Improved DDPM: cosine schedule (linear destroys information too fast at low res; last steps
  waste compute), learned variance (interpolate beta_t and beta_tilde_t), L_hybrid; better NLL;
  good samples with far fewer steps (50-100).
- ADM: architecture search on the U-Net (attention at several resolutions, more heads, BigGAN
  residual blocks, adaptive group norm) + classifier guidance: shift the reverse mean by
  s * Sigma * grad log p_phi(y|x_t) from a classifier trained on noisy images. Scale trades
  diversity (recall) for fidelity (precision). Beat BigGAN-deep FID on ImageNet.
- CFG: no classifier. Train one model conditional and unconditional (drop label with p_uncond);
  guided eps = (1+w) eps(z,c) - w eps(z). Implicit classifier = difference of the two. FID best at
  small w (0.3) and rises after; IS keeps rising: the same tradeoff, now free. Became universal.

## Scale: LDM (2112.10752v2) -> DiT (2212.09748v2)
- LDM: pixels are mostly imperceptible detail. Train an autoencoder (perceptual + patch
  adversarial loss, KL or VQ reg.) with downsampling f; run diffusion in latent space. f = 4-16 is
  the sweet spot (f=1-2 slow, f=32 loses detail). Cross-attention injects a condition encoder
  tau_theta(y) (text). Basis of Stable Diffusion.
- DiT: replace the U-Net with a ViT over latent patches (patch size p -> (I/p)^2 tokens);
  condition via adaLN-Zero (beats in-context and cross-attention). Quality tracks Gflops
  (not params). DiT-XL/2 ImageNet-256 FID 2.27 with CFG.

## Straight paths: Rectified Flow (2209.03003v1), Flow Matching (2210.02747v2) -> SD3 (2403.03206v1)
- RF: interpolate X_t = t X_1 + (1-t) X_0 between a noise sample and a data sample; regress a
  velocity v(X_t, t) onto X_1 - X_0 (least squares). The learned ODE flow cannot cross itself, so it
  re-pairs noise/data into a non-crossing coupling with no higher transport cost. "Reflow" retrains
  on those pairs -> straighter paths -> fewer Euler steps; distill to one step.
- FM: train continuous normalizing flows without simulation. Conditional flow matching loss
  (per-sample path p_t(x|x1), vector field u_t(x|x1)) has the same gradient as the intractable
  marginal loss. Diffusion paths are one choice; the optimal-transport path (straight lines) trains
  faster and samples with fewer NFE.
- SD3: rectified flow + MM-DiT (separate weights for text and image token streams, joint attention),
  three text encoders (2 CLIP + T5), logit-normal timestep sampling (more weight mid-noise) won a
  large formulation comparison; resolution-dependent timestep shift; validation loss falls smoothly
  with model size and tracks human preference; 16-channel autoencoder; up to 8B params.
  Time direction flipped vs RF/FM (t=0 data).

## One step: Consistency Models (2303.01469v2)
- Points on one probability-flow ODE trajectory share an endpoint. Learn f(x_t, t) -> x_0 with
  boundary condition f(x, eps) = x (enforced by c_skip/c_out). Consistency distillation (CD) from
  a pretrained diffusion model, or consistency training (CT) from scratch. One-step sampling; can
  alternate noise-and-jump for multistep refinement; zero-shot editing. Beat progressive
  distillation at one step.

## Limitations -> who fixed them (the arc)
- 1000 steps -> DDIM, EDM, consistency models, rectified flow/reflow.
- Linear schedule wastes steps -> cosine (Improved DDPM), EDM rho schedule, logit-normal (SD3).
- No control -> classifier guidance -> CFG.
- Pixel-space cost -> LDM latents; U-Net scaling -> DiT / MM-DiT.
- Curved paths -> rectified flow / flow matching (OT path).
