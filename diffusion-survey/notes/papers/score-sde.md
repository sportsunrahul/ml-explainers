# Score SDE: Score-Based Generative Modeling through Stochastic Differential Equations

Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole. ICLR 2021. arXiv 2011.13456v2. Sources: `sources/score-sde/tex/`. main.tex inputs abstract, intro, background, sdes_new, sampling, cond_gen, conclusion, appendix_new. `method.tex` and `appendix.tex` are NOT input by main.tex (stale drafts); the `score sde (v2)_arXiv/` subdir is an identical duplicate. All citations below are to the input files only.

## Core problem

Two successful families, SMLD (NCSN: denoising score matching + annealed Langevin) and DDPM (a reverse Markov chain trained with a reweighted ELBO), both perturb data with a finite ladder of noise levels. They look different, have ad hoc samplers, give no exact likelihood, and it is unclear how they relate.

## What prior methods did and why that hurt

- SMLD (Background Eq. 1-2): finite geometric sigma ladder, Langevin per level; many hand-tuned steps, no likelihood.
- DDPM (Eq. 3-4): finite beta ladder, ancestral sampling; the sampler is tied to one discretization, only an ELBO (bound) on likelihood.
- Both: fixed number of noise scales, so you cannot trade steps for quality freely, and deriving a sampler for a new noise process is non-trivial (sampling.tex, Sec. 4.1).

## The one key idea

Let the number of noise scales go to infinity: noising becomes a continuous-time SDE dx = f dt + g dw. By Anderson (1982) its time reversal is another SDE whose only unknown is the score grad_x log p_t(x). Learn that score with a time-conditioned network (denoising score matching), then generate by numerically solving the reverse SDE. SMLD and DDPM are just two particular discretized SDEs (VE and VP).

## Delta from predecessors: unification of SMLD and DDPM

- SMLD noise chain x_i = x_{i-1} + sqrt(sigma_i^2 - sigma_{i-1}^2) z (Eq. 8) -> Variance Exploding (VE) SDE, Eq. (9), f = 0, g = sqrt(d sigma^2/dt). Variance grows without bound as t -> infinity.
- DDPM noise chain x_i = sqrt(1 - beta_i) x_{i-1} + sqrt(beta_i) z (Eq. 10) -> Variance Preserving (VP) SDE, Eq. (11), f = -beta(t) x / 2, g = sqrt(beta(t)). Unit variance is preserved if the data have unit variance.
- New: sub-VP SDE, Eq. (12), variance always bounded by VP; best likelihoods.
- Both SMLD and DDPM objectives (Eq. 1 and 3) are weighted sums of denoising score matching losses with weights sigma_i^2 and (1 - alpha_i), both proportional to 1/E||grad log p(x_tilde|x)||^2; the continuous objective Eq. (7) generalizes both.
- Samplers: SMLD = identity predictor + annealed Langevin corrector; DDPM = ancestral-sampling predictor + identity corrector. Both are special cases of Predictor-Corrector sampling (sampling.tex, Sec. 4.2).
- Notation traps: here sigma_1 is the SMALLEST noise (sigma_min) and sigma_N the largest, the reverse of NCSN. And alpha_i here means the cumulative product prod(1 - beta_j), which DDPM calls alpha-bar. Langevin is written epsilon_i s + sqrt(2 epsilon_i) z (NCSN wrote alpha/2 s + sqrt(alpha) z; same thing).

## Mechanism, step by step

1. Choose a forward SDE (VE, VP, or sub-VP). Its affine drift gives Gaussian transition kernels p_0t(x(t)|x(0)) in closed form, so any x(t) can be sampled in one shot from x(0).
2. Train s_theta(x, t) with the continuous denoising score matching objective Eq. (7): sample t ~ U[0, T], noise x(0) to x(t), regress onto the kernel's score, weight lambda(t) proportional to 1/E||grad log p_0t||^2.
3. Plug s_theta into the reverse-time SDE Eq. (6) and integrate from t = T (prior sample) to t = 0.
4. Solvers: any general SDE solver (Euler-Maruyama etc.); "reverse diffusion" sampler that discretizes the reverse SDE the same way as the forward one; or Predictor-Corrector (Algorithm 1 in App. G): each step a numerical SDE solver predicts x at the next time, then M steps of score-based MCMC (Langevin) correct the marginal at that time.
5. Alternatively solve the deterministic probability flow ODE Eq. (13), which has the same marginals p_t as the SDE. It is a neural ODE, so it gives exact likelihood via the instantaneous change of variables (App. D.2, Eq. 39, with the Skilling-Hutchinson trace estimator), invertible encoding x(0) <-> x(T) (interpolation, temperature scaling), uniquely identifiable latents, and adaptive-step black-box ODE solvers (RK45) that cut function evaluations by over 90% without visible quality loss (Figure 3).
6. Controllable generation / inverse problems (cond_gen.tex, Sec. 5): replace the score by the conditional score grad log p_t(x) + grad log p_t(y|x) (Eq. 14), e.g. with a time-dependent classifier for class-conditional samples, plus inpainting and colorization.
7. Architectures: NCSN++ (for VE) and DDPM++ (for VP/sub-VP); "cont." = trained with the continuous objective; "deep" = doubled depth.

## Equations for the explainer

Macros (math_commands.tex / main.tex): `\bfx`=\mathbf{x}, `\bff`=\mathbf{f}, `\bfw`=\mathbf{w}, `\bfs`=\mathbf{s}, `\bftheta`={\boldsymbol{\theta}}, `\bfI`=\mathbf{I}, `\ud`=\mathrm{d}, `\mbb{#1}`=\mathbb{#1}, `\norm{#1}`=\left\lVert#1\right\rVert, `\pd`=p_{\mathrm{data}}, `\sm`=\sigma_{\text{min}}, `\sM`=\sigma_{\text{max}}, `\bbeta`=\bar{\beta}, `\bbetam`=\bbeta_{\text{min}}, `\bbetaM`=\bbeta_{\text{max}}, `\tilde{\bff}_\bftheta` is plain.

### Eq. (5) Forward SDE (sdes_new.tex, label eqn:forward_sde)
```tex
    \ud \bfx = \bff(\bfx, t) \ud t + {g}(t) \ud \bfw,
```
- x = x(t): the data point being gradually noised, t in [0, T]; x(0) ~ p_0 (data), x(T) ~ p_T (prior, e.g. a Gaussian).
- f(x, t): drift, a deterministic push (e.g. shrink toward 0 in VP).
- g(t): scalar diffusion coefficient, how much noise is injected per unit time.
- w: standard Wiener process (Brownian motion); dw is an infinitesimal Gaussian kick.

### Eq. (6) Reverse-time SDE, Anderson 1982 (sdes_new.tex, label eqn:backward_sde)
```tex
    \ud \bfx = [\bff(\bfx, t) - g(t)^2  \nabla_{\bfx}  \log p_t(\bfx)] \ud t + g(t) \ud \bar{\bfw},
```
- Time now flows from T back to 0; dt is an infinitesimal negative step.
- grad_x log p_t(x): score of the noised marginal at time t, the ONLY unknown; replaced by s_theta(x, t).
- -g^2 * score: the extra drift that pulls samples toward high density, undoing the noise.
- w-bar: a Brownian motion running backward in time.

### Eq. (7) Continuous training objective (sdes_new.tex, label eqn:training)
```tex
   \bftheta^* = \argmin_\bftheta 
   \mbb{E}_{t}\Big\{\lambda(t) \mbb{E}_{\bfx(0)}\mbb{E}_{\bfx(t) \mid \bfx(0) }
   \big[\norm{\bfs_\bftheta(\bfx(t), t) - \nabla_{\bfx(t)}\log p_{0t}(\bfx(t) \mid \bfx(0))}_2^2 \big]\Big\}%
   .
```
- t ~ Uniform[0, T]; lambda(t) > 0 weighting.
- p_0t(x(t) | x(0)): Gaussian transition kernel of the forward SDE; its score is known in closed form.
- s_theta(x(t), t): time-dependent score network.

### Eq. (9) VE SDE, continuum of SMLD (sdes_new.tex, label eqn:ncsn_sde)
```tex
\ud \bfx = \sqrt{\frac{\ud \left[ \sigma^2(t) \right]}{\ud t}}\ud \bfw.
```
- sigma(t): noise std at time t (continuous version of NCSN's sigma ladder); no drift, so variance explodes.
- Instantiated in App. C as sigma(t) = sigma_min (sigma_max/sigma_min)^t, t in (0, 1].

### Eq. (11) VP SDE, continuum of DDPM (sdes_new.tex, label eqn:ddpm_sde)
```tex
\ud \bfx = -\frac{1}{2}\beta(t) \bfx~ \ud t + \sqrt{\beta(t)} ~\ud \bfw.
```
- beta(t): continuous noise rate (limit of N * beta_i); instantiated as beta(t) = beta-bar_min + t (beta-bar_max - beta-bar_min).
- -beta x / 2: shrinks the signal exactly as fast as noise is added, keeping total variance fixed.

### Eq. (13) Probability flow ODE (sampling.tex, label eqn:flow)
```tex
    \ud \bfx = \Big[\bff(\bfx, t) - \frac{1}{2} g(t)^2\nabla_\bfx \log p_t(\bfx)\Big] \ud t,
```
- Same as Eq. (6) but with half the score term and NO noise; deterministic trajectories with the same marginals p_t(x) as the SDE.

### Eq. (39) Exact likelihood (appendix_new.tex, App. D.2, label eqn:likelihood)
```tex
    \log p_0(\bfx(0)) = \log p_T(\bfx(T)) + \int_0^T \nabla \cdot \tilde{\bff}_\bftheta(\bfx(t), t) \ud t,
```
- f-tilde_theta: the ODE's right-hand side with s_theta plugged in (Eq. 38).
- div f-tilde: how much the flow locally expands volume; integrating it along the ODE path converts the easy prior log-density into data log-density. Estimated with Skilling-Hutchinson.

Background context (background.tex): Eq. (1) SMLD objective, Eq. (2) SMLD Langevin, Eq. (3) DDPM objective, Eq. (4) DDPM ancestral sampling.

## Key hyperparameters

- VE (SMLD): sigma_min = 0.01; sigma_max chosen by "Technique 1" of NCSNv2 (numeric value not given in this paper) (App. C, appendix_new.tex).
- VP: beta-bar_min = 0.1, beta-bar_max = 20, matching Ho et al. (App. C).
- VE time interval [epsilon, 1] with epsilon = 1e-5 to avoid t = 0; ODE FIDs use epsilon = 1e-5 (VE) and 1e-3 (VP) (App. C, App. D.4).
- Table 1 sampler comparison: models trained with the original discrete objectives at 1000 noise scales; P2000 needs interpolation of noise scales.
- CIFAR-10 main results: PC sampler at 1000 steps, reverse diffusion predictor, one Langevin corrector step per predictor step with SNR 0.16 for VE; no corrector for VP (App. H).
- 1024x1024 CelebA-HQ: PC sampler at 2000 steps, one Langevin step per predictor, SNR 0.15, batch size 8, EMA 0.9999, ~2.4M iterations (App. H.3).
- Likelihood: RK45 (scipy solve_ivp) atol = rtol = 1e-5, averaged over five runs (App. D.2).

## Headline results

1. CIFAR-10 unconditional FID 2.20 and IS 9.89 with NCSN++ cont. (deep, VE), better FID than the best listed conditional model (Table 3; abstract).
2. 2.99 bits/dim on uniformly dequantized CIFAR-10 with DDPM++ cont. (deep, sub-VP), "without maximum likelihood training" (Table 2).
3. Exact likelihood improves on the ELBO for the same DDPM model: 3.28 vs <= 3.70/3.75 bits/dim (Table 2).
4. Adding one corrector step (PC1000) beats both P1000 and P2000 for every predictor; e.g. VE reverse diffusion 4.79 (P1000) -> 3.60 (PC1000); corrector-only C2000 is worst (20.43 VE, 19.06 VP) (Table 1).
5. Probability flow ODE with adaptive solver cuts score evaluations by over 90% without affecting visual quality (Sec. 4.3, Figure 3).
6. First high-fidelity 1024x1024 samples from a score-based model (CelebA-HQ, App. H.3, Figure 12). The paper itself notes visible flaws (e.g. facial symmetry).

## Limitations and later work

- Still slow: thousands of network evaluations for SDE/PC sampling; slower than GANs (conclusion.tex). Addressed by DDIM (concurrent), DPM-Solver, EDM, progressive distillation and consistency models. [later-work attributions not source-verified]
- Many sampler hyperparameters (SNR, number of corrector steps, epsilon cutoff); the conclusion calls for automatic tuning. EDM (Karras et al. 2022) systematizes this design space. [not source-verified]
- VE gives better samples, VP/sub-VP better likelihood; practitioners must experiment per domain (Sec. 4.4).
- Likelihood is on uniformly dequantized data; later work (e.g. Variational Diffusion Models, maximum-likelihood training of score models) improves it. [not source-verified]

## Official code

https://github.com/yang-song/score_sde (App. H, appendix_new.tex, "Code and checkpoints are open-sourced at").
