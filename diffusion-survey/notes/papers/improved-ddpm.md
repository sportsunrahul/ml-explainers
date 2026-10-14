# Improved DDPM: Improved Denoising Diffusion Probabilistic Models

Nichol and Dhariwal (OpenAI). arXiv 2102.09672v1 (Feb 2021). ICML 2021 (`\usepackage[accepted]{icml2021}`, main.tex line 37).
Source: `sources/improved-ddpm/tex/main.tex` (line numbers below refer to this file).
Official code: https://github.com/openai/improved-diffusion (abstract, line 102).

## Core problem

DDPM gave great samples but (a) poor log-likelihood relative to other likelihood models, raising the question of whether it covers all modes; (b) it was unclear if it scales to diverse data like ImageNet; (c) sampling needs hundreds to thousands of forward passes (Sec. 1, line 107).

## What prior methods did and why that hurt

- DDPM fixed the reverse variance to sigma_t^2 = beta_t or tilde beta_t and trained on L_simple, which gives no learning signal for Sigma (line 175). Learning Sigma directly was unstable in DDPM. But the first few diffusion steps contribute most of the bound (Fig. 2, line 222), and that is exactly where beta_t and tilde beta_t differ (Fig. 1), so a fixed variance leaves likelihood on the table.
- DDPM's linear beta schedule destroys information too fast at 32x32 and 64x64: the last ~quarter of the forward chain is almost pure noise; skipping up to 20% of the reverse process barely hurts FID (Sec. 3.2, line 284).
- Optimizing the true bound L_vlb directly is very noisy with uniform t sampling (Sec. 3.3).
- Fewer-step sampling with fixed variances degrades quality badly (Sec. 4).

## The one key idea

Let the network learn the reverse-step variance, but only as a per-dimension interpolation (in log space) between the two known extremes beta_t and tilde beta_t, trained by a small weight on the true bound (L_hybrid). This improves likelihood and, surprisingly, lets you sample with far fewer steps with little quality loss, because the variance rescales automatically for a strided schedule.

## Delta from DDPM

1. Learned variance Sigma_theta via interpolation v (Eq. 15) plus L_hybrid = L_simple + 0.001 L_vlb, with stop-gradient on mu for the L_vlb term (Eq. 16, line 236).
2. Cosine noise schedule defined on alpha_bar_t (Eq. 17), offset s = 0.008.
3. Importance-sampled L_vlb (Eq. 18) to cut gradient noise; best NLL.
4. T = 4000 instead of 1000 (NLL 3.99 -> 3.77 on ImageNet 64, line 184).
5. Strided sampling with K < T steps using rescaled betas (Eq. 19).
6. Architecture tweaks (App. A): 4-head attention, attention at 16x16 and 8x8, adaptive group norm `GroupNorm(h)(w + 1) + b` for timestep conditioning; class conditioning via class embedding added to timestep embedding (Sec. 5).
7. New evaluations: ImageNet 64x64, precision/recall vs BigGAN, compute scaling.

## Mechanism, step by step

1. Same forward process as DDPM, but the noise level follows a cosine curve in alpha_bar_t so information is destroyed more evenly across steps; beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}, clipped at 0.999.
2. Network outputs two things per pixel: eps_theta (noise guess, used for the mean as in DDPM) and v (variance interpolation weight).
3. Loss = L_simple (drives the mean) + 0.001 x L_vlb (drives only the variance, since gradient to mu is stopped).
4. Alternatively, pure L_vlb with t drawn from an importance distribution p_t proportional to sqrt(E[L_t^2]) estimated from the last 10 values per t; gives the best NLL but worse FID.
5. At sampling time pick K evenly spaced timesteps from 1..T (rounded), recompute beta and tilde beta for that subsequence (Eq. 19); since Sigma_theta is an interpolation between these, it adapts to the shorter chain. 50-100 steps then suffice.

## Key equations (verbatim TeX from `sources/improved-ddpm/tex/main.tex`)

Macros: `\kld{#1}{#2}` = `D_{KL}(#1 || #2)` (line 27). Otherwise standard LaTeX.

### E1. Closed-form marginal, Eqs. (8)-(9), lines 147-151

Verbatim:
```
With $\alpha_t \coloneqq 1 - \beta_t$ and $\bar{\alpha}_t \coloneqq \prod_{s=0}^{t} \alpha_s$, we can write the marginal
    q(x_t|x_0) &= \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I}) \label{eq:marginal} \\
    x_t &=  \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon \label{eq:jumpnoise}
```
Note: the product index starts at s=0 here, versus s=1 in DDPM (DDPM main.tex line 164). Treat as a notational slip; the meaning is the same cumulative product. Gloss as in ddpm.md E2.

### E2. Learned variance interpolation, Eq. (15), line 226

Verbatim:
```
\Sigma_{\theta}(x_t, t) = \exp(v \log \beta_t + (1-v) \log \tilde{\beta}_t)
```
Gloss: Sigma_theta is the variance of the model's reverse step. beta_t (upper extreme) and tilde beta_t (lower extreme, Eq. 10: `\tilde{\beta}_t \coloneqq \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t`) are both known from the schedule. v is a network output, one number per pixel/channel. v=1 gives beta_t, v=0 gives tilde beta_t, in between gives a geometric blend. v is unconstrained but in practice stayed in range (line 229).

### E3. Hybrid objective, Eq. (16), line 233 (lambda from line 236)

Verbatim:
```
L_{\text{hybrid}} = L_{\text{simple}} + \lambda L_{\text{vlb}}
```
with `$\lambda = 0.001$`, and a stop-gradient on `$\mu_{\theta}(x_t, t)$` inside the `$L_{\text{vlb}}$` term.
Gloss: L_simple is DDPM's noise-prediction MSE (Eq. 14 here: `L_{\text{simple}} = E_{t,x_0,\epsilon}\left[ || \epsilon - \epsilon_{\theta}(x_t, t) ||^2 \right]`). L_vlb is the true variational bound, the sum of per-step KL terms (Eqs. 4-7). The small lambda keeps the noisy L_vlb from dominating; it mainly teaches the variance.

### E4. Cosine schedule, Eq. (17), line 288, plus beta conversion (line 290-291, unnumbered)

Verbatim:
```
\bar{\alpha}_t = \frac{f(t)}{f(0)},\;\; f(t) = \cos\left(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\right)^2
```
and
```
To go from this definition to variances $\beta_t$, we note that $
\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$. In practice, we clip $\beta_t$ to be no larger than 0.999 to prevent singularities at the end of the diffusion process near $t=T$.
```
Gloss: t/T is progress through the chain (0 to 1). s = 0.008 is a small offset chosen so sqrt(beta_0) is slightly below the pixel bin size 1/127.5 (line 295), preventing nearly-zero noise at the start. The squared cosine goes smoothly from 1 to 0 with a near-linear middle, so alpha_bar_t (fraction of signal variance remaining) falls gently near both ends. Dividing by f(0) makes alpha_bar_0 = 1. The squared-cosine shape was an arbitrary choice (line 295).

### E5. Importance-sampled L_vlb, Eq. (18), line 324

Verbatim:
```
L_{\text{vlb}} = E_{t \sim p_t}\left[\frac{L_t}{p_t}\right]\text{, where } p_t \propto \sqrt{E[L_t^2]}\text{ and }\sum p_t = 1
```
Gloss: instead of drawing t uniformly, draw it more often where the loss term is large (early steps), and divide by p_t to keep the estimate unbiased. E[L_t^2] is estimated from a history of the last 10 values per t; uniform sampling until every t has 10 samples (line 326).

### E6. Strided sampling variances, Eq. (19), line 430

Verbatim:
```
\beta_{S_{t}} = 1 - \frac{\bar{\alpha}_{S_t}}{\bar{\alpha}_{S_{t-1}}},\;\;\;\tilde{\beta}_{S_t} = \frac{1-\bar{\alpha}_{S_{t-1}}}{1-\bar{\alpha}_{S_t}} \beta_{S_t}
```
Gloss: S is the subsequence of timesteps used at sampling (K of them). Recompute the per-step noise from the training alpha_bar values at those timesteps; the learned v then interpolates between these rescaled bounds.

## Key hyperparameters

- T = 4000 for all Sec. 3 and later models (lines 184, 424). Baseline check at T=1000.
- Linear baseline at T=4000: beta_1 = 0.0001/4 to beta_4000 = 0.02/4 (App. A, line 541), to preserve alpha_bar shape.
- Cosine schedule s = 0.008, beta clip 0.999.
- lambda = 0.001; importance sampling history 10 per t.
- ImageNet 64 UNet: channels [C, 2C, 3C, 4C], C = 128, 3 res blocks per stage, ~120M params, ~39 GFLOPs forward (line 535). CIFAR: [C, 2C, 2C, 2C], C = 128, 3 res blocks; dropout 0.1 (linear) / 0.3 (cosine).
- Adam lr 1e-4, batch 128, EMA 0.9999 (line 539); batch 2048 for large class-conditional ImageNet 64.
- Sampling sweeps: K in {25, 50, 100, 200, 400, 1000, 4000} (line 446); K evenly spaced reals in [1, T], rounded.
- Class-conditional: small 100M params / 1.7M steps; large 270M params / 250K iterations; 250 sampling steps (line 484).

## Headline results

1. ImageNet 64x64 ablation (Table 1, lines 349-358; FID on 10K samples): DDPM setup T=1K linear L_simple NLL 3.99 / FID 32.5; T=4K 3.77 / 31.3; cosine + L_hybrid 3.62 / 28.0; cosine + L_vlb 3.57 / 56.7; at 1.5M iters cosine L_hybrid 3.57 / 19.2, L_vlb 3.53 / 40.1.
2. CIFAR-10 ablation (Table 2, lines 376-382; 500K iters): T=1K linear L_simple 3.73 / 3.29; T=4K linear L_simple 3.37 / 2.90 (best FID); cosine L_hybrid 3.17 / 3.19; cosine L_vlb 2.94 / 11.47.
3. NLL vs other likelihood models (Table 3): Improved DDPM 3.53 (ImageNet 64) and 2.94 (CIFAR) bits/dim, vs DDPM 3.77 / 3.70 and Sparse Transformer 3.44 / 2.80. Competitive with best conv models, worse than transformers. Note Table 3's DDPM CIFAR entry 3.70 is DDPM's true-bound model, not its L_simple model (3.75).
4. Fewer steps: L_hybrid models trained with 4000 steps give good samples with as few as 50 forward passes (line 113); "100 sampling steps is sufficient to achieve near-optimal FIDs" (line 447). Fixed-variance L_simple models degrade much more. DDIM is better below 50 steps, worse at 50 or more (line 449; also LSUN, App. B). Specific per-K FID values exist only in figure PDFs (fid_vs_steps*.pdf), not in the tex.
5. vs BigGAN-deep, class-conditional ImageNet 64 (Table 4, 50K samples): BigGAN-deep FID 4.06 / precision 0.86 / recall 0.59; Improved Diffusion small 6.92 / 0.77 / 0.72; large 2.92 / 0.82 / 0.71. Higher recall = better mode coverage.
6. Class-conditional ImageNet 256 (App. C table): two-stage (64 + upsampler) FID 12.3, single-stage 31.5, vs BigGAN-deep 7.0, VQ-VAE-2 38.1. Linear schedule worked better at 256x256 (line 560).
7. Scaling: FID vs training compute is roughly a power law on log-log; NLL scales less cleanly (Sec. 6).

## Limitations and who addressed them

- Still behind GANs at 256x256 (FID 12.3 vs 7.0) and behind transformers on NLL. Addressed by Dhariwal and Nichol 2021 ("Diffusion Models Beat GANs": architecture + classifier guidance), later classifier-free guidance, VDM (likelihood).
- Best NLL (L_vlb) comes with bad FID; trade-off unresolved (App. D ensemble: FID 19.9, NLL 3.52). VDM and later continuous-time likelihood training address it.
- 50 steps still slow; below 50 steps DDIM wins. Addressed by DDIM, DPM-Solver, progressive distillation, consistency models.
- Overfitting on CIFAR-10 and even class-conditional ImageNet 64; they recommend early stopping (App. F, G).
- Unconditional ImageNet 64 FIDs use only 10K samples, biased higher by about 2 points (line 486, 543), so they are for relative comparison only.
- Cosine schedule is tuned for low resolution; the linear schedule was better at 256x256.
