# EDM -- Elucidating the Design Space of Diffusion-Based Generative Models

Karras, Aittala, Aila, Laine (NVIDIA), NeurIPS 2022. arXiv 2206.00364v2.
Sources read: sources/edm/tex/paper.tex, figures.tex (Table 1, Algorithms 1-2, Table 2),
supplemental-figures.tex (Tables 3-4), supplemental-body.tex (training details), macros.tex.

## Core problem

By 2022 there were several diffusion "families" (DDPM/VP, SMLD/VE, iDDPM, DDIM), each
derived from its own theory and each shipped as a tightly coupled package: the noise
schedule, the network input/output scaling, the training noise distribution, the loss
weights and the sampler all appeared to depend on each other. That made it unclear which
choices actually mattered, and sampling was slow (hundreds to thousands of network calls).

## What prior methods did and why that hurt

- Each method picked sigma(t) and a scaling s(t) from theory (e.g. VP's exponential-in-t^2
  schedule, VE's sqrt(t)), not from how well an ODE solver can follow the resulting
  trajectories. Curved trajectories need many small steps.
- Samplers used first-order Euler (or Euler-Maruyama) steps: O(h^2) local error per step,
  so many steps (VP 256 NFE, VE 8192 NFE, ADM ImageNet-64 250 NFE in the baselines).
- Networks were trained to predict the noise n scaled to unit variance, with
  D = x - sigma F. At large sigma any network error is amplified by sigma.
- Training noise levels and loss weights were chosen per theory, spreading effort over
  noise levels where nothing can be learned (very low or very high sigma).

## The one key idea

Rewrite every diffusion model as one probability-flow ODE in terms of the noise level
sigma, parameterised by a small table of independent, swappable functions (sigma(t), s(t),
time steps, solver, c_skip/c_out/c_in/c_noise, p_train, lambda). Then optimise each knob
empirically and independently. Best choices: sigma(t)=t, s(t)=1, Heun 2nd order steps,
rho=7 time steps, a preconditioned "predict something between x and the clean image"
network, and log-normal training noise.

## Delta from predecessors (Table 1 is the delta)

Table 1 ("Specific design choices employed by different model families", figures.tex
\tabSpecifics) puts VP (Song et al. Score SDE, DDPM-like), VE (Score SDE, SMLD-like),
iDDPM+DDIM, and EDM in four columns. Row by row:

| Row | VP | VE | iDDPM + DDIM | EDM |
|---|---|---|---|---|
| ODE solver | Euler | Euler | Euler | 2nd order Heun |
| Time steps t_{i<N} | 1 + i/(N-1) (eps_s - 1) | sigma_max^2 (sigma_min^2/sigma_max^2)^(i/(N-1)) | u_{floor(j0 + (M-1-j0)/(N-1) i + 1/2)} with recursive u_j | (sigma_max^(1/rho) + i/(N-1) (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho |
| Schedule sigma(t) | sqrt(e^(beta_d t^2/2 + beta_min t) - 1) | sqrt(t) | t | t |
| Scaling s(t) | 1/sqrt(e^(beta_d t^2/2 + beta_min t)) | 1 | 1 | 1 |
| Architecture F | DDPM++ | NCSN++ | DDPM | (any) |
| c_skip(sigma) | 1 | 1 | 1 | sigma_data^2/(sigma^2 + sigma_data^2) |
| c_out(sigma) | -sigma | sigma | -sigma | sigma sigma_data / sqrt(sigma_data^2 + sigma^2) |
| c_in(sigma) | 1/sqrt(sigma^2+1) | 1 | 1/sqrt(sigma^2+1) | 1/sqrt(sigma^2 + sigma_data^2) |
| c_noise(sigma) | (M-1) sigma^{-1}(sigma) | ln(sigma/2) | M - 1 - argmin_j abs(u_j - sigma) | (1/4) ln(sigma) |
| Noise distribution | sigma^{-1}(sigma) ~ U(eps_t, 1) | ln(sigma) ~ U(ln sigma_min, ln sigma_max) | sigma = u_j, j ~ U{0, M-1} | ln(sigma) ~ N(P_mean, P_std^2) |
| Loss weight lambda(sigma) | 1/sigma^2 | 1/sigma^2 | 1/sigma^2 (plus L_vlb) | (sigma^2 + sigma_data^2)/(sigma sigma_data)^2 |
| Parameters | beta_d=19.9, beta_min=0.1, eps_s=1e-3, eps_t=1e-5, M=1000 | sigma_min=0.02, sigma_max=100 | alpha_bar_j = sin^2(pi/2 j/(M(C2+1))), C1=0.001, C2=0.008, M=1000, j0=8 | sigma_min=0.002, sigma_max=80, sigma_data=0.5, rho=7, P_mean=-1.2, P_std=1.2 |

(Plain-text transcription of the table; M is the original number of training steps,
written \origN in the source.) Takeaway: DDIM already used sigma(t)=t, s(t)=1; VP and VE
are recovered by plugging their own sigma(t), s(t) into the same general ODE (Eq. 4).
Because the knobs are independent, EDM's sampler is a drop-in replacement for pre-trained
VP, VE and ADM models (Section 3), without retraining.

## Mechanism, step by step

1. Define p(x; sigma) = data blurred by Gaussian noise of std sigma. Pick sigma_max = 80 so
   p(x; sigma_max) is essentially pure noise.
2. The probability-flow ODE (Eq. 1) moves a sample between noise levels while keeping it
   distributed as p(x; sigma(t)). Its velocity uses the score grad log p.
3. The score is obtained from an ideal denoiser D: score = (D(x; sigma) - x)/sigma^2
   (Eq. 3). So "train a denoiser" = "learn the score".
4. Choose sigma(t)=t, s(t)=1. The ODE becomes dx/dt = (x - D(x; t))/t: the tangent always
   points at the denoiser's current guess of the clean image, so trajectories are nearly
   straight and large steps work.
5. Discretise noise levels with Eq. 5 (rho=7: short steps near sigma_min, long near
   sigma_max), final sigma_N = 0.
6. Integrate with Heun (Algorithm 1): Euler predictor, then average the slope at both ends
   (trapezoidal corrector); skip the corrector on the last step (sigma=0 would divide by
   zero). Cost: 2 network evaluations per step except the last, so NFE = 2N - 1
   (derived from Algorithm 1; 35 NFE corresponds to N = 18, 79 NFE to N = 40).
7. Optional stochastic sampler (Algorithm 2, Eq. 6): before each step, raise the noise
   level from t_i to t_i + gamma_i t_i by adding fresh noise ("churn"), then take a Heun
   step down. Controlled by S_churn, S_tmin, S_tmax, S_noise (grid-searched per model).
8. Network preconditioning (Eq. 7): D = c_skip x + c_out F(c_in x; c_noise). c_in and c_out
   make network inputs and training targets unit-variance; c_skip is chosen to minimise
   error amplification, so at low noise the net predicts a noise-like correction and at
   high noise it predicts the clean image directly.
9. Loss (Eq. 8) with lambda = 1/c_out^2 (effective weight 1 at every sigma) and training
   sigma drawn log-normal with P_mean = -1.2, P_std = 1.2, concentrating on the
   intermediate noise levels where loss can actually be reduced.
10. Optional non-leaky augmentation (from StyleGAN-ADA): augment before noising, pass the
    augmentation parameters as conditioning, set them to zero at inference.

## Equations for the explainer

Macro map (macros.tex): \xx = \boldsymbol{x}; \nablaxx = \nabla_{\hspace{-0.5mm}\xx};
\diff = \mathrm{d}; \odetime = t; \scale = s; \signal = \boldsymbol{y}; \noise =
\boldsymbol{n}; \boldzero = \mathbf{0}; \boldi = \mathbf{I}; \pdata = p_\text{data};
\ptrain = p_\text{train}; \sdata = \sigma_\text{data}; \cskip/\cout/\cin/\cnoise =
c_\text{skip}/c_\text{out}/c_\text{in}/c_\text{noise}; \smin/\smax =
\sigma_\stext{min}/\sigma_\stext{max} (\stext is a raised \text); \churn = \beta;
\wproc = \omega_t.

### Eq. (1), paper.tex, label eq:ode -- probability-flow ODE in sigma form

Verbatim:
```
\diff \xx = -\dot\sigma(t) ~\sigma(t) ~\nablaxx \log p \big( \xx; \sigma(t) \big) ~\diff\odetime \text{,}
```
Expanded (KaTeX-ready, macros substituted, spacing tildes dropped):
```
\mathrm{d}\boldsymbol{x} = -\dot\sigma(t)\,\sigma(t)\,\nabla_{\boldsymbol{x}} \log p\big(\boldsymbol{x}; \sigma(t)\big)\,\mathrm{d}t
```
Gloss: x = the image being generated; t = ODE time; sigma(t) = noise std at time t;
sigma-dot = how fast the noise level changes; grad_x log p(x; sigma) = score, a vector
pointing towards more likely (less noisy) images at that noise level. Running time
backwards (dt < 0) pushes x toward the data.

### Eqs. (2)-(3), paper.tex, labels eq:score, eq:scoredenoiser (one displayed line, two tags)

Verbatim:
```
\mathbb{E}_{\signal \sim \pdata} \mathbb{E}_{\noise \sim \mathcal{N}(\boldzero, \sigma^2 \boldi)} \lVert D(\signal + \noise; \sigma) - \signal \rVert^2_2
\text{,}\hspace*{2mm}\text{then}\hspace*{2mm}
\nablaxx \log p(\xx; \sigma) = \big( D(\xx; \sigma) - \xx \big) / \sigma^2 \text{,}
```
Expanded:
```
\mathbb{E}_{\boldsymbol{y} \sim p_\text{data}} \mathbb{E}_{\boldsymbol{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})} \lVert D(\boldsymbol{y} + \boldsymbol{n}; \sigma) - \boldsymbol{y} \rVert^2_2
\quad\Rightarrow\quad
\nabla_{\boldsymbol{x}} \log p(\boldsymbol{x}; \sigma) = \big( D(\boldsymbol{x}; \sigma) - \boldsymbol{x} \big) / \sigma^2
```
Gloss: y = clean training image; n = Gaussian noise with std sigma; D = denoiser that
guesses y from y + n. If D minimises the squared error (Eq. 2), then (D - x)/sigma^2 is
exactly the score (Eq. 3). Learning to denoise = learning the score.

### Eq. (5), paper.tex, label eq:discretization -- time steps with rho

Verbatim:
```
\sigma_{i<N} = \big( {\smax}^\frac{1}{\rho} + {\textstyle\frac{i}{N-1}} ( {\smin}^\frac{1}{\rho} - {\smax}^\frac{1}{\rho} ) \big)^\rho \hspace*{3mm}\text{and}\hspace*{3mm}\sigma_N = 0 \text{.}
```
Expanded:
```
\sigma_{i<N} = \Big( \sigma_\text{max}^{1/\rho} + \tfrac{i}{N-1} \big( \sigma_\text{min}^{1/\rho} - \sigma_\text{max}^{1/\rho} \big) \Big)^\rho \quad\text{and}\quad \sigma_N = 0
```
Gloss: N = number of steps; i = step index; sigma_max = 80, sigma_min = 0.002 (Table 1);
rho = 7 bends the spacing so steps are short near sigma_min (where errors hurt most) and
long near sigma_max. rho = 1 would be uniform; rho = 3 equalises truncation error but
rho in 5-10 gives better images (Sec. 3).

### Algorithm 1 (figures.tex \algHeun) -- Heun sampler, key lines

Verbatim (lines 4, 5, 7, 8 of the algorithm):
```
\dd_i \gets \bigg(\frac{\dot \sigma(\odetime_i)}{\sigma(\odetime_i)} + \frac{\dot \scale(\odetime_i)}{\scale(\odetime_i)}\bigg)\xx_i - \frac{\dot\sigma(\odetime_i)\scale(\odetime_i)}{\sigma(\odetime_i)} D_\theta\bigg(\frac{\xx_i}{\scale(\odetime_i)}; \sigma(\odetime_i)\bigg)
\xx_{i+1} \gets \xx_i + (\odetime_{i+1} - \odetime_{i}) \dd_i
\ddp_i \gets \bigg(\frac{\dot \sigma(\odetime_{i+1})}{\sigma(\odetime_{i+1})} + \frac{\dot \scale(\odetime_{i+1})}{\scale(\odetime_{i+1})}\bigg) \xx_{i+1}  - \frac{\dot\sigma(\odetime_{i+1})\scale(\odetime_{i+1})}{\sigma(\odetime_{i+1})} D_\theta\bigg(\frac{\xx_{i+1}}{\scale(\odetime_{i+1})}; \sigma(\odetime_{i+1})\bigg)
\xx_{i+1} \gets \xx_i + (\odetime_{i+1} - \odetime_{i}) \big( \frac{1}{2}\dd_i + \frac{1}{2}\ddp_i \big)
```
With sigma(t)=t, s(t)=1 this reduces to d_i = (x_i - D(x_i; t_i))/t_i (paper.tex Sec. 3,
"dx/dt = (x - D(x; t))/t"). Gloss: d_i = slope at the start of the step; Euler moves
along it; d'_i = slope at the landing point; the final update averages the two slopes.
Lines 6-8 (the correction) are skipped when sigma(t_{i+1}) = 0.

### Eq. (7), paper.tex, label eq:preconditioning

Verbatim:
```
D_\theta(\xx; \sigma) = \cskip(\sigma) ~\xx + \cout(\sigma) ~F_\theta \big( \cin(\sigma) ~\xx; ~\cnoise(\sigma) \big) \text{,}
```
Expanded:
```
D_\theta(\boldsymbol{x}; \sigma) = c_\text{skip}(\sigma)\,\boldsymbol{x} + c_\text{out}(\sigma)\,F_\theta\big( c_\text{in}(\sigma)\,\boldsymbol{x};\, c_\text{noise}(\sigma) \big)
```
EDM choices (Table 1, verbatim cells): `\sdata^2 / \left(\sigma^2 + \sdata^2 \right)`,
`\sigma \cdot \sdata / \sqrt{\sdata^2 + \sigma^2}`, `1 / \sqrt{\sigma^2 + \sdata^2}`,
`\frac{1}{4} \ln(\sigma)` for c_skip, c_out, c_in, c_noise; sigma_data = 0.5.
Gloss: F_theta = raw neural net; c_in rescales the noisy input to unit variance; c_noise
is how the net is told the noise level; c_skip says how much of the noisy input is passed
straight through; c_out scales the net's output. At small sigma, c_skip -> 1 (keep the
input, predict a small correction); at large sigma, c_skip -> 0 (predict the clean image).

### Eq. (8), paper.tex, label eq:precloss -- effective loss

Verbatim:
```
\mathbb{E}_{\sigma, \signal, \noise} \Big[
  \underbrace{\lambda(\sigma) ~ \cout(\sigma)^2}_{\text{effective weight}}
  \big\Vert
    \underbrace{F_\theta \big( \cin(\sigma) \cdot (\signal + \noise); \cnoise(\sigma) \big)}_{\text{network output}} -
    \underbrace{\tfrac{1}{\cout(\sigma)} \big(\signal - \cskip(\sigma) \cdot (\signal + \noise) \big)}_{\text{effective training target}}
  \big\Vert^2_2 \Big] \text{.}
```
Gloss: sigma ~ p_train (log-normal: ln sigma ~ N(-1.2, 1.2^2)); lambda(sigma) = loss weight,
set to 1/c_out^2 so the effective weight is 1 everywhere; the bracket shows what the raw
net is really asked to output.

Also available if needed: Eq. (4) (eq:odescale, general ODE with scaling s(t)) and
Eq. (6) (eq:sde, ODE + Langevin "churn" term) in paper.tex.

## Key hyperparameters

- sigma_min = 0.002, sigma_max = 80, sigma_data = 0.5, rho = 7 (Table 1).
- P_mean = -1.2, P_std = 1.2 (Table 1).
- Deterministic sampling: 35 NFE on CIFAR-10, 79 NFE at 64x64 (Table 2, NFE row).
- Stochastic knobs S_churn, S_tmin, S_tmax, S_noise: grid-searched per model (App.).
- CIFAR-10 training: ~200 Mimg, batch 512, ~400k iterations, lr 0.001 (supplemental-body.tex
  ~line 1299-1305). ImageNet-64: 32 A100s, batch 4096, ~2500 Mimg, ~600k iterations,
  lr 0.0001, two weeks (supplemental-body.tex ~line 1322-1324).

## Headline results

1. CIFAR-10 FID 1.79 class-conditional and 1.97 unconditional, both at 35 NFE with the
   deterministic sampler (Table 2, config F; text Sec. 5). Previous records 1.85 and 2.10.
2. Sampler alone (no retraining): NFE to reach near-best FID cut 7.3x for VP, 300x for VE,
   3.2x for DDIM/ADM (Sec. 3 Discussion; Table 3: VP 35 NFE, VE 27 NFE, ImageNet-64 79 NFE).
3. Sampler alone on pre-trained ADM ImageNet-64: FID 2.07 -> 1.55 with the tuned
   stochastic sampler at 511 NFE (Sec. 4 text; Table 4 "Optimal settings").
   Deterministic Heun + our t_i on the same model: 2.64 at 79 NFE (Table 3).
4. Retrained ImageNet-64 (ADM architecture unchanged, config E, no augmentation):
   FID 1.36, previous record 1.48 (Sec. 5 "ImageNet-64"); at NFE 511 with stochastic
   sampling (supplemental-figures.tex line 274, figure label only).
5. 26.3 CIFAR-10 images per second on one V100 (Sec. 3 Discussion).
6. With the improved CIFAR-10 models, any stochasticity hurt; ImageNet-64 still benefited
   from stochastic sampling (Sec. 5, Fig. 5b,c).

## Limitations and later fixes

- Stochastic sampling needs per-model grid search of 4 heuristics; too much churn loses
  detail and oversaturates colours, suspected from a slightly non-conservative learned
  denoiser (Sec. 4 Practical considerations). The authors warn that tuning stochastic
  samplers can bias model-evaluation conclusions.
- Architectures were largely left as-is (DDPM++, NCSN++, ADM); "the precise impact of
  architectural choices remains an interesting question" (supplemental-body.tex ~line 1348).
- Still ~35-79 network evaluations; one-step generation is out of scope (addressed by
  Consistency Models, 2303.01469, which reuse EDM's schedule, Heun solver and c_skip/c_out).
- Later work (not in this paper): EDM2 (Karras et al., 2312.02696) revisits architecture and
  training dynamics; noted from general knowledge, not verified here.

## Official code

https://github.com/NVlabs/edm (stated in paper.tex, end of Introduction).
