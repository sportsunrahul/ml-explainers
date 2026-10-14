# CM -- Consistency Models

Song, Dhariwal, Chen, Sutskever (OpenAI), ICML 2023. arXiv 2303.01469v2.
Sources read: sources/consistency/tex/main.tex (abstract), intro.tex, diffusion.tex,
consistency.tex, distillation.tex, generation.tex, experiments.tex, conclusion.tex,
appendix.tex (App. C experimental details, App. D editing), cmd.tex (macros).

## Core problem

Diffusion models sample by iteratively solving an ODE/SDE, needing 10-2000x more compute
than one-shot generators (GANs, VAEs, flows) (Sec. 1). The goal: one network evaluation
per sample, while keeping diffusion's perks: an optional compute-for-quality trade-off
and zero-shot editing (inpainting, colorization, super-resolution, ...).

## What prior methods did and why that hurt

- Faster ODE solvers (DDIM, DPM-Solver, DEIS) still need more than 10 steps for competitive
  samples (Sec. 2). E.g. on CIFAR-10: DDIM 8.23 FID at 10 NFE, 3-DEIS 4.17 at 10 NFE (Table 1).
- Distillation by synthetic data (Knowledge Distillation, DFNO, Rectified Flow + distill)
  first generates a large dataset of noise->image pairs with an expensive solver (Sec. 2).
- Progressive distillation (PD) avoids synthetic data but halves steps in repeated rounds;
  its 1-step quality is weak (CIFAR-10 FID 8.34, ImageNet-64 15.39; Tables 1-2).

## The one key idea

Learn a function f(x_t, t) that jumps from any point on a probability-flow ODE trajectory
directly to that trajectory's origin x_epsilon. Train it only through self-consistency:
outputs at two adjacent points of the same trajectory must match, anchored by the boundary
condition f(x, epsilon) = x. Then sampling is one call: x_eps = f(x_T, T).

## Delta from predecessors

- vs multi-step diffusion samplers: a diffusion model learns the ODE's velocity (score) and
  a solver must integrate it step by step; a consistency model learns the ODE's solution
  map (endpoint) directly, so 1 NFE suffices. Multi-step is still possible (Algorithm 1)
  by re-noising and re-applying f.
- vs progressive distillation: PD trains a student to match two teacher steps with one,
  then repeats (log2 rounds) (PD mechanism from Salimans and Ho 2022, not restated in CM;
  CM only notes PD needs no synthetic dataset). CD instead trains one model in one stage using single ODE
  solver steps of the teacher between adjacent discretization points, and is better at 1
  and 2 steps on every dataset except 1-step Bedroom with l2 (Sec. 6.2, Fig. 4).
- vs everything distillation-based: consistency training (CT) needs no pre-trained
  diffusion model at all, making CMs a standalone model family (Sec. 5).
- Builds heavily on EDM: same sigma(t)=t ODE, T=80, epsilon=0.002, rho=7 time steps,
  Heun solver, EDM architectures and c_skip/c_out form (Secs. 2, 4; App. C).

## Mechanism, step by step

1. Setup (EDM convention): PF ODE dx_t/dt = -t s_phi(x_t, t), t in [epsilon, T],
   T = 80, epsilon = 0.002, pixels in [-1, 1]; x_T ~ N(0, T^2 I) (Sec. 2, Eq. 3).
2. Consistency function f: (x_t, t) -> x_epsilon, with self-consistency
   f(x_t, t) = f(x_t', t') for points on one trajectory (Sec. 3 Definition).
3. Boundary condition f(x_epsilon, epsilon) = x_epsilon is built into the parametrisation
   f_theta = c_skip(t) x + c_out(t) F_theta(x, t), with c_skip(epsilon) = 1,
   c_out(epsilon) = 0 (Eq. 5). This prevents the trivial solution f = 0.
4. Discretize [epsilon, T] into N-1 intervals with EDM's rho = 7 formula (Sec. 4).
5. Consistency Distillation (CD, Algorithm 2): sample data x, pick n, noise it to
   x_{t_{n+1}} ~ N(x, t_{n+1}^2 I), take ONE teacher ODE step (Heun) to get
   x-hat_{t_n} (Eq. 6), and minimise d(f_theta(x_{t_{n+1}}, t_{n+1}),
   f_theta-(x-hat_{t_n}, t_n)) (Eq. 7). theta- is an EMA "target network" with stopgrad
   (Eq. 8), as in DQN / BYOL / MoCo.
6. Consistency Training (CT, Algorithm 3): replace the teacher step with the unbiased score
   estimate -(x_t - x)/t^2, which amounts to using the SAME noise z at both times:
   compare f_theta(x + t_{n+1} z, t_{n+1}) with f_theta-(x + t_n z, t_n) (Eq. 10). Grow N
   and adjust the EMA mu during training with schedules N(k), mu(k) (App. C).
7. Metric d: LPIPS works best (vs l1, l2) (Sec. 6.1, Fig. 3a).
8. Sampling (Algorithm 1): x = f(x_T, T) (1 step). For more steps, at each chosen
   tau_n add noise x + sqrt(tau_n^2 - epsilon^2) z and apply f again. The tau_n are found
   greedily by ternary search on FID.
9. Zero-shot editing (Algorithm 4, App. D): the multistep loop plus, after each f call,
   replace the known part of the image (mask Omega in a linear transform A's space) with
   the reference y. Covers inpainting, colorization, super-resolution, SDEdit; denoising
   and interpolation use f directly.

## Equations for the explainer

Macro map (cmd.tex): \rvx = {\mathbf{x}}, \rvz = {\mathbf{z}}, \rvw = {\mathbf{w}},
\rvy = {\mathbf{y}}; \vf = {\bm{f}}; \vtheta = {\bm{\theta}}; \vphi = {\bm{\phi}};
\vs = {\bm{s}} (line 302 redefines it; line 73 is the text "vs."); \mI = {\bm{I}};
\mA = {\bm{A}}; \mcal{X} = \mathcal{X}; \mbb{X} = \mathbb{X};
\ud = \mathop{}\!\mathrm{d}. \llbracket/\rrbracket need the stmaryrd package (KaTeX
supports them).

### Eq. (2), diffusion.tex, label eq:pfode -- PF ODE (general form)

Verbatim:
```
\ud \rvx_t = \left[\bm{\mu}(\rvx_t, t) - \frac{1}{2} \sigma(t)^2 \nabla \log p_t(\rvx_t)\right] \ud t.
```
And with EDM's settings mu = 0, sigma(t) = sqrt(2t) plus a learned score, Eq. (3)
(label eq:e_pfode), verbatim:
```
\frac{\ud \rvx_t}{\ud t} = -t \vs_\vphi(\rvx_t, t).
```
Expanded Eq. (3):
```
\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = -t\, \boldsymbol{s}_{\boldsymbol{\phi}}(\mathbf{x}_t, t)
```
Gloss: x_t = image at noise level t (t is also the noise std here); s_phi = learned score
model approx grad log p_t. This is EDM's Eq. 1 with sigma(t)=t (then sigma-dot = 1).

### Definition of self-consistency (consistency.tex, Sec. 3, inline)

Verbatim:
```
\vf: (\rvx_t, t) \mapsto \rvx_{\epsilon}
\vf(\rvx_t, t) = \vf(\rvx_{t'}, t') \text{ for all } t, t' \in [\epsilon,T]
```
(the second line is assembled from the inline text "$\vf(\rvx_t, t) = \vf(\rvx_{t'}, t')$
for all $t, t' \in [\epsilon,T]$".)
Gloss: f maps any noisy point to where its ODE trajectory started; any two points on the
same trajectory must give the same answer.

### Eq. (5), consistency.tex, label eq:param2 -- skip parametrisation / boundary condition

Verbatim:
```
\vf_\vtheta(\rvx, t) = c_\text{skip}(t) \rvx + c_\text{out}(t) F_\vtheta(\rvx, t)
```
with c_skip(epsilon) = 1, c_out(epsilon) = 0. The concrete choice (App. C, unnumbered
align*, appendix.tex line 453), verbatim:
```
c_\text{skip}(t) = \frac{\sigma_\text{data}^2}{(t-\epsilon)^2 + \sigma_\text{data}^2},\quad c_\text{out}(t) = \frac{\sigma_\text{data} (t-\epsilon) }{\sqrt{\sigma_\text{data}^2 + t^2}},
```
sigma_data = 0.5. Note: EDM's c_skip/c_out with t shifted by epsilon in c_skip's
denominator and c_out's numerator; the square root in c_out keeps t^2 (copied as in source).
Gloss: F_theta = free-form network (EDM/ADM/NCSN++ architecture); c_skip = how much of the
input passes straight through; c_out = how much the network adds. At t = epsilon, f is
exactly the identity.

### Eq. (6), distillation.tex, label eq:odesolve -- one teacher ODE step

Verbatim:
```
\hat{\rvx}_{t_n}^\vphi \coloneqq \rvx_{t_{n+1}} + (t_n - t_{n+1})\Phi(\rvx_{t_{n+1}}, t_{n+1}; \vphi),
```
Gloss: Phi = one-step ODE solver update using the pre-trained score model (Euler:
Phi = -t s_phi(x, t); experiments use Heun); moves the point from t_{n+1} to the
adjacent, less noisy t_n along the teacher's trajectory.

### Eq. (7), distillation.tex, label eq:distill_obj -- consistency distillation loss

Verbatim:
```
\mcal{L}_\text{CD}^N(\vtheta, \vtheta^{-}; \vphi) \coloneqq \\ \mbb{E}[\lambda(t_n) d(\vf_\vtheta({\rvx}_{t_{n+1}}, t_{n+1}), \vf_{\vtheta^-}(\hat{\rvx}_{t_n}^\vphi, t_n))],
```
(the `\\` is a multline line break). Gloss: theta = online model being trained; theta- =
EMA copy, no gradient; d = distance (LPIPS best); lambda = weight (1 works); expectation
over data x, n ~ U{1..N-1}, x_{t_{n+1}} ~ N(x, t_{n+1}^2 I). EMA update Eq. (8):
`\vtheta^- \leftarrow \operatorname{stopgrad}(\mu \vtheta^- + (1-\mu) \vtheta).`

### Eq. (10), generation.tex, label eq:dtct -- consistency training loss

Verbatim (resizebox wrapper removed):
```
\mbb{E}[\lambda(t_n) d(\vf_\vtheta(\rvx + t_{n+1}\rvz, t_{n+1}), \vf_{\vtheta^{-}}(\rvx + t_n\rvz, t_n))]
```
with `\rvz \sim \mcal{N}(\bf{0}, \mI)`. Gloss: same clean image x and same noise
direction z at two neighbouring noise levels; no teacher. Justified by the unbiased score
estimator (generation.tex, inline display):
`\nabla \log p_t(\rvx_t) = -\mbb{E}\left[ \frac{\rvx_t - \rvx}{t^2} \mathrel\bigg| \rvx_t \right],`
and Theorem 2: L_CD = L_CT + o(Delta t) with Euler and an exact score (Eq. 9).

### Algorithm 1 (consistency.tex) -- multistep consistency sampling

Verbatim body:
```
\rvx \gets \vf_\vtheta(\hat{\rvx}_T, T)
\FOR{$n=1$ {\bfseries to} $N-1$}
    Sample $\rvz \sim \mcal{N}(\bm{0}, \mI)$
    \hat{\rvx}_{\tau_n} \gets \rvx + \sqrt{\tau_n^2 - \epsilon^2} \rvz
    \rvx \gets \vf_\vtheta(\hat{\rvx}_{\tau_n}, \tau_n)
```
Gloss: one-step = just the first line. Two-step = one extra re-noise to level tau_1 and a
second f call. sqrt(tau^2 - eps^2) tops up the (epsilon-level) sample to total noise tau.

## Key hyperparameters (Table 3, App. C)

- T = 80, epsilon = 0.002, rho = 7, sigma_data = 0.5 (Sec. 2, Sec. 4, App. C).
- CD: Heun solver, LPIPS; N = 18 (CIFAR-10), 40 (ImageNet-64, LSUN); target EMA mu = 0
  (CIFAR-10), 0.95 (ImageNet-64, LSUN); models initialised from pre-trained EDMs.
- CT: randomly initialised; N(k) grows from s0 = 2 to s1 = 150 (CIFAR-10, LSUN) or 200
  (ImageNet-64); mu0 = 0.9 (CIFAR-10) or 0.95; schedules (App. C, verbatim):
  `N(k) &= \left\lceil \sqrt{\frac{k}{K} ((s_1 + 1)^2 - s_0^2) + s_0^2} - 1 \right\rceil + 1`
  `\mu(k) &= \exp\left(\frac{s_0 \log \mu_0}{N(k)}\right),`
- Batch 512 (CIFAR-10) / 2048 (others); lr 4e-4 / 8e-6 / 1e-5; RAdam, no decay/warmup;
  800k-1000k iterations; 8 or 64 A100s (Table 3).
- Architectures: NCSN++ (CIFAR-10), ADM networks (ImageNet-64, LSUN) (App. C).

## Headline results

1. CIFAR-10 CD: 1-step FID 3.55, 2-step 2.93 (IS 9.48 / 9.75) vs PD 8.34 / 5.58
   (Table 1). EDM teacher: 2.04 at 35 NFE (Table 1).
2. ImageNet 64x64 CD: 1-step FID 6.20, 2-step 4.70 vs PD 15.39 / 8.95 (Table 2).
   EDM teacher 2.44 at 79 NFE; ADM 2.07 at 250 NFE.
3. LSUN 256x256 CD 1-step / 2-step: Bedroom 7.80 / 5.22 (PD 16.92 / 8.47);
   Cat 11.0 / 8.84 (PD 29.6 / 15.5) (Table 2).
4. CT (no teacher): CIFAR-10 1-step 8.70, 2-step 5.83 (Table 1), beating all non-adversarial
   one-step models (best flow/VAE listed: DC-VAE 17.9) and comparable to PD 1-step (8.34).
   ImageNet-64 CT 13.0 / 11.1 (Table 2).
5. Zero-shot editing with one CD model on LSUN Bedroom: colorization, super-resolution
   (32x32 -> 256x256), stroke-guided generation (SDEdit), inpainting, interpolation,
   denoising (Sec. 6.3, Fig. 6, App. D).
6. CD with Heun and N = 18 is best on CIFAR-10; Heun beats Euler at equal N; CD is
   insensitive to N once N is large (Sec. 6.1, Fig. 3).

## Limitations and later fixes

- Still behind the multi-step teacher (CIFAR-10 3.55 vs EDM 2.04; ImageNet-64 6.20 vs 2.44)
  and behind the best GANs at one step (StyleGAN-XL 1.85 on CIFAR-10, Table 1).
- CT is much weaker than CD at higher resolution (ImageNet-64 13.0 vs 6.20).
- Uses LPIPS, a learned ImageNet-trained perceptual metric, which may flatter FID (also an
  Inception-based ImageNet feature metric); the paper does not discuss this. CT also
  depends on hand-tuned N(k), mu(k) schedules.
- Sampling time points tau_n are found by greedy ternary search on FID (Sec. 3).
- Later fixes (general knowledge, not verified here): "Improved Techniques for Training
  Consistency Models" (Song and Dhariwal, 2310.14189) drops LPIPS and EMA target, uses
  pseudo-Huber loss and lognormal noise; Latent Consistency Models (LCM) bring CD to
  Stable Diffusion; sCM / continuous-time CMs (2410.11081) scale the continuous version.

## Official code

Not stated in the paper's tex or in abs.html (grep found no URL).
