# Classifier-Free Diffusion Guidance (CFG)

Ho and Salimans, arXiv 2207.12598v1 (NeurIPS 2021 workshop paper posted to arXiv in 2022). Source: `sources/cfg/tex/main.tex` (all macros defined in its preamble; `math_commands.tex` ships in the e-print but is not \input). Code: none stated in the paper.

## Core problem

Classifier guidance (ADM-G, beat-gans) gives diffusion models a truncation-like knob that trades diversity for fidelity, but it needs an extra classifier trained on noisy images (no off-the-shelf classifier works). It also raises a doubt: since guidance follows a classifier's input gradient, maybe it only boosts classifier-based metrics (IS, FID) the way an adversarial attack would (Sec 1). Can you get the same knob with no classifier at all?

## What prior methods did and why that hurt

- Low-temperature tricks (scale the score, shrink the sampling noise) give blurry, low-quality samples (Sec 3, citing Dhariwal and Nichol).
- Classifier guidance (Dhariwal and Nichol): modify the noise prediction with w * sigma * grad log p(c | z). Costs: a second model to train on the same noise distribution; a more complicated pipeline; and the step direction is literally a classifier gradient, so the IS/FID gain could be "adversarial" against Inception-style classifiers.

## The one key idea

Train one network to be both conditional and unconditional (by randomly replacing the label with a null token during training), then at sampling time extrapolate away from the unconditional prediction and toward the conditional one: eps_tilde = (1+w) eps(z, c) - w eps(z). The difference eps(z,c) - eps(z) plays the role of an implicit classifier gradient, so no classifier is needed.

## Delta from its predecessor (classifier guidance, beat-gans)

- Classifier guidance in this paper's notation (Sec 3.1): eps_tilde = eps(z, c) - w sigma grad log p_theta(c | z). CFG replaces the classifier gradient term with w [eps(z, c) - eps(z)] (the unconditional model gets a negative weight).
- No extra model: one network, one extra training line (label dropout), one extra sampling line.
- The CFG step is not the gradient of any classifier (unconstrained networks give non-conservative vector fields), so it cannot be read as an adversarial attack on a classifier (Sec 3.2, Sec 5).
- Cost moves from training to sampling: two forward passes per step (conditional + unconditional) instead of one diffusion pass + one small classifier pass (Sec 4.3, Sec 5).
- Same setup as ADM-G otherwise: same architectures and hyperparameters as Dhariwal and Nichol, except continuous-time training (Sec 4).

## Mechanism step by step

1. Training (Algorithm 1): sample (x, c) from data; with probability p_uncond set c to the null token; sample a log-SNR lambda; corrupt z_lambda = alpha x + sigma eps; regress eps_theta(z_lambda, c) onto eps. This trains the conditional model and the unconditional model eps_theta(z) = eps_theta(z, c = null) in the same weights.
2. Sampling (Algorithm 2): at each step run the network twice, once with c and once with null, and combine: eps_tilde = (1+w) eps(z, c) - w eps(z).
3. Turn eps_tilde into a denoised estimate x_tilde = (z - sigma eps_tilde)/alpha and take the usual ancestral step (or any other sampler, e.g. DDIM).
4. Why it works (Sec 3.2): define an implicit classifier p^i(c | z) proportional to p(z | c) / p(z). Its gradient is -(1/sigma)[eps*(z, c) - eps*(z)] for exact scores. Plugging that into classifier guidance gives exactly (1+w) eps*(z, c) - w eps*(z). CFG uses learned scores in place of exact ones.
5. Intuition (Sec 5): guidance "decreases the unconditional likelihood of the sample while increasing the conditional likelihood".

## Equations (TeX copied verbatim from `tex/main.tex`)

Macro notes for rendering (all in `main.tex` preamble unless noted): `\bepsilon` -> `\boldsymbol{\epsilon}`, `\bz` -> `\mathbf{z}`, `\bc` -> `\mathbf{c}`, `\bx` -> `\mathbf{x}`, `\bI` -> `\mathbf{I}`, `\bzero` -> `\mathbf{0}`, `\bmu` -> `\boldsymbol{\mu}`, `\Eb{a}{b}` -> `\E_{a}\!\left[b\right]` with `\E` -> `\mathbb{E}` (line 31), `\grad` -> `\nabla` (line 30). Equation numbers from the compiled PDF.

### Eq. 1 (Sec 2 "Background") - forward process, first line
```
q(\bz_\lambda|\bx) &= \mathcal{N}(\alpha_\lambda \bx, \sigma_\lambda^2 \bI), \ \text{where}\ \alpha_\lambda^2 = 1/(1+e^{-\lambda}),\ \sigma_\lambda^2 = 1-\alpha_\lambda^2
```
- x: clean image. z_lambda: noisy image at noise level lambda.
- lambda: log signal-to-noise ratio, lambda = log(alpha^2 / sigma^2). Large lambda = nearly clean; small = nearly pure noise.
- alpha_lambda, sigma_lambda: signal and noise scales, alpha^2 + sigma^2 = 1 (variance preserving).

### Eq. 5 (Sec 2) - training objective
```
\Eb{\bepsilon,\lambda}{\|\bepsilon_\theta(\bz_\lambda) - \bepsilon \|^2_2}
```
- eps: the true Gaussian noise used to make z_lambda. eps_theta: network's guess of it. Same noise-prediction loss as DDPM, in continuous time.

### Classifier guidance in CFG notation (Sec 3.1, unnumbered display)
```
\tilde{\bepsilon}_\theta(\bz_\lambda, \bc) = \bepsilon_\theta(\bz_\lambda, \bc) - w\sigma_{\lambda}\nabla_{\bz_\lambda}\log p_{\theta}(\bc | \bz_\lambda) \approx -\sigma_{\lambda}\nabla_{\bz_\lambda}[\log p(\bz_\lambda |  \bc) + w \log p_{\theta}(\bc | \bz_\lambda) ],
```
and the resulting target (unnumbered):
```
\tilde{p}_{\theta}(\bz_\lambda | \bc) \propto p_{\theta}(\bz_\lambda | \bc)p_{\theta}(\bc | \bz_\lambda)^{w}.
```
- w: guidance strength. p_theta(c | z_lambda): the auxiliary noisy-image classifier.
- Note the convention: here w is applied on top of an already conditional model, so w = 0 means plain conditional sampling. The paper notes guidance weight w+1 on an unconditional model is theoretically equal to weight w on a conditional model.

### Algorithm 1, line 3 (Sec 3.2) - label dropout
```
\State $\bc \gets \varnothing$ with probability $p_\mathrm{uncond}$ \Comment{Randomly discard conditioning to train unconditionally}
```
- varnothing: the null token meaning "no label". p_uncond: dropout probability (0.1, 0.2, 0.5 tried).

### Eq. 6 (Sec 3.2) - the classifier-free guided noise prediction
```
\tilde{\bepsilon}_\theta(\bz_\lambda, \bc) = (1+w)\bepsilon_\theta(\bz_\lambda, \bc) - w\bepsilon_{\theta}(\bz_\lambda) \label{eq:classifier_free_score}
```
- eps_theta(z, c): the network's noise prediction when told the label c.
- eps_theta(z): the same network with c = null (unconditional prediction).
- w: guidance strength; w = 0 is unguided conditional sampling; larger w pushes further along (conditional minus unconditional).
- Equivalent rewrite (not in the paper, algebra only): eps(z,c) + w [eps(z,c) - eps(z)].
- Algorithm 2 uses the same formula per step: `\tilde{\bepsilon}_t = (1+w)\bepsilon_\theta(\bz_{t}, \bc) - w\bepsilon_{\theta}(\bz_{t})`.

### Implicit classifier (Sec 3.2, inline)
```
$p^{i}(\bc | \bz_\lambda) \propto p(\bz_\lambda | \bc)/p(\bz_\lambda)$
```
```
$\nabla_{\bz_\lambda} \log p^{i}(\bc | \bz_\lambda) = -\frac{1}{\sigma_{\lambda}}[\bepsilon^*(\bz_\lambda, \bc) - \bepsilon^*(\bz_\lambda)]$
```
- p^i: the classifier you would get from the generative model via Bayes' rule.
- eps*: exact (true) scores. With these, classifier guidance using p^i gives `\tilde{\bepsilon}^*(\bz_\lambda, \bc) = (1+w)\bepsilon^*(\bz_\lambda, \bc) - w\bepsilon^*(\bz_\lambda)`; CFG substitutes learned eps_theta.

## Key hyperparameters

- Architectures and hyperparameters: same as ADM-G (Dhariwal and Nichol) apart from continuous-time training (Sec 4).
- lambda_min = -20, lambda_max = 20; lambda sampled from a cosine-schedule-inspired distribution lambda = -2 log tan(au + b) (Sec 2, Sec 4.1).
- Sampler variance interpolation v = 0.3 (64x64), v = 0.2 (128x128) (Sec 4.1).
- Training steps: 400K (64x64), 2.7M (128x128) (Sec 4.1).
- p_uncond in {0.1, 0.2, 0.5} on 64x64 (Sec 4.2). w in {0, 0.1, 0.2, ..., 4} evaluated with 50000 samples (Sec 4.1); tables show 0.0-1.0 in steps of 0.1 plus 2.0, 3.0, 4.0.
- Sampling steps T in {128, 256, 1024} on 128x128 (Sec 4.3).

## Headline results

1. ImageNet 64x64 (Table 1): best FID 1.55 at w = 0.1, p_uncond = 0.1 (IS 66.11); unguided (w = 0) FID 1.8, IS 53.71. ADM 2.07, CDM 1.48 for reference. Max IS 260.2 at w = 4.0 (FID 26.22).
2. ImageNet 128x128 (Table 2): best FID 2.43 at w = 0.3 with T = 256 or 1024 (IS 158.47 / 156), vs ADM-G 2.97. Unguided T=256 FID 7.27, IS 82.45.
3. Strong guidance (Table 2): w = 4.0, T = 128: FID 21.55, IS 422.29, vs BigGAN-deep at its max-IS truncation FID 25, IS 253 ("outperforms BigGAN-deep at both FID and IS", Sec 4.1).
4. p_uncond (Sec 4.2): 0.5 is consistently worse across the IS/FID frontier; 0.1 and 0.2 are about equal. Only a small fraction of capacity needs to go to unconditional generation.
5. Trade-off (Sec 4.1): best FID at small guidance (w = 0.1 or 0.3), best IS at w >= 4, with a clear trade-off between; samples lose variety and gain fidelity as w grows; strongly guided samples have saturated colors (Fig 3 caption).
6. Compute caveat (Sec 4.3): each step costs 2 network evaluations; at matched cost (T = 128 vs ADM-G's ~250 steps) CFG FID 3.04 at w=0.3 underperforms ADM-G's 2.97.

## Limitations and later fixes

- Stated (Sec 5): sampling is slower than classifier guidance, since two full diffusion passes per step are needed; suggests injecting conditioning late in the network as a possible fix (left to future work).
- Stated: requires training the unconditional mode (avoidable only with few classes, by summing conditionals).
- Stated: reduced diversity may hurt under-represented parts of the data.
- Experiments are a proof of concept on ImageNet 64/128 only, with hyperparameters tuned for classifier guidance (Sec 4).
- Later fixes (external, not verified here): guidance distillation (Meng et al.) folds the two passes into one; batching cond/uncond in one forward pass is standard in practice; later work on guidance-interval / dynamic thresholding addresses saturation (Imagen). CFG became the default in GLIDE, LDM (this survey's ldm paper uses it at s=1.5), Imagen, Stable Diffusion.

## Surprising or worth flagging

- Text error in Sec 4.1: "FID monotonically decreasing and IS monotonically increasing with w". The tables show FID falls then rises (e.g. 128x128 T=256: 7.27 -> 2.43 at w=0.3 -> 21.53 at w=4). Only IS is monotone.
- Figure 5 legend says T=512 but Table 2 and text say T=1024 (source `\legend{$T=128$,$T=256$,$T=512$}`). Use 1024.
- Guidance-scale conventions differ across papers: here w=0 is unguided; GLIDE/LDM report "s" (LDM Table 2/3 use s=1.5, GLIDE s=3). Whether LDM's s equals 1+w is not stated in any of the three sources.
- The unconditional model gets a negative weight, which the authors call new ("to our knowledge has not yet been explored", Sec 5).
