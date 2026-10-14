# DDIM: Denoising Diffusion Implicit Models

Song, Meng, Ermon. ICLR 2021. arXiv 2010.02502v4. Sources: `sources/ddim/tex/main.tex` inputs `sections/1_introduction` ... `6_discussion` and `9_appendix`.

## Core problem

DDPM gives GAN-level samples but must run its reverse Markov chain for all T = 1000 steps sequentially. Intro: ~20 hours to sample 50k 32x32 images on an Nvidia 2080 Ti vs under a minute for a GAN; ~1000 hours for 50k 256x256 images (1_introduction.tex).

## What prior methods did and why that hurt

- DDPM defines generation as the reverse of one specific Markovian forward diffusion (Eq. 3). Because the generative chain approximates that reverse, it is forced to take the same T steps; large T is needed so each reverse step is near-Gaussian (2_background.tex).
- Skipping steps in DDPM (simulating a shorter chain) degrades quality badly: eta = 1 CIFAR-10 FID 41.07 at 10 steps, and the sigma-hat variant used by Ho et al. for CIFAR-10 gets 367.43 (Table 1).
- DDPM sampling is stochastic, so the same x_T gives very different images: no meaningful latent code, no encoding/reconstruction.

## The one key idea

The DDPM training loss L_gamma only depends on the marginals q(x_t | x_0), not on the joint forward process. So swap in a family of NON-Markovian forward processes with the same marginals, indexed by per-step noise sigma_t. They all share the same training objective (Theorem 1, up to weights/constant), so a pretrained DDPM network can be reused unchanged, and the matching generative process can be chosen freely: in particular a deterministic one (sigma_t = 0), and one that jumps over a short sub-sequence of timesteps.

## Delta from its predecessor (DDPM)

- Same network, same training (L_1 objective, T = 1000), same pretrained checkpoints (CIFAR-10, LSUN Bedroom/Church from Ho et al.; CelebA trained by the authors). Only the sampler changes.
- Sampler has two knobs: tau (which timesteps to visit, length S = dim(tau)) and eta (how much fresh noise; eta = 1 recovers DDPM, eta = 0 is DDIM).
- Deterministic (eta = 0) gives an implicit model: x_T is a latent code; consistent high-level features across step counts; semantic interpolation; near-lossless encode/decode.
- Notation trap: DDIM's alpha_t is DDPM's alpha-bar_t (cumulative product), stated explicitly in App. C.2 (9_appendix.tex). So x_t = sqrt(alpha_t) x_0 + sqrt(1 - alpha_t) eps (Eq. 4).

## Mechanism, step by step

1. Train (or reuse) a DDPM noise predictor eps_theta^(t)(x_t) with L_gamma, gamma = 1 (Eq. 5).
2. Define q_sigma(x_{t-1} | x_t, x_0) as a Gaussian (Eq. 7) whose mean is chosen so the marginals stay q(x_t|x_0) = N(sqrt(alpha_t) x_0, (1-alpha_t) I) for all t (Lemma 1, App. B). The forward process derived by Bayes' rule is non-Markovian (x_t depends on x_{t-1} and x_0).
3. Generative step: from x_t, predict the clean image x_0 = f_theta(x_t) (Eq. 9), then plug it into q_sigma(x_{t-1} | x_t, x_0). Written out, this is Eq. (12): predicted x_0 rescaled to level t-1, plus a "direction pointing to x_t" built from the predicted noise, plus optional fresh noise sigma_t eps.
4. Choose sigma via eta (Eq. 16). eta = 1 reproduces the DDPM posterior variance; eta = 0 is fully deterministic (DDIM).
5. Accelerate: run the same update on an increasing sub-sequence tau of [1..T] of length S, replacing t-1 by tau_{i-1} (App. C.1). tau is linear (tau_i = floor(c i)) or quadratic (floor(c i^2)); quadratic for CIFAR-10, linear elsewhere (App. D.2).
6. ODE view: eta = 0 DDIM is an Euler discretization (Eq. 13) of an ODE in the reparameterized variables sigma = sqrt(1-alpha)/sqrt(alpha), x-bar = x / sqrt(alpha) (Eq. 14). Running it forward encodes x_0 -> x_T; backward decodes.

## Equations for the explainer

Macros (math_commands.tex): `\vx`={\bm{x}}, `\gN`={\mathcal{N}}, `\mI`={\bm{I}}, `\vzero`={\bm{0}}, `\vone`={\bm{1}}, `\bb{#1}`={\mathbb{#1}}, `\diff`=\mathrm{d}, `\R`=\mathbb{R}, `\norm` from the standard ICLR math_commands.

### Eq. (4) Closed-form noising (2_background.tex, label eq:reparam_xt)
```tex
    \vx_t = \sqrt{\alpha_t} \vx_0 + \sqrt{1 - \alpha_t} \epsilon, \quad \text{where} \quad \epsilon \sim \gN(\vzero, \mI).
```
- x_0: clean image; x_t: noised image at step t.
- alpha_t: signal fraction remaining (DDPM's alpha-bar), decreasing from ~1 to ~0.
- epsilon: standard Gaussian noise; the network eps_theta learns to predict it from x_t.

### Eq. (7) Non-Markovian reverse conditional (3_method.tex, label eq:reversed-close-form)
```tex
   q_\sigma(\vx_{t-1} | \vx_t, \vx_0) = \gN\left(\sqrt{\alpha_{t-1}} \vx_{0} + \sqrt{1 - \alpha_{t-1} - \sigma^2_t} \cdot {\frac{\vx_{t}  - \sqrt{\alpha_{t}} \vx_0}{\sqrt{1 - \alpha_{t}}}}, \sigma_t^2 \mI \right).
```
- sigma_t >= 0: per-step stochasticity, a free choice that does not change the marginals.
- (x_t - sqrt(alpha_t) x_0)/sqrt(1 - alpha_t): the implied noise in x_t.
- Mean = clean image at level t-1 plus the "same" noise direction rescaled, leaving variance sigma_t^2 for fresh noise.

### Eq. (9) Predicted x_0 (3_method.tex, label eq:x0-pred-def)
```tex
    f_\theta^{(t)}(\vx_t) := (\vx_t - \sqrt{1 - \alpha_t} \cdot \epsilon_{\theta}^{(t)}(\vx_t)) / \sqrt{\alpha_t}.
```
- Invert Eq. (4) using the network's noise guess: the model's current best guess of the final clean image.

### Eq. (12) Generalized sampling update (3_method.tex, label eq:sample-eq-gen)
```tex
    \vx_{t-1} & = \sqrt{\alpha_{t-1}} \underbrace{\left(\frac{\vx_t - \sqrt{1 - \alpha_t} \epsilon_\theta^{(t)}(\vx_t)}{\sqrt{\alpha_t}}\right)}_{\text{`` predicted } \vx_0 \text{''}} + \underbrace{\sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \epsilon_\theta^{(t)}(\vx_t)}_{\text{``direction pointing to } \vx_t \text{''}} + \underbrace{\sigma_t \epsilon_t}_{\text{random noise}}
```
- Three pieces: rescaled predicted x_0; the predicted noise re-added at the level appropriate for t-1 (minus what is reserved for fresh noise); fresh noise epsilon_t ~ N(0, I) scaled by sigma_t. alpha_0 := 1.
- DDPM special case (inline text after Eq. 12): sigma_t = sqrt((1 - alpha_{t-1})/(1 - alpha_t)) sqrt(1 - alpha_t/alpha_{t-1}).
- DDIM: sigma_t = 0 for all t; deterministic map x_T -> x_0.
- For acceleration, replace t-1 by the previous element of tau.

### Eq. (16) sigma as a function of eta (4_experiments.tex)
```tex
    \sigma_{\tau_{i}}(\eta) = \eta \sqrt{(1 - \alpha_{\tau_{i-1}})/(1 - \alpha_{\tau_{i}})}\sqrt{1 - {\alpha_{\tau_i}}/{\alpha_{\tau_{i-1}}}}, %
```
- eta >= 0 interpolates: eta = 1 DDPM, eta = 0 DDIM. tau_i: the i-th visited timestep.
- The larger-variance DDPM variant sigma-hat_{tau_i} = sqrt(1 - alpha_{tau_i}/alpha_{tau_{i-1}}) (inline, unnumbered) is what Ho et al. used only for CIFAR-10.

### Eq. (14) DDIM ODE (3_method.tex, label eq:ddim-ode)
```tex
   \diff \bar{\vx}(t) = \epsilon_\theta^{(t)}\left(\frac{\bar{\vx}(t)}{\sqrt{\sigma^2 + 1}}\right) \diff \sigma(t)
```
- sigma(t) = sqrt(1 - alpha)/sqrt(alpha) (noise-to-signal ratio, NOT the per-step sigma_t above; reused symbol), x-bar = x/sqrt(alpha).
- DDIM (Eq. 13) is Euler on this ODE with steps in d sigma; Proposition 1 says it is equivalent to the probability flow ODE of the VE SDE of Song et al. 2020 (Score SDE), but Euler in d sigma differs from Score SDE's Euler in dt when steps are few (Eq. 15).

## Key hyperparameters

- T = 1000 training steps, L_1 objective, alpha schedule per Ho et al. heuristic (4_experiments.tex; App. D.1).
- S = dim(tau) in {10, 20, 50, 100, 1000}; eta in {0.0, 0.2, 0.5, 1.0} plus sigma-hat (Table 1).
- tau: quadratic for CIFAR-10, linear for CelebA/LSUN (App. D.2).
- Interpolation: slerp between two x_T draws, dim(tau) = 50 (Sec. 5.3, App. D.5).

## Headline results

Table 1 (4_experiments.tex), FID, same trained model:

| S | 10 | 20 | 50 | 100 | 1000 |
|---|---|---|---|---|---|
| CIFAR-10 DDIM (eta=0) | 13.36 | 6.84 | 4.67 | 4.16 | 4.04 |
| CIFAR-10 DDPM (eta=1) | 41.07 | 18.36 | 8.01 | 5.78 | 4.73 |
| CIFAR-10 DDPM sigma-hat | 367.43 | 133.37 | 32.72 | 9.99 | 3.17 |
| CelebA DDIM (eta=0) | 17.33 | 13.73 | 9.17 | 6.53 | 3.51 |
| CelebA DDPM (eta=1) | 33.12 | 26.03 | 18.48 | 13.93 | 5.98 |
| CelebA DDPM sigma-hat | 299.71 | 183.83 | 71.71 | 45.20 | 3.26 |

Caption caveat: eta = 1 at S < T "can be seen as simulating DDPMs trained with S steps"; only the sigma-hat S=1000 cell (3.17) is Ho et al.'s published setting.

1. DDIM wins at every S < 1000 on both datasets; at S = 1000 sigma-hat DDPM is marginally better (3.17 vs 4.04 CIFAR-10) (Table 1).
2. Speedup: "quality comparable to 1000 step models within 20 to 100 steps, which is a 10x to 50x speed up"; CelebA 100-step DDPM (13.93) ~ 20-step DDIM (13.73) (Sec. 5.1). Sampling time scales linearly with trajectory length (Figure 4).
3. LSUN 256x256 (Table 3, App. D): Bedroom DDIM 6.75 at 50 steps vs DDPM 1000-step 6.36; Church DDIM 10.84 at 50 vs 1000-step DDPM 7.89. Church at 100 steps DDPM (8.27) beats DDIM (10.58).
4. Consistency: same x_T with different tau gives images with the same high-level features; 20-step samples already close to 1000-step ones (Sec. 5.2, Figure 5).
5. Semantic interpolation in x_T (Sec. 5.3, Figure 6).
6. Encode/decode reconstruction error on CIFAR-10 test set falls from 0.014 (S=10) to 0.0001 (S=500, 1000) per-dim MSE (Table 2).

Speedup is stated three different ways in the source: abstract "10x to 50x faster in terms of wall-clock time"; Sec. 5 opening and intro "10x to 100x"; Sec. 5.1 "10x to 50x" for comparable quality in 20-100 steps.

## Limitations and later work

- At very few steps (10) FID is still far from 1000-step quality (13.36 vs 4.04). Discussion suggests higher-order ODE solvers (Adams-Bashforth); later addressed by PNDM, DPM-Solver, EDM Heun sampler, and distillation / consistency models. [later-work attributions not source-verified]
- Deterministic sampling lowers diversity-driven quality at full length (sigma-hat 1000 steps beats it), and on Church at 100 steps.
- Proposition equivalence is to the VE probability flow ODE, even though the model is a VP/DDPM model; the text calls it "a continuous-time analog of DDPM". Worth explaining carefully (the x-bar = x/sqrt(alpha) rescaling turns VP into VE).
- Discrete multinomial non-Markovian forward process proposed (App. A) but not evaluated.
- DDIM inversion (encode x_0 -> x_T) became the basis of later image-editing methods. [not source-verified]

## Official code

Not stated in the LaTeX source (only links to the DDPM implementation and StyleGAN preprocessing). The commonly cited repo is github.com/ermongroup/ddim [not source-verified].
