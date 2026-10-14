# NCSN: Generative Modeling by Estimating Gradients of the Data Distribution

Song and Ermon, NeurIPS 2019. arXiv 1907.05600v3. Sources: `sources/ncsn/tex/` (main.tex inputs abstract, intro, pre, difficulty, method, exp, related, conclude, app).

## Core problem

Generate new samples from an unknown data distribution p_data(x) given only i.i.d. samples, without (a) a normalized likelihood model, (b) surrogate losses, or (c) adversarial training.

## What prior methods did and why that hurt

- Likelihood models (autoregressive, flows) need specialized architectures to stay normalized; VAEs / EBMs need surrogate losses (ELBO, contrastive divergence) (intro.tex).
- GANs avoid that but adversarial training is unstable (intro.tex).
- Earlier Markov-chain transition-operator methods (GSN, NET = Sohl-Dickstein 2015, Infusion Training, Variational Walkback) must simulate a chain during training and tend to give blurry samples; NET needs thousands of tiny steps (related.tex).
- Plain score matching (Hyvarinen 2005) needs tr(grad_x s_theta(x)), which does not scale to deep nets / high dimension (pre.tex, Sec. 2.1).
- Naive "score matching + Langevin" fails for two reasons (difficulty.tex, Sec. 3):
  1. Manifold hypothesis: real data sits on a low-dimensional manifold, so the score (a gradient in ambient space) is undefined off it, and score matching is only consistent when the data support is the whole space. Figure 1: sliced score matching loss on raw CIFAR-10 fluctuates irregularly; with tiny N(0, 0.0001) noise it converges. (0.0001 is the variance, i.e. std 0.01.)
  2. Low-density regions: (a) the expectation in the loss is estimated from samples, so in regions with ~no data the score is basically unconstrained; Figure 2 toy (mixture 1/5 N((-5,-5),I) + 4/5 N((5,5),I)) shows the estimate is only accurate near modes. (b) Langevin mixes slowly between modes separated by low density; for disjoint supports the score does not even depend on the mixing weight pi, so Langevin cannot recover relative mode weights (Figure 3b).

## The one key idea

Perturb the data with Gaussian noise at many scales, train ONE noise-conditional network s_theta(x, sigma) to estimate the score of every perturbed distribution, then sample with annealed Langevin dynamics: start at the biggest noise (where the landscape is smooth and fills empty space) and gradually anneal down to tiny noise (where the distribution is almost the data).

## Delta from predecessors

- vs score matching: uses denoising score matching (Vincent 2011) on noise-perturbed data, which avoids the Jacobian trace entirely, and conditions on sigma so many noise levels share one net. Sliced score matching also works but costs ~4x compute (pre.tex).
- vs Langevin dynamics: annealed (multi-temperature) Langevin in the spirit of simulated annealing / annealed importance sampling, with step size scaled per noise level.
- vs likelihood models / GANs: no likelihood, no adversarial game, no MCMC during training, no special architecture; training and sampling are decoupled (related.tex).

## Mechanism, step by step

1. Pick a geometric sequence sigma_1 > sigma_2 > ... > sigma_L (ratio constant, > 1). NOTE: in NCSN sigma_1 is the LARGEST noise; Score SDE flips the index (sigma_1 smallest, sigma_N largest).
2. Network s_theta(x, sigma): output has the same shape as the image; RefineNet-style U-Net with dilated convolutions and a modified conditional instance normalization (CondInstanceNorm++) for sigma conditioning (method.tex, app.tex).
3. Training: for each minibatch pick sigma_i, add N(0, sigma_i^2 I) noise, regress s_theta onto -(x_tilde - x)/sigma_i^2 (Eq. 5), weighted by lambda(sigma) = sigma^2 so all levels have similar loss magnitude (Eq. 6). The sigma^2 choice comes from the empirical observation ||s_theta(x, sigma)||_2 proportional to 1/sigma at optimum.
4. Sampling (Algorithm 1): initialize from uniform noise; for each level i = 1..L run T Langevin steps with step size alpha_i = epsilon * sigma_i^2 / sigma_L^2, using the final sample of level i-1 to initialize level i. The sigma^2 scaling keeps the "signal-to-noise" ratio of the Langevin update independent of sigma_i.
5. Inpainting: same procedure with the known pixels clamped to (noised) observed values (app.tex).

## Equations for the explainer

Macros (from main.tex): `\bfx`=\mathbf{x}, `\bfs`=\mathbf{s}, `\bfz`=\mathbf{z}, `\bftheta`={\boldsymbol{\theta}}, `\mbb{#1}`=\mathbb{#1}, `\mcal`=\mathcal, `\ud`=\mathrm{d}, `\norm{#1}`={\lVert {#1} \rVert}. `\text{\iid}` etc. are text macros.

Score definition (inline, pre.tex, Sec. 2): `$\nabla_\bfx \log p(\bfx)$`. The score network is `$\bfs_\bftheta: \mbb{R}^D \rightarrow \mbb{R}^D$`.

### Eq. (4) Langevin dynamics (pre.tex, label eqn:langevin)
```tex
\tilde{\bfx}_{t} = \tilde{\bfx}_{t-1} + \frac{\epsilon}{2}\nabla_\bfx \log p(\tilde{\bfx}_{t-1}) +  \sqrt{\epsilon}~ \bfz_t,
```
- x_tilde_t: the current sample after t steps. x_tilde_0 drawn from a prior pi(x).
- epsilon: fixed step size > 0.
- grad_x log p(.): the score; pushes the sample uphill toward higher density.
- z_t ~ N(0, I): fresh Gaussian noise each step; keeps the chain exploring instead of collapsing to a mode.
- Exact as epsilon -> 0 and T -> infinity (Metropolis-Hastings correction otherwise, ignored in practice).

### Eq. (2) Denoising score matching (pre.tex, label eqn:dsm)
```tex
\frac{1}{2}\mbb{E}_{q_\sigma(\tilde{\bfx}\mid \bfx)p_\text{data}(\bfx)}[\| \bfs_\bftheta(\tilde{\bfx}) - \nabla_{\tilde{\bfx}} \log q_\sigma(\tilde{\bfx}\mid \bfx) \|_2^2].
```
- x: clean data point; x_tilde: noisy version drawn from q_sigma(x_tilde | x).
- grad log q_sigma(x_tilde | x): score of the known noise kernel, computable in closed form.
- Minimizer equals grad log q_sigma(x) (score of the noisy data distribution), which approximates the data score only for small noise.

### Eq. (5) Per-level DSM loss with Gaussian noise (method.tex, label eqn:dsm2)
```tex
 \ell(\bftheta; \sigma) \triangleq \frac{1}{2}\mbb{E}_{p_\text{data}(\bfx)} \mbb{E}_{\tilde{\bfx}\sim \mcal{N}(\bfx,\sigma^2 I)} \bigg[\norm{ \bfs_\bftheta(\tilde{\bfx}, \sigma) + \frac{\tilde{\bfx} - \bfx}{\sigma^2}}_2^2\bigg].
```
- s_theta(x_tilde, sigma): the NCSN, told which noise level it is looking at.
- (x_tilde - x)/sigma^2: minus the target; the target score -(x_tilde - x)/sigma^2 points from the noisy point back toward the clean one. So the net learns to "point toward the denoised image".

### Eq. (6) Unified objective (method.tex, label eqn:all)
```tex
    \mcal{L}(\bftheta; \{\sigma_i\}_{i=1}^L) \triangleq \frac{1}{L}\sum_{i=1}^L \lambda(\sigma_i)\ell(\bftheta; \sigma_i),
```
- L: number of noise levels; lambda(sigma_i) > 0: weight, chosen as sigma^2.
- A conical combination, so the optimum matches every level's score simultaneously (given capacity).

### Algorithm 1, annealed Langevin dynamics (method.tex, label alg:anneal)
Step size line: `$\alpha_i \gets \epsilon \cdot \sigma_i^2/\sigma_L^2$`
Update line (resizebox wrapper stripped):
```tex
$\tilde{\bfx}_{t} \gets \tilde{\bfx}_{t-1} + \dfrac{\alpha_i}{2} \bfs_\bftheta(\tilde{\bfx}_{t-1}, \sigma_i) + \sqrt{\alpha_i}~ \bfz_t$
```
- Outer loop i = 1..L over noise levels (large to small), inner loop t = 1..T steps; x_tilde_0 <- x_tilde_T between levels.
- alpha_i: per-level step size; epsilon is the base step at the smallest noise sigma_L.
- Note the convention: NCSN uses alpha/2 * score + sqrt(alpha) z; Score SDE Eq. (2) writes the same thing as epsilon_i * score + sqrt(2 epsilon_i) z (epsilon_i = alpha_i/2). Not a bug.

## Key hyperparameters

- Images (exp.tex, Sec. 5 Setup): L = 10, geometric sigma_1 = 1 to sigma_10 = 0.01; pixel values in [0, 1]; T = 100 steps per level; epsilon = 2e-5 (5e-6 to 5e-5 "generally works fine"); uniform noise init. So 1000 total network evaluations per sample.
- Toy mixture (method.tex, Sec. 4.3): L = 10, sigma_1 = 10, sigma_10 = 0.1. Do not confuse with the image setting.
- lambda(sigma) = sigma^2; alpha_i proportional to sigma_i^2.
- Datasets: MNIST, CelebA (center-crop 140, resize 32x32), CIFAR-10. IS/FID on 50k samples (app.tex).

## Headline results

1. CIFAR-10 unconditional Inception Score 8.87 +/- .12, claimed state of the art, better than most class-conditional models (Table 1; exp.tex).
2. CIFAR-10 FID 25.32, "comparable to" SNGAN (21.7) (Table 1). Not SOTA on FID: MoLM 18.9 is better in the same table.
3. Single-noise baseline (sigma = 0.01, vanilla Langevin) fails to produce reasonable images (App. baseline, Figures 7-8).
4. Annealed Langevin recovers correct relative mode weights on the toy mixture where plain Langevin fails (Figure 3).
5. Arbitrary-shape inpainting with the same model (Figure 6).

## Limitations and later work

- Sampling is slow (L*T = 1000 net evals) and hyperparameters (sigma_1, L, epsilon, T) were hand-picked; unstable at higher resolution. NCSNv2 (Song and Ermon 2020, "Improved Techniques...") gives principled choices for sigma_max etc. (Score SDE cites its "Technique 1" for sigma_max). [not source-verified here]
- FID not competitive with GANs; Score SDE (NCSN++) gets FID 2.20 and reinterprets NCSN as a discretized VE SDE.
- No likelihood; Score SDE's probability flow ODE adds exact likelihood.
- Discrete noise levels; Score SDE makes them a continuum.

## Official code

https://github.com/ermongroup/ncsn (app.tex, "Code and checkpoints are available at"). PyTorch.
