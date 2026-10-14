# Sohl-Dickstein et al.: Deep Unsupervised Learning using Nonequilibrium Thermodynamics

Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli (Stanford, UC Berkeley). arXiv 1503.03585 (first posted March 2015), notes from v8. ICML 2015 (`\usepackage[accepted]{icml2015}`, tex line 39).
Source: `sources/sohl-dickstein/tex/icml2015_dpm.tex` (line numbers below refer to this file).
Official code: https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models (Sec. 3, line 807), built on Theano/Blocks.

## Core problem

The tractability vs flexibility trade-off (Sec. 1, line 150): simple models (a Gaussian) are easy to evaluate and fit but cannot describe rich data; flexible models p(x) = phi(x)/Z are expressive but the normalizer Z is intractable, so training, sampling and evaluating need expensive Monte Carlo.

## What prior methods did and why that hurt

Mean field, variational Bayes, contrastive divergence, score matching, pseudolikelihood, etc. "ameliorate, but do not remove, this tradeoff" (line 155). VAE-style variational methods train an inference model and a generative model against each other, but the asymmetry between them makes the inference model hard to train (line 264), and they use only a handful of layers (line 267). Multiplying a learned model by another distribution (for posteriors, inpainting) is "costly and difficult" for VAEs, GSNs, NADEs and most graphical models (line 546).

## The one key idea

"Systematically and slowly destroy structure in a data distribution through an iterative forward diffusion process. We then learn a reverse diffusion process that restores structure in data" (abstract, line 145). Because each forward step adds only a tiny amount of noise, each reverse step has the same simple functional form (Gaussian or binomial), so learning a full distribution reduces to regressing the mean and covariance of many small Gaussian steps (line 501).

## Physics analogy

- Taken from nonequilibrium statistical physics: a Markov chain that gradually converts one distribution into another (Jarzynski equality; known in ML as annealed importance sampling, AIS) (lines 178, 293).
- Forward trajectory = letting a structured distribution diffuse toward equilibrium (like ink spreading in water) until it is an identity-covariance Gaussian.
- Kolmogorov forward/backward equations: for many diffusions the time-reversed process has the same functional form (line 301, 417).
- Quasi-static process: if forward and reverse trajectories become identical (infinitesimal beta), a single sample of the forward trajectory evaluates the likelihood exactly and the bound becomes tight (lines 446-450, 487-488).
- Choosing beta_t is like choosing the schedule between equilibrium states, which in thermodynamics determines how much free energy is lost (line 509).
- Langevin dynamics as the stochastic realization of the Fokker-Planck equation is cited as related (line 298).

## Delta from its predecessors

First paper of the line. Relative to VAEs/wake-sleep: the inference (forward) process is fixed to a simple form so the generative (reverse) process has the same form; thousands of layers/time steps instead of a handful; easy multiplication with other distributions; analytic upper/lower bounds on per-step entropy (lines 259-271).

## Mechanism, step by step

1. Pick a tractable target pi(y): identity-covariance Gaussian (continuous data) or independent binomial (binary data).
2. Forward trajectory: apply T small diffusion steps q(x^(t)|x^(t-1)) with rate beta_t, turning data into pi (Eqs. 1-3, Table App.1).
3. Reverse trajectory: a learned chain starting from pi, with the same form: Gaussian steps with learned mean f_mu(x^(t),t) and covariance f_Sigma(x^(t),t) (or learned bit-flip probability f_b) (Eqs. 4-5, Table App.1).
4. Model probability p(x^(0)) is an intractable integral over all latent trajectories; rewrite it as an average of the forward/reverse probability ratio over forward trajectories, as in AIS (Eqs. 6-9).
5. Training: maximize a Jensen lower bound K on the log-likelihood, which reduces to KL divergences between forward posteriors q(x^(t-1)|x^(t),x^(0)) and reverse steps plus analytic entropies (Eqs. 10-14). Gaussian: beta_{2..T} also learned by gradient ascent on K with "frozen noise" (beta_1 fixed small); binomial: beta_t = (T-t+1)^{-1} (Sec. 2.4.1).
6. Posteriors (denoising/inpainting): multiply each reverse step by r(x^(t)); for Gaussians this perturbs the mean by f_Sigma times the gradient of log r (Sec. 2.5, Table App.1).

## Key equations (verbatim TeX from `sources/sohl-dickstein/tex/icml2015_dpm.tex`)

Macros (lines 61-83), needed to render: `\mb`=`\mathbf`, `\mc`=`\mathcal`,
`\pf` = `q\left( \mb x^{(t)} | \mb x^{(t-1)} \right)` (forward step),
`\pr` = `q\left( \mb x^{(t-1)} | \mb x^{(t)} \right)`,
`\qr` = `p\left( \mb x^{(t-1)} | \mb x^{(t)} \right)` (learned reverse step),
`\pst` = `q \left( \mb x^{(0)} \right)` (data distribution),
`\qst` = `p \left( \mb x^{(T)} \right)`,
`\ptraj` = `q\left( \mb x^{(0\cdots T)} \right)`,
`\pcondtraj` = `q\left( \mb x^{(1\cdots T)} | \mb x^{(0)} \right)`,
`\qtraj` = `p\left( \mb x^{(0\cdots T)} \right)`,
`\ptarget` = `\pi\left( \mb x^{(T)} \right)`,
`\qmarg` = `p\left( \mb x^{(0)} \right)`.
Note the naming: q is the forward (data -> noise) process, p is the learned reverse model, same as DDPM.

### E1. Forward trajectory, Eqs. (2)-(3), lines 396 and 401

Verbatim:
```
\pf &= T_\pi\left( \mb x^{(t)} | \mb x^{(t-1)}; \beta_t \right)
```
```
\ptraj &= \pst \prod_{t=1}^T \pf
```
Gaussian case, Table App.1 row "Forward diffusion kernel" (line 1074):
```
\mc N \left( \mb x^{(t)} ; \mb x^{(t-1)} \sqrt{1 - \beta_t}, \mb I \beta_t \right)
```
Renderable: `q(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}) = \mathcal{N}(\mathbf{x}^{(t)}; \mathbf{x}^{(t-1)}\sqrt{1-\beta_t}, \mathbf{I}\beta_t)` and `q(\mathbf{x}^{(0\cdots T)}) = q(\mathbf{x}^{(0)})\prod_{t=1}^T q(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)})`.
Gloss: T_pi is a Markov diffusion kernel whose equilibrium is pi. beta_t is the diffusion rate at step t (how much noise). x^(t) is the state after t steps, x^(0) is data. The whole forward path's probability is the data probability times each small step. The Gaussian kernel is the same one DDPM uses (shrink by sqrt(1-beta_t), add variance beta_t).

### E2. Reverse trajectory, Eqs. (4)-(5), lines 412-413, and Gaussian kernel (Table App.1, line 1075)

Verbatim:
```
\qst &= \ptarget \\
\qtraj &= \qst \prod_{t=1}^T \qr
```
```
\mc N \left( \mb x^{(t-1)} ;\mb f_\mu\left( \mb x^{(t)}, t \right), \mb f_\Sigma\left( \mb x^{(t)}, t \right) \right)
```
Gloss: the generative model starts from the simple distribution pi at time T and runs backward. Each reverse step is Gaussian with mean f_mu and covariance f_Sigma, both output by a neural network given the current noisy state and time t. These (plus beta_{1..T}) are the only things learned.

### E3. Model probability via forward/reverse ratio, Eq. (9), line 442

Verbatim:
```
&= \int d\mb x^{(1\cdots T)} \pcondtraj \cdot \nonumber \\ &\qquad \quad \qst \prod_{t=1}^T \frac{\qr}{\pf}
```
Renderable: `p(\mathbf{x}^{(0)}) = \int d\mathbf{x}^{(1\cdots T)}\, q(\mathbf{x}^{(1\cdots T)}|\mathbf{x}^{(0)})\; p(\mathbf{x}^{(T)}) \prod_{t=1}^T \frac{p(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)})}{q(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)})}`
Gloss: to score a data point, noise it with the forward process and multiply, step by step, how likely the reverse model finds each backward step divided by how likely the forward step was. Averaging over forward samples gives p(x^(0)); this is the AIS / Jarzynski trick.

### E4. The lower bound K, Eqs. (13)-(14), lines 469-483

Verbatim:
```
L
&\geq K \label{eq ineq LK}\\
K = & 
-\sum_{t=2}^T \int d\mb x^{(0)}d\mb x^{(t)} q\left( \mb x^{(0)}, \mb x^{(t)} \right) \cdot
\nonumber\\  & \qquad \qquad
	D_{KL}\left( 
		q\left( \mb x^{(t-1)} | \mb x^{(t)}, \mb x^{(0)} \right)
			\middle|\middle|
		\qr
	\right)  \nonumber \\
&	+ H_q\left( \mb X^{(T)} | \mb X^{(0)} \right) - H_q\left( \mb X^{(1)} | \mb X^{(0)} \right)
	- H_p\left( \mb X^{(T)} \right)
```
where L (Eq. 10, line 455) is `L &= \int d\mb x^{(0)} \pst \log \qmarg`, the expected log-likelihood of the data.
Gloss: L is the average log-probability the model gives real data (to maximize). K is a computable lower bound. Main term: for each step t, a KL divergence (a mismatch score) between the true "one step back" distribution when the clean image is known, q(x^(t-1)|x^(t),x^(0)), and the model's reverse step p(x^(t-1)|x^(t)), averaged over data and noisy states. The H terms are entropies fixed by the forward process and the prior (constants if beta is fixed). This is the same bound DDPM rewrites as its Eq. 5 (DDPM credits it to this paper, DDPM App. A). Training = argmax over reverse steps of K (Eq. 15).

### E5. Entropy bounds on each reverse step, Eq. (24), lines 663-671 (Sec. 2.6)

Verbatim:
```
H_q\left( \mb X^{(t)} | \mb X^{(t-1)} \right) + H_q\left( \mb X^{(t-1)} | \mb X^{(0)} \right)  - H_q\left( \mb X^{(t)} | \mb X^{(0)} \right) 
\nonumber \\ \qquad
 \le
H_q\left( \mb X^{(t-1)} | \mb X^{(t)} \right)
\le
H_q\left( \mb X^{(t)} | \mb X^{(t-1)} \right)
```
Gloss: the randomness of the true reverse step is sandwiched between two quantities computable from the forward process alone. DDPM's two fixed variance choices, sigma_t^2 = beta_t and tilde beta_t, are exactly these upper and lower extremes (DDPM line 195).

## Key hyperparameters

- Gaussian image models: T = 1000 (T = 500 for bark) (App. D.2, line 1285, "For all image experiments a number of timesteps T=1000 was used, except for the bark dataset which used T=500").
- Swiss roll: 40 Gaussian steps; RBF network, 1 hidden layer, 16 units (App. D.1).
- Binary heartbeat: 2000 binomial steps; MLP 3 hidden layers x 50 units; beta_t = (T-t+1)^{-1}.
- Gaussian beta_{2..T} learned by gradient ascent on K, beta_1 fixed small; footnote (line 513) says a fixed schedule works just as well.
- Image network: multi-scale convolution (mean-pool to multiple scales, conv, upsample, sum, soft ReLU), then 1x1 conv layers; time dependence via J Gaussian "bump" readout functions g_j(t); mean and variance parameterized as perturbations of the forward kernel with Sigma diagonal (App. D.2). CIFAR-10 adds a parallel dense pathway; MNIST uses only the dense pathway.
- Optimizer: SFO, except CIFAR-10 with RMSprop via the open-source code (line 804).

## Datasets

Toy: 2-D swiss roll; binary "heartbeat" sequences (length 20, pulse every 5th bin). Images: MNIST, CIFAR-10, dead leaves, bark textures (T01-T04 from Lazebnik et al.).

## Headline results

1. Table 1 (lower bound K on holdout log-likelihood; data scaled to variance 1 except heartbeat): Swiss roll 2.35 bits; heartbeat -2.414 bits/seq (true -2.322, "nearly perfect"); bark -0.55 bits/pixel; dead leaves 1.489 bits/pixel; CIFAR-10 5.4 +/- 0.2 bits/pixel (dequantized; footnote says an earlier arXiv version reported higher bounds due to learning 8-bit quantization).
2. Table 2: dead leaves 1.489 bits/pixel vs MCGSM 1.244 (state of the art at the time).
3. Table 2: MNIST Parzen-window estimate 317 +/- 2.7 bits vs Deep GSN 309, DBN 199, Stacked CAE 174, adversarial net 325, "perfect model" (Parzen on training data) 349. Parzen estimate, not the bound.
4. Figures: swiss roll learned (Fig. 1); CIFAR-10 denoising at SNR 1 and samples (Fig. 3); bark inpainting of central 100x100 region with long-range structure (Fig. 5).

## Limitations and who addressed them

- The paper states no explicit limitation. The main one, per DDPM (DDPM main.tex line 127): "there has been no demonstration that they are capable of generating high quality samples". CIFAR-10 samples were blurry and no sample-quality metric (FID/IS) is reported. Addressed by DDPM (epsilon-prediction, fixed schedule, L_simple, U-Net): CIFAR-10 FID 3.17.
- Learning covariances and beta schedule adds complexity; DDPM fixes both. Learned variances later reinstated stably by Improved DDPM.
- Log-likelihood bounds are in a scaled continuous convention and not directly comparable to bits/dim of 8-bit data; DDPM's table lists it as NLL <= 5.40 without showing the conversion (uncertain).
- Slow sampling (1000 steps for images); addressed much later (Improved DDPM, DDIM).
- Networks (MLP / multi-scale conv) were weak; DDPM's U-Net with attention was a large part of the jump in quality.
