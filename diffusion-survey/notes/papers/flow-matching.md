# Flow Matching (Lipman, Chen, Ben-Hamu, Nickel, Le 2022) -- arXiv 2210.02747v2

"Flow Matching for Generative Modeling". Meta AI / Weizmann. ICLR 2023.
Source: tex/fm_arxiv_v2.tex (single file), macros in tex/math_commands.tex and the preamble of fm_arxiv_v2.tex.

**Time convention: t=0 is noise p_0 = N(0, I), t=1 is (approximately) data q.** Same as Rectified Flow; opposite to SD3.

## Core problem
Train a Continuous Normalizing Flow (CNF: a neural ODE d/dt phi_t(x) = v_t(phi_t(x)) that pushes noise to data) at image scale, *without simulating the ODE during training*, and without being tied to diffusion-derived probability paths.

## What prior methods did and why that hurt
- CNFs trained by maximum likelihood (Chen et al. 2018, FFJORD): every training step needs forward and backward ODE solves; sequential and expensive; does not scale to high-dimensional images (Sec 1, Sec 5 Related Work).
- Earlier simulation-free CNF attempts: Moser Flow (Rozen et al. 2021) has intractable integrals in high dimension; Ben-Hamu et al. 2022 has biased minibatch gradients.
- Diffusion models: simulation-free via denoising score matching, but restricted to paths that come from simple diffusion SDEs. These paths are curved, "overshoot" and backtrack, never reach true noise in finite time, need long training and special fast samplers.

## The one key idea
You cannot compute the vector field that moves the whole noise distribution to the whole data distribution, but you can easily write down the vector field that moves noise to *one* data point x_1. Regressing the network on these per-sample (conditional) targets gives exactly the same gradient as regressing on the intractable whole-distribution target. Then you are free to choose the simplest per-sample path: a straight, constant-speed "optimal transport" path.

## Delta from predecessors
- vs CNFs trained by simulation: training becomes a plain regression (sample t, sample x_1, sample a noisy point, compute a closed-form target) with no ODE solves. The ODE is solved only at sampling time.
- vs diffusion paths: diffusion (VE and VP) is recovered as a special choice of Gaussian path (Example I, Sec 4.1); Flow Matching with the *same* diffusion path already trains more stably than score matching. The new OT path (Example II) makes each conditional trajectory a straight constant-speed line and the conditional target direction constant in time.
- vs Rectified Flow (concurrent): the OT CFM loss with sigma_min = 0 is the rectified flow loss. (This remark is in a commented-out line of the tex, line 639, not in the rendered paper; the rendered Related Work only says Liu et al. 2022 and Albergo & Vanden-Eijnden 2022 "arrived at similar conditional objectives".) FM frames it through probability paths and marginal vector fields; RF frames it through couplings and adds reflow.

## Mechanism step by step
1. Pick a target probability path p_t from noise (t=0) to data (t=1) and the vector field u_t that generates it. The ideal objective is to regress v_t onto u_t (FM, Eq. 5). Intractable: neither p_t nor u_t is known.
2. Build p_t as a mixture: for each data point x_1, a conditional path p_t(x|x_1) that starts at N(0,I) and ends as a tiny Gaussian around x_1 with std sigma_min (Eq. 6, 7, 10).
3. Define the marginal vector field u_t(x) as the posterior-weighted average of the conditional fields u_t(x|x_1) (Eq. 8). Theorem 1: this marginal field generates the marginal path (proof via the continuity equation).
4. Theorem 2: the Conditional Flow Matching loss (Eq. 9), which only needs per-sample quantities, equals the FM loss up to a constant independent of theta, so the gradients are identical.
5. Choose Gaussian conditional paths N(mu_t(x_1), sigma_t(x_1)^2 I) and the simplest flow that realizes them: the affine map psi_t(x) = sigma_t(x_1) x + mu_t(x_1) applied to noise x (Eq. 11). Theorem 3 gives the closed-form conditional vector field (Eq. 15).
6. Plug in a schedule. Diffusion: mu_t = alpha_{1-t} x_1, sigma_t = sqrt(1 - alpha_{1-t}^2) (VP, Eq. 18-19) or mu_t = x_1, sigma_t = sigma_{1-t} (VE, Eq. 16-17). OT: mu_t = t x_1, sigma_t = 1 - (1 - sigma_min) t (Eq. 20-22).
7. Training loop for OT: sample t ~ U[0,1], x_1 ~ data, x_0 ~ N(0,I); form psi_t(x_0) = (1 - (1 - sigma_min) t) x_0 + t x_1; regress v_t(psi_t(x_0)) onto x_1 - (1 - sigma_min) x_0 (Eq. 23).
8. Sampling: draw x_0 ~ N(0,I) and integrate the learned ODE from t=0 to 1 (dopri5 adaptive solver in experiments, or fixed-step Euler/midpoint for low-NFE studies). Exact likelihood is available through the CNF change-of-variables (Appendix C).

## Equations (TeX copied verbatim from fm_arxiv_v2.tex)

Macro key: `\gN` = calligraphic N (Gaussian); `\gL` = calligraphic L; `\FM`, `\CFM` = small "FM"/"CFM" subscripts; `\sigmamin` = `\sigma_{\text{min}}`; `\E` = `\mathbb{E}`; `\gU` = uniform distribution; `\parr{x}` = `\left(x\right)`; `\brac{x}` = `\left[x\right]`; `\too` = right arrow.

### Eq. (5) -- Flow Matching objective (label e:fm, Sec 3)
```tex
    \mathcal{L}_{\FM }(\theta) = \mathbb{E}_{t,p_t(x)}\| v_t(x)-u_t(x) \|^2,
```
Gloss: theta = network weights; v_t(x) = the network's velocity at point x and time t; u_t(x) = the true velocity that generates the desired path p_t; t ~ U[0,1]; x ~ p_t. "Make the network's arrows match the true arrows everywhere." Intractable because u_t is unknown.

### Eq. (9) -- Conditional Flow Matching objective (label e:cfm, Sec 3.2)
```tex
    \mathcal{L}_{\CFM}(\theta) = \mathbb{E}_{t,q(x_1),p_t(x\vert x_1)} \big\| v_t(x) - u_t(x\vert x_1) \big\|^2,
```
Gloss: q(x_1) = the data distribution (sample a training image x_1); p_t(x|x_1) = the per-image path (sample a noisy version x of x_1 at time t); u_t(x|x_1) = the velocity that moves x toward that one image. Everything is computable per sample.

### Theorem 2 -- equal gradients (label thm:grad_VR_equals_CVR, Sec 3.2)
```tex
Assuming that $p_t(x)>0$ for all $x\in\Real^d$ and $t\in [0,1]$, then, up to a constant independent of $\theta$, $\gL_{\CFM}$ and $\gL_{\FM}$ are equal. Hence,  
$\nabla_\theta \mathcal{L}_{\FM}(\theta) = \nabla_\theta \mathcal{L}_{\CFM}(\theta)$.
```
Gloss: nabla_theta = gradient with respect to the weights. Because the marginal field is an average of the conditional fields (Eq. 8), the cross term in the squared error is the same for both losses; the losses differ only by a theta-independent constant. So SGD on the easy loss is SGD on the right loss. Same trick as denoising score matching.

### Eq. (10) and Eq. (15) -- Gaussian conditional path and its vector field (labels e:pt_gau, e:u_t_cond_x1; Sec 4, Theorem 3)
```tex
    p_t(x\vert x_1) = \gN(x\, \vert\, \mu_t(x_1), \sigma_t(x_1)^2 I ),
```
```tex
    u_t(x\vert x_1) = 
    \frac{\sigma'_t(x_1)}{\sigma_t(x_1)}\parr{x-\mu_t(x_1)} + \mu'_t(x_1).
```
Gloss: mu_t(x_1) = the center of the Gaussian cloud at time t (moves from 0 to x_1); sigma_t(x_1) = its width (shrinks from 1 to sigma_min); I = identity (same width in every pixel); primes = time derivatives. The velocity has two parts: mu'_t moves the center, and (sigma'_t / sigma_t)(x - mu_t) shrinks the cloud toward its center. Boundary conditions: mu_0 = 0, sigma_0 = 1 (pure noise), mu_1 = x_1, sigma_1 = sigma_min.

### Eq. (20)-(21) and Eq. (23) -- the OT path, its field, and the OT CFM loss (Sec 4.1 Example II; labels e:ut_ot, e:cfm_cond_ot; Eq. 20 unlabeled)
```tex
    \mu_t(x) = tx_1, \text{ and } \sigma_t(x) = 1-(1-\sigmamin)t.
```
```tex
    u_t(x|x_1)= \frac{x_1-(1-\sigmamin)x}{1-(1-\sigmamin)t},
```
```tex
     \gL_{\CFM}(\theta) 
     =\E_{t, q(x_1), p(x_0)} \Big\|v_t(\psi_t(x_0)) - \Big (x_1 - (1-\sigma_{\min})x_0\Big) \Big\|^2.     
```
Gloss: mean and width both change linearly in t. Note the source writes mu_t(x) and sigma_t(x) on the left of Eq. 20; everywhere else these are functions of x_1 (mu_t(x_1)), so read it as mu_t(x_1) = t x_1. In Eq. 23, x_0 ~ N(0,I) is the noise sample, psi_t(x_0) = (1-(1-sigma_min)t) x_0 + t x_1 (Eq. 22) is the point on the straight line, and the target x_1 - (1-sigma_min) x_0 is constant in t: a straight, constant-speed trajectory from noise to a slightly blurred x_1. Setting sigma_min = 0 gives the rectified-flow target x_1 - x_0. psi_t is the OT displacement map between the two Gaussians N(0,I) and N(x_1, sigma_min^2 I) (McCann 1997, Example 1.7).

For contrast, the VP diffusion conditional field (Eq. 19, label e:ut_dif_with_our_method) is
```tex
    u_t(x\vert x_1) = \frac{\alpha'_{1-t}}{1-\alpha^2_{1-t}} \parr{\alpha_{1-t}x-x_1} = -\frac{T'(1-t)}{2}\brac{
    \frac{e^{-T(1-t)}x - e^{-\frac{1}{2}T(1-t)}x_1}{1-e^{-T(1-t)}}}.
```
with alpha_t = e^{-T(t)/2}, T(t) = integral of the noise scale beta; its direction changes with t (Figure 2 in PDF), and it coincides with the probability-flow ODE field of Song et al. 2020.

## Key hyperparameters (Appendix E.2, Table 3)
- Architecture: U-Net of Dhariwal & Nichol 2021 with minimal changes; all methods (FM-OT, FM-Diffusion, SM-Diffusion, DDPM, ScoreFlow) share architecture, hyperparameters and epochs.
- ImageNet-128: 500k iterations, effective batch 1536, 32 GPUs, lr 1e-4 polynomial decay, 20k warmup, 571 epochs. CIFAR-10: 391k iterations, batch 256, lr 5e-4. ImageNet-64: 157k iterations, batch 2048, lr 1e-4 constant.
- Adam beta1 0.9, beta2 0.999, eps 1e-8, no weight decay; fp32 for CIFAR/ImageNet-32, fp16 mixed for larger.
- Evaluation solver: dopri5, atol = rtol = 1e-5 (torchdiffeq).
- **sigma_min: numeric value NOT stated anywhere in the v2 tex** (only "set sufficiently small"). Do not quote a value without checking the official code.

## Headline results (Table 1 = downsampled ImageNet, unconditional; NLL in bits/dim; NFE of adaptive dopri5)
1. ImageNet 32x32: FM w/ OT NLL 3.53, FID 5.02, NFE 122 vs FM w/ Diffusion 3.54 / 6.37 / 193, Score Matching 3.56 / 5.68 / 178, DDPM 3.54 / 6.99 / 262. (Table 1 left.)
2. ImageNet 64x64: FM w/ OT NLL 3.31, FID 14.45, NFE 138 vs FM w/ Diffusion 3.33 / 16.88 / 187, DDPM 3.32 / 17.36 / 264, Score Matching 3.40 / 19.74 / 441. (Table 1 left.)
3. CIFAR-10: FM w/ OT NLL 2.99, FID 6.35, NFE 142 vs DDPM 3.12 / 7.48 / 274. Authors note CIFAR FIDs are higher than usual because the architecture was not tuned for CIFAR. (Table 1 left, Sec 6.1.)
4. ImageNet 128x128 unconditional: FM w/ OT NLL 2.90, FID 20.9, best among listed GANs (PGMGAN 21.7); IC-GAN excluded because it uses self-supervised conditioning. (Table 1 right.)
5. Training efficiency: for ImageNet-128 FM used 500k iterations at batch 1.5k vs Dhariwal & Nichol's 4.36m at batch 256 (33% less image throughput, 25% larger model). (Sec 6.1 "Faster training", Table 3.)
6. Low-NFE sampling: FM-OT needs roughly 60% of the NFEs of diffusion models to reach the same numerical error (midpoint solver, ImageNet-32; read off a plot). (Sec 6.2, Figure 7 left.) Super-resolution 64->256: FID 3.4, IS 200.8 vs SR3 5.2 / 180.1 (Table 2).

## Limitations and later fixes
- Stated: the conditional flow is OT per sample, but that "by no means impl[ies] that the marginal VF is an optimal transport solution" (end of Sec 4.1). Marginal trajectories can still curve because conditional lines from different x_1 overlap.
- Stated as future work: non-isotropic Gaussians or more general conditional paths (Conclusion).
- Later work (external knowledge, not in this paper): minibatch-OT couplings / multisample flow matching (Tong et al. 2023; Pooladian et al. 2023) pair noise and data to straighten the marginal paths; SD3 (2403.03206) cites them as curvature-reduction work and uses the sigma_min=0 rectified-flow form with logit-normal t-sampling at scale; Lipman et al.'s later "Flow Matching Guide and Code" (2024) and the facebookresearch/flow_matching library.

## Official code
Not stated in the v2 tex/PDF. Only third-party URLs appear (torchdiffeq, tensorflow/gan, openai/guided-diffusion for evaluation).
