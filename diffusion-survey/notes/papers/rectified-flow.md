# Rectified Flow (Liu, Gong, Liu 2022) -- arXiv 2209.03003v1

"Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow". UT Austin. NeurIPS 2022 style file.
Sources: tex/main.tex, tex/arxiv_textfiles/method_new.tex (method + theory), experiments.tex, appendix.tex. Macros in tex/qiangstyle.sty.

**Time convention: t=0 is the source (noise, pi_0), t=1 is the target (data, pi_1).** SD3 uses the opposite convention.

## Core problem
The "transport mapping problem" (method_new.tex, Introduction): given only samples X_0 ~ pi_0 and X_1 ~ pi_1, find a map T so that T(Z_0) ~ pi_1 when Z_0 ~ pi_0. Generative modeling (pi_0 = Gaussian noise, pi_1 = images) and domain transfer (pi_0 = human faces, pi_1 = cat faces) are both instances.

## What prior methods did and why that hurt
- GANs: minimax training, unstable, mode collapse, heavy tuning.
- MLE models (VAE, normalizing flows, autoregressive): intractable likelihood or restricted architectures.
- Diffusion / score SDEs (DDPM, score-based) and their deterministic versions (probability flow ODE, DDIM): high quality, but sampling requires many calls of the network because the ODE/SDE trajectories are curved, so a numerical solver needs many small steps. Also "substantial hyper-parameter search in an involved design space".
- Optimal transport solvers: slow in high dimension, and exact OT cost does not track learning performance well.

## The one key idea
Train an ODE whose velocity field copies the direction of the straight line from X_0 to X_1, as much as a causal (non-cheating) flow can. Then repeat the procedure on the flow's own (noise, output) pairs ("reflow"), which makes the paths straighter each time until one Euler step is almost exact.

## Delta from predecessor (vs diffusion SDE/ODE with curved paths)
- Interpolation is a straight line X_t = t X_1 + (1-t) X_0 at constant speed, not a variance-preserving/exploding noise schedule. Diffusion PF-ODEs and DDIM are shown to be (approximately) special cases of a *nonlinear* rectified flow with curved interpolants X_t = alpha_t X_1 + beta_t X_0 (Sec 2.3, 2.3.1).
- Purely ODE-based; no SDE, no score function, no noise schedule to tune.
- With the straight interpolant it gets two guarantees diffusion paths do not have: rectification never increases any convex transport cost, and reflow provably straightens the paths. With curved interpolants these guarantees are lost (Sec 2.3).
- It works between any two distributions, not only noise to data (image-to-image translation, domain adaptation).

## Mechanism step by step
1. Draw pairs (X_0, X_1). For generation they are independent: X_0 Gaussian noise, X_1 a training image (pi_0 x pi_1).
2. Draw t ~ Uniform[0,1], form X_t = t X_1 + (1-t) X_0.
3. Regress the network v(X_t, t) onto the constant direction X_1 - X_0 with squared error (Eq. 1). PyTorch one-liner in appendix.tex Algorithm: `Loss = ( Model(t*x1+(1-t)*x0, t) - (x1-x0) ).pow(2).mean()`.
4. Why this is not trivial: the straight-line process X_t is non-causal. To move along it you must already know X_1. Different lines cross each other, and at a crossing point the "right direction" is ambiguous. The least-squares optimum is the average direction of all lines passing through that point at that time (Eq. 2).
5. Non-crossing: a well-defined ODE dZ_t = v(Z_t,t)dt with a unique solution cannot have two trajectories through the same point at the same time in different directions. So the learned flow "rewires" trajectories at intersections (Figure 2 in PDF: panels a-d) instead of crossing them, while keeping the same marginal distribution at every time t (Theorem 3.3).
6. Sampling: start Z_0 ~ pi_0, integrate dZ_t = v(Z_t,t)dt from t=0 to 1 (Euler with N steps or RK45). The pair (Z_0, Z_1) is the *rectified coupling*: a deterministic pairing of noise to output.
7. Rectified coupling has no larger convex transport cost than the original random pairing, for every convex cost at once (Theorem 3.5). Intuition: the flow's paths are a rewiring of the original straight segments, so by Jensen the end-to-end displacement can only shrink.
8. Reflow: retrain a new flow on pairs (Z_0^k, Z_1^k) produced by the previous flow. Because these pairs were already produced by a non-crossing flow, their straight-line interpolations cross much less, so the new flow's paths are straighter. Straightness S (Eq. 3) decays like O(1/K) in the number of reflows (Theorem 3.7).
9. Distill (optional): fit a one-step model z_1 = z_0 + v(z_0, 0) on the k-rectified pairs. For the one-step (k=1) distillation on CIFAR-10 they swap the L2 loss for LPIPS (appendix.tex, "Experiment Configuration on CIFAR10").

## Equations (TeX copied verbatim from source)

Macro key (qiangstyle.sty / def.tex): `\bbb ... \eee` = numbered `align`; `\tg` = `\pi`; `\norm{x}` = `\left\|x\right\|`; `\E` = `\mathbb{E}`; `\e{x}` = `\E\left[x\right]`; `\dt` = `\mathrm{d}t`; `\d` = `\mathrm{d}`; `\traj{Z}` = bold Z (whole trajectory).

### Eq. (1) -- the rectified flow objective (method_new.tex, label equ:mainf, Sec 2.1)
```tex
\min_{v} %
\int_0^1\E \left [ \norm{( X_1 - X_0) - v\big (X_t,~ t\big)}^2
\right ] \dt, 
~~~~~\text{with}~~~~
X_t = t X_1 + (1-t) X_0, 
```
Gloss: X_0 = a noise sample (or source image); X_1 = a data sample (or target image); t = time in [0,1]; X_t = the point a fraction t of the way along the straight segment from X_0 to X_1; X_1 - X_0 = the segment's direction (also its velocity, since the segment is crossed in unit time); v(X_t, t) = the network's predicted velocity at that point and time; the norm-squared is the squared error; the integral over t and expectation E average over all times and all pairs. In words: "at every point on every straight line, predict which way the line is going".

### Eq. (2) -- the optimal velocity (method_new.tex, label equ:vxoracle, Sec 2.2)
```tex
 v^X(x,t) = \E[X_1- X_0 ~|~ X_t=x],
```
Gloss: v^X(x,t) = the best possible answer to Eq. 1 at location x and time t; it is the average (conditional expectation) of the directions X_1 - X_0 over all line segments that pass through x at time t. Where lines cross, it averages their directions, which is exactly what un-crosses them.

### Eq. (3) -- straightness measure (method_new.tex, label equ:straight, Sec 2.2)
```tex
S(\traj Z) = \int_0^1 \e{\norm{(Z_1-Z_0) -\dot Z_t}^2} \dt.
```
Gloss: Z = a whole trajectory of the flow; Z_1 - Z_0 = the straight shortcut from start to end; dot Z_t = the instantaneous velocity at time t; S = average squared deviation of the velocity from the shortcut. S = 0 means perfectly straight paths, so one Euler step Z_1 = Z_0 + v(Z_0, 0) is exact.

### Theorem 3.5 -- rectification does not increase convex transport cost (method_new.tex, label thm:cost, Sec 3.2)
```tex
$$\E[c(Z_1 - Z_0)] \leq \E[c(X_1 - X_0)].$$ 
```
Gloss: c = any convex cost of displacement (e.g. distance, squared distance); (X_0,X_1) = the original random pairing; (Z_0,Z_1) = the rectified pairing. The new pairing moves mass no farther than the old one, simultaneously for every convex c ("Pareto descent"), without targeting a specific c.

### Theorem 3.7 (informal statement in Sec 2.2) -- reflow straightens (method_new.tex, after label equ:straight)
```tex
\min_{k\in\{0\cdots K\}} S(\traj Z^k) \leq \frac{\E[\norm{X_1-X_0}^2]}{K}.  
```
Gloss: Z^k = the k-th rectified flow (k reflows applied); K = number of reflow rounds; the right-hand side is the average squared length of the original pairs divided by K. So after K rounds at least one flow has straightness error at most O(1/K). (The formal Theorem 3.7 in Sec 3.3 bounds the sum over k of S(Z^{k+1}) + V((Z_0^k, Z_1^k)).)

## Key hyperparameters (CIFAR-10, appendix.tex)
- Network: DDPM++ U-Net from Song et al. 2020 (score SDE codebase); EMA ratio 0.999999; Adam, lr 2e-4, dropout 0.15.
- Reflow: generate 4 million (z_0, z_1) pairs, fine-tune the i-rectified flow for 300,000 steps to get the (i+1)-rectified flow.
- Distillation to k steps: sample t from {0, 1/k, ..., (k-1)/k}; for k=1 use LPIPS instead of L2.
- Sampling: Euler with constant step 1/N, or scipy RK45 with the same tolerances as Song et al.
- t ~ Uniform[0,1] during training (the uniform choice SD3 later replaces).

## Headline results
1. One-step CIFAR-10: 2-rectified flow + distillation, NFE 1, FID 4.85 (IS 9.01, recall 0.50). Claimed SOTA for one-step diffusion/flow models; beats TDPM's 8.91, the best one-step U-Net model they cite. (Table 1(a), one-step block; experiments.tex.)
2. Without distillation, one Euler step: 1-rectified flow FID 378 (useless), 2-rectified flow FID 12.21, 3-rectified flow FID 8.15. VP ODE at one step: 451. This is the "reflow makes one step work" result. (Table 1(a).)
3. Best one-step recall: 3-rectified flow + distill, recall 0.51 (FID 5.21), beating StyleGAN2+ADA's 0.49. (Table 1(a); Table 1(b).)
4. Full RK45 solve: 1-rectified flow NFE 127, FID 2.58, recall 0.57, IS 9.60 -- best among ODE methods; VP ODE needs NFE 140 for FID 3.93, sub-VP ODE NFE 146 for FID 3.16. VP SDE at 2000 steps: FID 2.55. (Table 1(a).)
5. Reflow trades off: 2- and 3-rectified flows need fewer RK45 steps (NFE 110, 104) but have worse full-solve FID (3.36, 3.96). (Table 1(a); Sec 5.2 text: "worsens the results in the large step regime due to the accumulation of error on estimating v^x".)
6. Qualitative: 1-rectified flow gives good results with few Euler steps on 256x256 datasets (cat faces etc.), and the same algorithm does unpaired image-to-image translation (human to cat faces) and domain adaptation (Figure 1, Sec 5.3, 5.4).

## Limitations and later fixes
- Stated in paper: each reflow accumulates error in estimating v, so full-solve quality degrades (FID 2.58 -> 3.36 -> 3.96); reflow is expensive (simulate millions of pairs, retrain).
- Stated: rectification is Pareto descent on all convex costs, not optimal for any given cost; repeated rectification does not generally reach the optimal-transport coupling (except in 1-D).
- Stated: with curved interpolants (the nonlinear extension that covers PF-ODE/DDIM) neither cost reduction nor straightening is guaranteed.
- Uniform t sampling: SD3 (2403.03206) later shows logit-normal timestep sampling beats uniform for rectified flow at scale.
- Later work (external knowledge, not in this paper): InstaFlow (Liu et al. 2023) applies reflow + distillation to Stable Diffusion; SD3 cites it. Improved reflow training and minibatch-OT couplings (e.g. Tong et al. 2023; Pooladian et al. 2023) are other attempts to get straighter paths without full reflow.
- Concurrent, equivalent formulations: Flow Matching with sigma_min = 0 (2210.02747), Stochastic Interpolants (Albergo and Vanden-Eijnden 2022).

## Official code
Not stated in the v1 tex or PDF text (conclusion.tex is empty; no github URL). The experiments say the implementation is modified from the open-source code of Song et al. 2020. (Commonly known repo github.com/gnobitab/RectifiedFlow is NOT cited in v1; verify before using.)
