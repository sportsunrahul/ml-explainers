# Joint Embedding Predictive Architectures Focus on Slow Features (2211.10831v1)

## 1. Identity
- Title: Joint Embedding Predictive Architectures Focus on Slow Features
- arXiv: 2211.10831v1 (version read). Published 2022-11-20 (abs.html: citation_date 2022/11/20, "Submitted on 20 Nov 2022").
- Authors: Vlad Sobal (first) ... Yann LeCun (last); 6 authors (Sobal, S V, Jalagam, Carion, Cho, LeCun). NYU / Meta AI.
- Venue: not stated on the abs page. The LaTeX uses `neurips_2022.sty` with the `[final]` option.
- Repo: https://github.com/vladisai/JEPA_SSL_NeurIPS_2022 (footnote 1, Section 1 of the paper). `meta.json` says `repo: null`; the paper itself gives this URL.
- Source read: `src/main.tex` (equations), `src/table_vals` (Table 1 numbers), `paper.pdf` (figures, page numbers).

## 2. Core problem
World models for pixel-based environments are usually trained with pixel reconstruction, which suffers from "object vanishing" (the objective does not distinguish important objects). JEPA offers a reconstruction-free alternative. The paper asks whether a JEPA trained offline on (image, action) sequences with VICReg or SimCLR objectives learns a representation that keeps the controllable object, and what it keeps instead when the background has distractor noise.

## 3. What prior methods did and why that hurt (per the paper)
- Reconstruction world models (Dreamer, DreamerV2, PlaNet, Atari video models): the pixel loss weights every pixel the same, so small important objects vanish (Section 1, citing Okada & Taniguchi "Dreaming").
- JEPA with VICReg/SimCLR (this paper): no decoder, but the temporal prediction loss "can be easily minimized by paying attention to noise that does not change with time, making the system ignore all other information" (Section 2.1).

## 4. The one key idea
A JEPA trained to predict the next-step embedding is minimised by features that do not change over time ("slow features", after Wiskott & Sejnowski's slow feature analysis). When the slowest thing in the input is fixed background noise, the representation contains only the noise and drops the moving dot. When the noise changes every frame, JEPA ignores it and does as well as or better than reconstruction.

Precise meaning of "slow" and "noisy/fast" in this paper:
- Slow feature = anything constant across the 17 frames of one sequence. In the experiments this is the FIXED distractor noise (same noise image on every frame, re-sampled only between sequences) and, in the 3-dots dataset, the stationary dot.
- Fast/noisy feature = distractor noise re-sampled every frame ("changing" noise). The dot's position is a fast but predictable feature (it moves by the known action each frame).
- Polarity: "focus on slow features" is a FAILURE MODE here. JEPA keeps the slow distractor and loses the dot.

## 5. Mechanism, step by step
1. Data: offline sequences (o_1..o_{T+1}, a_1..a_T), T = 16 actions, 17 frames. One dot in a unit square, 28 x 28 images; action = delta in dot coordinates.
2. Encoder g_phi: o_t -> s_t in R^D (3 conv layers + BatchNorm + ReLU, avg pool, linear to D = 512).
3. Forward model (predictor) f_theta: single-layer GRU, hidden 512, input = action (size 2). Unrolled from s~_1 = g_phi(o_1): s~_t = f_theta(s~_{t-1}, a_{t-1}). Gradients propagate through time. No image augmentations.
4. Loss: prediction loss ||s~_t - s_t||_2^2 at every step t = 2..T+1, PLUS collapse prevention:
   - VICReg variant: variance + covariance terms applied to the encoder outputs S_t separately at each time step (eqs (5)-(9), Appendix A.3.1).
   - SimCLR variant: InfoNCE with (S_t,i, S~_t,i) as the positive pair, other batch elements as negatives (eqs (10)-(12), Appendix A.3.2).
   There is NO EMA target, NO stop-gradient, and the predictor is the GRU forward model. Collapse is prevented by the explicit VICReg/InfoNCE terms, not by an asymmetry.
5. Baselines trained the same way: Reconstruction (decoder symmetric to the encoder, pixel L2, eq (16)), IDM (linear inverse-dynamics head predicting a_t from s_t, s_{t+1}, eqs (13)-(15)), Supervised (probe error backpropagated end to end, lower bound), Random (frozen random weights), Center (always predict the centre, upper bound).
6. Evaluation: freeze encoder and predictor, unroll from o_1 with the actions, train a linear probe q: R^512 -> R^2 to recover the dot location at every step (eq (18)); report RMSE across the 17 steps, 3 seeds.
7. The trivial solution (Section 2.1): with fixed Gaussian noise the encoder can copy the noise, g_phi(o_t) = s ~ N(0, sigma^2 I); then s_t = s_{t+1}, the forward model is the identity, and every VICReg term is exactly zero (eqs (1)-(4)). For SimCLR, the same solution satisfies perfect alignment + uniformity (Wang & Isola theorem 1), so InfoNCE is minimised too.

## 6. Equations (TeX verbatim from `src/main.tex`)
No custom macros are needed; `\mathds{1}` comes from `dsfont`.

(1) Prediction loss vanishes under the trivial solution (Section 2.1, p3). Paper's wording: "the model can extract features by directly copying the values of the noise from the input ... We assume that the forward model has converged to identity".
```
\mathcal{L}_\mathrm{prediction} &= \frac{1}{TN}\sum_{t=1}^{T}\sum_{i=1}^N\Vert f_\theta(S_{t, i}, A_{t,i}) - g_\phi(O_{t+1, i})\Vert_2^2  \nonumber \\
&= \frac{1}{TN}\sum_{t=1}^{T}\sum_{i=1}^N\Vert S_{t, i} - S_{t+1, i}\Vert_2^2 = \frac{1}{TN}\sum_{t=2}^{T+1}\sum_{i=1}^N\Vert S_{t, i} - S_{t, i}\Vert_2^2 = 0
```
Gloss: S_{t,i} = encoder output for batch element i at step t (matrix S_t is N x D); A_{t,i} = action; f_theta = forward model; g_phi = encoder; T = 16; N = batch size.

(2) Variance of the copied noise (p3):
```
&\mathrm{Var}(s_t) = \frac{1}{N-1}\sum_{i=1}^N (s_i - \bar s)^2 = \sigma^2 & \text{As } s \sim \mathcal{N}(0, \sigma^2 I)
```

(3) `\label{eq:var}` and (4) `\label{eq:cov}` (p3). Paper: "Equation (3) holds for large enough sigma, (4) holds because the noise variables are independent across episodes."
```
&\mathcal{L}_\mathrm{variance} = \frac{1}{(T+1)D}\sum_{t=1}^{T+1}\sum_{j=1}^D \max \left(0, \gamma - \sqrt{\mathrm{Var}(S_{t, :, j}) + \epsilon}\right)  = 0 \label{eq:var} \\
&\mathcal{L}_\mathrm{covariance} =  \frac{1}{(T+1)(N-1)}\sum_{t=1}^{T+1}\sum_{i=1}^D \sum_{j=i+1}^D (S_t S_t^\top)_{i,j} = 0 & \label{eq:cov}
```
Gloss: gamma = VICReg variance target; epsilon = numerical constant; D = 512 embedding size; S_{t,:,j} = column j of the batch at step t; the covariance term sums the off-diagonal entries (upper triangle).

(5)-(9) Full VICReg-JEPA objective (Appendix A.3.1, p9-10). Paper: "We consider representations at each time step separately for calculating variance and covariance losses, while the representation loss becomes the prediction error between forward model and encoder outputs."
```
&\mathcal{L}_\mathrm{VICReg} = \alpha \mathcal{L}_\mathrm{prediction} + \beta \mathcal{L}_\mathrm{variance} + \mathcal{L}_\mathrm{covariance} \\
&\mathcal{L}_\mathrm{prediction} = \frac{1}{NT}\sum_{t=1}^T\sum_{i=1}^N\Vert f_\theta(S_{t, i}, A_{t, i}) - g_\phi(O_{t+1, i})\Vert_2^2 =
\frac{1}{NT}\sum_{t=2}^{T+1}\sum_{i=1}^N \Vert \tilde S_{t, i} - S_{t, i}\Vert_2^2 \\
&\mathrm{Var}(v) = \frac{1}{N-1}\sum_{i=1}^N (v_i - \bar v)^2 \\
&\mathcal{L}_\mathrm{variance} = \frac{1}{T+1}\sum_{t=1}^{T+1}\frac{1}{D}\sum_{j=1}^D \max \left(0, \gamma - \sqrt{\mathrm{Var}(S_{t, :, j}) + \epsilon}\right)  \\
&\mathcal{L}_\mathrm{covariance} = \frac{1}{T+1} \sum_{t=1}^{T+1}\frac{1}{N-1}\sum_{i=1}^D \sum_{j=i+1}^D (S_t S_t^\top)_{i,j}
```
Gloss: alpha, beta = loss weights (tuned per setting); \tilde S_t = predictor outputs; the prediction term grades steps 2..T+1 only (step 1 is the encoder output itself).

(10)-(12) SimCLR-JEPA objective (Appendix A.3.2, p10):
```
&\mathrm{expsim}(u, v) = \exp \left( \frac{u^\top v}{\tau \Vert u \Vert \Vert v \Vert} \right)\\
&\mathrm{InfoNCE}(S_t, \tilde S_t) = -\frac{1}{N}\sum_{i=1}^{N}\log \frac{\mathrm{expsim}(S_{t, i}, \tilde S_{t, i})}{
\sum_{k=1}^N \mathrm{expsim}(S_{t, i}, \tilde S_{t, i})) + \mathds{1}_{k \neq i} \mathrm{expsim}(S_{t, i}, S_{t, k}) } \\
&\mathcal{L}_\mathrm{SimCLR} = \frac{1}{2T} \sum_{t=2}^{T+1} \mathrm{InfoNCE}(S_t, \tilde S_t) + \mathrm{InfoNCE}(\tilde S_t, S_t)
```
Gloss: tau = temperature; positives are (encoder output, prediction) at the same step; negatives are other batch elements at the same step.

Data model (Appendix A.2, unnumbered): `c_{t+1} = \max(0, \min(c_t + a_t, 1))`, `\Vert a_t \Vert_2 \leq D` with D = 0.14, direction `u_t \sim \mathrm{VonMises}(\omega, 1.3)`, step `d_t \sim \mathrm{Uniform}(0, D)`, `a_t = d_t u_t`; noisy frame `\hat o_t = o_t + \alpha Z_t` (text says "coefficient beta" then writes alpha; alpha is the symbol used everywhere else).

Probe (18), Appendix A.4: `\mathcal{L}_\mathrm{probing} = \frac{1}{TN}\sum_{t=1}^{T}\sum_{i=1}^N \Vert q(\tilde S_{t,i}) - C_{t, i} \Vert_2^2`, C_{t,i} = true dot location.

## 7. Measurement setup (analysis lens)
| Item | Value | Where |
|---|---|---|
| Environment | one dot in a 1 x 1 square, Gaussian blur sigma = 0.05, 28 x 28 frames | A.2 |
| Episode | 17 frames = 16 actions, max step D = 0.14 | Sec 3, A.2 |
| Data sizes | 1 million pretraining sequences; 300,000 prober; 10,000 eval | Sec 3 |
| Noise types | uniform (each pixel ~ Uniform) or structured (CIFAR-10 image overlay) | Sec 3, A.2 |
| Noise timing | changing (re-sampled each frame) or fixed (same across the sequence, re-sampled per sequence) | Fig 2 caption |
| Noise level | alpha = ratio of brightest noise pixel to brightest dot pixel; x-axis 0.0 to 3.0 in steps of 0.5 (Fig 3) | Sec 3, Fig 3 |
| Encoder | 3 conv (5x5 s2 p2 -> 32, 5x5 s2 p2 -> 64, 3x3 s1 p1 -> 64) + BN + ReLU, avgpool 2x2, linear -> 512 | A.7 |
| Predictor | 1-layer GRU, hidden 512, input size 2 (the action) | A.7 |
| Collapse prevention | VICReg var+cov terms per time step, or InfoNCE; no EMA, no stop-grad | Sec 2, A.3 |
| Metric | RMSE of linear-probed dot location, mean over 17 steps, 3 seeds | Sec 3, Fig 3 |
| HP search | 100 random hyperparameter configs per (method, noise type, level); best re-run with 3 seeds | A.9 |
| Compute | AMD MI50 or Nvidia RTX 8000; each run under one hour of GPU time | A.9 |
| Held fixed | encoder and predictor architectures across methods; no augmentations | Sec 2 |

## 8. Results worth remembering
- Fig 3 (p4), main result, four panels (changing uniform, changing structured, fixed uniform, fixed structured), RMSE vs alpha. Qualitative reading of the plot (no exact values in the source):
  - No noise (alpha = 0): all methods near the Supervised floor.
  - Changing noise (top row): VICReg and SimCLR stay near Supervised and reconstruction for all alpha; text: "JEPA methods perform on par or better than reconstruction when distractor noise changes every time step".
  - Fixed noise (bottom row): VICReg and SimCLR rise to the Random/Center ceiling by alpha = 1.0 (uniform) and already by alpha = 0.5-1.0 (structured). Text: "Reconstruction performs well in all settings with alpha <= 1.5, while JEPA-based methods fail in the presence of fixed noise, both structured and uniform."
  - IDM is good everywhere (but only captures the controlled dot, see Table 1).
- Fig 5 (p11, Appendix A.5): without per-noise re-tuning, VICReg/SimCLR fail at lower fixed-noise levels, while reconstruction fails on changing uniform noise.
- Table 1 (p12, Appendix A.6.2), 3-dots dataset (red = action-controlled dot, green = random dot with unknown actions, blue = stationary dot), RMSE over 17 steps, mean +- std over 3 seeds. Columns: Average, Action, Random, Stationary.
  - VICReg: 0.229+-0.031, 0.277+-0.041, 0.273+-0.044, 0.066+-0.026
  - SimCLR: 0.158+-0.001, 0.193+-0.001, 0.193+-0.002, 0.025+-0.001
  - IDM: 0.234+-0.001, 0.035+-0.000, 0.298+-0.002, 0.272+-0.000
  - Supervised: 0.104+-0.000, 0.010+-0.001, 0.180+-0.001, 0.005+-0.000
  - Reconstruction: 0.107+-0.000, 0.021+-0.001, 0.182+-0.000, 0.026+-0.002
  - Random: 0.260+-0.001, 0.235+-0.002, 0.278+-0.002, 0.265+-0.004
  - Center: 0.299+-0.000, 0.304+-0.000, 0.304+-0.000, 0.289+-0.000
  Reading: VICReg/SimCLR locate the STATIONARY dot almost perfectly (0.066, 0.025) and lose the action-controlled dot (0.277, 0.193, near the Random baseline 0.235). Reconstruction gets both (0.021, 0.026). IDM gets only the controlled dot (0.035) and ignores the stationary one (0.272).
- What would falsify the claim: a VICReg- or SimCLR-JEPA trained on the fixed-noise moving dot (alpha >= 1) that, after the same 100-config search, reaches dot-location RMSE comparable to reconstruction; or the same JEPA capturing the action-controlled dot in the 3-dots table as well as the stationary one. A reconstruction baseline that also failed on fixed noise would weaken the contrast, not the JEPA claim.

## 9. Limitations the paper states
- Toy dataset only; unclear whether it holds for complex video and bigger models (A.8).
- Only VICReg and SimCLR objectives tested; BYOL / Barlow Twins not covered (A.8). In particular, no EMA/stop-gradient self-distillation JEPA (I-JEPA style) is tested.
- Proposed fixes (Section 4): image differences / optical flow as input; hierarchy (H-JEPA, LeCun 2022); or an added constraint that stops representations from being constant across time.
- Later paper in this set: 2407.03475 (noisy-features) gives the linear-dynamics account and cites this paper in its Related Work: JEPA "will prioritize 'slow features' for which the marginal distribution has the smallest variance" (straight quotes substituted for the source's curly quotes).

## 10. Outline feedback (ch-collapse)
1. Polarity is reversed in the outline. The outline says "with noisy fast features and clean slow ones, the predictive loss keeps the slow ones" as if it were the good case. In this paper the slow feature is the fixed background noise (a distractor), JEPA keeps it and LOSES the dot (Fig 3 bottom row; Table 1 VICReg Stationary 0.066 vs Action 0.277). The positive half is: JEPA ignores noise that changes every frame without re-tuning. The claim card should read: "JEPA keeps whatever is slowest; that is a gift when the noise is fast and a trap when the noise is fixed."
2. The chapter thesis "JEPAs avoid collapse not by a term in the loss but by an asymmetry (EMA target, stop-gradient, a predictor)" does not describe this paper. Its collapse prevention is the explicit VICReg variance + covariance terms (eqs (3)-(4), (8)-(9)) or InfoNCE. No EMA, no stop-gradient. The GRU predictor is there, but it is not what prevents collapse. The thesis fits 2407.03475 only; say so in the chapter or move this card to a "what the loss prefers" framing.
3. "Slow" here means constant within a 17-frame sequence; the paper does not use the (lambda, rho) vocabulary of 2407.03475. The bridge is 2407.03475's Related Work sentence quoted in Section 9 above.
4. The browser toy should NOT copy this paper's setup (conv encoder + GRU + VICReg on 28 x 28 frames, 1M sequences). Copy 2407.03475 (see `noisy-features.md`, section "Toy spec"). The toy can still honour this paper with a preset "fixed background": a dimension with gamma = 1 and zero per-frame noise gets regression coefficient rho = 1 and the largest encoder weight, exactly the Sobal failure mode in two numbers.
5. Repo: the outline/meta say none; the paper gives github.com/vladisai/JEPA_SSL_NeurIPS_2022.
