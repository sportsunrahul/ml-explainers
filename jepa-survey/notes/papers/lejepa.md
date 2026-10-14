# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics

## 1. Identity
- arXiv 2511.08544, version read: v3. Submission history (abs.html): v1 2025-11-11, v2 2025-11-12, v3 2025-11-14. citation_date 2025/11/11.
- Authors: Randall Balestriero (Brown University, Meta-FAIR) and Yann LeCun (NYU, Meta-FAIR), marked equal contribution. Two authors only.
- Venue: none stated (ICML-style template). PDF is 50 pages.
- Repo: github.com/rbalestr-lab/lejepa (README read), pip package `lejepa`, project page rbalestr-lab.github.io/lejepa.
- Sources used: src/main.tex, src/content/{abstract,intro,background,body,conclusion,appendix}.tex, paper.pdf, README.md.
- Custom macros in src/math_commands.tex needed to read the TeX below: `\vtheta={\bm{\theta}}`, `\vx={\bm{x}}`, `\vz={\bm{z}}`, `\va={\bm{a}}`, `\vy={\bm{y}}`, `\vq={\bm{q}}`, `\vmu={\bm{\mu}}`, `\mZ={\bm{Z}}`, `\mI={\bm{I}}`, `\sA={\mathbb{A}}`, `\R=\mathbb{R}`, `\E=\mathbb{E}`, `\Cov=\mathrm{Cov}`, `\Tr` = operator Tr, `\grad=\nabla`, `\norm{x}=\left\lVert x\right\rVert`.

## 2. Core problem
JEPAs train an encoder so the embedding of one view predicts the embedding of another view. The prediction task alone admits shortcut solutions: complete collapse (all inputs map to one point) or dimensional collapse (a low-dimensional subspace). Sec 1 and 2.2 say the anti-collapse side of current recipes is "a game of Whac-A-Mole": stacked heuristics with no theory of which embedding distribution the encoder should produce.

## 3. What prior methods did and why that hurt (per the paper)
- Sec 1 lists the heuristics: stop-gradient, asymmetric view generation, teacher-student networks with tuned EMA schedules, explicit normalization and whitening layers, "and a delicate balance of hyperparameters".
- Sec 2.2 groups anti-collapse mechanisms as (i) feature whitening (W-MSE, VICReg), (ii) negative samples (SimCLR, MoCo), (iii) asymmetric views and teacher-student with stop-gradient (DINO, I-JEPA). Each suffers at least two of: under-specification (criterion minimized while embeddings degenerate), quadratic time or memory in batch size or dimension, sensitivity to data, hyperparameters and architecture, no guarantees.
- Sec 6.1: prior work shows predictors (without asymmetric information) and teacher-student architectures "serve primarily to prevent collapse"; removing them gives chance-level encoders. Register tokens were added to fix instabilities that the paper attributes to "poorly conditioned training objectives".
- Sec 6.2: in latest JEPAs the training loss has low correlation with downstream accuracy and may not decrease monotonically, so model selection needs labels.

## 4. The one key idea
Two axioms (Sec 1): (i) solve the prediction task, (ii) make the embedding distribution an isotropic Gaussian. Sec 3 proves the isotropic Gaussian uniquely minimizes downstream risk for linear, kNN and kernel probes. SIGReg (Sec 4) enforces it with a sketched statistical test: project embeddings on random unit directions, compare each 1D projection to N(0,1) with the Epps-Pulley characteristic-function statistic, average. Prediction loss + SIGReg = LeJEPA, one trade-off hyperparameter lambda, no collapse heuristics.

## 5. Mechanism, step by step
1. Views. DINO-style multi-crop: V_g global views and V_l local views per image, V = V_g + V_l (Sec 5.1). Global views 224x224, local 96x96 per the Experiment Details box (p14); README (code-only) says local 98x98 with crop scale (0.05, 0.3), global scale (0.3, 1.0), plus flip, color jitter, grayscale, blur, solarize.
2. Encoder. One network f_theta: R^D -> R^K shared by all views, any timm backbone (Sec 2.1, Algorithm 2 p11). A projector head exists in the experiments (Table 1(d) "proj. dim." 64..1024 on top of "emb. dim." 512/2048; Table 4 "1/2/3-layer" projector) but is not in Algorithm 2 and the text never says which output SIGReg or the prediction term grades. Our reading: both losses are applied to the projector output, as in VICReg/DINO practice; linear probes use the backbone (README: concatenated CLS of last two layers).
3. Prediction term. All V views predict the mean of the V_g global-view embeddings, squared L2, eq (5)-(7). Algorithm 2: `centers = g_emb.view(-1, bs, K).mean(0)`; `sim = (centers - a_emb).square().mean()`. No predictor network, no detach on `centers`, so the gradient flows into the encoder from both sides of the difference. For ResNets, all_views = global_views (no local crops).
4. SIGReg term. Per view and per minibatch (Algorithm 1 p10): sample M random directions A in R^{K x M}, Gaussian then column-normalized, seeded by global_step so all GPUs draw the same A; project `x @ A` -> (N, M); for 17 knots t in linspace(-5, 5) compute the empirical characteristic function mean_n exp(i t u_n), all_reduce-averaged across GPUs; squared distance to the N(0,1) CF exp(-t^2/2), weighted by the same exp(-t^2/2); trapezoid integrate over t; multiply by N. Returns one statistic per direction; the SIGReg definition averages over directions (eq SIGReg). Memory and time O(N) in minibatch size (Sec 4.2.3).
5. Total loss, eq (LeJEPA): lambda times the mean SIGReg over views plus (1 - lambda) times the mean prediction loss over the batch. Default lambda = 0.05 (Sec 6.1).
6. Gradient. Everything trainable gets a gradient from both terms through every view: there is no frozen or EMA branch. Theorem 4 bounds the Epps-Pulley gradient per sample by 4 sigma^2 / N. Theorem 6: minibatch estimate of the EP loss is biased by O(1/N); the paper says this is negligible even at batch 16 (Sec 5.1).
7. Optional SWA on the encoder that produces mu in eq (6) gives a small gain for ViTs (Sec 6.1, Table 4) but is not needed to avoid collapse.

### What the paper says it REMOVES, and what remains
| Removed (paper's words, location) | Remains |
|---|---|
| Stop-gradient (abstract; Sec 1; Sec 5.1; Algorithm 2 has no detach) | Shared encoder, gradient on both sides |
| Teacher-student / EMA target encoder (abstract; Sec 1; Sec 5.1; Sec 6.1, Table 4 "w/ SWA" False rows) | Optional SWA on the mu-producing encoder for ViTs (Sec 6.1) |
| Predictor network (Sec 2.1: "only justified when there exists an asymmetry"; Sec 6.1 "remove both the predictor and teacher-student"; Table 4 "w/ predictor" False) | A projector head (Table 1(d), Table 4) |
| Prototypes (Sec 5.1 "absence of prototypes, stop-gradients, and teacher-student networks") | -- |
| Register tokens (Sec 6.1; Table 1(e): 75.14 to 75.34 for 0..8 tokens at 1024 slices) | -- |
| Whitening / normalization layers, negative samples (Sec 1, Sec 2.2 as heuristics LeJEPA does not need) | -- |
| "Hyper-parameter schedulers" (abstract). Our reading: EMA, temperature and weight-decay schedules; the Experiment Details box still uses linear warm-up + cosine lr, "no scheduler on weight-decay" | lr warm-up + cosine (p14; README) |
| Hyperparameter tuning per dataset/architecture (Sec 1 contribution 3) | One lambda, plus standard lr/wd choice |
| -- | DINO-style multi-crop (2 global + local views). Our reading: view generation is still asymmetric in crop size; what is removed is architectural asymmetry |

## 6. Equations (TeX verbatim from src)
Numbered equations in the PDF: (1) JEPA definition, (2)-(4) hypothesis test construction, (5)-(7) prediction loss, (8) loss-accuracy correlation. SIGReg, Epps-Pulley and LeJEPA carry name tags, not numbers.

(1) Definition 1 "JEPA", p4, `\label{def:SSL}`:
```
{\rm JEPA}(\vx) \iff& {\rm Enc}\left(\vx_{n,t+1,.}\right) \text { is predictable from }{\rm Enc}\left(\vx_{n,t,.}\right), \forall n,t\text{ and } {\rm Enc}\left(\vx_{.,.,.}\right) \text{ is not degenerate}.
```
Meaning: a JEPA is any system where one view's embedding predicts another's and the embeddings are not degenerate. Gloss: x_{n,v,d} = sample n, view v, dimension d.

(OLS) Sec 3.1, `\label{eq:OLS}`:
```
\hat{\beta} = \underset{\beta \in \mathbb{R}^K}{\arg\min} \|\vy - \mZ\beta\|_2^2+\lambda \|\beta\|_2^2,\tag{OLS}
```
Z in R^{N x K} embeddings, y in R^N unknown labels, lambda >= 0 Tikhonov strength (this lambda is the probe's, not LeJEPA's).

Lemma 1 (p5) "Anisotropy amplifies bias": whenever lambda_K > lambda_1 (covariance eigenvalues), there always exists a downstream task y for which Z_aniso gives a higher-bias estimator than Z_iso for lambda > 0. Lemma 2 (p5) "Anisotropy amplifies variance", lambda = 0:
```
\text{tr}(\text{Var}(\hat{\boldsymbol{\beta}}_{\text{aniso}})) > \text{tr}(\text{Var}(\hat{\boldsymbol{\beta}}_{\text{iso}}))
```
Z_aniso and Z_iso have the same column span and the same total variance (Z_iso eigenvalues all equal the mean of Z_aniso's).

Theorem 1 (p6) "isotropic Gaussian Optimality", `thm:nonlinear_optimal`:
```
\text{ISB}_{k\text{-NN}} =\frac{r_0^4}{(K+2)^2}\tau_g^2J(p)+O(r_0^4),&&\text{(k-NN)}\\
\text{ISB}_{\text{kernel}} \le \Big(\frac{h^2\mu_2(K)}{2}\Big)^2 \Big(2 B^2 + 8 L^2J(p)\Big)+o(h^4),&&\text{(kernel)}
```
"and among distributions with a scalar-based covariance constraint, the isotropic Gaussian is the unique minimizer of the integrated square bias." One-sentence statement: under a fixed total variance (Tr(Cov Z) = kappa_1 or ||Cov Z||_F = kappa_2), the isotropic Gaussian is the unique embedding distribution that minimizes the integrated squared bias of radius-kNN and Nadaraya-Watson kernel probes, and (Lemmas 1-2) the bias and variance of ridge/OLS linear probes, over all downstream targets. Gloss: ISB = E_q[Bias(q)^2] with queries q ~ p_z; r_0 = kNN radius; h = kernel bandwidth; mu_2(K) = kernel second moment; K = embedding dimension (also the kernel symbol, overloaded in the source); tau_g^2 = isotropic prior on target gradients, E[grad eta grad eta^T] = tau_g^2 I (App A.1); J(p) = Fisher information functional int ||grad log p||^2 p dx (App B.4, "recovering the Fisher-information functional J(p)"); B, L = bounds on |Laplacian eta| and ||grad eta||. "Downstream risk" therefore means: bias (and variance) of a frozen-encoder probe averaged over the embedding distribution and taken worst-case or prior-averaged over unknown target functions. Our reading: the Gaussian wins because it minimizes Fisher information at fixed covariance.

(2)-(4) Sec 4.1, `eq:null`, `eq:HUV`, `eq:T_max`:
```
H_0: P_{\vtheta}=Q \quad \text{vs.} \quad H_1: P_{\vtheta}\neq Q,
```
```
H_0(\va):  P_\vtheta^{(\va)} = Q^{(\va)}\;\;\text{vs.} \;\; H_1(\va): P_\vtheta^{(\va)} \not = Q^{(\va)},
```
```
T_{\sA}(\{f_{\vtheta}(\vx_n)\}_{n=1}^N)\triangleq \max_{\va \in \sA} T(\{\va^\top f_{\vtheta}(\vx_n)\}_{n=1}^N).
```
P_theta = embedding distribution, Q = target, P^{(a)} = push-forward along unit direction a in S^{K-1}, T = any univariate test statistic, A = set of M directions. Lemma 3 (hyperspherical Cramer-Wold) and Theorem 2 (p7): the max over directions is a valid test with level alpha and power 1 as n -> infinity.

(SIGReg) Definition 2, p7, `\label{eq:BCS}`:
```
{\rm SIGReg}_{T}(\sA,\{f_{\vtheta}(\vx_n)\}_{n=1}^{N})\triangleq\frac{1}{|\sA|}\sum_{\va\in\sA}T(\{\va^\top f_{\vtheta}(\vx_n)\}_{n=1}^{N}),\tag{SIGReg}
```
"SIGReg sketches a statistical test T towards isotropic Gaussian ... where we recommend the Epps-Pulley test for T." The max of (4) is replaced by an average "to avoid sparse gradient over the directions". Gloss: |A| = M directions; T applied to the N scalar projections a^T f_theta(x_n).

(Epps-Pulley) Sec 4.2.3, `\label{eq:epps_pulley}`:
```
EP = N \int_{-\infty}^{\infty} \left| \hat{\phi}_X(t) - \phi(t) \right|^2 w(t)  dt.\tag{Epps–Pulley}
```
with ECF `\hat{\phi}_X(t) = \frac{1}{n} \sum_{j=1}^{n} e^{itX_j}` and "The weight function is typically Gaussian, such as $w(t) = e^{- t^2/\sigma^2}$ with $\sigma$ commonly set to $1$." phi(t) = target CF = exp(-t^2/2) for N(0,1). Discrepancy: Algorithm 1 uses `exp_f = exp(-0.5 t^2)` as both the target CF and the window, i.e. w(t) = exp(-t^2/2), not exp(-t^2).

Theorem 4 (p10) "Stability of Epps-Pulley Test":
```
\left|\frac{\partial EP(\mathbf{a})}{\partial z_i}\right| \le \frac{4\sigma^2}{N}, \quad \left|\frac{\partial^2 EP(\mathbf{a})}{\partial z_i^2}\right| \le \frac{C\sqrt{\pi}\sigma^3}{2N},
```
Bounded gradient and curvature for any input distribution; z_i = projected sample.

Theorem 5 (p11) "Unified Error Bounds": if EP = 0 on the M sampled directions and p_theta has Sobolev smoothness alpha,
```
\mathbb{E}_{\va} \left[ \int_{\mathbb{R}} \left| \varphi_a(t) - \varphi_{\mathcal{N}}(t) \right|^2 dt \right] \leq C(K, \alpha) |\sA|^{-2\alpha/(K-1)} \times\int_0^\infty \left\| \varphi_{\cdot}(r) - \varphi_{\mathcal{N}}(r) \right\|_{H^\alpha(\mathcal{S}^{K-1})}^2 dr,
```
"|A| = O(K) directions suffice for epsilon-approximation when alpha is large." Plus the SGD argument (Sec 4.3): directions are resampled each step, so "|A| as low as 16 can easily outperform a fixed set with |A| of order of thousands".

Theorem 6 (p12) "Vanishing gradient bias":
```
\mathbb{E}\left[\widehat{L}_n(\theta)\right] = L(\theta)+\frac{1}{N}\int_{\mathbb{R}} w_s(t)\big(1-|\varphi_P(t)|^2\big)dt,
```
Minibatch EP loss and gradient are biased by O(1/N). Consequence for the visual: under the null the per-direction statistic does not reach 0; with Algorithm 1 weighting the N-scaled floor is about int exp(-t^2/2)(1 - exp(-t^2)) dt = sqrt(2 pi) - sqrt(2 pi / 3) ~ 1.06, independent of N (our reading; numerically checked: mean T ~ 0.7-1.2 for N = 64..4096 on a 3D Gaussian cloud, versus hundreds on a stretched or collapsed axis).

(5)-(7) Sec 5.1, `eq:pred1`, `eq:pred2`, `eq:pred_as`:
```
\mathcal{L}_{\rm pred}(\{\vz_{n,v}\}_{v=1}^{V})=&\frac{1}{V_g}\sum_{v=1}^{V_g}\frac{1}{V}\sum_{v'=1}^{V}\left\| \vz_{n,v} - \vz_{n,v'} \right\|_2^2\label{eq:pred1}\\
=&\frac{1}{V}\sum_{v'=1}^{V}\left\| \frac{1}{V_{\rm g}}\sum_{v=1}^{V_{\rm g}}\vz_{n,v} - \vz_{n,v'} \right\|_2^2\label{eq:pred2}\\
\triangleq& \frac{1}{V}\sum_{v'=1}^{V}\left\| \bm{\mu}_{n} - \vz_{n,v'} \right\|_2^2,\label{eq:pred_as}
```
"the prediction loss is then given by having all views predict the global views". Gloss: z_{n,v} = embedding of view v of sample n; views 1..V_g are global; mu_n = mean of the global-view embeddings. Our reading: (5) and (6) are not equal; (5) = (6) + (1/V_g) sum_v ||z_{n,v} - mu_n||^2, a within-global-view spread term that depends on theta. Appendix B.6 (eqs 14-19) asserts (1/V_g) sum_v ||z_v||^2 = ||z_bar||^2, which holds only when all global embeddings coincide. Algorithm 2 implements (7).

(LeJEPA) Sec 5.1, `\label{eq:lejepa}`:
```
\mathcal{L}_{\rm LeJEPA}(\{\vx_{n,v}\}_{n,v=1}^{B,V})=\frac{\lambda}{V}\sum_{v=1}^{V}{\rm SIGReg}(\{\{\vz_{n,v}\}_{n=1}^{B}\})\\
+\frac{1-\lambda}{B}\sum_{n=1}^{B}\mathcal{L}^{(V_{\rm g})}_{\rm pred}(\{\vz_{n,v}\}_{v=1}^{V}).\tag{LeJEPA}
```
Algorithm 2: `return (1-lambd)*sim + lambd*sigreg`. Gloss: B = minibatch size, V = views, lambda in [0,1] the single trade-off hyperparameter. Note the weighting is lambda on SIGReg and (1 - lambda) on prediction, not "prediction + lambda * SIGReg".

(8) Sec 6.2, `\label{eq:corr}`:
```
C^{(\alpha)} = \rho_s\left(\frac{\text{train\_loss}}{\lambda^{\alpha}}, \text{test\_accuracy}\right).
```
rho_s = Spearman correlation; alpha ~ 0.4 gives nearly 99% correlation (Sec 6.2); alpha = 0 is the plain loss.

Relation to VICReg (Sec 5.2, Theorem 9 App): with T = mean^2 + (std - 1)^2 as the univariate test, SIGReg in the many-slice limit enforces E[Z] = 0 and Cov(Z) = I, i.e. VICReg; the paper advises against it (Theorem 3: finitely many moments do not identify the distribution).

## 7. Config / setup table
| Item | Value | Location |
|---|---|---|
| lambda default | 0.05 | Sec 6.1 p13 ("recommend lambda = 0.05 as a robust default"); Table 7 sweep 0.001..0.5 |
| Number of directions M = |A| | 1024 recommended (Sec 6.1 p13); Algorithm 1 default `num_slices=256`; Table 1(a) 512 vs 2048; README `num_slices=1024` (code-only) | Sec 6.1, Alg 1, README |
| Epps-Pulley knots | 17 points on [-5, 5], trapezoid (Sec 6.1; Alg 1 `linspace(-5, 5, 17)`); Table 1(a) ablates 5/17/41 and [-1,1]/[-3,3]/[-5,5]; "symmetry of the integrand to double the number of knots for free" in official code (Sec 5.1) | Sec 5.1, 6.1 |
| EP weight | text: w(t) = exp(-t^2 / sigma^2), sigma = 1; Algorithm 1: exp(-t^2 / 2) | Sec 4.2.3, Alg 1 |
| Views | Sec 6.1 recommends V_g = 2, V_l = 8 (V = 10); Experiment Details box p14: V = 8 with V_g = 2; README (code-only): 2 global + 6 local | three places disagree |
| Resolutions | global 224x224; local 96x96 (p14) vs 98x98 (README, code-only) | p14, README |
| Crop scales (code-only) | global (0.3, 1.0), local (0.05, 0.3) | README |
| Batch size | >= 128 recommended; Table 1(c): 128 -> 72.20, 256 -> 74.15, 512 -> 74.72, 1024 -> 74.07 | Sec 6.1 |
| Optimizer | AdamW, lr in {5e-3, 5e-4}, wd in {1e-1, 1e-2, 1e-5}, linear warm-up + cosine lr, no wd schedule | p14 box |
| Optimizer (code-only) | lr 5e-4; wd 5e-2 ViT, 5e-4 ResNet; bf16; final lr = initial / 1000 | README |
| Linear probe (code-only) | concat CLS of last two layers (mean of patch tokens if no CLS), LayerNorm, AdamW, wd 1e-6, warm-up + cosine | README |
| Projector | 1/2/3 layers (Table 4); proj. dim 64..1024, emb. dim 512/2048 (Table 1(d)) | Tables 1, 4 |
| Backbones | all from timm; ViTs, ConvNeXts, ResNets, MaxViTs, Swin; ImageNet-10 sweep: ~50 models from 8 families, < 20M params | Sec 1, 6.1 |
| Largest model | ViT-g/14, 1.8B params (Fig 1 top-right, Sec 6.4, conclusion); intro says "approaching 1 billion" | Fig 1, Sec 6.4 |
| Pretraining epochs | 100 on ImageNet-1k (Tables 1, 2); 400 on ImageNet-100 (Table 4, Fig 8) and Galaxy10 (Sec 6.3) | |
| SIGReg cost | N = 512, M = 512, 16 knots: 0.465 ms on a Tesla V100-SXM2-16GB; N = 32768: 26.37 ms | Table 6 p43 |
| Compute (GPU hours) | not stated | -- |
| Lines of code | "approximately 50 lines of code" (abstract); "fewer than 50 lines of core implementation" (conclusion); "~50 lines of core code" (README); body: "a few dozens lines in PyTorch" (Sec 5.1) | |

## 8. Results worth remembering
- ImageNet-1k frozen linear probe, ViT-H/14: 79% appears ONLY in the abstract. No table or figure gives a LeJEPA ViT-H row. Sec 6.4 (p17) gives online linear probe 77.1% for ViT-L (0.3B) and 78.5% for ConvNeXtV2-Huge (0.6B), 100 epochs. Table 1 (ViT-L/14, 100 ep) tops out at 75.84 (4096 slices, 8 register tokens).
- Table 1 (p13), ViT-L/14, IN-1k, 100 ep: (a) 512 slices / [-5,5] / 17 pts 74.21; 2048 slices 74.80; (b) V_g = 1, V = 4: 53.06 vs V_g = 2, V = 8: 74.24 vs V_g = 4, V = 10: 75.08; (c) batch 128: 72.20 .. 1024: 74.07; (e) register tokens 0..8 at 1024 slices: 75.14 .. 75.34; at 4096: 75.61 .. 75.84.
- Table 2 (p16), frozen transfer, 8 datasets: LeJEPA ViT-L (304M, IN-1K, 100 ep) avg all-shot 79.48 vs I-JEPA ViT-H (632M, 300 ep) 78.50; 1-shot 29.55 vs 30.20; 10-shot 60.95 vs 60.51. Caption: "100 pretraining epochs compared to I-JEPA's 300 epochs, a 3x reduction".
- Galaxy10 in-domain vs frontier transfer, Table 3 (p42; Fig 12 p16; excerpt in Fig 1 bottom-right): frozen backbone, all samples: LeJEPA ResNet-34 78.17, ConvNeXt-V2 Nano 76.52, ResNet-18 75.95, LeViT-128 69.00 vs DINOv2 Small 67.62, DINOv3 ViT-S/16 71.38. Frozen 1-shot: ResNet-34 31.08, ConvNeXt-V2 Nano 28.74 vs DINOv2 27.68, DINOv3 30.17. Full fine-tune, all: ResNet-34 83.28, ConvNeXt-V2 Nano 82.72 vs DINOv2 78.34, DINOv3 81.60; 1-shot: 24.27 / 29.42 vs 21.05 / 24.71. Dataset: 11,000 training samples, 10 galaxy types (Sec 6.3; Table 5 says 11008); LeJEPA pretrained 400 epochs. Fig 1 names the baseline "DINOv2 ViT-S/16", Table 3 "DINOv2 Small".
- Table 5 (p43), in-domain small datasets, frozen linear: flowers102 (1020 train samples) LeJEPA resnext26ts 82.19 vs IJEPA ViT-H/14 (630M) 85.76; galaxy10 LeJEPA resnet34 77.29 vs IJEPA 62.93; food101 LeJEPA swin_tiny 78.40 vs IJEPA 81.06; cifar100 LeJEPA resnet34 70.44 vs IJEPA 86.93. The IJEPA row is labelled "IJEPA-inet22k" while its pretraining column says "inet1k".
- Loss as a model-selection signal (Sec 6.2): Spearman about 85% between training loss and downstream accuracy; nearly 99% with alpha ~ 0.4 in eq (8); Fig 1 top-left: 94.52% (ViT/base-8, inet1k).
- Table 4 (p42), ImageNet-100, 400 ep: removing predictor and teacher does not collapse. resnet50 3-layer projector: no predictor / no SWA 83.93, with predictor 83.57; vit_small_patch8: no predictor / no SWA 81.07, with SWA 84.12.
- Table 7 (p43), resnet50, ImageNet-100, 8 views: lambda 0.001 -> 76.67, 0.05 -> 84.32, 0.5 -> 77.72. Best lambda grows with the number of views (Fig 8).
- Stability across architectures (Sec 6.1, Fig 9): ~50 timm models, 8 families, ImageNet-10, 91.5% to 95% top-1 frozen linear.
- Stability of training: "very stable and smooth training curve" on ViT-g 1.8B (Sec 6.4, Fig 1 top-right); Theorem 4 bounded gradients.
- Synthetic SIGReg check (Sec 4.3, Fig 6): N = 100 samples in 1024 dims with two adversarial "X" dims, M = 10 directions per step, recovered; text also cites D = 512, M = 16.
- Emergent segmentation from CLS attention (Fig 13) and PCA features (Fig 14), qualitative.

## 9. Limitations the paper states
- Minibatch Epps-Pulley loss and gradient are biased O(1/N) (Theorem 6); unbiased U-statistic or sample-splitting variants "we do not explore".
- Theorem 5 requires smoothness alpha of the embedding density; the number of directions needed grows otherwise.
- SWA and the number of slices and projector size still move accuracy by about 1 point (Table 1, Table 4); the "single hyperparameter" claim concerns collapse, not peak accuracy.
- The paper gives no compute budget and no ImageNet-1k table for its headline 79%.
- Later source in this project: 2605.26379 (lejepa-worldmodel) assumes SIGReg succeeded exactly and shows identifiability degrades with the alignment gap (its Theorem 3); 2510.05949 (gaussian-embeddings) is earlier (2025-10-07) and does not test LeJEPA itself.

## 10. Outline feedback (ch-lejepa)
- "Loss = prediction + lambda * SIGReg" is wrong. Eq (LeJEPA): lambda * SIGReg + (1 - lambda) * prediction. Build the term-by-term equation with (1 - lambda) on the prediction term.
- "SIGReg as written (their equation number)": it has no number; it is Definition 2 with tag (SIGReg), p7. Epps-Pulley is tag (Epps-Pulley), Sec 4.2.3 p9. Numbered are (2)-(4) for the test, (5)-(7) prediction loss.
- "IN-1k linear for ViT-H/14": 79% is abstract-only, no table. Either cite it as "abstract" or use Sec 6.4's 77.1 (ViT-L) and 78.5 (ConvNeXtV2-H), which are "online linear probe".
- "many architectures/scales": say "60+ architectures" (abstract) or "about 50 timm models from 8 families on ImageNet-10" (Sec 6.1); both exist, cite one.
- Parameter range: largest is 1.8B ViT-g (Fig 1, Sec 6.4), but the intro says "approaching 1 billion"; use 1.8B with Fig 1 as location.
- In-domain vs DINOv2/v3 comparison is Table 3 (Galaxy10) and Table 5, not Table 2 (which is vs I-JEPA).
- Visual: the per-direction statistic does not fall to zero after "apply SIGReg"; it settles near a floor of about 1 (Theorem 6 bias, N-scaled). Show "drops to ~1" not "drops to 0". The random direction count in the figure is 1 (one draggable direction); the paper's M is 1024 resampled every step; say that in the caption.
- The "what is removed" table has a subtlety: a projector head and multi-crop remain, and lr warm-up + cosine remains despite "no schedulers". Say "no EMA / stop-grad / predictor / prototypes / register tokens" and keep the projector in the "stays" column.
- Equation (5) = (6) in the paper is not an exact identity (see Sec 6 note); use (7) as the implemented loss.
- Prediction prompt "which direction will the test fail on?": with the toy below, the answer is any axis whose 1D projection is non-Gaussian (stretched, squashed, bimodal); an isotropic-but-non-Gaussian cloud fails on every direction (Fig 5 "X" example). Good prompt.

## 11. Browser toy: SIGReg statistic for one direction (our reading of Algorithm 1)
Given N embeddings z_n in R^3 and a direction a in R^3:
1. Normalize: a <- a / ||a||_2 (Algorithm 1 column-normalizes A).
2. Project: u_n = a . z_n = a_1 z_{n,1} + a_2 z_{n,2} + a_3 z_{n,3}, n = 1..N.
3. Knots: t_k = -5 + 10 (k - 1) / 16, k = 1..17 (so dt = 0.625).
4. Empirical characteristic function at each knot:
   Re_k = (1/N) sum_n cos(t_k u_n), Im_k = (1/N) sum_n sin(t_k u_n).
5. Target CF of N(0,1): phi_k = exp(-t_k^2 / 2).
6. Weighted squared error: err_k = ((Re_k - phi_k)^2 + Im_k^2) * w_k, with w_k = phi_k = exp(-t_k^2 / 2) as in Algorithm 1 (the text's sigma = 1 form would be w_k = exp(-t_k^2); label the choice).
7. Trapezoid integral: I = sum_{k=1}^{16} (err_k + err_{k+1}) / 2 * dt.
8. Statistic for that direction: T(a) = N * I.
9. SIGReg = average of T(a) over the M sampled directions (eq SIGReg); the toy shows one T(a).
Notes: Steps 4-5 are what the paper calls comparing the ECF to the Gaussian CF "in weighted l2-norm". Under the null (isotropic Gaussian cloud) T(a) sits near 1.06 for any N (Theorem 6 floor, our computation), so the "apply SIGReg" animation should ease the statistic to about 1, not 0. For display, dropping the factor N makes the number scale-free (our choice; the paper keeps N). Numeric sanity check (our run, N = 1024, 3D cloud with axis stds 3 / 1 / 0.1): T = 612.9 on the stretched axis, 0.99 on the unit axis, 402.1 on the collapsed axis.
