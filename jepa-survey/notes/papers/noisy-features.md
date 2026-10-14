# How JEPA Avoids Noisy Features: The Implicit Bias of Deep Linear Self Distillation Networks (2407.03475v1)

## 1. Identity
- Title: How JEPA Avoids Noisy Features: The Implicit Bias of Deep Linear Self Distillation Networks
- arXiv: 2407.03475v1 (version read). Published 2024-07-03 (abs.html: citation_date 2024/07/03, "Submitted on 3 Jul 2024").
- Authors: Etai Littwin (first) ... Joshua Susskind (last); 7 authors (Littwin, Saremi, Advani, Thilak, Nakkiran, Huang, Susskind).
- Venue: not stated on the abs page. The LaTeX uses `icml2021.sty`.
- Repo: none given (meta.json `repo: null`; no URL in the source).
- Source read: `src/ssl_dynamics_arxiv.tex` (main text lines 180-1230, appendix proofs 1539-1760 and 2755-2900), `paper.pdf` for printed equation/theorem numbers and figures.
- Custom macros used in the quoted TeX (preamble): `\V` -> `V`, `\E` -> `\mathbb{E}`, `\R` -> `\mathbb{R}`, `\si` -> `\Sigma^{xx}`, `\la` -> `\Sigma^{xy}`, `\w` -> `\bar{W}`, `\avg{#1}` -> `\mathbb{E}\left[ #1 \right]`. The preamble also redefines `\[`/`\]` as `eqnarray`.

## 2. Core problem
JEPA (predict the target's latent) and MAE (predict the target in data space) are the two main SSL paradigms. The common motivation for JEPA is that it "prioritizes abstract features over fine-grained pixel information (which can be unpredictable and uninformative)", but the mechanism is not understood. The paper builds a fully solvable deep linear model of both objectives and asks which features each one learns first and how strongly.

## 3. What prior methods did and why that hurt (per the paper)
- MAE (eq (1)): penalises reconstruction error in the input space, so encoding quality depends on how (x, y) are sampled; it learns the subspace that explains most of the input variance, which Balestriero & LeCun 2024 showed is sub-optimal for perception.
- Plain JEPA objective (eq (2)) with a shared encoder is "susceptible to collapse": encoder and decoder mapping everything to one vector minimise it. Contrastive methods fix this with negatives (costly); non-contrastive methods use the StopGrad operator on the target branch (BYOL, data2vec, SimSiam, I-JEPA, V-JEPA). The paper studies exactly that StopGrad family and calls it JEPA.
- Prior linear-dynamics work (Saxe et al., Simon et al. 2023 "stepwise" SSL) showed greedy step-wise learning ordered by covariance, but did not separate the two objectives.

## 4. The one key idea
In deep linear self-distillation, the stop-gradient target branch changes a single exponent in the per-feature learning ODE. That exponent makes JEPA greedy in the REGRESSION COEFFICIENT rho_i = lambda_i / sigma_i^2 ("influence": how predictive feature i is of its target per unit variance), while MAE is greedy only in the covariance lambda_i. Features that are predictive but low-variance (semantic, "slow") are learned first and kept at full weight; high-variance, weakly predictive (noisy) features are learned later and converge to the smaller weight rho^L, more suppressed the deeper the encoder. Paper's informal statement (Section 1 contributions): "JEPA prioritizes 'influential features' (features which are most informative in prediction), whereas MAE only prioritizes highly covarying features (even if they are noisy and thus less informative)."

## 5. Mechanism, step by step
1. Setting (Section 2, eqs (3)-(5)): encoder f(x) = W-bar x with W-bar = W^L ... W^1 (each W^a in R^{d x d}, depth L); linear predictor/decoder g(f) = V f (V in R^{d x d}). x, y centred with covariances Sigma^{xx}, Sigma^{yy}, Sigma^{xy}. The same encoder encodes x and y; the target branch is SG(W-bar) y. There is no EMA in the paper; the teacher is the current encoder with gradients blocked.
2. Feature = coordinate i (after assuming Sigma^{xx}, Sigma^{yx} diagonal). Feature parameters: sigma_i^2 = Sigma^{xx}_ii (variance), lambda_i = Sigma^{yx}_ii (covariance with the target), rho_i = lambda_i / sigma_i^2 (regression coefficient). "Learned" means w-bar_i = ||W-bar e_i|| is large.
3. Train all weights by gradient flow (eq (6)) from a small balanced orthogonal initialisation (Assumption 4.1).
4. Theorem 4.2: the dynamics decouple per feature into scalar ODEs (7) JEPA and (8) MAE that differ only in one exponent (3 - 1/L vs 2 - 1/L).
5. Corollary 4.3: fixed points w-bar_JEPA(inf, L) = rho^L, w-bar_MAE(inf, L) = rho^{L/(L+1)}; one-layer JEPA equals infinite-depth MAE.
6. Critical time t* (eq (11)): time to reach a fraction p of the fixed point. Theorems 4.4-4.7 solve both ODEs in closed form and expand t* in the init scale epsilon: for MAE the leading term goes as 1/lambda and the rho-dependence is negligible (eq (16)); for JEPA the next-to-leading term goes as 1/rho and grows with depth (eq (13)). Corollary 4.8 (eq (17)) gives the ratio of critical times for two features with equal lambda and rho' > rho.
7. Numerical check: Figure 1 (full deep linear nets, SGD, d = 100, L = 5), Figure 2 (ODE simulations, L = 5), Figure 4 (depth sweep L = 2, 5, 6, 7), Figure 5 (random Gaussian init, L = 5). All show: both objectives greedy in lambda; only JEPA greedy in rho; order inverts between the objectives when lambda and rho trend oppositely.
8. Where "noise" enters (Section 5.1, temporal model eqs (18)-(19)): views are consecutive frames of flickering images plus per-pixel noise of amplitude sigma_a; lambda_a = gamma_a ||v^a||^2 and rho_a = gamma_a ||v^a||^2 / (sigma_a^2 + ||v^a||^2). More per-frame noise lowers rho_a but not the ordering by lambda_a, so "JEPAs can be slower to learn noisy features in a way that MAEs are not. JEPA will also learn noisy features by converging to lower amplitude weights (9), making such features potentially less likely to be used downstream from the embedding."

## 6. Equations (TeX verbatim from `src/ssl_dynamics_arxiv.tex`)

(1) `\label{eqn:mae}`, Section 1, p2. MAE objective.
```
    \mathcal{L}_{\text{MAE}} = \frac{1}{2}\E_{x,y \sim p(x,y)} \|g_{V}(f_{W}(x)) - y\|^2,
```
(2) `\label{eqn:jepa}`, p2. JEPA objective with shared encoder. Paper: "susceptible to collapse: it can be minimized trivially by employing an encoder and decoder that map all inputs to the same vector."
```
    \mathcal{L}_{\text{JEPA}} = \frac{1}{2}\E_{x,y \sim p(x,y)} \|g_{\V}(f_{W}(x)) - f_{W}(y)\|^2.
```
Gloss: f_W = encoder, g_V = decoder (MAE) or predictor (JEPA), p(x, y) = joint over two views of the same underlying sample, x, y in R^d.

(3), (4) Section 2, p3 (unlabelled). Linear parameterisation and data covariances.
```
    \bar{W} = \prod_{a=1}^L W^{a},~~~f(x) = \bar{W} x,~~~g(f(x)) = \V f(x),
```
```
    \E[xx^\top] = \Sigma^{xx},~~~\E[yy^\top]= \Sigma^{yy},~~~\E[xy^\top] = \Sigma^{xy}.
```
Gloss: W^a = layer a weight (d x d), product is right-to-left (W^L ... W^1); V = predictor; later assumed diagonal covariances with sigma_i^2 = Sigma^{xx}_ii, lambda_i = Sigma^{yx}_ii, rho_i = lambda_i / sigma_i^2.

(5) `\label{eqn:lin}`, p3. The two linear objectives; SG = StopGrad.
```
 \mathcal{L}_\text{jepa} &=   \frac{1}{2}\E_{x,y \sim p(x,y)}\|\V \bar{W} x - \text{SG}(\bar{W})y\|^2,~~~
 \mathcal{L}_\text{mae} =   \frac{1}{2}\E_{x,y \sim p(x,y)}\|\V \bar{W} x - y\|^2.
```
(6) `\label{eqn:gf}`, p5. Gradient flow on all weights at once.
```
    \forall_a,~~\dot{W}^a = -\nabla_{W^a}\mathcal{L},~~~ \dot{\V} = -\nabla_{\V}\mathcal{L},
```
Assumption 4.1 (`\label{ass:init}`, p5), verbatim: (1) "Sigma^{xx}, Sigma^{yx} are diagonal and positive definite." (2) "All weight matrices W^a(0), a=1...L, V(0) are initialized as W^a = epsilon^{1/L} U^a, V(0) = epsilon^{1/L} U^v where {U^a}_a, U^v are orthogonal matrices, and 0 < epsilon << 1 is an initialization scale." (3) "For JEPA, we set U^v to be diagonal, and for MAE we set U^v such that U^v (prod_a U^a) is diagonal."

(7) `\label{eqn:dynj}` and (8) `\label{eqn:dynm}`, Theorem 4.2 (ODE Equivalence), p5-6. THE main dynamics equation. Paper: "Remarkably, the only difference between the JEPA and MAE dynamical equations is in an exponent."
```
        \text{JEPA}: \dot{\bar{w}}_i(t) = \bar{w}_i(t)^{3 - \frac{1}{L}}\lambda_i - \bar{w}_i(t)^3\lambda_i\rho_i^{-1}\label{eqn:dynj}.
```
```
        \text{MAE}:  \dot{\bar{w}}_i(t) = \bar{w}_i(t)^{2 - \frac{1}{L}}\lambda_i - \bar{w}_i(t)^3\lambda_i\rho_i^{-1} \label{eqn:dynm}.
```
Gloss: w-bar_i(t) = ||W-bar(t) e_i|| = how much the encoder keeps feature i; lambda_i = covariance of feature i with its target; rho_i = regression coefficient; L = encoder depth; first term drives growth, second term saturates it. Setting the right side to zero gives the fixed points in (9). In words: a feature grows at a rate set by its covariance lambda, and stops growing when w-bar reaches rho^L (JEPA) or rho^{L/(L+1)} (MAE).

(9) `\label{eq:fixed}` and (10), Corollary 4.3, p6. Paper: "When L>1, JEPA will suppress the encoding of noisier directions with a lower regression coefficient and this suppression increases with the network depth. At large depths, the encoder becomes low-rank to the point a single dominating eigenvalue corresponding to the feature with the largest regression coefficient remains."
```
    \bar{w}_\text{MAE}(\infty,L) = \rho^\frac{L}{L+1} ,~~~ \bar{w}_\text{JEPA}(\infty,L) = \rho^{L}.
```
```
    \lim_{L' \to \infty}\bar{w}_\text{MAE}(t,L') = \bar{w}_\text{JEPA}(t,1).
```
(11) `\label{t-crit-def}`, p6. Critical time definition, p = fixed fraction "not too close to 1 or 0".
```
\bar{w}(t^*) = p \bar{w}(\infty).
```
(12) Theorem 4.4 (JEPA dynamics), p7, with psi = (1/rho) w-bar^{1/L}, psi(0) = epsilon^L / rho, C = integration constant.
```
\psi(t) =  \frac{\exp\Big[\frac{\lambda}{L}\rho^{2L-1} t - \sum^{2L-1}_{n=1}\frac{1}{n-2L}\psi(t)^{n-2L}   + \mathcal{C}\Big]}{1 + \exp\Big[\frac{\lambda}{L}\rho^{2L-1} t - \sum^{2L-1}_{n=1}\frac{1}{n-2L}\psi(t)^{n-2L}   + \mathcal{C}\Big]}.
```
(13) `\label{eqn:jepacritical}`, Theorem 4.5 (JEPA critical time), p7.
```
t^*_\text{jepa} = \frac{1}{\lambda}\sum^{2L-1}_{n=1}\frac{L}{n \rho^{2L-n-1}\epsilon^{\frac{n}{L}}} + \Theta\Big[\log(\epsilon)\Big],
```
Gloss: epsilon = init scale; the n = 2L-1 term is the leading order in 1/epsilon and depends on lambda only; the next terms carry 1/rho^{...} and grow with L.

(16) Theorem 4.7 (MAE critical time), p7, L > 1.
```
t^{*}_\text{mae}=\frac{L}{\lambda (L-1)\epsilon^\frac{L-1}{L}}  + \Theta(1).
```
(17) `\label{eqn:ratio_jepa}`, Corollary 4.8, p8. rho' > rho, Delta_rho = 1/rho - 1/rho'. Paper: "indicating greedy learning dynamics with respect to rho, a property absent in the case of MAE."
```
\frac{t^{*}_\text{jepa}(\epsilon, \lambda, \rho)}{t^{*}_\text{jepa}(\epsilon, \lambda, \rho^{'})} &= 1+\frac{2L-1}{2L-2}\Delta_\rho\epsilon^{\frac{1}{L}} +\Theta(\epsilon^{\frac{2}{L}}),~~~\frac{t^{*}_\text{mae}(\epsilon, \lambda, \rho)}{t^{*}_\text{mae}(\epsilon, \lambda, \rho^{'})} = 1+ \Theta\Big[\epsilon^{\frac{L-1}{L}}\Big]
```
(18) `\label{eq:gen_temp}`, Section 5.1, p10. Temporal generative model, x = z^t, y = z^{t+1}.
```
z^t_i = \sum_a{ u^a(t) v^a_i} + \xi^t_i,\quad t = 1,...T.
```
(19) `\label{eq:temp_val}`, p10. Feature parameters of the temporal model.
```
\lambda_a = \gamma_a \|v^a\|^2, \quad \rho_a = \frac{\gamma_a \|v^a\|^2}{\sigma_a^2 + \|v^a\|^2}.
```
Gloss: v^a = image a (orthogonal supports), u^a(t) = its flickering amplitude with lag-1 autocorrelation gamma_a, xi = per-pixel noise with std sigma_a on the pixels of image a. Slow + clean => gamma near 1, sigma small => rho near 1. Fast or noisy => small gamma or large sigma => small rho.

Appendix "Temporal model" (`\label{sec:tcm}`): the AR(1) process that realises gamma_a (equation number not captured):
```
u^a(1) = \eta_0^a, \quad u^a(t+1) = \gamma_a u^a(t) + \sqrt{1- \gamma_a^2}\eta_t^a,\quad \eta_t^a \sim \mathcal{N}(0,1).
```
(21), (22) `\label{eqn:gf_jepa_2}`, Appendix proof of Theorem 4.2, p19. The JEPA gradients (what the toy must implement):
```
\forall_a,~~~\nabla_{W^a(t)}\mathcal{L}_{jepa} &= \E\Big[(\prod_{b=a+1}^{L}W^b(t))^\top V(t)^\top\Big[V(t)\bar{W}(t)x - \bar{W}(t)y\Big]x^\top \Big[\prod_{b=1}^{a-1}W^b(t)\Big]^\top\Big], \\
\nabla_{V(t)}\mathcal{L}_{jepa} &= \E\Big[(V(t)\bar{W}(t)x - \bar{W}(t)y)x^\top \bar{W}(t)^\top\Big].
```
Gloss: the error V W-bar x - W-bar y is back-propagated through V and the online encoder only (x^T on the right); the target W-bar y contributes no gradient, which is the StopGrad.

## 7. Measurement setup (analysis lens)
| Experiment | What is varied | Held fixed | Where |
|---|---|---|---|
| Fig 1 (p4): full deep linear nets | Distribution 1: lambda_i and rho_i both increasing in i; Distribution 2: lambda_i increasing, rho_i decreasing | d = 100, L = 5, SGD, minibatch 1000, 5000 iterations, Gaussian data with diagonal covariances, Gaussian init | Sec 3, Appendix A.1 |
| Fig 2 (p8): ODE simulations of (7), (8) | (a),(d) lambda in 0.65..0.95 with rho = 1; (b),(e) rho in 0.65..0.95 with lambda = 1; (c),(f) higher lambda with lower rho | L = 5, small init | Sec 5 |
| Fig 3 (p9): temporal model | T (log10 T on x-axis); feature 1: gamma_1 = 0.99, noise std 1; feature 2: gamma_2 = 0.95, noise std 0.5 | 500k samples for the empirical covariances, 10 runs, 2 std bands | Sec 5.1 |
| Fig 4 (p17): depth sweep | L = 2, 5, 6, 7; darker = higher rho | lambda = 1 | Appendix A |
| Fig 5 (p17): random init | lambda, rho per feature (left column) | standard Gaussian init with default scale, L = 5 | Appendix A |
| Metric | encoder magnitude ||W-bar e_i|| per feature over iterations (heatmaps in Fig 1, curves in Figs 2, 4, 5) | | |

## 8. Results worth remembering
- Fig 1(b),(c): on Distribution 1 both objectives learn features in the same order (high lambda first). Fig 1(e),(f): on Distribution 2 MAE keeps the lambda order; JEPA "exhibits a complete inversion of the learning order where features with a higher rho_i, rather than a high lambda_i are learned first" (Section 3).
- Fig 2(b) vs (e): with lambda = 1 fixed, JEPA curves separate by rho (greedy in rho) and plateau at different heights; MAE curves rise together and plateau at different heights only. Fig 2(c) vs (f): learning order inverts between objectives.
- Corollary 4.3 (eq (9)): JEPA fixed point rho^L vs MAE rho^{L/(L+1)}; eq (10): one-layer JEPA = infinite-depth MAE. The rho-bias "takes hold only when the encoder is deep and diminishes substantially when the encoder is shallow" (Section 1).
- Corollary 4.8 (eq (17)): JEPA critical-time ratio 1 + (2L-1)/(2L-2) Delta_rho epsilon^{1/L} + ...; MAE ratio 1 + Theta(epsilon^{(L-1)/L}) (no leading rho term).
- Fig 3(e): in the temporal model, feature 1 has higher lambda but lower rho than feature 2 (noise std 1 vs 0.5), so JEPA learns feature 2 first and MAE learns feature 1 first; theory matches simulation.
- Discussion: the bias "allows it to focus on features that are predictive, yet contain minimal noise, as measured by their variance across the training set"; "Conversely, it may shed light on JEPAs vulnerability to spurious or slow features" (straight quotes substituted for the source's \textit).
- Related Work bridge to 2211.10831: "Our findings about JEPA are related to empirical observations in [Sobal et al.] which show JEPA-based methods focus on 'slow features' that vary slowly over time. This is confirmed in our studies where JEPA-based methods will prioritize 'slow features' for which the marginal distribution has the smallest variance." (curly quotes in the source replaced by straight quotes).
- What would falsify it: in the Assumption 4.1 setting, (a) a feature with the same lambda but smaller rho reaching fraction p of its fixed point no later than a higher-rho feature under the JEPA objective; (b) a JEPA fixed point that does not scale as rho^L with depth; (c) MAE showing the same rho-ordering at the same depth; or, empirically, (d) Fig 1(f)'s inversion failing to appear for L = 5 random-init deep linear nets. Outside the setting, the paper itself says the claims are unproven for non-simultaneously-diagonalisable data and nonlinear models.

## 9. Limitations the paper states (Section 7)
- Results need Assumption 4.1: deep linear MLP, balanced orthogonal small init, and simultaneously diagonalisable Sigma^{yx}, Sigma^{xx} ("a potentially stronger assumption ... we expect data distributions of interest to significantly deviate from this assumption").
- Random (He) init: results generalise qualitatively (Fig 5), not proven.
- Nonlinear models and real data: unexplored. Also open: whether the mechanism holds for joint-embedding methods that do not use self-distillation (e.g. VICReg, SimCLR).
- No later paper in this source set addresses these.

## 10. Outline feedback (ch-collapse)
1. "noisy ones are not [learned]" is too strong. The paper's statement is: noisy (low-rho) features are learned LATER (eq (13), Cor 4.8) and converge to the SMALLER weight rho^L (eq (9)); they vanish only as rho -> 0 or L -> large. Suggested card text: "high-influence features first; noisy ones later and shrunk to rho^L, more so with depth."
2. The bias is depth-dependent: a one-layer JEPA has the same dynamics as an infinite-depth MAE (eq (10)). If the toy uses a one-layer encoder it will show the fixed point rho (noisy axis shrinks) but not a learning-ORDER effect; use L >= 2 if the page wants to show order.
3. "the linear-dynamics statement ... (their main equation)" = eq (7), Theorem 4.2, with the fixed point eq (9). Introduce it as "growth rate set by covariance lambda, plateau set by regression coefficient rho".
4. This is the only one of the three papers whose collapse prevention is the asymmetry the chapter describes (StopGrad on the target branch + a predictor V). There is no EMA in the paper; EMA is a toy addition (see below).
5. Vocabulary: "influence" in the paper = regression coefficient rho_i = lambda_i / sigma_i^2 ("features with a large regression coefficient", Section 1). Use that phrase on the page; the word "influence" alone is undefined for the reader.

## Toy spec for the ch-collapse visual (which paper to copy, and the update rules)
Copy THIS paper: Section 2 model (eqs (3)-(5)), gradient descent as the discrete form of eq (6), gradients from eqs (21)-(22), data from the temporal model (eqs (18)-(19)). Do not copy 2211.10831 (conv encoder + GRU + VICReg on 28 x 28 frames) or 2210.07277 (clustering-based, ViT).

Data (paper's temporal model with d = 2, v^a = e_a so ||v^a|| = 1; "our reading": sample stationary pairs directly instead of running a long chain):
```
u_i ~ N(0, 1),  u'_i = gamma_i u_i + sqrt(1 - gamma_i^2) eta_i,  eta_i ~ N(0, 1)      (paper's AR(1), appendix sec:tcm)
x_i = u_i + s_i xi_i,   y_i = u'_i + s_i xi'_i,   xi, xi' ~ N(0, 1)                   (eq (18) with per-frame noise std s_i)
=> lambda_i = gamma_i,  sigma_i^2 = 1 + s_i^2,  rho_i = gamma_i / (1 + s_i^2)          (eq (19) with ||v^a|| = 1)
```
Suggested presets: slow axis gamma_1 = 0.99, s_1 = 0.1 (rho_1 = 0.980); noisy axis gamma_2 = 0, s_2 = 1 (rho_2 = 0, pure per-frame noise). For a depth effect use gamma_2 = 0.5, s_2 = 1 (rho_2 = 0.25, so w_2 -> 0.25 at L = 1 and 0.0625 at L = 2). For the 2211.10831 "fixed background" trap: gamma_bg = 1, s_bg = 0 (rho = 1) against a "dot" axis with gamma = 0.9: the JEPA weights the background at 1 and the dot at 0.9^L.

Model (eq (3)): W-bar = W^L ... W^1 (2 x 2 each), predictor V (2 x 2). Online branch h = W-bar x, prediction p = V h. Target branch W-bar_tgt y with no gradient. Loss (eq (5)) on a batch of n pairs: (1/2n) sum ||V W-bar x - W-bar_tgt y||^2.

Per-step update rules (paper: eqs (6), (21), (22); batch means replace expectations):
```
e        = V W-bar x - W-bar_tgt y                                   per sample, 2-vector
G_V      = mean[ e (W-bar x)^T ]                                      eq (22)
G_Wbar   = mean[ V^T e x^T ]                                          eq (21), gradient w.r.t. the product; stop-grad: no y^T term
G_{W^a}  = (W^L ... W^{a+1})^T  G_Wbar  (W^{a-1} ... W^1)^T           eq (21), chain to layer a
W^a     <- W^a - eta G_{W^a};    V <- V - eta G_V                      eq (6), discretised ("our reading": Euler step, eta ~ 0.05)
```
Toggles ("our reading"; the paper has only stop-grad ON, predictor ON, no EMA):
```
stop-grad ON, EMA OFF :  W-bar_tgt = W-bar (current online weights), no gradient through it   (paper's SG, eq (5))
EMA ON (momentum m)   :  after each step, W^a_tgt <- m W^a_tgt + (1 - m) W^a; target uses W-bar_tgt;  m = 0 recovers stop-grad
stop-grad OFF         :  W-bar_tgt = W-bar AND G_Wbar += mean[ -e y^T ]  (the target branch now trains too; loss is symmetric)
predictor OFF         :  V = I fixed, skip the V update
```
Init (Assumption 4.1): W^a = epsilon^{1/L} U^a, V = epsilon^{1/L} U^v with U orthogonal, epsilon small (0.05 works); a small random jitter is fine (paper's Fig 5 shows random init gives the same picture).
Readouts: w-bar_i = ||W-bar e_i|| (column norms of the product) for the two axes; singular values of the batch embedding matrix (W-bar X) for the side chart; expected fixed point with stop-grad + predictor: w-bar_i -> rho_i^L (eq (9)).

Predicted toggle outcomes (our reading; checked with a 20k-step numpy run of the rules above, eta = 0.05, batch 256, epsilon = 0.05, slow axis rho_1 = 0.980, noisy axis rho_2 = 0):
| stop-grad | predictor | EMA | depth L | ||W-bar e_1|| | ||W-bar e_2|| | reading |
|---|---|---|---|---|---|---|
| on | on | off | 1 | 0.986 | 0.007 | slow axis kept at ~rho_1, noisy axis flattened (eq (9)) |
| on | on | off | 2 | 0.973 | 0.011 | same; rho_1^2 = 0.961 predicted |
| off | on | off | 1 | 0.000 | 0.000 | full collapse to a point |
| on | off (V = I) | off | 1 | 0.000 | 0.000 | full collapse: stop-grad alone does not save a linear JEPA |
| on | on | m = 0.99 | 1 | 0.986 | 0.008 | EMA behaves like stop-grad |
| on | off (V = I) | m = 0.99 | 1 | 0.002 | 0.000 | EMA without a predictor still collapses |
| on | on | m = 0.99 | 2 | 0.959 | 0.010 | as row 2, slower |
Why (from the gradient expressions): with V = I and stop-grad, G_Wbar = W-bar (Sigma^{xx} - Sigma^{yx}) = W-bar diag(1 + s_i^2 - gamma_i), which is positive for every axis, so every w-bar_i decays to 0. With stop-grad off the loss is (1/2) E||W-bar (x - y)||^2 whose gradient is W-bar E[(x - y)(x - y)^T], again positive, so all axes decay (the slow one slowest). Only a trainable V lets the fixed point V_ii = rho_i, w-bar_i = rho_i^L exist (eq (9)). These three sentences are our derivation from eqs (21)-(22), not a statement in the paper; the paper does not analyse the V = I or no-stop-grad cases.
