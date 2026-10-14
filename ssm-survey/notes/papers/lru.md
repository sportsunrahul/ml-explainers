# LRU: Resurrecting Recurrent Neural Networks for Long Sequences

Lens: architecture/training-method, with the ablation leading. The spine of the paper is an ablation
ladder (Tables 1 -> 2 -> 3, expanded in Tables 6 -> 7 -> 8); each step is one table row, and the theory
(Thm 3.1, Lemma 3.2, Prop 3.3, App E) is support for the rows. Serves ch 1 (ch-recurrence).

Sources read: `sources/lru/src/content_main.tex`, `appendix.tex`, `main.tex`, `math_commands.tex`,
`main.bbl` (full); `paper.pdf` for pages, Fig 1, App A code listing. The duplicate folder
`src/Resurrecting RNNs  Arxiv/` is byte-identical to `src/`.

## 1. Identity
- Title: Resurrecting Recurrent Neural Networks for Long Sequences.
- arXiv 2303.06349, version read: v1 (abs.html "Submitted on 11 Mar 2023"; only v1 listed).
- Published: 2023-03-11 (abs.html citation_date; main.tex `\today` = 2023-03-11).
- Authors: Antonio Orvieto (ETH Zurich, work done at DeepMind) ... Soham De (DeepMind). Full list: Orvieto,
  Smith, Gu, Fernando, Gulcehre, Pascanu, De.
- Venue: not stated (abs.html has no journal-ref; comments "30 pages, 9 figures").
- Repo: not stated (no README in `sources/lru`, no URL in the paper). App A gives a JAX listing.

## 2. Core problem
RNNs give fast inference on long sequences but are hard to optimize and slow to train. Deep SSMs (S4 and
variants) match RNN inference cost, train in parallel and do very well on Long Range Arena (LRA). The
paper asks where the SSM gain comes from, and whether a plain deep RNN, changed step by step with standard
signal-propagation arguments, can match it (Abstract; Sec 1, p.1-2).

## 3. What prior methods did and why that hurt (per the paper)
All Sec 1, p.1, unless noted. Cited works by bibkey -> name.
- RNNs are "hard to train in practice"; they "suffer from the vanishing and exploding gradient problem",
  which "makes it difficult for these models to learn about the long-range dependencies".
  Cites: Hochreiter 1991 (diploma thesis "Untersuchungen zu dynamischen neuronales netzen", sic in bib), Bengio,
  Simard, Frasconi 1994 ("Learning long-term dependencies with gradient descent is difficult"), Pascanu,
  Mikolov, Bengio 2013 ("On the difficulty of training recurrent neural networks").
- Fixes tried: orthogonal/unitary RNNs (Arjovsky, Shah, Bengio 2016 "Unitary evolution RNNs"; Helfrich,
  Willmott, Ye 2018 "Orthogonal RNNs with scaled Cayley transform"), gating: LSTM (Hochreiter and
  Schmidhuber 1997), GRU (Cho et al. 2014a "On the properties of neural machine translation").
- "Nonetheless, these models are still slow to optimize due to the inherently sequential nature of their
  computation" (cites Kalchbrenner et al. 2016 "Neural machine translation in linear time"), "and are
  therefore hard to scale."
- Transformers (Vaswani et al. 2017): "attention layers are easier to scale and parallelize during
  training, and crucially they do not suffer from the vanishing gradient problem, since the interaction
  between any two tokens in the sequence is modeled by direct edges in the network." Cost: "computational
  and memory costs scale quadratically as $O(L^2)$". RNNs "are therefore typically faster than
  transformers at inference time even for modest sequence lengths" (cites Liu et al. 2019 "MKD").
- What SSMs did differently (Sec 1 p.1-2; Sec 2.1 "Some key differences", p.4): S4 (Gu, Goel, Re 2021
  "Efficiently modeling long sequences with structured state spaces") and DSS/S4D/S5 (Gupta, Gu, Berant
  2022a; Gu, Gupta, Goel, Re 2022a; Smith, Warrington, Linderman 2022) model token interactions through a
  hidden state "under proper discretization", unroll like an RNN at inference, and "since SSMs are linear
  in the temporal dimension, they are easily parallelizable during training". Eq. (3) "can be efficiently
  parallelized until $k=L-1$ using parallel scans" (cites Martin and Cundy 2017 "Parallelizing linear
  recurrent neural nets over sequence length"; Smith et al. 2022) "unlike a nonlinear RNN where the
  computation has to be performed sequentially." Also: A and B are tied through Delta; A is complex
  diagonal, initialized deterministically from HiPPO (Gu et al. 2020).
- Root of vanishing/exploding (Sec 3.2.1, p.6, after Eq. 4): "Exponentiations of the matrix $A$ in the
  equation above are the source of the well-known vanishing/exploding gradient issue in RNNs" (cites
  Bengio 1994; Pascanu 2013). Nonlinear RNN state is bounded ("forced to live on the compact image of the
  activation function"); the linear RNN state "can potentially explode or vanish exponentially as $k$
  increases".
- Related-work restatement (App B, p.18): vanishing/exploding gradients (Bengio 1994, Pascanu 2013) "is one
  barrier to training" RNNs "with gradient descent"; unitary RNNs keep eigenvalues on the unit circle but
  (1) restrict the function class and (2) need a Stiefel-manifold projection each step (expensive).

## 4. The one key idea
A deep stack of LINEAR, complex-diagonal recurrences, with nonlinearity only in the position-wise MLP/GLU
between them, parameterized so every eigenvalue stays inside the unit disk (|lambda| = exp(-exp(nu_log))),
initialized on a ring near |lambda| = 1, and with a per-channel input scale gamma that cancels the forward
blow-up. This matches S4/S4D/S5 on all six LRA tasks without discretization or HiPPO (Sec 3.4 last para,
p.10; Sec 5, p.12).

## 5. Mechanism, step by step (the ablation ladder; numbers in Sec 8)
Same backbone throughout: 6 residual blocks, Norm -> Recurrence -> GLU -> Skip (Fig 1 left, p.2; Sec 2.2,
p.4; App D.1, p.21). Only the recurrence changes.
0. Start: vanilla dense RNN, Eq. (1) with sigma = tanh or ReLU, Glorot init (Haiku vanilla RNN, Sec 3.1
   footnote, p.5). Fails PathFinder/PathX (Table 1 caption).
1. Linearize: sigma = identity, $x_{k} = A x_{k-1} + B u_k$ (Sec 3.1, p.5). Improves sCIFAR, ListOps, Text,
   Retrieval (Table 1). Gains: can control how fast gradients vanish/explode, and can parallelize (Sec 1
   bullet "Linear Recurrences", p.2).
2. Diagonalize (complex): $A = P\Lambda P^{-1}$, learn $(\Lambda,\bar B,\bar C,D)$, output takes real part
   (Sec 3.2.1, p.6). Init Lambda uniform on the unit disk via Lemma 3.2 with $[r_{\min},r_{\max}]=[0,1]$, which
   matches the Glorot spectrum in the large-N limit (Thm 3.1, strong circular law; Fig 2, Fig 3, p.7).
   Lambda stored as Re + Im floats (footnote 7, p.7). About 8x faster to train on sCIFAR than a dense ReLU
   RNN (Sec 3.2.2, p.6). Table 2 row "Lambda Real + Im".
3a. Exponential (polar) parameterization: $\Lambda = \diag(\exp(-\nu + i \theta))$, learn nu, theta
   (Sec 3.3, p.8). Decouples magnitude and phase, easier for Adam (App E.2, Fig 8, p.27). First run above
   chance on PathFinder. Table 2 row "Lambda Exp".
3b. Stable exponential: $\lambda_j:=\exp(-\exp(\nu_j^{\log})+i\theta_j)$, so $|\lambda_j|<1$ always; exp chosen
   over ReLU "to increase granularity around $|\lambda| = 1$" (Sec 3.3 "Enforcing stability", p.8). Table 2
   row "Lambda Stable Exp" (PathFinder above 93%).
4. Ring init: raise $r_{\min}$, $r_{\max}$ toward 1 to "bias the network towards longer-range interactions
   and avoid vanishing gradients" (Sec 3.3, p.8). Without stability, sCIFAR degrades past r_max = 0.9; with
   stability, r_max up to 0.99 helps (Table 5a vs 5b, p.21). Table 2 row "+ Ring Init". Per-task rings in
   Table 8.
5. gamma normalization: near the unit circle the forward pass blows up by $1/(1-r^2)$ (Prop 3.3, Eq. 5,
   Eq. 6, p.9). Fix: scale input per channel by $\exp(\gamma^{\log})$, init
   $\gamma_i^{\log}\leftarrow \log(\sqrt{1-|\lambda_i|^2})$, Eq. (7) (p.10). Table 3 / Table 8 row LRU.
6. Small phase (PathX only): restrict theta init to $[0, \pi/10]$ and log-parameterize $\theta =
   \exp(\theta^{\log})$ (Sec 3.4, p.10). Gives 94.2% on PathX. Both gamma and small phase were needed:
   "We were unable to learn when using restricted phase at initialization without also introducing
   $\gamma$ normalization" (Sec 3.4, p.10; Fig 5).
Training-time parallelism: the linear recurrence is unrolled with an associative scan (App A listing, p.17,
`jax.lax.associative_scan` with operator `binary_operator_diag`).

Why nonlinearity between layers suffices (Sec 3.1 p.5; App E.1, p.23-25):
- Linear RNNs approximate any shift-invariant LINEAR map (cites Li, Han, E, Li 2022 "Approximation and
  optimization theory for linear continuous-time RNNs"), but "one thing a linear RNN cannot do is store
  information under frequencies which are not present in the input signal" (App E.1.1, p.23).
- A position-wise ReLU after the RNN "leaks" information to higher frequencies (Prop E.1, Fig 6, p.24).
  Bolded claim: "frequencies can be scaled up and down arbitrarily by the RNN, and can then be translated
  in the space using the ReLU" (p.24).
- Koopman view (cites Koopman and von Neumann 1932): "any regular nonlinear dynamical system is
  representable by a linear RNN after proper nonlinear reparameterization of the inputs -- which can be
  performed by an MLP" (App E.1.2, p.25-26). Paper says a quantitative version would need more work.
- Trade-off stated: single-layer tanh/sigmoid RNNs are Turing complete, the linear one is not (cites Chung
  and Siegelmann 2021), but the deep stack has nonlinearity after each block (Sec 3.1, p.5).
- Optimization: on a toy task (learn one length-100 convolution kernel, 32 trajectories, 100-dim state,
  Adam, Glorot), "For all learning rates in our grid, the linear variant is faster to converge" than tanh
  (App E.2, Fig 7, p.26-27).

## 6. Equations
Notation: LRU (like S4) writes input u, state x, output y. Page notation: input x, state h (size N),
output y. So LRU $x_k$ = page $h_k$, LRU $u_k$ = page $x_k$. LRU's Lambda is the page's (diagonal) A.
Sign/scale: $|\lambda| = e^{-\nu}$ with $\nu = \exp(\nu^{\log}) > 0$, so $|\lambda|<1$ for any real
$\nu^{\log}$; $|\lambda| \to 1$ as $\nu^{\log}\to-\infty$. Custom macros used: `\R`, `\Cmp` (= C), `\E`,
`\diag` (operator), from `math_commands.tex`.

E1. RNN layer, Eq. (1), `eq:RNN`, Sec 2.1, p.3:
```
&x_{k} = \sigma(Ax_{k-1}+ Bu_k),\quad y_k = C x_k + D u_k,
```
"If $\sigma$ is the identity function, then we say the RNN layer is linear." Gloss: next state h_k =
nonlinearity(A h_{k-1} + B x_k); output y_k = C h_k + D x_k. $A\in\R^{N\times N}$ state-to-state,
$B\in\R^{N\times H_{in}}$ input-to-state, $C\in\R^{H_{out}\times N}$ state-to-output, D skip. No bias
(footnote: biases go in the surrounding MLPs). $x_0 = 0$.

E2. S4-like continuous system, Eq. (2), `eq:S4`, Sec 2.1, p.3 (S5-like simplification, one shared Delta):
```
&\frac{d}{dt}x_{\text{ct}}(t) = \tilde A x_{\text{ct}}(t) + \tilde B u_{\text{ct}}(t),\nonumber\\
&y_{\text{ct}}(t) =  \Re\left[\tilde C  x_{\text{ct}}(t)\right] + \tilde D u_{\text{ct}}(t),
```
$\tilde A = \text{diag}(\tilde a)$, $\tilde a\in\Cmp^N$. S4D-Lin $\tilde a_n = -\frac{1}{2}+ i\pi n$; S4D-Inv
$\tilde a_n = -\frac{1}{2}+ i\frac{N}{\pi}\left(\frac{N}{n+1}-1\right)$ (p.3).

E3. ZOH-discretized recurrence, Eq. (3), `eq:S4-disc`, Sec 2.1, p.3:
```
&x_{k} = Ax_{k-1}+ Bu_k, \quad y_k = C x_k + D u_k,
```
with "$A = \exp(\Delta \tilde A)$, $B = (A-I)\tilde A^{-1}\tilde B$, $ C = \tilde C$ and $ D = \tilde D$". Gloss:
the same shape as a linear RNN, but A and B are both functions of Delta, so they share a parameter.

E4. Unrolled linear recurrence, Eq. (4), `eq:lin_rnn_unroll`, Sec 3.2.1, p.6:
```
x_{0} = Bu_0,\quad x_1 = ABu_0 + Bu_1,\quad x_2 = A^2Bu_0 + ABu_1 + B u_2, \quad \dots \quad \implies \quad x_k = \sum_{j=0}^{k-1} A^jBu_{k-j}.
```
Gloss: the state is a weighted sum of all past inputs; input j steps ago is multiplied by A^j. This is the
outline's h_t = sum_k a^{t-k} b x_k. The printed upper limit k-1 is a source typo (the expansion needs j=0..k); quote the expansion terms, not the sum.
Diagonal version (inline, p.6): $\bar x_k = \sum_{j=0}^{k-1} \Lambda^j\bar B u_{k-j}$, $\bar x_k := P^{-1}x_k$,
$\bar B := P^{-1}B$, $y_k = \Re[\bar C \bar x_k] + D u_k$.

E5. Stability (inline, Sec 3.2.1 "Stability", p.6):
```
|x_{k,j}|=O(|\bar x_{k,j}|)= O(|\lambda_j|^k)
```
"a sufficient condition to ensure stability (i.e. $x_k$ does not explode) is therefore $|\lambda_j|<1$ for
all $j$" (cites Gu et al. 2021). Gloss: each eigen-channel scales like |lambda|^k; above 1 it explodes,
below 1 it decays.

E6. Strong circular law, Theorem 3.1 (`thm:ginibre`), p.7: real N x N matrix, iid Gaussian entries, mean
0, variance 1/N; empirical spectral measure -> uniform on $\{|z| \le 1\}$ as $N\to\infty$ (cites Ginibre
1965). Gloss: a Glorot-initialized dense A has eigenvalues spread uniformly over the unit disk, so the
diagonal model should start uniform on the disk.

E7. Ring sampling, Lemma 3.2 (`lemma:sampling_exp`), p.7:
```
\nu = -\frac{1}{2}\log\left(u_1(r_{\max}^2-r_{\min}^2)+r_{\min}^2\right) and \theta = 2\pi u_2
```
(as printed: "Compute $\nu = ...$ and $\theta = 2\pi u_2 $"), with $u_1,u_2\sim U[0,1]$,
$0\le r_{\min}\le r_{\max}\le1$; then $\exp(-\nu+i\theta)$ is uniform on the ring between radii r_min and r_max.
Definition used: $\exp(-\nu + i\theta) := e^{-\nu}(\cos(\theta) + i \sin(\theta))$. Gloss: nu sets the
magnitude (e^{-nu}), theta the rotation per step.

E8. Exponential parameterization (Sec 3.3, p.8): `\Lambda = \diag(\exp(-\nu + i \theta))`, learn nu, theta.

E9. Stable exponential parameterization (Sec 3.3 "Enforcing stability", p.8):
```
\lambda_j:=\exp(-\exp(\nu_j^{\log})+i\theta_j)
```
"$\nu^{\log}\in\R^N$ is the parameter we optimize, and we set $\nu_j^{\log} := \log(\nu)$ at initialization."
$|\lambda| = 1$ "achieved at $\nu^{\log} = -\infty$ (while $|\lambda| =0$ is achieved at $\nu^{\log} =\infty$)".
Final LRU form adds a log phase (Sec 3.4, p.10): "$\theta = \exp (\theta^{\log})$, where $\theta^{\log}$ is
optimized". So the final eigenvalue is $\lambda_j = \exp(-\exp(\nu_j^{\log}) + i\exp(\theta_j^{\log}))$; this
combined form appears only in Fig 1 right panel (p.2) and the App A code (p.17):
`Lambda = jnp.exp(-jnp.exp(nu_log) + 1j*jnp.exp(theta_log))`.
Source typo: the Sec 3.4 "Reducing Eigenvalue Phase" paragraph (p.10) prints
`\Lambda = \text{diag}(\exp(-\exp(\nu^{\log}) + \theta))` with the `i` missing. Quote the Sec 3.3 form or
the code, not this one.

E10. Forward-pass blow-up, Proposition 3.3 (`prop:r_min_max_effect`), p.9 (display unnumbered):
```
\E[\|x_\infty\|^2_2] = \frac{1}{r_{\max}^2-r_{\min}^2}\log\left(\frac{1-r_{\min}^2}{1-r_{\max}^2}\right) \E[\| B u\|_2^2].
```
Conditions: Lambda diagonal, uniform on ring with $r_{\min}<r_{\max}<1$, constant or white-noise input,
Glorot B. Verified numerically (Fig 9, p.30: N=500, L=10k, 10 runs). Gloss: expected state energy =
(input energy) x a gain that grows without bound as r_max -> 1.

E11. Thin-ring limit, Eq. (5) (unlabelled), p.9:
```
\lim_{\epsilon\to 0}\frac{\E[\|x_\infty\|^2_2]}{\E[\|B u\|_2^2]} = \lim_{\epsilon\to 0}\left[ \frac{1}{\epsilon}\log\left(1+\frac{\epsilon}{\rho}\right)\right] = \lim_{\epsilon\to 0} \left[\frac{1}{\epsilon}\left(\frac{\epsilon}{\rho} + O(\epsilon^2)\right)\right] = \frac{1}{\rho} = \frac{1}{1-r^2}.
```
$\epsilon = r_{\max}^2-r_{\min}^2$, $\rho = 1-r_{\max}^2$. Paper's wording: "if we initialize $\rho$-close to the
unit disk, the forward pass blows up by a factor $1/\rho$ (since the contributions from previous states
take longer to decay)".

E12. One-dimensional white-noise derivation, Eq. (6), `eq:gamma_derivation`, p.9 ($\Lambda=\lambda$, $B=1$):
```
\E|x_k|^2= \left(\sum_{i=0}^{k-1} \lambda^i \E[u_{k-i}]\right)\left(\sum_{j=0}^{k-1} \lambda^j \E[u_{k-j}]\right)^*
 = \sum_{i,j=0}^{k-1} \lambda^i  (\lambda^j)^* \E[u_{k-i} u_{k-j}]
= \sum_{i=0}^{k-1}|\lambda|^{2i} \overset{\infty}{\to} \frac{1}{1-|\lambda|^{2}}.
```
Gloss: with uncorrelated unit inputs the state variance is a geometric sum of |lambda|^{2i}; it tends to
1/(1-|lambda|^2), which is huge when |lambda| is near 1.

E13. Normalized LRU recurrence, Eq. (7), `eq:normalized_rnn`, Sec 3.4, p.10:
```
x_{k} = \Lambda x_{k-1} +\exp(\gamma^{\log})\odot (B u_{k}),
```
init "$\gamma_i^{\log}\leftarrow \log(\sqrt{1-|\lambda_i|^2})$". Output $y_k = C x_k + D u_k$ (App A, real
part taken: `(C @ x).real + D * u`). Gloss: each eigen-channel's input is scaled by sqrt(1-|lambda|^2), which
cancels the 1/(1-|lambda|^2) variance gain, so state variance stays near input variance at init. gamma is
trainable (log-parameterized because 1-|lambda|^2 -> 0 near the circle; footnote 9 p.10: recomputing gamma
each step "work[s] similarly").

E14. Parallel scan operator (App A listing, p.17, code not TeX):
```
def binary_operator_diag(element_i, element_j):
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j
```
Gloss: combining two segments (a_i, b_i) then (a_j, b_j) gives decay a_j a_i and offset a_j b_i + b_j; this
is associative, so `jax.lax.associative_scan` computes all states in parallel. Same operator as the
outline's ch 7 (a1,b1) o (a2,b2) = (a1 a2, a2 b1 + b2).

E15. S4D-Lin eigenvalues after ZOH (Sec 4 "Discretization changes initialization spectrum", p.11):
eigenvalues $\exp(-\Delta/2 + i\pi \Delta n)$; with "$\Delta = 1e{-3}, N = 128$" magnitude
"$\exp(-\Delta/2)\approx 0.9995$" and phase "$\theta = \pi \Delta n\overset{\sim}{\in}[0, \pi/8]$". Gloss: S4D
also starts on a thin ring near the unit circle with small phase; LRU picks these directly.

E16. ReLU spectrum, Proposition E.1 (`prop:relu_effect`), App E.1.1, p.24:
```
\mathcal{F}_{\text{ReLU}(u)} = \mathcal{F}_{u}(\omega) \star\left[\sum_{i} 2L_i e^{-i \omega p_i} \text{sinc}(\omega L_i)\right].
```
Gloss: the ReLU output spectrum is the input spectrum convolved with a kernel that moves energy to other
frequencies, which a linear recurrence alone cannot do.

## 7. Config / setup
Architecture (App D.1, p.21; Fig 1, p.2):
- Encoder to H features -> 6 residual blocks -> linear head. Block: BatchNorm (pre-norm, "we always use
  batch normalization") -> recurrent layer -> GLU with dropout (PathX: GLU "without one additional linear
  transform", as in S5) -> skip.
- Bidirectional for PathFinder and PathX; unidirectional otherwise.
- LRU layer shapes (App A, p.17): Lambda in C^N (nu_log, theta_log in R^N); B in C^{N x H}; C in C^{H x N};
  D in R^H (elementwise, despite Eq. (1) writing a matrix); gamma_log in R^N. Input/output (L, H).
- Trainable: nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log (all).
- Init (App A code): B_re, B_im ~ N(0, 1/(2H)); C_re, C_im ~ N(0, 1/N); D ~ N(0, 1); Lambda by Lemma 3.2,
  default r_min=0, r_max=1, max_phase=6.28. Text (p.17) says B "scaled additionally by a factor 2"; Sec 3.2.2
  (p.7) says halved variance per component for B and C. The code halves only B's. Not reconciled.

Per-task LRU hyperparameters (Table 9, p.23):
| Task | Depth | H | N | Iterations | Batch | LR factor | WD | Dropout |
|---|---|---|---|---|---|---|---|---|
| sCIFAR | 6 | 512 | 384 | 180k | 50 | 0.25 | 0.05 | 0.1 |
| ListOps | 6 | 128 | 256 | 80k | 32 | 0.5 | 0.05 | 0.0 |
| Text | 6 | 256 | 192 | 50k | 32 | 0.1 | 0.05 | 0.1 |
| Retrieval | 6 | 128 | 256 | 100k | 64 | 0.5 | 0.05 | 0.1 |
| PathFinder | 6 | 192 | 256 | 500k | 64 | 0.25 | 0.05 | 0.0 |
| PathX | 6 | 128 | 256 | 250k | 32 | 0.25 | 0.05 | 0.0 |

Per-task LRU rings and phase (Table 8, p.22, row "$[r_{\min}, r_{\max}]$" under LRU; caption for phase):
| Task | [r_min, r_max] | phase init |
|---|---|---|
| sCIFAR | [0.9, 0.999] | [0, 2pi] |
| ListOps | [0.0, 0.99] | [0, 2pi] |
| Text | [0.5, 0.9] | [0, 2pi] |
| Retrieval | [0.5, 0.9] | [0, 2pi] |
| Pathfinder | [0.9, 0.999] | [0, 2pi] |
| PathX | [0.999, 0.9999] | [0, pi/10] |
Fig 4 caption (p.10) uses $[0,\pi/50]$ and $\nu_j=0.00005$ only as an illustration (impulse response over
L = 16k); the PathX run used $[0, \pi/10]$ (Sec 3.4, p.10; Table 8 caption).

Optimization (Sec 2.2, p.4; App D.2-D.3, p.22-23):
- AdamW. LR warmup linear from 1e-7 over the first 10% of training, then cosine to 1e-7.
- Smaller LR (base x "LR factor" < 1) on recurrent A (Lambda) and B, and on gamma. No weight decay on
  A, B, gamma. (Cites Steil 2004; Gu et al. 2021 for the smaller recurrent LR.)
- Base LR tuned on a logarithmic grid of 2. H and N taken from the tuned S5 baseline.
- 3 seeds, mean and standard error reported.
- Hardware: A100 GPUs (App D.2). GPU-hours: not stated. Loss: not stated (LRA tasks are classification;
  our reading: cross-entropy).
- Data: the 6 LRA tasks (Tay et al. 2020), except sCIFAR uses COLOURED images, not LRA's grayscale
  (Sec 2.2, p.4; App D.4, p.23). PathX sequence length "$16k$ tokens" (Sec 3.4, p.9).

## 8. Results worth remembering
Test accuracy %, mean (std err over 3 seeds). X = not above chance.

A. Ablation ladder, all six tasks (Table 6 p.21 rows RNN-*, Table 7 p.22, Table 8 p.22). Main-text
versions: Table 1 p.5 (4 tasks), Table 2 p.8 (sCIFAR, ListOps, Pathfinder), Table 3 p.9.
| Step (row label) | sCIFAR | ListOps | Text | Retrieval | Pathfinder | PathX |
|---|---|---|---|---|---|---|
| RNN-ReLU | 69.7 (0.2) | 37.6 (8.0) | 88.0 (0.1) | 88.5 (0.1) | X | X |
| RNN-Tanh | 69.9 (0.3) | 43.9 (0.1) | 87.2 (0.1) | 88.9 (0.2) | X | X |
| RNN-Lin = "Dense A" | 72.2 (0.2) | 50.4 (0.2) | 89.1 (0.1) | 89.1 (0.1) | X | X |
| Lambda Real + Im | 86.5 (0.1) | 58.8 (0.3) | 87.4 (0.3) | 87.8 (0.5) | X | X |
| Lambda Exp | 85.4 (0.7) | 60.5 (0.3) | 86.5 (0.4) | 89.4 (0.1) | 65.4 (9.0) | X |
| Lambda Stable Exp | 87.2 (0.4) | 59.4 (0.3) | 87.6 (0.3) | 89.1 (0.2) | 93.5 (0.5) | X |
| + Ring Init | 88.1 (0.0) | 59.4 (0.3) | 89.4 (0.1) | 90.1 (0.1) | 94.4 (0.3) | X |
| + gamma norm (+ small phase on PathX) = LRU | 89.0 (0.1) | 60.2 (0.8) | 89.4 (0.1) | 89.9 (0.1) | 95.1 (0.1) | 94.2 (0.4) |
"Dense A" in Table 2/7 is the same model and numbers as "RNN-Lin" in Table 1/6: one step, not two.
Table 8 relabels rows: "Linear Dense RNN", "Diagonal Complex RNN", "Stable Exp Param w/ Ring Init",
"$+\gamma$ Normalization (LRU)". Fig 1 right (p.2) plots per-step LRA curves; read no numbers off it.

B. LRU vs deep SSMs (Table 3, p.9; same rows in Tables 6-8):
| Model | sCIFAR | ListOps | Text | Retrieval | Pathfinder | PathX |
|---|---|---|---|---|---|---|
| LRU | 89.0 (0.1) | 60.2 (0.8) | 89.4 (0.1) | 89.9 (0.1) | 95.1 (0.1) | 94.2 (0.4) |
| S4D (our reprod.) | 91.5 (0.2) | 60.2 (0.3) | 86.4 (0.0) | 89.5 (0.0) | 94.2 (0.3) | 97.5 (0.0) |
| S5 (our reprod.) | 88.8 (0.1) | 58.5 (0.3) | 86.2 (0.1) | 88.9 (0.0) | 95.7 (0.1) | 96.0 (0.1) |
| S4 (paper results) | 91.1 | 59.6 | 86.8 | 90.9 | 94.2 | 96.4 |
| S4D-LegS (paper results) | 89.9 | 60.5 | 86.2 | 89.5 | 93.1 | 91.9 |
| S5 (paper results) | 90.1 | 62.2 | 89.3 | 91.4 | 95.3 | 98.6 |
- No average column exists in the paper. If the page shows a mean, it must be labelled "computed from
  Table 3, not in paper". (Computed, our reading: LRU 86.3; S4 paper-results row 86.5. Not in claims.)
- S4 row is rounded from S4's own source (S4 v3 experiments.tex: 59.60 / 86.82 / 90.90 / 94.20 / 96.35),
  and its sCIFAR 91.1 is S4's sequential-CIFAR result (91.13, S4 Table tab:image; colour not stated there), NOT S4's LRA Image (grayscale)
  88.65. So LRU's row is not comparable with S4's published LRA average 86.09. Cite "as reported in LRU
  Table 3".
- Sequential CIFAR: only the coloured sCIFAR column above (LRU 89.0). Best sCIFAR 89.0 at r_min=0.9,
  r_max=0.999 (App C.2, p.20).

C. Speed (Sec 3.2.2 p.6; Table 4 p.20, steps/sec on A100):
- "diagonal linear RNNs to be $\sim$8 times faster to train than a dense RNN with ReLUs" on sCIFAR,
  "matching the speed of our implementations of S4D and S5".
| Model | sCIFAR | ListOps | Text | Retrieval | Pathfinder | PathX |
|---|---|---|---|---|---|---|
| Tanh RNN | 2.0 | 1.1 | 0.5 | 0.5 | 2.1 | 0.14 |
| LRU | 15.9 (8x) | 2.1 (1.9x) | 14.7 (29x) | 5.7 (11.4x) | 15.5 (7.4x) | 2.4 (17x) |
| S4D (repro) | 13.5 | 2.2 | 10.6 | 3.0 | 24.5 | 2.6 |
| S5 (repro) | 15.9 | 2.2 | 14.4 | 5.7 | 15.6 | 2.3 |

D. Stability x ring x gamma sweep, sCIFAR, LR fixed 0.004 (Table 5, p.21; text App C.2 p.20):
| r_max \ r_min | (a) no stab: 0 / 0.5 / 0.9 | (b) stable: 0 / 0.5 / 0.9 | (c) + gamma: 0 / 0.5 / 0.9 |
|---|---|---|---|
| 0.9 | 87.6 / 87.8 / 87.9 | 86.2 / 86.6 / 87.3 | 86.4 / 86.5 / 88.3 |
| 0.99 | 83.8 / 85.8 / 81.9 | 87.8 / 87.7 / 88.1 | 88.1 / 88.4 / 89.0 |
| 0.999 | 83.9 / 84.8 / 84.8 | 87.4 / 87.4 / 87.5 | 88.1 / 88.6 / 89.0 |
Reading: without stability, accuracy drops once r_max > 0.9; with stability the best moves to 0.99;
with gamma there is "no degradation in performance even when $r_{\max} = 0.999$" (p.20).

E. Eigenvalue-magnitude story (for viz-recur), with locations:
- Long memory needs $|\lambda|$ near 1: "for reasonings requiring consideration of interactions between
  distant tokens, eigenvalues in the recurrence need to have magnitude close to $1$. Otherwise ... when
  taking powers of eigenvalues close to the origin, the signal from past tokens quickly dies out" (Sec 3.3,
  p.8; cites Gu et al. 2022b "How to train your HiPPO", Gupta et al. 2022a). Same paragraph: these
  eigenvalues "are also likely to make the system unstable during training".
- $|\lambda|>1$: state explodes as $O(|\lambda|^k)$ (E5, p.6). Exp parameterization makes $|\lambda|>1$
  unreachable (E9).
- Even at $|\lambda|<1$, close to 1, the forward pass blows up by $1/(1-r^2)$ (Prop 3.3, Eq. 5, p.9); at
  init the training loss "started to blow up" as r_min, r_max -> 1 (Sec 3.4, p.9; Fig 5, p.10).
- Phase: uniform $\theta\in[0,2\pi]$ on L = 16k makes most channels oscillate many times (Fig 4, p.10), a
  bias to "spurious features"; PathX then sits at chance (Fig 5).

F. Insights on S4 (Sec 4, p.11-12), stated as findings:
- Exponential parameterization helps through magnitude-phase decoupling, "not necessarily attributable to
  accurate integration".
- "Structured initialization is not necessary": uniform ring + normalization solves PathX; HiPPO "should
  not be thought of as the main source of S4 success".
- "Discretization performs normalization": replacing S4D's $(\exp(\Delta \tilde A)-I)\tilde A^{-1}\tilde B$ with
  $\Delta \tilde B$ gives "a close match in performance"; the Delta in front of B "effectively scales the
  recurrence to avoid blow-ups -- similar to our $\gamma$". (Source typo: the inline recurrence there omits
  $x_{k-1}$.)
- "Parameter sharing is not necessary": decoupling Delta in A and B "does not decrease performance".
- Conclusion (p.12): SSM success comes from "linear recurrences and complex diagonal exponential matrices,
  combined with the normalization and initialization induced by discretization".

## 9. Limitations the paper states
- Aim is not to beat S4 (Sec 1, p.3): "the main purpose of our work is not to surpass the performance of
  S4-based models". LRU trails S4D/S5 on PathX (94.2 vs 97.5/96.0 repro) and S4D on sCIFAR (Table 3).
- Small phase tried only on PathX; "We did not explore using a smaller phase at initialization for the
  other LRA tasks" (Sec 3.4, p.10).
- Linear RNN loses single-layer Turing completeness (Sec 3.1, p.5). Koopman argument is qualitative
  ("a quantitative discussion would require substantial technical efforts", App E.1.2, p.26).
- Evaluated only on LRA (classification); no language modelling. Later fix (from sources given): Griffin
  (2402.19427) adds an input-dependent gate to this recurrence (RG-LRU) and scales it to language models.
- Recurrence is time-invariant (Lambda does not depend on the input). Our reading; the paper does not
  discuss selection. Mamba and RG-LRU make the decay input-dependent.

## 10. Outline feedback (ch 1)
- The "why dropped" paragraph should cite LRU Sec 1 (p.1) for: vanishing/exploding (Hochreiter 1991,
  Bengio et al. 1994, Pascanu et al. 2013), sequential computation "slow to optimize ... hard to scale"
  (Kalchbrenner et al. 2016), attention's "direct edges" and $O(L^2)$. Parallel-scan contrast: Sec 2.1
  "Some key differences" (p.4).
- The scan's cost (O(L) work, O(log L) depth) is NOT stated in LRU. LRU only gives the operator (App A),
  "~8 times faster" (Sec 3.2.2) and steps/sec (Table 4). Source the cost from `parscan` (Martin and Cundy
  2017, which LRU cites) or S5.
- The gradient "a^T" readout: LRU states only that powers of A cause vanishing/exploding gradients
  (Sec 3.2.1, p.6) and that the state scales as $O(|\lambda|^k)$. It gives no explicit gradient formula.
  Frame a^T as the forward impulse response, and cite Bengio 1994 / Pascanu 2013 via LRU.
- viz-recur: a real a in [-1.2, 1.2] gives decay, sign flips or explosion, not true ringing. LRU's point is
  complex lambda = r e^{i theta}. Consider a phase slider (Fig 4 impulse responses) and a second readout
  for the forward gain 1/(1-|lambda|^2) (Eq. 6), which motivates gamma. Both are exact toy arithmetic.
- LRU delta box: use Table 3 (p.9). Show a per-step mini-ladder from Table 8 rather than one number. Label
  sCIFAR as coloured, give no LRA "average" unless marked computed, and do not put LRU's row next to S4's
  own LRA table (Image 88.65 vs sCIFAR 91.1).
- LRU is also a natural cite for ch 4's "Delta: DSS and S4D show a diagonal A is enough" and for ch 7's
  scan operator (App A listing is the same operator).
- Family tree (ch 12): LRU (2023-03-11) -> Griffin/Hawk (RG-LRU, 2024).
