# LSSL: Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers

Read: arXiv 2110.13985v1, branch depth (abstract, Sec. 1 Fig. 1, Sec. 2 discretization, Sec. 3.1-3.2 views and
Lemma 3.1, Sec. 4 Theorems 1-2, Table 1, App. B.2-B.3 init, Sec. 6 limitations). LaTeX `sources/lssl/src/`,
PDF 46 pages. Serves ch 2 (discretization, Delta) and ch 4 (three views).

## 1. Identity
- Title: Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers.
- arXiv 2110.13985, version read v1, submitted 2021-10-26 (abs.html; only version).
- Authors: Albert Gu (first), Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, Christopher Re (last).
- Venue: NeurIPS 2021 (abs.html Comments).
- Code: https://github.com/HazyResearch/state-spaces (footnote 2, App. E.2 area, p41). No README in sources.

## 2. Core problem
RNNs (stateful, cheap inference, hard to train), CNNs (parallel training, fixed context) and neural ODEs
(continuous time, slow) each have one strength. Find one layer with all three, and make it remember long
dependencies.

## 3. What prior methods did and why that hurt (Sec. 1)
- RNNs: constant cost per step but slow to train and vanishing gradients.
- CNNs: parallel but not sequential, costly inference, limited context.
- NDEs: principled but very inefficient.
- CKConv, UnICORNN, LMU, HiPPO combine some of these "at the price of reduced expressivity".
- HiPPO's A matrices were fixed: only three special cases were known, none trainable (Sec. 4.1).

## 4. The one key idea
Simulate a linear continuous-time state space x' = Ax + Bu, y = Cx + Du with a step size Delta t. The same layer
is then a continuous model, a linear recurrence, and a convolution with kernel (CB, CAB, ..., CA^{L-1}B).
Initialize A to HiPPO-LegS so it remembers.

## 5. Mechanism, step by step
1. Per feature channel h in [H]: a 1-D input sequence u^(h) of length L, its own Delta t_h (Sec. 3.1).
2. Discretize (1) with the bilinear GBT (alpha = 1/2) to get A-bar, B-bar (Sec. 2, Eq. (3)).
3. Recurrent view: x_t = A-bar x_{t-1} + B-bar u_t, y_t = C x_t + D u_t; state x in R^{H x N} (Eq. (4)-(5)).
4. Convolutional view: with x_{-1} = 0, y = K_L(A-bar, B-bar, C) * u + D u (Eq. (6)-(7)); FFT convolution.
5. Continuous view: change Delta t to handle irregular sampling or a new sampling rate (Fig. 1 left; Sec. 5.3).
6. Stack LSSL layers with normalization and residuals (Sec. 3.3). LSSL-f freezes A and Delta t; full LSSL learns
   them (Sec. 4).

## 6. Equations
Notation: LSSL writes input u, state x, output y, step `\dt` (macro `\newcommand{\dt}{\Delta t}`). Page: u -> x,
x -> h, y -> y, Delta t -> Delta. Sign: here A enters with a plus (x' = Ax + Bu), same as Mamba/page convention.

### Eq. (1), (2) continuous state space (Sec. 1, p2)
```tex
\dot{x}(t) &= Ax(t) + Bu(t) \label{eq:1} \\
y(t) &= Cx(t) + Du(t) \label{eq:2}
```
"where \( A \) controls the evolution of the system and \( B, C, D \) are projection parameters."

### Eq. (3) GBT (Sec. 2, p4)
```tex
x(t + \dt) = (I - \alpha \dt \cdot A)^{-1}(I + (1 - \alpha) \dt \cdot A) x(t) + \dt (I - \alpha \dt \cdot A)^{-1} B \cdot u(t)
```
alpha = 0 Euler, alpha = 1 backward Euler, alpha = 1/2 bilinear, "which preserves the stability of the system".
"the bilinear discretization will be our main method"; A-bar, B-bar are the matrices in (3) at alpha = 1/2.

### Eq. (4), (5) recurrence (Sec. 2, p4)
```tex
x_{t} &= \overline{A} x_{t-1} + \overline{B} u_t \label{eq:1-discrete} \\
y_t &= C x_t + D u_t \label{eq:2-discrete}
```

### Eq. (6), (7) convolution and Krylov kernel (Sec. 3.1, p5)
```tex
y_k = C \left( \overline{A} \right)^k \overline{B} u_0 + C \left( \overline{A} \right)^{k-1} \overline{B} u_1 + \dots + C \overline{A} \overline{B} u_{k-1} + \overline{B} u_k
+ D u_k
```
```tex
\mathcal{K}_L(A, B, C) = \left(C A^i B\right)_{i \in [L]} \in \mathbb{R}^L = (CB, CAB, \dots, CA^{L-1}B)
```
"the entire output ... can be computed at once by a convolution, which can be efficiently implemented with three
FFTs." NOTE: Eq. (6) as printed (TeX and PDF) has `+ \overline{B} u_k` with no C; by Eq. (7) the term is
C B-bar u_k. Our reading: typo; the page should write C B-bar u_k.
Bottlenecks named (Sec. 3.1): recurrent view = MVM by A-bar; convolutional view = computing K_L.

### Delta t as timescale (Sec. 2, p4)
"the length of dependencies they can capture is roughly proportional to \( \frac{1}{\dt} \)"; viewed as a CNN,
Delta t "can be viewed as controlling the width of the convolution kernel".

### Lemma 3.1, gates are discretization (Sec. 3.2, p6)
"A (1-D) gated recurrence \( x_{t} = (1-\sigma(z)) x_{t-1} + \sigma(z) u_t \) ... can be viewed as the
GBT(\( \alpha=1 \)) (i.e., backwards-Euler) discretization of a 1-D linear ODE \( \dot{x}(t) = -x(t) + u(t) \)."
Proof: Delta t = exp(z), A = -1, B = 1, alpha = 1 in (3).

### Theorem 1 (informal, Sec. 4.1, p7)
"For an arbitrary measure \( \omega \), the optimal memorization operator \( \operatorname*{hippo}(\omega) \)
has the form \( \dot{x}(t) = A x(t) + B u(t) \) \eqref{eq:1} for a low recurrence-width (LRW) state matrix
\( A \)." Corollary: classical OPs give 3-quasiseparable A.

### A initialization (App. B.2, p17), verbatim
"The LSSL initializes the \( A \) parameter in \eqref{eq:1} to the HiPPO-LegS operator":
```tex
A_{nk}
&=
\begin{cases}
  (2n+1)^{1/2}(2k+1)^{1/2} & \mbox{if } n > k \\
  n+1 & \mbox{if } n = k \\
  0 & \mbox{if } n < k
\end{cases}
```
SIGN MISMATCH (do not reconcile on the page without saying so): this is HiPPO's positive A, but here it is
plugged into x' = Ax + Bu with no minus sign, while HiPPO Eq. (3) has -A. Our reading: the stable choice is
-A. Also: LSSL uses this A in an LTI system with step Delta t, without HiPPO-LegS's 1/t factor.
Delta t init (App. B.3, p18): log-uniform in [dt_min, dt_max], "generally chosen to be a factor of 100 apart".

## 7. Config / measurement (Table 1 only)
Table 1 (p8): pixel-by-pixel classification accuracy, sMNIST / pMNIST / sCIFAR; baselines are reported numbers.
Sec. 5.1: "all results were achieved with at least 5x fewer parameters than the previous SoTA". Per-dataset
hyperparameters are in App. E (not extracted).

## 8. Results worth remembering
- Table 1 (p8), sMNIST / pMNIST / sCIFAR: LSSL 99.53 / 98.76 / 84.65; LSSL-fixed 99.50 / 98.60 / 81.97;
  HiPPO-RNN 98.9 / 98.3 / 61.1; Transformer 98.9 / 97.9 / 62.2; CKConv 99.32 / 98.54 / 63.74; TrellisNet
  99.20 / 98.13 / 73.42; URGRU 99.27 / 96.51 / 74.4; LMUFFT pMNIST 98.49.
- Sec. 5.1: "LSSL sets SoTA on sCIFAR by more than 10 points."
- Abstract: speech classification on length-16000 sequences, "outperforms prior approaches by 24 accuracy
  points". (Sec. 1 says "by over 20 accuracy points in 1/5 the training time".)
- Sec. 5.4 (p10): random A "perform very poorly (e.g., 62% on pMNIST)"; sCIFAR LSSL-f with poorly-specified
  Delta t "gets only 49.3% accuracy".

## 9. Limitations (Sec. 6, p10) and later fixes
- Recurrent inference speed was not tested.
- Theorem 2's fast Krylov algorithm (exact arithmetic) "is not numerically stable and thus not usable on
  hardware" (per a follow-up).
- LSSL and LSSL-f need O(NL) space instead of O(L); multi-GPU for the largest runs.
- "These fundamental issues ... were revisited and resolved in follow-up work ..., where a new state space model
  (the Structured State Space) provided a new parameterization and algorithms" (S4).

## 10. Outline feedback
1. The three views (continuous / recurrent / convolutional) come from LSSL Fig. 1 (p2) and Eq. (6)-(7) (p5).
   Ch 4's viz-views should cite LSSL for the views and S4 for making K cheap.
2. Ch 2: LSSL Sec. 2 is a clean source for "Delta t is a timescale; memory length ~ 1/Delta t" and for GBT
   alpha = 1/2 bilinear (S4-era) vs ZOH (Mamba).
3. Lemma 3.1 (gate = backward Euler with Delta t = exp(z)) plus HiPPO Sec. 2.5 bridges ch 1 (gates) and ch 6
   (Mamba Theorem 1: selection as gating).
4. The family tree should show HiPPO (2020-08-17) -> LSSL (2021-10-26, NeurIPS 2021) -> S4.
5. Eq. (6) typo and the B.2 sign mismatch: write C B-bar u_k and -A on the page.
