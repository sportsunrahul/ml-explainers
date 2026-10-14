# HiPPO: Recurrent Memory with Optimal Polynomial Projections

Read: arXiv 2008.07669v2, full (LaTeX `sources/hippo/src/`, PDF 47 pages). Lens: method/theory, plus a
measurement section for the cited experiments. Serves ch 2 (ch-continuous) and ch 3 (ch-hippo).

## 1. Identity
- Title: HiPPO: Recurrent Memory with Optimal Polynomial Projections.
- arXiv 2008.07669, version read v2. v1 submitted 2020-08-17; v2 2020-10-23 (abs.html submission history).
- Authors: Albert Gu and Tri Dao (equal contribution, coin flip), Stefano Ermon, Atri Rudra, Christopher Re
  (last). Stanford; Rudra at University at Buffalo.
- Venue: not stated in abs.html or the LaTeX. Leave it out.
- Repo: https://github.com/HazyResearch/hippo-code (paper Sec. 1). The README says it is deprecated and points
  to https://github.com/HazyResearch/state-spaces. The repo code is not in `sources/`.

## 2. Core problem
Keep a bounded-size summary of the whole history of a signal, updated online as each new value arrives. The
paper poses this as online function approximation: at every time t, store the coefficients of the best
polynomial fit (degree < N) of f on the past, where a measure mu^(t) says how much each past time counts.

## 3. What prior methods did and why that hurt (per the paper, Sec. 1, 2.3)
- RNNs have a limited memory horizon (vanishing gradients). Gates (LSTM, GRU) and higher-order bases (FRU,
  LMU) are heuristics with no unified explanation.
- Existing methods need a prior on sequence length or timescale and fail outside that range (Sec. 1). The
  LMU = HiPPO-LegT needs a window length theta; LegT and LagT also need a step size Delta t.
- Most lack guarantees such as gradient bounds.
- LegT (sliding window) needs f(t - theta), which is no longer available, so it adds a second approximation
  (Sec. 2.3, p5). Its window forgets the start of the signal; for small t it puts weight on x < 0, where f
  is undefined (Fig. 5, p26).
- LagT (exponential decay) does not forget, but also puts weight on x < 0 (Fig. 5).

## 4. The one key idea
Differentiate the optimal projection coefficients c(t) with respect to t. For orthogonal-polynomial bases this
gives a closed-form linear ODE dc/dt = A(t) c + B(t) f(t), so the optimal compression of the past is updated
by a linear recurrence. Choosing a measure that stretches with t (LegS, uniform on [0, t]) gives a memory with
no timescale hyperparameter.

## 5. Mechanism, step by step
1. Choose a time-varying measure mu^(t) supported on (-inf, t]: the weight on each past time (Sec. 2.1).
2. Choose the approximation space: polynomials of degree < N. Use the orthonormal polynomials g_n for mu^(t).
3. The optimal coefficients are c_n(t) = <f_{<=t}, g_n>_{mu^(t)} (Sec. 2.2; App. C.2 Eq. (18)).
4. Differentiate under the integral. The derivative of g_n is a lower-degree polynomial and the derivative of
   the measure folds back into the measure, so dc/dt is linear in c(t) and f(t) (App. C.3 Eq. (20)).
5. Discretize the ODE (Euler, backward Euler, bilinear, GBT, or ZOH; App. B.3) to get
   c_{k+1} = A_k c_k + B_k f_k. Experiments use bilinear for LegS (App. B.3, p21).
6. At any time, rebuild the history f on [0, t] from c(t) with the reconstruction formula (App. D.3, p32).
7. In a model (Sec. 4, Fig. 2; App. F.1 Eq. (31)): a gated RNN emits a scalar feature f_t = L_f(h_t); HiPPO
   memorizes the history of f_t in c_t (A_t, B_t fixed, not trained); c_{t-1} is fed back into the next RNN
   step. N is tied to the hidden size d.
8. Special case (Sec. 2.5): N = 1 LagT with Euler is c(t + Delta t) = (1 - Delta t) c(t) + Delta t f(t). With an
   input-dependent Delta t this is a gated RNN (GRU as a special case).

## 6. Equations

Notation translation (say once on the page). HiPPO does NOT write the input as u and the state as x:
- input signal f(t), samples f_k -> page x (input).
- coefficients c(t), c_k in R^N -> page h (state, size N).
- x inside HiPPO integrals is a dummy variable for a PAST TIME (x <= t), not the input.
- Delta t -> page Delta. N is the same.
- Sign: HiPPO's A has positive entries and enters as -A (Theorems 1, 2). The page's A (Mamba convention,
  dh/dt = A h + B x) is the negative of HiPPO's A. HiPPO-LegS A has eigenvalues 1, 2, ..., N (App. E.3 p37:
  A is lower triangular with diagonal n + 1), so -A has eigenvalues -1, ..., -N.
- LegS is time-varying: page form is dh/dt = (-A/t) h + (B/t) x. LegT and LagT are LTI.

Custom macros (hippo.tex): `\ddt` = `\frac{d}{d t}`; `\ppt` = `\frac{\partial}{\partial t}`;
`\d`, `\dd` = upright d for integrals; `\defeq` = `\coloneqq`; `\norm{x}` = `\left\|x\right\|`;
`\proj`, `\coef`, `\hippo`, `\diag` = operator names; `\R` = `\mathbb{R}`.

### E1. Inner product and the online approximation problem (Sec. 2.1, p2-3; Definition 1, p3)
```tex
\langle f, g \rangle_\mu = \int_0^\infty f(x) g(x) \d \mu(x)
```
Paper: "we seek some $g^{(t)}\in\mathcal{G}$ that minimizes $\|f_{\le t} - g^{(t)}\|_{L_2(\mu^{(t)})}$."
Definition 1: proj_t maps f_{<=t} to the best polynomial g^(t) in G; coef_t maps g^(t) to its coefficients
c(t) in R^N in the orthogonal basis for mu^(t); `hippo = coef o proj`.
Gloss: mu^(t) is the weight on each past time x <= t; G is polynomials of degree < N; c(t) is the state h.

### E2. Measures (Sec. 2.3 p4, unnumbered; LegS Sec. 3 p5)
```tex
\textbf{LegT}: \mu^{(t)}(x) = \frac{1}{\theta} \mathbb{I}_{[t-\theta, t]}(x)
\qquad
\textbf{LagT}: \mu^{(t)}(x) = e^{-(t-x)} \mathbb{I}_{(-\infty, t]}(x)
```
```tex
\mu^{(t)} = \frac{1}{t} \mathbb{I}_{[0, t]}
```
Plain words:
- LegT (translated Legendre): equal weight on the last theta time units only. A sliding window; theta is a
  hyperparameter. This is the LMU.
- LagT (translated Laguerre): weight e^{-(t-x)}, so recent history counts most and old history fades
  exponentially. It never drops anything but still needs a step size.
- LegS (scaled Legendre): equal weight on ALL history [0, t]. The window grows with t, so nothing is
  forgotten and there is no timescale to set.

### E3. Theorem 1, LegT and LagT (p4-5), Eq. (1) and Eq. (2)
```tex
\ddt c(t) = - A c(t) + B f(t)
```
```tex
A_{nk} = \frac{1}{\theta}
  \begin{cases}
    (-1)^{n-k} (2n+1) & \mbox{if } n \ge k \\
    2n+1 & \mbox{if } n \le k
  \end{cases},
  \quad
  B_n = \frac{1}{\theta} (2n+1) (-1)^n
```
```tex
A_{nk} =
  \begin{cases}
    1 & \mbox{if } n \ge k \\
    0 & \mbox{if } n < k \\
  \end{cases},
  \quad
  B_n = 1
```
Paper: "For LegT and LagT, the $\hippo$ operators ... are given by linear time-invariant (LTI) ODEs". Eq. (1)
"proves the LMU update". Gloss: n, k index degrees 0..N-1; theta is the window length.

### E4. Theorem 2, HiPPO-LegS (p5-6), Eq. (3) continuous and Eq. (4) discrete
```tex
\ddt c(t) &= -\frac{1}{t} A c(t) + \frac{1}{t} B f(t)
\label{eq:scaled-legendre-dynamics}
\\
c_{k+1} &= \left( 1-\frac{A}{k} \right) c_k + \frac{1}{k} B f_k
\label{eq:legs-discrete}
```
```tex
A_{nk}
=
\begin{cases}
  (2n+1)^{1/2}(2k+1)^{1/2} & \mbox{if } n > k \\
  n+1 & \mbox{if } n = k \\
  0 & \mbox{if } n < k
\end{cases},
\qquad
B_n = (2n+1)^{\frac{1}{2}}
```
Paper: Eq. (4) "uses the Euler method for illustration; HiPPO-LegS is invariant to other discretizations"
(footnote 4, p6). Gloss: c(t) = state h; f = input x; n, k = 0..N-1; A lower triangular, diagonal 1..N;
the 1/t factor makes this time-varying. Concrete N = 4 (computed from the formula):
A = [[1,0,0,0],[sqrt3,2,0,0],[sqrt5,sqrt15,3,0],[sqrt7,sqrt21,sqrt35,4]], B = [1, sqrt3, sqrt5, sqrt7].
Our reading (checked numerically): column 0 of A equals B, so for a constant input f0 the state
c = f0 * e_0 is a fixed point of Eq. (3) (the constant is fit exactly by P_0).

Alternative form, App. D.3 Eq. (30), p31:
```tex
\ddt c(t) = -t^{-1} D
\left[
  M D^{-1}
  c(t)
  +
  \mathbf{1}
  f(t)
\right]
```
with D = diag[(2n+1)^{1/2}], M_{nk} = 2k+1 if k < n, k+1 if k = n, 0 if k > n. NOTE: as printed, the
`+ \mathbf{1} f(t)` sign conflicts with Eq. (29) (`+\frac{1}{t} B f(t)` means `-t^{-1} D [.. - 1 f]`).
Our reading: a sign typo in Eq. (30). Use Eq. (3)/(29).

### E5. LegS measure and basis, App. D.3 Eq. (27), (28), p30
```tex
\omega(t, x) &= \frac{1}{t} \mathbb{I}_{[0, t]}
\\
g_n(t, x) &= p_n(t, x) = (2n+1)^{\frac{1}{2}} P_n\left( \frac{2x}{t} - 1 \right)
```
"We use no tilting ... and $\lambda_n=1$ so that the functions $g_n(t, x)$ are an orthonormal basis."
Gloss: P_n = standard Legendre polynomial on [-1, 1]; 2x/t - 1 maps past time x in [0, t] to [-1, 1].
Normalization (App. B.1.1, Eq. (5), p18): `\frac{2n+1}{2} \int_{-1}^1 P_n(x)P_m(x) \dd x = \delta_{nm}`, and
P_n(1) = 1, P_n(-1) = (-1)^n. Hence (2n+1)^{1/2} P_n(2x/t - 1) is orthonormal for the probability measure
1_{[0,t]}/t.

### E6. Coefficients (what c_n(t) is), App. C.2 Eq. (18) p24, LegS case written out in App. E.1 p36
```tex
c_n(t) = \langle f_{\leq t}, g_n^{(t)} \rangle_{\nu^{(t)}}
```
LegS case (our reading of Eq. (18) with chi = 1, zeta = 1, lambda_n = 1, matching the integrand in E.1):
c_n(t) = (1/t) * integral_0^t f(x) (2n+1)^{1/2} P_n(2x/t - 1) dx.

### E7. Reconstruction, LegS (App. D.3 "Reconstruction", top of p32; general form Eq. (19), p24)
```tex
f(x) \approx g^{(t)}(x)
&= \sum_n c_n(t) g_n(t, x).
\\
&= \sum_n c_n(t) (2n+1)^{\frac{1}{2}} P_n\left( \frac{2x}{t} - 1 \right).
```
Gloss: sum over n = 0..N-1; x in [0, t] is the past time to rebuild; the (2n+1)^{1/2} factor is the full
normalization (basis is orthonormal, lambda_n = 1, so no extra 1/||g_n||^2). General Eq. (19):
`f_{\leq t} \approx g^{(t)} \defeq \sum_{n=0}^{N-1} \langle f_{\leq t}, g_n^{(t)} \rangle_{\nu^{(t)}} \frac{g_n^{(t)}}{\| g_n^{(t)} \|_{\nu^{(t)}}^2}`.

### E8. Discretization, App. B.3 (p20-22)
General linear ODE in B.3 is written `\frac{d}{dt} c(t) = A c(t) + Bf(t)` (A carries its own sign here).
Euler: `c(t + \Delta t) = (I + \Delta t A) c(t)+ \Delta t B f(t).`
GBT, Eq. (13), p21 (alpha = 0 forward Euler, 1/2 bilinear, 1 backward Euler):
```tex
c(t + \Delta t) = (I - \Delta t \alpha A)^{-1}(I + \Delta t (1 - \alpha) A) c(t) + \Delta t (I - \Delta t \alpha A)^{-1} B f(t).
```
ZOH (unnumbered, p21): `c(t + \Delta t) = e^{\Delta t A} c(t) + A^{-1} (e^{\Delta t A} - I) B f(t)` if A invertible.
LegS under GBT, "HiPPO-LegS invariance to discretization step size" (p21), verbatim:
```tex
c^{(k+1)} = \left(I - \frac{1}{k+1}\alpha A\right)^{-1}\left(I + \frac{1}{k} (1 - \alpha) A\right) c^{(k)}
+ \frac{1}{k} \left(I - \frac{1}{k+1} \alpha A\right)^{-1} B f_k,
```
with c^(k) := c(k Delta t), f_k := f(k Delta t); "which does not depend on $\Delta t$."
Choice: "We will use bilinear as the discretization method for the LegS updates for the experiments" (p21;
Fig. 4). Footnote 2 (p5) says experiments use "the more numerically stable Bilinear and ZOH methods".

TRAP 1 (sign). This B.3 paragraph writes the LegS ODE as `\frac{d}{dt} c(t) = \frac{1}{t} A c(t) + ...`, i.e.
WITHOUT the minus of Theorem 2. Its recurrence therefore uses A = -A_Thm2. With Theorem 2's positive A,
the bilinear (alpha = 1/2) LegS step is (our reading, verified numerically):
  c_{k+1} = (I + A/(2(k+1)))^{-1} [ (I - A/(2k)) c_k + B f_k / k ].
TRAP 2 (start index). Eq. (4) and the B.3 recurrence divide by k; k = 0 is undefined and the paper does not
say how to start. Our check used: first update at k = 1 with c_1 = f_0 * e_0 (exact projection of a constant).
An alternative (index shifted, t = k + 1, c_0 = 0) gave final coefficients within 0.0067 of exact
quadrature (vs 0.0047), reconstruction rmse 0.0194 vs 0.0133 at N = 64. Both are `our reading`.

Our numerical check (not a paper number; numpy, script not saved). Signal = the paper's
App. F.8 function f(x) = 1/4 sin x + 1/2 sin(x/3) + sin(x/7), 1000 samples on [0, 100]. Reconstruct at
t = 100 and report rmse vs the true f:
| N | forward Euler | backward Euler | bilinear | exact projection (quadrature) |
|---|---|---|---|---|
| 4 | 0.7503 | 0.7503 | 0.7503 | 0.7503 |
| 16 | 0.2356 | 0.2356 | 0.2336 | 0.2334 |
| 64 | 0.2962 | 0.1074 | 0.0133 | 0.0000 |
Forward Euler fails at N = 64 (early steps have A/k with eigenvalues up to 64/k, so 1 - A/k is unstable for
small k). This reproduces the qualitative claim of Fig. 4 (p22). Step-size invariance: the same function at
1000 vs 4000 samples (N = 16) gave final c within 0.0026.

### E9. LegS properties (Sec. 3, p6; proofs App. E)
- Prop. 3 (timescale equivariance): `For any scalar $\alpha > 0$, if $h(t) = f(\alpha t)$, then
  $\hippo(h)(t) = \hippo(f)(\alpha t)$.` Plain words: play the signal faster or slower and you get the same
  coefficients at the matching time. Proof App. E.1 p36 (change of variables). Consequence stated in text:
  LegS has "no timescale hyperparameters"; Eq. (4) is invariant to the step size.
- Prop. 4 (efficiency): under any GBT discretization each LegS step costs O(N). Reason (E.2 p36-37):
  A = D_1 (L + D_0) D_2 with L the all-ones lower-triangular matrix (cumsum) and D_i diagonal.
- Prop. 5 (gradient flow): `\left\| \frac{\partial c(t_1)}{\partial f(t_0)} \right\| = \Theta\left( 1/t_1 \right)`.
  Plain words: the influence of an old input decays like 1/t (polynomially), not exponentially.
- Prop. 6 (approximation error): L-Lipschitz f gives `O(tL/\sqrt{N})`; order-k bounded derivatives give
  `O(t^k N^{-k+1/2})`.

### E10. Low-order case = gating (Sec. 2.5, p5)
```tex
c(t + \Delta t) = c(t) + \Delta t(-Ac(t) + Bf(t)) = (1 - \Delta t)c(t) + \Delta t f(t)
```
N = 1 LagT (A = B = 1), Euler. With Delta t chosen from the input and state, this "becomes exactly a gated RNN".

### E11. Model cell (App. F.1, Eq. (31), Fig. 6, p41)
```tex
h_t \in \R^d &= \tau(h_{t-1}, [c_{t-1}, x_t]) \\
f_t \in \R^1 &= \mathcal{L}_f(h_{t}) \\
c_t \in \R^N &= \hippo_t(f) \\
&= A_t c_{t-1} + B_t f_t
```
with `\tau(h, x) = (1-g(h, x)) \circ h + g(h, x) \circ \tanh(\mathcal{L}_\tau(h, x))`,
`g(h, x) = \sigma(\mathcal{L}_g(h, x))`. Here x_t is the RNN input and h_t the RNN hidden state (the RNN's
own notation, not the page's). Trainable: L_tau, L_g, L_f and the head; A_t, B_t are fixed by the measure.

## 7. Measurement setup (analysis lens, for the cited experiments)
| Item | pMNIST (Sec. 4.1, App. F.2) | Character Trajectories (Sec. 4.2, App. F.4) | Function approx + speed (Sec. 4.3, App. F.5) |
|---|---|---|---|
| Data | MNIST pixels, fixed permutation, length 784; 60,000 train / 10,000 test; 10% of train as val | 2858 pen trajectories, 3-dim input (x, y, force), 20 classes, length up to 182, 200Hz; split 70/15/15 | band-limited white noise, length 10^6, Delta t = 10^-4, band limit 1Hz |
| Model | MGU (GRU without reset gate) + hippo layer; N = d | same cell family | memory update only (LegS, LMU); LSTM + trained linear decoder (100 sequences, L2 loss) |
| Size | hidden 512 for all methods | 256 hidden (LSTM, GRU, LMU); GRU-D, ODE-RNN, NCDE use Kidger et al. code and hyperparameters | N = 256 for all |
| Training | cross-entropy; Adam lr 0.001 frozen; 50 epochs; batch 100; 3 seeds, max average val acc reported | cross-entropy; 100 epochs | no training for HiPPO/LMU |
| Metric | val acc (Table 1 left / Table 4); test acc vs reported literature (Table 1 right / Table 5) | test acc under timescale shift (Table 2) | MSE after 10^6 steps; elements/sec single-thread Intel Xeon E5-2690 v4 2.60GHz, C++ with PyTorch binding |
| Hardware | PyTorch 1.5, Nvidia P100 (App. F.1) | same | CPU as stated |
Timescale shift protocol (App. F.4): (1) downsample train or eval sequences by 2 (100Hz -> 200Hz means train
at 100Hz, test at 200Hz); (2) halve train or eval timestamps (missing values, irregular sampling). Models
cannot detect the shift (variable lengths). What would falsify the robustness claim: LegS accuracy under shift
dropping to the baselines' level. Without shift, all methods reach >= 95% (Sec. 4.2).
Code-only (README): pMNIST command uses `runner.ntrials=5`, conflicting with the paper's "3 seeds"; the other
flags (hidden 512, 50 epochs, batch 100, lr 0.001) match. CT uses hidden 256, 100 epochs.

## 8. Results worth remembering
- pMNIST test acc, HiPPO-LegS 98.3 (Table 1 right, p8; Table 5 p43; abstract). Previous recurrent SoTA in the
  same table: LMU 97.15 (Table 1 right) and URLSTM + Zoneout 97.58 (Table 5 only). Non-recurrent: Transformer
  97.9, TCN 97.2, TrellisNet 98.13. Text: "beating the previous RNN SoTA by over 1 point" (Sec. 1).
- pMNIST val acc (Table 1 left, p8): LegS 98.34, LagT 98.15, LegT theta=200 98.0, LegT theta=20 91.75, Rand
  69.93, LMU 97.08, ExpRNN 94.67, GRU 93.04, MGU 89.37, RNN 52.98. Table 4 (p42) adds LegT theta=2000 97.90,
  LagT Delta t = 0.1 96.44, LagT Delta t = 0.01 90.71, LSTM 92.54 (and prints LegT theta=200 as 98.00).
  Point for the page: the derived A matters (Rand 69.93), and LegT is fragile in theta (91.75 at theta=20).
- Timescale robustness, Table 2 (p9), test acc. LegS: 88.8 (100Hz -> 200Hz), 90.1 (200Hz -> 100Hz), 94.5
  (missing values upsample), 94.9 (downsample). Best baseline per row: NCDE 44.7, GRU 64.6, NCDE 63.9, NCDE
  69.7. Abstract: "outperforms RNN and neural ODE baselines by 25-40% accuracy".
- Function approximation after 1 million steps, 256 units (Table 3, p9): MSE LSTM 0.25, LMU 0.05, LegS 0.02;
  speed (elements/sec) 35,000 / 41,000 / 470,000. Text: "10x faster than the LSTM and LMU".
- Copying, L = 200 (Fig. 7, p44): LegS solves fastest; LegT only with theta = 200; Rand "does much worse";
  RNN and LSTM "nearly stuck at baseline" (random-guess loss log(8) ~ 2.08). No numeric table.
- Mackey-Glass (Fig. 8 table, p46), test NRMSE: LegS 0.04752, Hybrid LSTM+LMU 0.06862, LMU 0.09722, LSTM
  0.10123, identity baseline 1.62274. Text: "reducing normalized MSE by 30%" (Sec. 4.4) / "over 30%" (F.7).
- IMDB (Table 6, p45), test acc: LegS 87.8 +- 0.2, LagT 88.0 +- 0.2, LSTM 87.3 +- 0.4, expRNN 84.3, Rand 82.9.
- Function approx ablation (App. F.8, Fig. 9, p47): LegS lowest error; LegT error explodes if theta is too
  small "by 0.5% relative to the total sequence length".

## 9. Limitations (stated) and later fixes
- Conclusion: LegS "takes a first step toward timescale robustness"; future work is to use the memory in
  models other than RNNs and on large-scale tasks.
- The memory is 1-D input to N-D output inside an RNN, with A, B fixed. LSSL (App. B.3, p18) says its added
  output projection C, D makes the layer 1-D to 1-D, unlike HiPPO and the LMU, so H copies can run in
  parallel with different Delta t; LSSL also learns A and Delta t.
- LegT/LagT need timescale hyperparameters; LMU fast only with forward Euler (Sec. 4.3 footnote).
- Forward/backward Euler are inaccurate (Fig. 4); bilinear needed.
- The generalized LagT family is unstable for larger beta (App. F.8, Fig. 10).

## 10. Outline feedback
1. Notation line (outline header, ch 3; also the extract brief) is wrong for HiPPO. HiPPO writes input f(t),
   state c(t), and uses x as the past-time integration variable. LSSL (and S4) use u and x. The ch 3
   translation must be f -> x, c -> h, with a warning that HiPPO's x means time.
2. HiPPO-LegS is time-varying: dc/dt = -(1/t) A c + (1/t) B f. LegT and LagT are the LTI ones. LSSL uses the
   LegS A as a fixed LTI matrix with a step Delta and drops the 1/t (LSSL App. B.2). Ch 2 should not cite
   LegS as its constant-A example; ch 3 should say "S4/LSSL keep LegS's A but drop the 1/t".
3. LegS has no Delta knob: its discrete recurrence does not depend on Delta t (B.3, Prop. 3). This contrasts
   with ch 2's "Delta is the knob". One sentence in ch 3 should note that LSSL/S4 bring Delta back when they
   drop 1/t.
4. Discretization: HiPPO experiments use bilinear (GBT alpha = 1/2); Mamba uses ZOH. Ch 2 builds Mamba Eq. 4
   (ZOH) while ch 3 computes with bilinear; the page needs one sentence on that. viz-hippo must use bilinear:
   forward Euler breaks at N = 64 (Sec. 6, E8 table).
5. pMNIST 98.3 is a HiPPO-RNN (gated nonlinear RNN + fixed linear memory, Fig. 2), not a linear SSM. LSSL
   Table 1 (p8) gives the first stacked-linear-layer pMNIST numbers (98.76; LSSL-fixed 98.60).
6. Timescale robustness claim lives in Prop. 3 (p6) with evidence in Table 2 (p9). Suggest a claim card with
   the four LegS numbers vs best baseline.
7. viz-hippo: a good default signal is the paper's own F.8 function (above); Fig. 3 uses band-limited white
   noise. Reconstruction formula E7 plus orthonormal scaling is all the visual needs. Mention the A sign and
   start index traps in the scene code comments.
8. Sec. 2.5 (N = 1 LagT + adaptive Delta t = gated RNN) is a free bridge for ch 1 (gates) and ch 6 (Mamba
   Theorem 1), together with LSSL Lemma 3.1.
