# S4D -- On the Parameterization and Initialization of Diagonal State Space Models (branch)

Branch extraction for ch 4 (the "diagonal A is enough" paragraph). Notation as in S4: input u
(page x), state x (page h), output y. TeX verbatim; custom macros: `\dt` = `\Delta`, `\bm{}` = bold.
Equation numbers are printed PDF numbers.

## 1. Identity
- Title: On the Parameterization and Initialization of Diagonal State Space Models.
- arXiv 2206.11893, read **v2**. v1 2022-06-23, v2 2022-08-05 (abs.html submission history).
- Authors: Albert Gu, Ankit Gupta, Karan Goel, Christopher Re (Stanford, IBM Research).
- Venue: not stated in abs.html (do not state one).
- Repo: same as S4 (`sources/s4/README.md` lists S4D); no S4D README in sources.

## 2. Core problem
S4's DPLR parameterization and Cauchy/Woodbury algorithm are hard to implement. DSS found that a
diagonal A (HiPPO with the low-rank term removed) nearly matches S4, but without an explanation and
with a more complex softmax kernel (Abstract, Sec 1).

## 3. What prior methods did and why that hurt
- S4: DPLR A needs "a sophisticated algorithm with many linear algebraic techniques" (Sec 1).
- DSS: complex-softmax kernel (row-normalized Vandermonde) ties the model to one length L and uses
  O(N) more memory than S4(D); B not trained; Re(A) unconstrained; ZOH required (Sec 3.4).
- Random diagonal A is fully expressive (Prop 2) but performs poorly (S4 Sec 4.4; S4D-Rand here).

## 4. The one key idea
Use a diagonal A with S4's other choices; the kernel is then a Vandermonde product. The init is what
matters: dropping P P^T from HiPPO-LegS (S4D-LegS) gives the same basis as S4 as N -> infinity
(Theorem 3); S4D-Inv and S4D-Lin are closed-form approximations.

## 5. Mechanism
1. A = diag(A_n), complex; Re(A) = -exp(A_Re) (left half-plane), Im(A) free (Sec 3.3).
2. B init 1 and trainable (DSS freezes it); C random std 1 (Sec 3.3).
3. Discretize: bilinear (default, as S4) or ZOH; "little empirical difference" (Sec 3.1, Table 2).
4. Kernel by Eq (7): K-bar = (B-bar^T o C) V_L(A-bar); naive O(NL) time without materializing V
   gives O(N+L) space; theory O~(N+L), same as DPLR (Prop 1).
5. Conjugate symmetry: N/2 complex parameters, output = 2 Re(sum) (Sec 3.3, Listing 1).
6. Delta: log-uniform in (dt_min, dt_max) = (0.001, 0.1) (Listing 1, Table 12); same as S4 and DSS (Sec 3.4).
7. Broadcast H independent copies; rest of the network as in S4 (Sec 3.4).

## 6. Equations

**Eq (4)** (`eq:hippo-legs`), HiPPO-LegS with B and P:
```
\bm{A}_{nk}
&=
-
\begin{cases}
  (2n+1)^{\frac{1}{2}}(2k+1)^{\frac{1}{2}} & n > k \\
  n+1 & n = k \\
  0 & n < k
\end{cases}
\\
\bm{B}_n &= (2n+1)^{\frac{1}{2}}
\quad
\bm{P}_n = (n + 1 / 2)^{\frac{1}{2}}
```
**Eq (5)** (`eq:hippo-legsd`), normal / DPLR form:
```
\bm{A}^{(N)}_{nk}
&=
-
\begin{cases}
  (n+\frac{1}{2})^{1/2}(k+\frac{1}{2})^{1/2} & n > k \\
  \frac{1}{2} & n = k \\
  (n+\frac{1}{2})^{1/2}(k+\frac{1}{2})^{1/2} & n < k
\end{cases}
\\
\bm{A} &= \bm{A}^{(N)} - \bm{P}\bm{P}^\top,
\qquad
\bm{A}^{(D)} := \operatorname*{eig}(\bm{A}^{(N)})
```
Note: as typeset, the n < k case has no sign flip; with the leading minus, A^(N) would be symmetric,
not skew. Our check: A + P P^T from Eq (4) gives -(...) for n > k, -1/2 on the diagonal, and
+(n+1/2)^{1/2}(k+1/2)^{1/2} for n < k (skew part plus -1/2 I), as in S4 App C.1. Quote verbatim; flag only.

**Eq (7)** (`eq:vandermonde`), diagonal kernel:
```
\bm{\overline{K}}_\ell = \sum_{n = 0}^{N-1} \bm{C}_n \bm{\overline{A}}_n^\ell \bm{\overline{B}}_n
\implies \bm{\overline{K}} = (\bm{\overline{B}}^\top \circ \bm{C}) \cdot \mathcal{V}_L(\bm{\overline{A}})
\qquad \qquad
\text{where } \mathcal{V}_L(\bm{\overline{A}})_{n, \ell} = \bm{\overline{A}}_n^\ell
```
Gloss: kernel entry l = sum over the N state dims of C_n B-bar_n times A-bar_n to the power l; each
state dim is an independent 1-D decaying (complex: decaying-oscillating) mode.

**Theorem 3** (`thm:legsd`): "Let A = A^(N) - P P^T and B be the HiPPO-LegS matrices, and K_{A,B}(t)
be its basis. As the state size N -> infinity, the SSM basis K_{A^(N), B/2}(t) limits to K_{A,B}(t)".

**Proposition 4**: `\Re(\bm{A}) = -\frac{1}{2}\bm{1}` (for A^(D)).
**Conjecture 5**: Im(A)_0 -> N^2/pi + c, c approx 0.5236; Im(A)_n = Theta(n^{-1}).

**Eq (8)** (`eq:s4d-inv`): `(\textbf{S4D-Inv}) \quad \bm{A}_n = -\frac{1}{2} + i \frac{N}{\pi} \left( \frac{N}{2n+1}-1 \right)`
**Eq (9)** (`eq:s4d-lin`): `(\textbf{S4D-Lin}) \quad \bm{A}_n = -\frac{1}{2} + i \pi n`
Gloss: real part -1/2 sets one decay envelope e^{-t/2}; imaginary parts set spread-out frequencies
(inverse law for Inv; Fourier frequencies for Lin, a damped Fourier basis). Largest imaginary part:
N^2/pi (Inv), pi N (Lin) (Sec 5.2).

Listing 1 (p.8) init lines, verbatim:
```
log_dt = np.random.rand() * (np.log(dt_max)-np.log(dt_min)) + np.log(dt_min) # Geometrically uniform timescale
A = -0.5 + 1j * np.pi * np.arange(N//2) # S4D-Lin initialization
B = np.ones(N//2) + 0j
C = np.random.randn(N//2) + 1j * np.random.randn(N) # Variance preserving initialization
```
(Defaults `dt_min=1e-3, dt_max=1e-1`. The C line mixes N//2 and N as printed.)

## 7. Setup
- Table 1 (p.6): S4 = DPLR, Cauchy, bilinear, Re(A) via exp, trainable B, HiPPO init; DSS = diagonal,
  softmax, ZOH, no constraint, frozen B, HiPPO-D; S4D = diagonal, Vandermonde, either, exp/ReLU,
  optional, various.
- Ablation protocol: 4 layers, H=128, about 100K params; A, B tied across the H copies (App B).
- LRA (App B, Table 12): bidirectional; GLU mixing (W1 y) o sigma(W2 y), W1, W2 in R^{H x H};
  cosine schedule; WD 0.05; mostly 6 layers, H=256, N=64, BN, pre-norm; ListOps 8 layers H=128;
  Image H=512 LN dropout 0.1; Path-X LR 0.0005, batch 32, 50 epochs, Delta range (0.0001, 0.01).

## 8. Results (key table: Table 5, LRA, p.14)

| Model | ListOps | Text | Retrieval | Image | Pathfinder | Path-X | Avg |
|---|---|---|---|---|---|---|---|
| S4-LegS | 59.60 | 86.82 | 90.90 | 88.65 | 94.20 | 96.35 | 86.09 |
| S4-FouT | 57.88 | 86.34 | 89.66 | 89.07 | 94.46 | x | 77.90 |
| S4D-LegS | 60.47 | 86.18 | 89.46 | 88.19 | 93.06 | 91.95 | 84.89 |
| S4D-Inv | 60.18 | 87.34 | 91.09 | 87.83 | 93.78 | 92.80 | 85.50 |
| S4D-Lin | 60.52 | 86.97 | 90.96 | 87.93 | 93.96 | x | 78.39 |
| S4 (original) | 58.35 | 76.02 | 87.09 | 87.26 | 86.05 | 88.10 | 80.48 |
| Transformer | 36.37 | 64.27 | 57.46 | 42.44 | 71.40 | x | 53.66 |

(Std devs omitted.) S4-LegS row = S4 v3 Table 4 updated row. Abstract: "averaging 85%". Source
inconsistency: Sec 5.3 says "only S4D-Inv ... achieves above random chance" on Path-X, but the table
shows S4D-LegS 91.95. Text also: "more than 30 points better than the original Transformer".

Others: Table 4 sCIFAR test: S4-LegS 91.80, S4D-LegS 89.92, S4D-Inv 90.69, S4D-Lin 90.42.
Table 11 SC (35-class): S4-LegS 96.08 (307K), S4D-Inv 96.18, S4D-Lin 96.25 (306K). Table 6 Path-X
ablation (<200K params): S4D 92.12, DSS 89.72. Kernel "requires just 2 lines of code" (Abstract).
"DPLR versions are often slightly better than the diagonal version" (Sec 5.3).

## 9. Limitations stated
Diagonal SSMs need precise inits: scaling Im(A) by 0.01 or 100, random Im/Re parts, S4D-Rand, and
S4D-Real (same spectrum as LegS, A_n = -(n+1)) all degrade (Sec 5.2). Frozen A, B hurt on SC (Sec 5.3).

## 10. Outline feedback
- ch 4 delta: "a diagonal A is enough" holds with HiPPO-derived inits (LegS-D, Inv, Lin), not
  random ones; DPLR is often slightly better; on Path-X only S4D-LegS and S4D-Inv beat chance.
- Delta log-uniform init (0.001, 0.1) for S4 is sourced here, not in S4 v3.
