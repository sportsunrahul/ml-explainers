# Mamba-2: "Transformers are SSMs" (Dao and Gu, 2024)

## 1. Identity
- Title: Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.
- arXiv 2405.21060, version read: v1 (only version in abs.html), submitted 2024-05-31.
- Authors: Tri Dao (first, Princeton) and Albert Gu (last, CMU). Footnote: "Alphabetical by last name."
- Venue: ICML 2024 (abs.html Comments; README citation).
- Repo: https://github.com/state-spaces/mamba (Sec 1). README: `Mamba2` module, `ssd_minimal.py` = Listing 1.
- Sources read: all of `src/structure/*.tex` (scan.tex skimmed: headings and modes only), `paper.pdf` (52 pages;
  Figs 6, 8, 9, 10 viewed; fig PDFs rendered), `README.md`.

## 2. Core problem
SSMs (Mamba) match Transformers at small/medium scale but their development is disjoint from attention research, so
they cannot reuse Transformer theory or systems work. Mamba's selective scan does not use matrix-multiply units
(tensor cores), so it trains slower than attention at moderate length and its cost grows linearly with state size N (Sec 1, 2.1).

## 3. What prior methods did and why that hurt (per the paper)
- Mamba (S6): selective SSM computed only in recurrent/scan mode with a hardware-aware kernel; "still less efficient
  than hardware-friendly models such as CNNs and Transformers because it does not leverage matrix multiplication units" (Sec 2.1).
- Naive diagonal-SSM computation materializes expanded tensors Z, H of shape (T, P, N) (Remark after Eq. 8, Sec 3.4.1).
- Linear attention (Katharopoulos et al.): states the masked recurrent formula without proof; its mask is all ones (Sec 4.2).
- Mamba block: B, C, Delta are computed from the conv output x_c, so tensor parallelism needs 2 all-reduces per block
  instead of 1 (Sec 8.1).

## 4. The one key idea
A selective SSM is multiplication by a T x T lower-triangular semiseparable matrix M. If each A_t is a scalar times
the identity, M = L o (C B^T) with L the 1-semiseparable decay mask of cumulative products of a_t: this is masked
linear attention with Q = C, K = B, V = X. One matrix, two algorithms: quadratic (attention-like, matmuls) or linear
(recurrence). A block (chunk) decomposition mixes both: diagonal blocks by the quadratic form, off-diagonal blocks are
low rank and pass through the state.

## 5. Mechanism, step by step
1. Selective SSM recurrence h_t = A_t h_{t-1} + B_t x_t, y_t = C_t^T h_t (Eq. 2). Notation: A_t is the discrete transition
   (Remark 1 renames prior work's A-bar to A); time index starts at 0 with h_0 = B_0 x_0 (Sec 3.1).
2. Unroll: y_t = sum_{s<=t} C_t^T A_t...A_{s+1} B_s x_s, i.e. y = M x with M_ji = C_j^T A_j ... A_{i+1} B_i (Eq. 3).
3. That M is N-semiseparable: every block on or below the diagonal has rank <= N (Def 3.1, Lemma 3.3, Prop 3.4,
   Theorem 3.5). Off-diagonal blocks factor as (C-factors) x (A product) x (B-factors) (Eq. 5).
4. N = 1, no B, C: the 1-SS matrix of cumulative products of a_t (Eq. 6), equivalent to the scalar recurrence
   y_t = a_t y_{t-1} + x_t (Eq. 7).
5. Masked kernel attention y = (L o Q K^T) V (Eq. 10) is a single 4-way contraction (Eq. 12). Quadratic order (Eq. 13)
   = standard attention; linear order (Eq. 15) = expand by K, multiply by L, contract by Q. With L = causal ones mask,
   multiplying by L is a cumsum (Sec 4.2): this proves linear attention (Prop 4.1).
6. Structured masked attention (SMA, Def 4.2): any structured L with fast matvec keeps the linear form.
7. Restrict A_t = a_t I (scalar times identity). Then M_ji = a_{j:i} (C_j^T B_i), so M = L o (C B^T), L = 1SS(a)
   (Sec 5.1, unnumbered display; contraction form Eq. 16). This is masked kernel attention with the 1-SS mask.
8. Duality: 1-SS SMA is a diagonal SSM with scalar-identity A (Cor 5.1). Conversely, any SMA with bounded-order
   efficient autoregression must have a semiseparable mask (Theorem 5.2; proof Theorem C.3).
9. SSD algorithm (Sec 6): split M into (T/Q) x (T/Q) blocks of size Q x Q.
   a. Diagonal blocks: per-chunk output assuming zero initial state; quadratic SMA form, all chunks in parallel.
   b. Right factors (B-block-factors): per-chunk final state assuming zero initial state; (N, Q) x (Q, P) matmul -> (N, P).
   c. Center factors (A-block-factors): scalar SSM scan over T/Q chunk states, (T/Q, N, P); Q times shorter than a full scan.
   d. Left factors (C-block-factors): per-chunk output from the true incoming state; contract(QN, NP -> QP).
   Output = diagonal-block part + off-diagonal part (Listing 1: Y_diag + Y_off).
10. With N = P = Q all matmuls are BMM(T/N, N, N, N): O(TN^2) FLOPs, O(TN) memory, mostly matmuls (Theorem 6.1, Sec 6.3).
11. Mamba-2 block (Sec 7, Fig 6): one parallel input projection makes X, z (gate), B, C and the A/Delta input; conv;
    SSD with multi-input (MVA) heads; gate; extra norm; output projection.

## 6. Equations
Custom macros (structure.tex preamble) kept verbatim: `\dt` = `\Delta`; `\mathsf{SSM}`, `\mathsf{SSS}`, `\mathsf{1SS}`,
`\mathsf{contract}`, `\mathsf{cumsum}` are operator names; `\coloneqq` = ":=" (mathtools); `\circ` = Hadamard product;
`\mathtt{T,N,P,S,H,D,Q}` are dimension names. Python-style indexing (Sec 2.5): `v_{j:i}^\times` = `v_j x ... x v_{i+1}`
for j >= i (superscript often omitted, so `A_{j:i}` means a product; `A_{i:i}` = empty product = 1).

Dimension glossary (Table 8, App A): T time/target length, S source length, D = d_model, N = state/feature dim
(d_state), P = head dim (d_head), H = number of heads. Page notation already matches (x, h, y, A, B, C, Delta).
Sign convention: a_t "input-dependent scalars bounded in [0, 1]" (Sec 2.4). Listing 1 passes `A` in log space:
`L = torch.exp(segsum(A))` turns cumulative sums of `A` into cumulative products (our reading of Listing 1: the `A`
argument is log a_t <= 0, i.e. Delta_t times a negative continuous A, following Mamba's parameterization per Remark 1).

**Eq. (1a,b)** `eq:ssm` and **Eq. (2a,b)** `eq:s6`, Sec 2.1 (p3):
```
h_{t} &= A h_{t-1} + B x_t \\
y_t &= C^{\top} h_t
```
```
h_{t} &= A_t h_{t-1} + B_t x_t \\
y_t &= C_t^{\top} h_t
```
A in R^(N,N), B, C in R^(N,1). (1) is LTI; (2) is the selective SSM "where the parameters (A, B, C) can also vary in time".
Broadcast over P channels = head dimension P (Def 2.1-2.2). x is a scalar per channel per step; h is N-dim.

**Eq. (3)** `eq:ssm-matrix`, Sec 3.1 (p7), matrix form of SSMs:
```
y_t &= \sum_{s=0}^t C_t^{\top} A_{t:s}^\times B_s x_s
\\
y &= \mathsf{SSM}(A, B, C)(x) = Mx
\\
M_{ji} &\coloneqq C_j^{\top} A_{j} \cdots A_{i+1} B_{i}
```
Gloss: entry (j, i) is how much input i reaches output j: write with B_i, decay through A_{i+1}..A_j, read with C_j.

**Def 3.1** `def:semiseparable-rank` (p7), verbatim: "A (lower triangular) matrix $M$ is $\mathtt{N}$-semiseparable if
every submatrix contained in the lower triangular portion (i.e. on or below the diagonal) has rank at most $\mathtt{N}$.
We call $\mathtt{N}$ the order or rank of the semiseparable matrix."

**Eq. (4)** `eq:sss`, Def 3.2 (p7), N-SSS representation: `M_{ji} = C_j^{\top} A_{j} \cdots A_{i+1} B_{i}`, with
B_i, C_i in R^N and A_i in R^(N,N). Lemma 3.3: N-SSS => N-semiseparable. Prop 3.4: every N-semiseparable matrix has an
N-SSS representation.

**Eq. (5)** `eq:sss-rank-factor`, proof of Lemma 3.3 (p8): an off-diagonal block M_{j:j', i':i} (j' > j >= i > i')
equals [C_j^T A_{j:j} ; ... ; C_{j'-1}^T A_{j'-1:j}] * A_{j:i-1} * [A_{i-1:i'} B_{i'} ... A_{i-1:i-1} B_{i-1}], an explicit
rank-N factorization (full TeX in ssm.tex lines 80-89). "will be used extensively in deriving our fast algorithms".

**Theorem 3.5** `thm:ssm-sss` (p9), verbatim: "The state space model transformation $y = \mathsf{SSM}(A, B, C)(x)$ with
state size $\mathtt{N}$ is identical to matrix multiplication by an $\mathtt{N}$-SS matrix in sequentially semiseparable
representation $y = \mathsf{SSS}(A, B, C) \cdot x$."
Prop 3.6 (Pernet): an N-SS matrix of size T has O(NT) parameters and O(NT) matvec. Theorem 3.7: any SSM of state size N on
length T can be computed in O(TN) time (not counting preprocessing).

**Eq. (6)** `eq:1ss`, Sec 3.2.2 (p8), the 1-semiseparable mask:
```
M =
\mathsf{1SS}(a_{0:T}) \coloneqq
\begin{bmatrix}
  1 & \\
  a_1 & 1 & \\
  a_2a_1 & a_2 & 1 \\
  \vdots & \vdots & \ddots & \ddots \\
  a_{T-1}\dots a_1 & a_{T-1}\dots a_2 & \dots & a_{T-1} & 1 \\
\end{bmatrix}
```
Entry (j, i) = a_{j:i} = a_j ... a_{i+1}; diagonal = 1. Paper: "equivalence to the minimal form of a scalar recurrence".

**Eq. (7)** `eq:1ss-recurrence` (p8):
```
y_t &= a_{t:0}x_0 + \dots + a_{t:t}x_t \\
    &= a_t \left(a_{t-1:0}x_0 + \dots + a_{t-1:t-1}x_{t-1}\right) + a_{t:t}x_t \\
    &= a_t y_{t-1} + x_t
```
Called the "scalar SSM recurrence" or `cumprodsum`. a_t = 1 gives cumsum (the causal mask, Sec 5.2).

**Eq. (8a-c)** `eq:ssm-diagonal`, Sec 3.4.1 (p10), linear (recurrent) mode of a diagonal SSM:
```
Z &= \mathsf{contract}(\mathtt{SP},\mathtt{SN} \to \mathtt{SPN})(X, B) & \mathtt{(S,P,N)} \\
H &= \mathsf{contract}(\mathtt{TSN},\mathtt{SPN} \to \mathtt{TPN})(L, Z) & \mathtt{(T,P,N)} \\
Y &= \mathsf{contract}(\mathtt{TN},\mathtt{TPN} \to \mathtt{TP})(C, H) & \mathtt{(T,P)}
```
Expand input by B; run N independent scalar recurrences (L = 1SS(A), one per state dim); contract by C.

**Section 2.4 dual form** (unnumbered display, p6):
```
(L \circ QK^{\top}) \cdot V \qquad
L_{ij} = \begin{cases}
  a_i \times \dots \times a_{j+1} & i \ge j \\
  0 & i < j
\end{cases}
```
"where $a_i$ are input-dependent scalars bounded in $[0, 1]$." Two differences from softmax attention: softmax dropped;
attention matrix multiplied elementwise by mask L, "a different data-dependent positional mask".

**Eq. (9)** `eq:kernel-attention`, Sec 4.1.1 (p12): Q (T,N), K (S,N), V (S,P), G = QK^T (T,S), M = f(G) (T,S),
`Y = GV` (T,P). Source writes `Y &= GV`; with M = f(G) defined on the line above this is likely a typo for MV (our reading).

**Eq. (10)** `eq:sha-quad-matrix`, Sec 4.1.4 (p13): `y = (L \circ (QK^\top)) \cdot V`.

**Eq. (12)** `eq:sha` (p13): `Y = \mathsf{contract}(\mathtt{TN},\mathtt{SN},\mathtt{SP},\mathtt{TS} \to \mathtt{TP})(Q, K, V, L)`.
Masked attention as one 4-way contraction.

**Eq. (13a-c)** `eq:sha-quad` (p13), quadratic mode:
```
G &= \mathsf{contract}(\mathtt{TN, SN} \to \mathtt{TS})(Q, K) && \qquad \mathtt{(T,S)} \\
M &= \mathsf{contract}(\mathtt{TS, TS} \to \mathtt{TS})(G, L) && \qquad \mathtt{(T,S)} \\
Y &= \mathsf{contract}(\mathtt{TS, SP} \to \mathtt{TP})(M, V) && \qquad \mathtt{(T,P)}
```

**Eq. (14)** `eq:sha-lin-matrix` (p13): `Y = Q \cdot \mathsf{cumsum}(K^\top V)`. Linear attention's claimed formula.

**Eq. (15a-c)** `eq:sha-lin`, Sec 4.2.1 (p14), linear mode:
```
Z &= \mathsf{contract}(\mathtt{SP},\mathtt{SN} \to \mathtt{SPN})(V, K) & \mathtt{(S,P,N)} \\
H &= \mathsf{contract}(\mathtt{TS},\mathtt{SPN} \to \mathtt{TPN})(L, Z) & \mathtt{(T,P,N)} \\
Y &= \mathsf{contract}(\mathtt{TN},\mathtt{TPN} \to \mathtt{TP})(Q, H) & \mathtt{(T,P)}
```
(15b) is the only step touching both T and S; with L = all-ones causal mask it is a feature-wise cumsum, so linear time.
"the crux of the original linear attention is the fact that matrix-vector multiplication by the causal mask is
equivalent to the cumulative sum operator" (Sec 4.3).

**Prop 4.1** `prop:linear-attention` (p14): "Autoregressive kernel attention, i.e. masked kernel attention with the causal
mask, can be computed in $O(T)$ time by a recurrence taking constant time per step."

**Def 4.2** `def:sma` (p14): SMA = the 4-way contraction (12) with any structured L (sub-quadratic matmul). Quadratic mode
= (13); linear mode = (15) with (15b) done by fast structured multiplication. Examples (Fig 3): linear attention (causal
mask); RetNet decay mask `L_{ij} = \gamma^{i-j} \cdot \mathbb{I}[j \ge i]` (source indicator reads j >= i; for a
lower-triangular mask this should be i >= j, our reading); Toeplitz; Fourier.

**SSD statement, Sec 5.1** (unnumbered displays, p16), scalar-identity SSM:
```
M_{ji} = A_{j:i} \cdot (C_j^{\top}B_i)
```
```
L &\coloneqq \mathsf{1SS}(a) \\
M &= L \circ (C B^{\top}) \\
```
"where $B, C \in \R^{\mathtt{(T,N)}}$", with `A = aI` "for scalar $a$ and identity matrix $I$". Output Y = MX.

**Eq. (16)** `eq:ssm-quad` (p16), the numbered form of the SSD quadratic mode:
```
G     & = \mathsf{contract}(\mathtt{TN, SN \to TS})(C, B) & & \qquad \mathtt{(T,S)} \\
M     & = \mathsf{contract}(\mathtt{TS, TS \to TS})(G, L) & & \qquad \mathtt{(T,S)} \\
Y     & = \mathsf{contract}(\mathtt{TS, SP \to TP})(M, X) & & \qquad \mathtt{(T,P)}
```
"But this is exactly the same as original definition of masked kernel attention definition (13)!"
Dictionary (Fig 4, p18): C <-> Q (contraction / queries), B <-> K (expansion / keys), X <-> V (input / values),
A_{j:i} <-> L_ji (state matrix / mask), N <-> N (state expansion dim / kernel feature dim).

**Cor 5.1** `cor:1ss-sma` (p17): "1-SS SMA (masked attention with 1-semiseparable structured matrices $L$) (15) is a
special case of a diagonal SSM (8) where the diagonal matrix is a scalar multiple of the identity."

**Theorem 5.2** `thm:ss-sma` (p17): "For any instantiation of structured masked attention (Definition 4.2) that is an
autoregressive process with bounded order, the structured mask $L$ must be a semiseparable matrix." Formal version
Theorem C.3 (p49): an efficient autoregressive transformation of order k is an SSM of order k+1. Intro phrasing: "any
kernel attention method possessing a fast recurrent form must be an SSM."

**Theorem 6.1** `thm:algorithm` (p17): "Consider an SSD model with state expansion factor $\mathtt{N}$ and head dimension
$\mathtt{P}=\mathtt{N}$. There exists an algorithm for computing the model on any input $X \in \R^{\mathtt{(T, P)}}$ which
only requires $O(\mathtt{TN}^2)$ training FLOPs, $O(\mathtt{TN})$ inference FLOPs, $O(\mathtt{N}^2)$ inference memory, and
whose work is dominated by matrix multiplications." Bounds are tight: total state size is N^2 (head P = N).
Mismatch in source: the theorem statement says inference FLOPs O(TN), but the sentence after it gives the inference
FLOPs lower bound as O(N^2), and the Sec 6.3 table gives N^2 per step. Cite the table for per-token inference cost
(our reading; not reconciled).

**Block decomposition** (Sec 6, unnumbered displays, p18-19): diagonal block `M^{(j,j)} = \mathsf{SSM}(A_{j\mathtt{Q}:(j+1)\mathtt{Q}},
B_{j\mathtt{Q}:(j+1)\mathtt{Q}}, C_{j\mathtt{Q}:(j+1)\mathtt{Q}})`; low-rank block M^{(j,i)} = [C-factors] A_{jQ-1:(i+1)Q-1}
[B-factors]^T. Worked example T = 9, Q = 3 (p19). Fig 5 (p20) colors: diagonal block input->output (orange), low-rank
input->state (green, B), state->state (yellow, A), state->output (blue, C).

**Cost per piece** (Sec 6.3, p21-22), BMM(B, M, N, K) = batch B of (M x K)(K x N): O(BMNK) FLOPs:
- Diagonal: C^T B is BMM(T/Q, Q, Q, N); mask elementwise on (T/Q, Q, Q); times X is BMM(T/Q, Q, P, N).
- Right factors: BMM(T/Q, N, P, Q). Center: scalar scan of length T/Q on (N, P) channels, work TNP/Q. Left: BMM(T/Q, Q, P, N).
- N = P = Q: all BMM(T/N, N, N, N); total O(TN^2) FLOPs, O(TN) memory; scan O(T/Q N^2) FLOPs, O(log(T/Q)) depth.

**Complexity table** (Sec 6.3, unnumbered table, `\label{tab:compute}` inside `center`, p22), verbatim cells:

| | Attention | SSM | SSD |
|---|---|---|---|
| State size | T | N | N |
| Training FLOPs | T^2 N | T N^2 | T N^2 |
| Inference FLOPs | T N | N^2 | N^2 |
| (Naive) memory | T^2 | T N^2 | T N |
| Matrix multiplication | yes | no | yes |

**Eqs. (17)-(20)** `eq:multihead`, `eq:multiquery`, `eq:multikey`, `eq:multivalue`, Sec 7.2 (p24), head-pattern shapes:
- (17) Multi-head SSM (MHA): X (T,H,P), A (T,H), B (T,H,N), C (T,H,N).
- (18) Multi-contract SSM (MQA): X (T,1,P), A (T,H), B (T,1,N), C (T,H,N).
- (19) Multi-expand SSM (multi-key attn): X (T,1,P), A (T,H), B (T,H,N), C (T,1,N).
- (20) Multi-input SSM (MVA), used by Mamba-2: X (T,H,P), A (T,H), B (T,1,N), C (T,1,N).

**Prop 7.2** `prop:mamba-multihead` (p24): Mamba's S6 has head dim P = 1 (every channel has its own A) and MIS/MVA
structure (B, C shared across all channels of X).

## 7. Config / setup (architecture lens)

Mamba-2 block forward (Fig 6 p23; Sec 7.1; Sec 8.1 equations), single sequence u in R^(L x d), expansion e:
1. x = u W^(x)T (L x ed); z = u W^(z)T (L x ed); Delta, B, C = projection(u), "one or more groups of Delta, B, C per GPU"
   (Sec 8.1). Fig 6 labels the parallel outputs A, X, B, C; Sec 7.1 says "produce A, X, B, C in parallel with a single
   projection at the beginning of the block", analogous to Q, K, V.
2. x_c = conv1d(x), depthwise (Sec 8.1). Discrepancy: Fig 6 draws X, B, C all leaving the Conv and the nonlinearity
   together; Sec 8.1 equations apply conv1d to x only. Not resolved here. Conv width not stated in the paper
   (README `d_conv=4`, code-only).
3. Kernel feature map psi = Swish/SiLU on B and C (optionally X) (Sec 7.3). Source says "B and C branches
   (corresponding to the K and V branches in attention)"; by Fig 4 the analogs are K and Q (our reading: typo).
4. y = SSM_{A,B,C,Delta}(x_c) (L x ed), heads in MIS/MVA pattern (Eq. 20): X (T,H,P), per-head scalar A (T,H), B and C
   shared (or G groups: grouped-input SSM / grouped-value attention, GIS/GVA, Sec 7.2).
5. y_g = y . phi(z), phi = SiLU gate.
6. y_n = groupnorm(y_g), "number of groups divisible by degree of tensor parallel"; Sec 7.1: extra norm (LayerNorm,
   GroupNorm or RMSNorm) "right before the final output projection", "after the multiplicative gate branch". Source
   writes out = y_g W^(o)T after defining y_n (our reading: should be y_n).
7. out = y_n W^(o)T (L x d). One all-reduce per block under tensor parallelism (vs two for Mamba-1).

| Item | Value | Loc |
|---|---|---|
| A structure | scalar times identity per head; "Each A_t can also be identified with just a scalar" | Sec 2.4 (p5); Sec 5.1 |
| Head dim P | "Typically P = {64, 128}"; Mamba-1 has P = 1 | Sec 2.4 (p5); Prop 7.2 |
| State size N | N and P "constants around 64 or 128"; heads grow with D, head dims fixed | Sec 7.2 (p24) |
| N in experiments | MQAR N = 16, 64, 256; head ablation N = 64, P = 64; (Re)Based ablation N = 64 (130M), N = 256 (380M); Fig 10 sweep N = 4..256 | Fig 8; Table 5; Sec D.4; Fig 10 right |
| N for the main LMs | not stated in the paper (README Mamba2 example `d_state=64`, "typically 64 or 128", code-only) | README |
| Expansion e | "typically 2" (stated for Mamba arch.) | Sec 8.1 |
| Chunk size Q | not stated for experiments; Theorem 6.1 / Sec 6.3 analysis sets Q = N = P; Listing 1 default `block_len=64` | Sec 6.3; Listing 1 (p21) |
| Total state per layer | H P N (product of heads, head dim, state dim) | Sec 9.4.2 |
| Heads, 125M ablation | 24 heads, 24 layers, 126.5M params, N = P = 64 (24 x 64 = 1536 = 2 x 768: e = 2, d_model 768, our reading) | Table 5 top |
| Heads, 360M ablation | 32 heads, 48 layers, 361.8M params (32 x 64 = 2048 = 2 x 1024, our reading) | Table 5 bottom |
| Scaling-law sizes (Transformer dims; "Model dimension and number of heads applies only to Transformer models") | 125M: 12 layers, d 768, 12/64, 4800 steps, lr 6e-4, 2.5B tok; 350M: 24, 1024, 16/64, 13500, 3e-4, 7B; 760M: 24, 1536, 16/96, 29000, 2.5e-4, 15B; 1.3B: 24, 2048, 32/64, 50000, 2e-4, 26B; batch 0.5M tokens | Table 9 (App D.2, p51) |
| Scaling-law data | the Pile, GPT2 tokenizer, Chinchilla token counts, sequence length 8192 (Fig 9 title) | App D.2; Fig 9 |
| Downstream LMs | 300B tokens on the Pile, GPT-NeoX tokenizer; batch 1M for 1.3B and 2.7B; 2.7B follows GPT3 spec "32 layers, dimension 2560" (64 Mamba-2 blocks per Sec 9.2.3, our reading: 2 blocks per Transformer layer) | App D.3; Sec 9.2.3 |
| Optimizer | AdamW, grad clip 1.0, weight decay 0.1, no dropout, linear warmup + cosine decay to 1e-5, peak 5x GPT3 lr, no linear bias, RMSNorm, beta = (.9, .95) | App D.2, D.3 |
| Loss | standard autoregressive LM (next-token) | Sec 9.2 |
| Compute | GPU type/hours for training not stated; speed benchmarks on A100 80GB PCIe (Fig 10 panel titles) | Fig 10 |
| Trainable | input projection (W^(x), W^(z), Delta/B/C projection), conv1d, norm, W^(o); A and Delta use Mamba's parameterization and discretization ("our main models adopt the same parameterization and discretization step as prior work", Remark 1). The exact list is our reading; the paper gives no parameter table. | Remark 1 (Sec 2.1); Fig 6 |

MQAR setup (App D.1, p50): T in {256, 512, 1024}, T/4 key-value pairs, vocab 8192; non-query/key/value tokens replaced
by random tokens (harder than Based's version); curriculum over (T/32, T/16, T/8, T/4) pairs, 2^18 examples per dataset,
8 epochs each, total 2^28 ~ 270M examples; batch 2^18 ~ 0.25M tokens; 2 layers; attention baseline also gets positional
embeddings; sweep D in {32, 64, 128, 256} and lr {10^-3.5, 10^-2, 10^-2.5}; lr decays each epoch. Baselines: softmax
multi-head attention, Based, Mamba (N=16). Falsifier: Mamba-2 at equal N not beating Mamba-1, or accuracy not rising with N.

## 8. Results worth remembering
- Abstract: core layer "2-8x faster" than Mamba's selective SSM.
- Intro (p2): SSD "2-8x faster than the optimized selective scan implementation of Mamba", allows "8x the size of
  Mamba or even higher" state with minimal slowdown; vs FlashAttention-2 "crossing over at sequence length 2K and 6x
  faster at sequence length 16K".
- Fig 10 left (p29, A100 80GB PCIe, state dim 64): curves for FlashAttention-2, Convolution, Scan (PyTorch), Scan (Mamba),
  SSD, sequence length 512 to 512k. Caption: SSD "2-8x faster than a Mamba fused scan for large state expansion (N = 64)
  and faster than FlashAttention-2 for sequence length 2k and above". Read from plot (approximate): at 16k FA2 ~3 ms,
  SSD ~0.55 ms; at 512k SSD ~20 ms vs Mamba scan ~125 ms; FA2 ~3800 ms at 512k.
- Fig 10 right (sequence length 4K): Mamba scan time grows linearly with state dim (read: ~2 ms at N=64, ~4 ms at 128,
  ~7.6 ms at 256); SSD stays ~0.2-0.7 ms; FA2 flat ~0.85 ms (all approximate, read from plot).
- Sec 9.3: SSD uses tensor cores; "2-8x faster than Mamba's fused associative scan"; faster than FA2 "starting at sequence
  length 2K". Caveat: a full Mamba-2 model may train slower than a Transformer at 2K because a Transformer has L/2 MLP
  layers (very hardware-efficient) and Mamba-2 has L SSD layers.
- Fig 8 MQAR (p28): Mamba (N=16) fails almost everywhere; Mamba-2 good across settings; better than Mamba-1 even at
  N = 16; N = 16 -> 64 -> 256 consistently improves. Read from plot (approximate): T=1024, D=32: Mamba-2 N=256 ~0.52,
  N=64 ~0.18, attention ~0.0; T=256, D=32: N=256 ~0.95, attention ~0.50; Mamba N=16 ~0.48 only at T=256, D=256.
- Fig 9 (p29): scaling laws ~125M to ~1.3B, Pile, seq 8192; Mamba-2 "Pareto dominant on performance (perplexity),
  theoretical FLOPs, and actual wall-clock time" vs Transformer++.
- Table 1 (p29) zero-shot, 300B tokens: Mamba-2-2.7B Pile ppl 6.09, LAMBADA ppl 4.10, avg acc 60.2 vs Mamba-2.8B 6.22 /
  4.23 / 59.9 and Pythia-2.8B 6.73 / 5.04 / 55.7; Mamba-2-1.3B 6.66 / 5.02 / 56.4 vs Mamba-1.4B 6.80 / 5.04 / 56.4;
  Mamba-2-780M 7.26 / 5.86 / 53.5. Table 10 (p52): Pythia-6.9B avg 58.3; Mamba-2-130M avg 42.6; Mamba-2-370M avg 49.0.
  Intro: Mamba-2-2.7B "outperforms Mamba-2.8B, Pythia-2.8B and even Pythia-6.9B".
- Table 4 (block ablation, p32): Mamba-1 block (sequential, no norm) 129.3M ppl 11.76; +norm 11.54; parallel no norm
  126.5M 11.66; Mamba-2 (parallel + norm) 126.5M 11.49.
- Table 5 (head patterns, p33, N = P = 64): 125M: MIS/MVA 11.66, MCS/MQA 12.62, MES/MKA 12.59, MHS 12.06, MSS 12.00;
  360M: MVA 8.73, MQA 9.33, MKA 9.36, MHS 9.01, MSS 9.04. Same total state HPN for all. (360M MHS/MSS rows list B/C/X heads
  1/1/1 at 70 layers and 32/32/32 at 29 layers, the reverse of the top block's pattern; labels may be swapped in source,
  our reading.)
- Tables 6-7 (kernel ablations, p33): no feature map beats simple activations by much (none 11.58, Swish 11.66, Performer
  12.21); adding a softmax-like denominator "introduced instabilities to most variants".
- Hybrids (Sec 9.2.3): Table 2 (350M, 48 layers, 7B tokens, GPT-2 tokenizer): ppl 8.60 with 0 attention layers, 8.38 with 1,
  best 8.26 with 6, 8.50 with 24, Transformer++ 8.68; "around 10%" attention best. Table 3 (2.7B, 64 layers, 300B
  tokens): Mamba-2-Attention (58 SSD + 6 attention at indices 9, 18, 27, 36, 45, 56) Pile ppl 5.95, avg 61.0 vs
  Mamba-2 6.09 / 60.2 and Transformer++ 6.13 / 60.2.
- Systems (Sec 8): TP with one all-reduce per block ("reducing the number of synchronization point per block by half");
  context parallelism passes states between devices (same as the chunk decomposition); variable length by setting A_t = 0
  at sequence boundaries.

## 9. Limitations stated
- Scalar-identity A_t is less expressive than Mamba's diagonal A_t; "slightly decreasing the expressive power in return
  for significant training efficiency improvements" (Sec 2.4); "main restriction of SSD is on the expressivity of the
  state transitions A_t" (Sec 10.1). Hope to extend the matrix algorithms to general diagonal SSMs (Sec 10.1).
- "SSD does not generalize standard softmax attention, or any other transformation on the attention kernel matrix that
  does not have a finite feature map" (Sec 10.3).
- Full model may be slower than a Transformer at short length (2K) (Sec 9.3).
- MQAR: unclear "which aspect of the architecture is the predominant factor" (Sec 9.1); copying / in-context learning
  tradeoffs "more remains to be understood" (Sec 10.3).
- Instabilities at larger scale needed the extra norm (Sec 7.1); denominator normalization unstable (Sec 9.4.3).
- Later fixes in the sources given: README lists Mamba-3 (arXiv 2603.15569, "Inference-first", MIMO mode option),
  code-only; no claims about it extracted here.

## 10. Outline feedback
- ch 8 "SSD matrix form (Mamba-2 Eq.)": the display M = L o (C B^T) is unnumbered (Sec 5.1, p16). Cite "Sec 5.1" plus the
  numbered contraction form Eq. (16), and Eq. (3) for M_ji. The attention-style dual (L o QK^T) V with L_ij = a_i...a_{j+1}
  is also unnumbered (Sec 2.4, p6); its numbered general form is Eq. (10).
- ch 8 complexity table is unnumbered (Sec 6.3, p22). Cite as "Sec 6.3 table", not "Table N".
- ch 8 notation box: linear attention N = sequence length; Mamba-2 N = state size, T = length. Also linear attention's
  S_i/Z_i vs s_i/z_i.
- ch 8 viz-ssdmat: L_ii = 1 (empty product); a_t in [0, 1]; "linear attention" button = L all ones below the diagonal;
  "softmax" has no finite-rank structure, so the chunk low-rank overlay only applies to linear attention and SSD.
  Linear attention's denominator Z is not part of SSD.
- ch 8 viz-mamba2flow "N 64-256": paper default is N, P "around 64 or 128" (Sec 7.2); 256 appears in MQAR and ablations.
  Conv width and N of the released LMs are not in the paper (README `d_conv=4`, `d_state=64` example, code-only).
  Fig 6 vs Sec 8.1 disagree on what passes through the conv; draw per Fig 6 but note it.
- ch 8 add one line: Theorem 5.2, any efficient autoregressive masked attention needs a semiseparable mask.
- ch 10 recall: on Mamba-2's harder MQAR (2 layers, small D), Mamba-2 N=256 beats the softmax attention baseline at small
  D (Fig 8). The phonebook/recall chapter should not imply SSMs always lose; Mamba-2 itself says copying/ICL tradeoffs
  remain open (Sec 10.3).
- ch 11 hybrids: Mamba-2 Table 2 (6 of 48 attention layers best, "around 10%") and Table 3 (6 of 64 layers, Pile ppl
  6.09 -> 5.95) are clean matched-data hybrid points; add to ch 11 sources (not ch 8).
- ch 12 tree: Sec 10.4 says RetNet/TransNormerLLM = SSD with time-invariant A_t (L = gamma^(i-j)); GateLoop and GLA have
  input-dependent decay; useful edges.
