# Linear attention: "Transformers are RNNs" (Katharopoulos et al., 2020)

## 1. Identity
- Title: Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.
- arXiv 2006.16236, version read: v3. v1 submitted 2020-06-29; v3 (read) 2020-08-31 (abs.html submission history). v2 2020-06-30.
- Authors: Angelos Katharopoulos (first) ... Francois Fleuret (last); also Apoorv Vyas, Nikolaos Pappas. Idiap / EPFL / UW / UNIGE.
- Venue: ICML 2020 (abs.html Comments).
- Code: https://linear-transformers.com/ (paper Sec 4); repo README = `fast-transformers` (idiap), `pip install pytorch-fast-transformers`.
- Sources read: `src/arxiv.tex` (full), `paper.pdf` (17 pages; Fig 1 on p6 viewed), `README.md`.

## 2. Core problem
Softmax self-attention costs O(N^2) time and memory in the sequence length N, so long sequences are slow and
context is limited. Autoregressive generation is worse: each new step re-attends to all previous steps, so per-step
cost is not constant (Sec 1, Sec 3.3.2).

## 3. What prior methods did and why that hurt (per the paper)
- Sparse factorized attention (Child et al.): O(N sqrt(N)). Reformer (LSH): O(N log N). Both speed up training on long
  sequences but "do not speed-up autoregressive inference" (Sec 1).
- Reformer needs keys identical to queries for LSH, so it cannot be used for decoding where keys differ (Sec 2.1).
- Pruning, factorization, quantization, distillation: still quadratic in N (Sec 2.1).
- Transformer-XL / adaptive span: same asymptotic complexity as vanilla (Sec 2.1).

## 4. The one key idea
Replace the softmax similarity with a kernel sim(q, k) = phi(q)^T phi(k). Then associativity lets you compute
phi(K)^T V once and reuse it for every query: (phi(Q) phi(K)^T) V = phi(Q) (phi(K)^T V). With a causal mask the running
sum phi(K_j) V_j^T becomes a fixed-size matrix state updated per token, so a causal transformer layer is an RNN.

## 5. Mechanism, step by step
1. Project input x (N x F) to Q = x W_Q (N x D), K = x W_K (N x D), V = x W_V (N x M) (Eq. 2).
2. Write attention for any non-negative similarity as a weighted average (Eq. 3). Softmax is sim = exp(q^T k / sqrt(D)).
3. Choose a kernel with feature map phi (C-dimensional): sim(Q_i, K_j) = phi(Q_i)^T phi(K_j) (Eq. 4).
4. Pull phi(Q_i) out of the sum (associativity) (Eq. 5); vectorized numerator (Eq. 6). Sum_j phi(K_j) V_j^T and
   Sum_j phi(K_j) are computed once and shared by all queries -> O(N) time and memory (Sec 3.2).
5. Feature map used in all experiments: phi(x) = elu(x) + 1 (Eq. 7). Positive similarity; elu instead of relu so the
   gradient is not 0 for negative x.
6. Causal mask: sums run to j <= i only (Eqs. 8-9). Define S_i = sum_{j<=i} phi(K_j) V_j^T and Z_i = sum_{j<=i} phi(K_j)
   (Eqs. 10-11); output V'_i = phi(Q_i)^T S_i / phi(Q_i)^T Z_i (Eq. 12). S_i, Z_i come from S_{i-1}, Z_{i-1} in constant time.
7. Backward pass: gradients of the numerator are also cumulative sums (Eqs. 13-15): forward sum for Q, reverse sum
   for K and V (like BPTT). Linear time, constant memory in N for the cumulative-sum terms. About 200 lines of CUDA (Sec 4).
8. RNN form (Sec 3.4, Eqs. 16-20): two hidden states, the "attention memory" s_i (matrix, C x M) and the "normalizer
   memory" z_i (vector, C). At inference store s_i and z_i only -> constant time and memory per generated token.

## 6. Equations
Custom macros used verbatim below (from arxiv.tex preamble): `\fe{x}` = `\phi\left(x\right)`; `\bigO{x}` =
`\mathcal{O}\left(x\right)`; `\R{x}` = `\mathbb{R}^{x}`; `\similarity{x}` = `\text{sim}\left(x\right)`; `\e{x}` =
`\exp\left(x\right)`; `\softmax{x}` = `\text{softmax}\left(x\right)`; `\calL` = `\mathcal{L}`.

Notation trap (state once on the page): this paper's **N is the sequence length** (Mamba-2 and the page call it T or L).
Its D = query/key dim, M = value dim, C = feature-map dim, F = model feature dim, L = number of layers.
Page mapping (via Mamba-2 Fig 4 / Sec 5): phi(Q_i) ~ C_t, phi(K_j) ~ B_s, V_j ~ x_s, S_i ~ the state h (a C x M matrix,
i.e. N x P in Mamba-2 terms), feature dim C ~ state size N. The normalizer z_i has no counterpart in SSD.
Row-vector convention: rows of Q, K, V are tokens; S_i is (C x M) and phi(Q_i)^T S_i is a 1 x M output row.
The paper writes the state as capital S_i, Z_i in Eqs. 10-12 and lower-case s_i, z_i in the RNN form Eqs. 16-20.

**Eq. (1)** `eq:transformer`, Sec 3.1:
`T_l(x) = f_l(A_l(x) + x).`
Meaning: a layer = self-attention A_l plus residual, then a per-position feedforward f_l. A_l is "the only part of the
transformer that acts across sequences".

**Eq. (2)** `eq:attn`, Sec 3.1 (one number for the aligned block):
```
Q &= x W_Q, \\
K &= x W_K,  \\
V &= x W_V, \\
A_l(x) = V' &= \softmax{\frac{Q K^T}{\sqrt{D}}} V.
```
W_Q, W_K in R^{F x D}, W_V in R^{F x M}; softmax applied rowwise to Q K^T.

**Eq. (3)** `eq:general-attn`, Sec 3.1:
`V'_i = \frac{\sum_{j=1}^N \similarity{Q_i, K_j} V_j}{\sum_{j=1}^N \similarity{Q_i, K_j}}.`
Paper: "a generalized attention equation for any similarity function". Equals Eq. 2 with sim(q,k) = exp(q^T k / sqrt(D)).
The only constraint for this to be attention: sim non-negative (Sec 3.2).

**Eq. (4)** (no label in source; numbered (4) in the PDF), Sec 3.2, kernelized attention:
`V'_i = \frac{\sum_{j=1}^N \fe{Q_i}^T \fe{K_j} V_j}{\sum_{j=1}^N \fe{Q_i}^T \fe{K_j}},`
Gloss: similarity = dot product of feature-mapped query and key.

**Eq. (5)** `eq:linear_attn`, Sec 3.2, associativity rewrite:
`V'_i = \frac{\fe{Q_i}^T \sum_{j=1}^N \fe{K_j} V_j^T}{\fe{Q_i}^T \sum_{j=1}^N \fe{K_j}}.`
Paper: "time and memory complexity O(N) because we can compute sum phi(K_j) V_j^T and sum phi(K_j) once and reuse them
for every query."

**Eq. (6)** (no label in source; numbered (6) in the PDF), Sec 3.2, vectorized numerator:
`\left(\fe{Q} \fe{K}^T\right) V = \fe{Q} \left(\fe{K}^T V\right).`
phi applied rowwise to Q and K. Left side: N x N matrix first (quadratic). Right side: C x M matrix first (linear in N).

**Eq. (7)** `eq:elu`, Sec 3.2.1, feature map:
`\fe{x} = \text{elu}(x) + 1,`
Positive similarity; "We prefer elu over relu to avoid setting the gradients to 0 when x is negative." With this map,
cost O(N D M) (C = D).

**Eq. (8)** `eq:masked_attn`, Sec 3.3, causal masking:
`V'_i = \frac{\sum_{j=1}^i \similarity{Q_i, K_j} V_j}{\sum_{j=1}^i \similarity{Q_i, K_j}}.`
Position i is influenced by j iff j <= i.

**Eq. (9)** `eq:masked_linear_attn`, Sec 3.3:
`V'_i = \frac{\fe{Q_i}^T \sum_{j=1}^i \fe{K_j} V_j^T}{\fe{Q_i}^T \sum_{j=1}^i \fe{K_j}}.`

**Eq. (10)** `eq:attn_state` and **Eq. (11)** (unlabelled, numbered (11) in PDF), Sec 3.3:
```
S_i &= \sum_{j=1}^i \fe{K_j} V_j^T, \\
Z_i &= \sum_{j=1}^i \fe{K_j},
```
S_i: running sum of outer products key-feature x value (C x M). Z_i: running sum of key features (C), the normalizer.

**Eq. (12)** `eq:masked_linear_attn2`, Sec 3.3:
`V'_i = \frac{\fe{Q_i}^T S_i}{\fe{Q_i}^T Z_i}.`
Paper: "S_i and Z_i can be computed from S_{i-1} and Z_{i-1} in constant time hence making the computational
complexity of linear transformers with causal masking linear with respect to the sequence length."

**Eqs. (13)-(15)** `eq:grad_q`, `eq:grad_k`, `eq:grad_v`, Sec 3.3.1 (gradients of the numerator V-bar_i):
```
\nabla_{\fe{Q_i}} \calL &= \nabla_{\bar{V}_i}\calL \left(\sum_{j=1}^i \fe{K_j} V_j^T\right)^T, \\
\nabla_{\fe{K_i}}\calL &= \left(\sum_{j=i}^N \fe{Q_j} \left(\nabla_{\bar{V}_j}\calL\right)^T\right) V_i\ , \\
\nabla_{V_i}\calL &= \left(\sum_{j=i}^N \fe{Q_j} \left(\nabla_{\bar{V}_j}\calL\right)^T\right)^T \fe{K_i}.
```
Q-gradient uses the forward cumulative sum; K and V gradients use a reverse cumulative sum (from N down to i).
Result: compute O(N C M), memory O(N max(C, M)). Algorithm 1 (p5) gives forward/backward pseudocode.

**Eqs. (16)-(20)** `eq:trnn-1` ... `eq:trnn-5`, Sec 3.4, "Transformers are RNNs":
```
s_0 &= 0, \\
z_0 &= 0, \\
s_i &= s_{i-1} + \fe{x_i W_K} \left(x_i W_V\right)^T, \\
z_i &= z_{i-1} + \fe{x_i W_K}, \\
y_i &= f_l\left(\frac{\fe{x_i W_Q}^T s_i}{\fe{x_i W_Q}^T z_i} + x_i\right).
```
Paper: "two hidden states, namely the attention memory s and the normalizer memory z". x_i is the i-th input, y_i the
i-th output of one layer. Gloss: write = add outer product (key feature) x (value); read = query feature times state,
divided by query feature times normalizer; residual + feedforward. No decay or forgetting: every past token stays in
the sum with weight 1 (our reading; this is what Mamba-2's mask L generalizes). The paper says the form holds for
"any transformer model, in theory even the ones using softmax attention" (infinite-dimensional phi for softmax, Sec 3.2.1).

## 7. Config / setup
| Item | Value | Loc |
|---|---|---|
| Feature map | elu(x)+1, all experiments | Eq. 7; Sec 4 |
| Benchmark GPU | NVidia GTX 1080 Ti, 11GB | Sec 4.1.2 |
| Benchmark lengths | N in {2^9, ..., 2^16}; batch scaled inversely with N; per-sample time/memory | Sec 4.1.2 |
| Max length fitting GPU | softmax 4,096; lsh-4 and lsh-8 16,384; linear to 2^16 (Fig 1 x-axis, our reading) | Sec 4.1.2, Fig 1 |
| Copy task | max length 128, 10 symbols + separator; 4 layers, 8 heads; batch 64; RAdam lr 1e-3 -> 1e-4 after 3000 updates | Sec 4.1.1 |
| MNIST | 8 layers x 8 heads, embed 256 (32/head), FF 4x embed; mixture of 10 logistics; RAdam lr 1e-4; 250 epochs; batch 10; seq 784 pixels; Reformer 64 buckets, 27 chunks of 29 | Sec 4.2.1 |
| CIFAR-10 | 16 layers, per-layer config as MNIST; NVidia P40 24GB; softmax batch 1, linear/reformer batch 4; 7 days each; Reformer 64 buckets, 83 chunks of 37 | Sec 4.2.2 |
| ASR | WSJ 80 h, 40-dim mel filterbanks, avg 800 frames, max 2,400; CTC loss; transformers 9 layers x 6 heads; RAdam lr 1e-4 halved on plateau; Bi-LSTM 3 layers hidden 320, Adam lr 1e-3 | Sec 4.3 |
| Complexity, softmax | O(N^2 max(D, M)) | Sec 3.2.1 |
| Complexity, linear | O(N C M); degree-2 polynomial map O(N D^2 M), favorable when N > D^2; elu map O(N D M) | Sec 3.2.1 |
| Causal linear, train | compute O(N C M), memory O(N max(C, M)); naive storage of all S_i costs max(D, M) times more memory | Sec 3.3.1 |
| Inference | constant time and memory per prediction; store phi(K_j) V_j^T matrix as state | Sec 3.3.2 |
| Trainable | W_Q, W_K, W_V, feedforward f_l (standard transformer); phi has no parameters (our reading) | Sec 3.1, Eq. 7 |

## 8. Results worth remembering
- Abstract: "up to 4000x faster on autoregressive prediction of very long sequences"; complexity O(N^2) -> O(N).
- Fig 1 (p6): forward/backward time and GPU memory vs N; linear is fastest and smallest at every N; softmax quadratic.
- Fig 2 (p6): copy task; linear reaches the same loss as softmax; lsh-4 plateaus higher (hashing noise).
- Table 1 (MNIST, p7): bits/dim softmax 0.621, LSH-1 0.745, LSH-4 0.676, linear 0.644. Images/sec 0.45 (1x), 0.68 (1.5x),
  0.27 (0.6x), linear 142.8 (317x). Text: "more than 300 times faster"; 10,000 MNIST images at once on one GPU.
- Table 2 (CIFAR-10, p7): bits/dim softmax 3.47, LSH-1 3.39, LSH-4 3.51, linear 3.40. Images/sec 0.004, 0.015 (3.75x),
  0.005 (1.25x), linear 17.85 (4,462x). Text: "for every image generated by the softmax transformer, our method can
  generate 4,460 images". Fig 4 caption: "more than 4,000 times higher throughput". Linear ran 3 times more epochs in 7 days.
- Table 3 (ASR, p8): PER Bi-LSTM 10.94 (1047 s/epoch), softmax 5.12 (2711), LSH-4 9.33 (2250), linear 8.08 (824).
  Linear "more than 3x faster per epoch"; softmax has the best PER.
- Supp. Table 4 (Sec C.1, p13): stateful-softmax (KV-cache baseline) images/sec MNIST 7.56 (16.8x), CIFAR-10 0.32 (80x);
  linear "more than 50x faster for CIFAR-10" than stateful-softmax. Fair comparison for the page (the transformers page
  reader knows the KV cache).
- Supp. Table 5 (Sec C.2, p13), batch 1 latency per image, CIFAR-10: GPU linear 61.3 s vs softmax 300.1 s (4.9x) vs
  stateful-softmax 70.4 s (1.14x); CPU linear 45.1 s vs softmax 8651.4 s (191.8x). Text: "almost 6.6x faster than softmax
  transformers for generating an image on CIFAR-10" (the table does not show a 6.6x cell; 300.1/45.1 is about 6.65, our
  reading). Linear is the only method faster on CPU than GPU.

## 9. Limitations stated
- The exponential kernel's feature map is infinite-dimensional, so exact softmax cannot be linearized (Sec 3.2.1).
- elu+1 chosen for "smaller sequences"; polynomial maps suggested for N > D^2 (Sec 3.2.1).
- ASR: softmax "achieves lower phone error rate in comparison to all baselines" (Sec 4.3).
- Future work: choice of feature map; random Fourier features for RBF to reuse softmax-pretrained models (Sec 5).
- Our reading, from the equations: the state only adds (no forgetting) and is fixed size C x M, so it cannot store
  unbounded key-value pairs. Fix in the sources given: Mamba-2 replaces the all-ones causal mask with a data-dependent
  1-semiseparable decay mask L (Mamba-2 Sec 4.3, Sec 5.2); Mamba-2 also drops the denominator and adds a norm before the
  output projection (Mamba-2 Sec 7.1, Sec 7.3).

## 10. Outline feedback
- ch 8 equation build: cite Eqs. (3), (5)/(6), (7), (10)-(12) and the RNN form (16)-(20). Eqs. (4) and (6) have no LaTeX
  label but are numbered (4) and (6) in the PDF.
- ch 8 must translate notation once: linear attention's N = sequence length (page T/L); its feature dim C plays the role
  of SSM state size N. Without this box, "O(N)" reads as "O(state size)".
- ch 8 "headline speed": use 317x (MNIST) and 4,462x (CIFAR-10) with the caveat that the baseline is a no-cache PyTorch
  softmax; the KV-cache baseline (stateful-softmax) gap is 80x vs 4,462x on CIFAR-10 throughput (Supp. Table 4). The
  abstract's "4000x" is the rounded CIFAR-10 figure.
- ch 10 viz-phonebook "outer-product write, linear attention style": the write rule is Eq. (18), s_i = s_{i-1} +
  phi(k_i) v_i^T; cite it there.
- ch 12 family tree: Mamba-2 Sec 10.4 places RetNet/TransNormerLLM as SSD with time-invariant A_t (decay gamma) and GLA
  as gated linear attention; good edges for "linear attention -> RetNet/GLA".
