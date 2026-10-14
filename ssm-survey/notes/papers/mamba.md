# Mamba (2312.00752v2)

## 1. Identity
- Title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
- arXiv 2312.00752, read v2. v1 submitted 2023-12-01; v2 2024-05-31 (abs.html submission history).
- Authors: Albert Gu (CMU) and Tri Dao (Princeton), alphabetical, equal contribution.
- Venue: none stated on the abs page. `src/main.tex` has a `colm2024_conference` toggle (source artifact, not a venue claim).
- Repo: github.com/state-spaces/mamba (`sources/mamba/README.md`). Checkpoints `mamba-130m` .. `mamba-2.8b` on HF.
- Read from: `src/src/*.tex` (equations, tables), `paper.pdf` (numbering, pages, figures), README (configs).
- Notation: Mamba already uses page notation (input x, state h, output y). Alg. 1/2 "D" = number of channels fed
  to the SSM; inside a Mamba block that is the expanded width E*d_model (our reading of Fig. 3 + Sec 3.4).
  "B" in shape tuples = batch size, not the matrix B. Sign convention: A is real and negative (S4D-Real, A_n = -(n+1)).

## 2. Core problem
Subquadratic models (linear attention, gated convolutions, RNNs, LTI SSMs) trail attention on discrete,
information-dense data such as text. The paper says the cause is that they cannot do content-based reasoning:
their dynamics are fixed, so they cannot choose what to keep in a fixed-size state (abstract; Sec 3.1).

## 3. What prior methods did and why that hurt
- S4 and all derivatives are LTI: (Delta, A, B, C) constant over time, so the model is a recurrence or one global
  convolution kernel (Sec 2 "Linear Time Invariance"). LTI was required for efficiency (Sec 2, Sec 3.3).
- The convolution trick avoids materializing the (B, L, D, N) state; it only builds a (B, L, D) kernel (Sec 3.3.1).
- A static kernel weighs by position, not content: it solves vanilla Copying (time-aware) but not Selective
  Copying, where spacing varies (Sec 3.1, Fig. 2). LTI models also "bleed" across packed documents and cannot
  ignore noisy long context (Sec 3.5.2).
- Attention does not compress at all: the KV cache stores the whole context, so inference is linear-time per
  token and training quadratic (Sec 3.1).
- "Gating" in H3/Hyena is elementwise multiplication that does not act along the sequence axis, so it cannot
  change token spacing (Sec 4.1.1; App. A).

## 4. The one key idea
Make Delta, B, C functions of the current input (selection). The SSM becomes time-varying, so it can reset and
store a token (large Delta) or skip it (small Delta). Then compute it with a fused, hardware-aware parallel scan
that keeps the expanded state only in SRAM.

## 5. Mechanism, step by step
1. Per channel, a diagonal SSM with state size N maps x -> y (Sec 2 "Structure and Dimensions"). Total hidden state
   is D*N per input; over a sequence that is O(BLDN) time and memory.
2. Selection (Sec 3.2, Alg. 2): B = s_B(x) = Linear_N(x); C = s_C(x) = Linear_N(x);
   Delta = softplus(Parameter + Broadcast_D(Linear_1(x))). Shapes gain a length axis L.
3. Generalized Delta (Sec 3.6): Linear_1 widened to rank R ("a small fraction of D"), giving
   s_Delta(x) = Linear_D(Linear_R(x)), a low-rank projection. The "Parameter" acts as a bias, initialized to
   tau_Delta^{-1}(Uniform([0.001, 0.1])).
4. Discretize per token with ZOH (Eq. 4): A_bar_t = exp(Delta_t A), B_bar_t from Delta_t, A, B_t. Shapes (B, L, D, N).
5. Recurrence h_t = A_bar_t h_{t-1} + B_bar_t x_t, y_t = C_t h_t. No convolution form exists now (Alg. 2 line 6
   comment: "Time-varying: recurrence (scan) only").
6. Hardware-aware kernel (Sec 3.3.2; App. D): read Delta, A, B, C (O(BLD + DN) bytes) from HBM to SRAM; discretize
   in SRAM; run a work-efficient parallel associative scan in SRAM; multiply by C and sum; write only (B, L, D)
   outputs to HBM. IO drops by a factor O(N). Sequences too long for SRAM are split into chunks; the scan state
   carries across chunks.
7. Backward: intermediate (B, L, D, N) states are not saved but recomputed when inputs are reloaded into SRAM.
   Block-level recomputation also drops cheap activations (activation output, short conv) (App. D "Memory").
8. Mamba block (Sec 3.4, Fig. 3): two input projections expand d_model by E; main branch: Conv -> sigma -> SSM;
   second branch: sigma; the two are multiplied; an output projection maps back. sigma = SiLU/Swish. Optional
   LayerNorm "in a similar location" to RetNet (exact position not given). Blocks are stacked homogeneously with
   standard normalization and residual connections. No separate MLP and no attention.

### Shape trace, one Mamba block (our reading; Fig. 3 + Sec 3.4 + Alg. 2 + README defaults)
Training (B, L, D_m) with D_m = d_model, E = 2, N = 16, conv width 4 (code-only), R = dt rank (not given):
| step | op | output shape | weight shape | source |
|---|---|---|---|---|
| 1 | norm (residual stream) | (B, L, D_m) | D_m | Sec 3.4 ("standard normalization"); type not pinned |
| 2 | in_proj, 2 branches | x: (B, L, E D_m), z: (B, L, E D_m) | 2 x (D_m, E D_m) = 2ED^2 | Sec 3.4 |
| 3 | causal conv1d on x | (B, L, E D_m) | width 4 per channel | Fig. 3 "Conv"; width = README `d_conv=4` (code-only) |
| 4 | SiLU | (B, L, E D_m) | - | Fig. 3, Sec 3.4 |
| 5 | s_B, s_C | B, C: (B, L, N) each | (E D_m, N) each | Alg. 2 lines 2-3 |
| 6 | s_Delta + bias, softplus | Delta: (B, L, E D_m) | (E D_m, R) then (R, E D_m) + bias | Alg. 2 line 4; Sec 3.6 |
| 7 | discretize | A_bar, B_bar: (B, L, E D_m, N) (SRAM only) | A: (E D_m, N) | Alg. 2 line 5; App. D |
| 8 | selective scan, y = C h | (B, L, E D_m) | - | Alg. 2 line 6 |
| 9 | gate y * SiLU(z) | (B, L, E D_m) | - | Fig. 3 |
| 10 | out_proj + residual | (B, L, D_m) | (E D_m, D_m) = ED^2 | Sec 3.4 |
- Paper: most parameters are the 3ED^2 projections; "the number of SSM parameters (projections for Delta, B, C,
  and the matrix A) are much smaller in comparison" (Sec 3.4). README: "roughly 3 * expand * d_model^2 parameters".
- Not in any given source: the D skip parameter, a log parameterization of A ("A_log"), the dt_rank default,
  the vocabulary size, the norm type inside the released checkpoints. Do not state them as sourced.
- Trainable (our reading of Alg. 2 + Sec 3.6): A (line 1 "Parameter"), the Delta bias (line 4 "Parameter"), and all
  projection, conv and norm weights. Loss: next-token prediction (LM follows the GPT3 recipe, Sec 4.2; DNA text
  says "standard causal language modeling (next token prediction)", Sec 4.3); cross-entropy is our reading.
  Every weight gets gradient from that one loss; A gets it only through A_bar = exp(Delta A) (Sec 3.5.2
  "Interpretation of A").

### Recurrent state per sequence, Mamba-2.8B (our reading; config is code-only)
Paper fact: "the total hidden state has dimension DN per input" (Sec 2). Inference keeps no cache of previous
elements (Sec 1 (ii)). Config: 64 layers, d_model 2560 (README, code-only); E = 2 (Sec 3.4); N = 16 (README
example default and paper's ablation/benchmark setting; the README does not say the checkpoints use 16).
- Inner width E*D = 5120. SSM state per layer = 5120 x 16 = 81,920 values. x 64 layers = 5,242,880 values.
- Conv state per layer: needs the last (width - 1) = 3 inputs per inner channel = 15,360 values (x 64 = 983,040).
  An implementation that buffers all 4 taps would hold 20,480 per layer (unverified here).
- Total (3-tap conv): 6,225,920 values. At 2 bytes: 12,451,840 B = 11.875 MiB (SSM 10.0 MiB + conv 1.875 MiB).
  At 4 bytes: 24,903,680 B = 23.75 MiB. Fixed for any context length.
- Vs the outline's Llama-3-8B KV cache (128 KiB/token): 12,451,840 / 131,072 = 95.0 tokens of KV cache (2-byte
  state). This dtype choice is ours; the paper states none for inference state.

## 6. Equations
Custom macros needed (from `src/main.tex`): `\dt`=`\Delta`, `\A`=`\bm{A}`, `\B`=`\bm{B}`, `\C`=`\bm{C}`,
`\dA`=`\overline{\bm{A}}`, `\dB`=`\overline{\bm{B}}`, `\dtAB`=`(\dt, \A, \B)`, `\dAB`=`(\dA, \dB)`, `\R`=`\mathbb{R}`.

Eq. (1a,b) `eq:ssm` (Sec 2):
```tex
h'(t) &= \A h(t) + \B x(t) \\
y(t) &= \C h(t)
```
Paper: a continuous system "that maps a 1-dimensional function or sequence x(t) in R -> y(t) in R through an
implicit latent state h(t) in R^N". Gloss: h is the memory (N numbers per channel); A sets how it decays; B writes
the input in; C reads the output out.

Eq. (2a,b) `eq:ssm:recurrence`:
```tex
h_{t} &= \dA h_{t-1} + \dB x_t \\
y_t &= \C h_t
```
Paper: the "linear recurrence" mode, used "for efficient autoregressive inference (where the inputs are seen one
timestep at a time)". Gloss: A_bar, B_bar are the discrete versions; one step = multiply old state, add new input.

Eq. (3a,b) `eq:ssm:convolution`:
```tex
\bm{\overline{K}} &= (\bm{C}\bm{\overline{B}}, \bm{C}\bm{\overline{A}}\bm{\overline{B}}, \dots, \bm{C}\bm{\overline{A}}^{k}\bm{\overline{B}}, \dots) \\
y &= x \ast \bm{\overline{K}}
```
Paper: the "global convolution" mode, used "for efficient parallelizable training". Gloss: K_bar is the impulse
response; valid only when A_bar, B_bar, C are the same at every step (LTI).

Eq. (4) `eq:zoh` (zero-order hold):
```tex
\dA = \exp(\dt \bm{A})
\qquad
\dB = (\dt \bm{A})^{-1} (\exp(\dt \bm{A}) - \bm{I}) \cdot \dt \bm{B}
```
Paper: discretization "can simply be viewed as the first step of the computation graph in the forward pass".
Gloss: Delta is the step size; with A negative, exp(Delta A) is in (0, 1): small Delta keeps the state, large
Delta wipes it.

Alg. 1 "SSM (S4)" vs Alg. 2 "SSM + Selection (S6)" (`alg:s4`, `alg:s6`, Sec 3.2, p. 6). Input x: (B, L, D),
output y: (B, L, D) in both.
| line | Alg. 1 (S4) | Alg. 2 (S6) |
|---|---|---|
| 1 | `A : (D, N) <- Parameter` (structured N x N) | same |
| 2 | `B : (D, N) <- Parameter` | `B : (B, L, N) <- s_B(x)` |
| 3 | `C : (D, N) <- Parameter` | `C : (B, L, N) <- s_C(x)` |
| 4 | `Delta : (D) <- tau_Delta(Parameter)` | `Delta : (B, L, D) <- tau_Delta(Parameter + s_Delta(x))` |
| 5 | `A_bar, B_bar : (D, N) <- discretize(Delta, A, B)` | `A_bar, B_bar : (B, L, D, N) <- discretize(...)` |
| 6 | `y <- SSM(A_bar, B_bar, C)(x)`; "Time-invariant: recurrence or convolution" | same; "Time-varying: recurrence (scan) only" |
| 7 | return y | return y |
Verbatim TeX of the changed S6 lines:
```tex
\State $\B : \textcolor{BrickRed}{\mathtt{(B, L, N)}} \gets \textcolor{BrickRed}{s_B(x)}$
\State $\C : \textcolor{BrickRed}{\mathtt{(B, L, N)}} \gets \textcolor{BrickRed}{s_C(x)}$
\State $\dt : \textcolor{BrickRed}{\mathtt{(B, L, D)}} \gets \tau_\dt(\mathsf{Parameter} \textcolor{BrickRed}{+ s_\dt(x)})$
\State $\dA, \dB : \textcolor{BrickRed}{\mathtt{(B, L, D, N)}} \gets \mathsf{discretize}(\dt, \A, \B)$
```
Selection functions (Sec 3.2), verbatim:
```tex
s_B(x) = \mathsf{Linear}_N(x), s_C(x) = \mathsf{Linear}_N(x),
s_\dt(x) = \mathsf{Broadcast}_D(\mathsf{Linear}_1(x)), \tau_\dt = \mathsf{softplus}
```
Gloss: B and C are shared across the D channels (shape (B, L, N), no D axis); Delta has one value per channel
per token. Linear_1 then Broadcast_D: "if a given input x_t should be completely ignored ... all D channels should
ignore it" (Sec 3.5.1). Sec 3.6 generalizes to `s_\dt(x) = \mathsf{Linear}_D(\mathsf{Linear}_R(x))`.

Theorem 1 / Eq. (5) `thm:gating`, `eq:gates` (Sec 3.5.1, p. 8):
```tex
When $N=1, \A=-1, \B=1, s_\dt=\mathsf{Linear}(x)$, and $\tau_\dt=\mathsf{softplus}$,
then the selective SSM recurrence (\cref{alg:s6}) takes the form
g_t &= \sigma(\mathsf{Linear}(x_t)) \\
h_{t} &= (1-g_t) h_{t-1} + g_t x_t
```
Proof key lines (App. C, verbatim):
```tex
\dA_t &= \exp(\dt \A) = \frac{1}{1 + \exp(\mathsf{Linear}(x_t))} = \sigma(-\mathsf{Linear}(x_t))
    \\&= 1 - \sigma(\mathsf{Linear}(x_t))
\dB_t &= (\dt \bm{A})^{-1} (\exp(\dt \bm{A}) - \bm{I}) \cdot \dt \bm{B} = -(\exp(\dt \bm{A}) - \bm{I}) = 1 - \dA
    \\&= \sigma(\mathsf{Linear}(x_t))
```
Paper: "discretization of SSMs is the principled foundation of heuristic gating mechanisms". Gloss: with one
state, A = -1 and softplus step, ZOH gives exactly a leaky gated RNN; g_t -> 0 ignores x_t, g_t -> 1 resets to x_t.
The bias "Parameter" is folded into the Linear. (App. C writes the continuous system as `h(t) = -h(t) + x(t)`,
a typo for h'(t); "leaky integrator".)

Scan cost statements (Sec 3.3.2): naive recurrence `O(BLDN)` FLOPs vs convolution `O(BLD\log(L))` FLOPs, "the
former has a lower constant factor". Parameters per block: `3ED^2` (`2ED^2` in, `ED^2` out) (Sec 3.4).
Scan operator for the page (a1,b1) o (a2,b2) = (a1 a2, a2 b1 + b2): not written in this paper; it cites Blelloch
1990, Martin and Cundy 2018, S5 for the parallel scan (Sec 3.3.2).

## 7. Config / setup
Architecture constants:
| item | value | location |
|---|---|---|
| expansion E | 2 ("We always fix to E=2") | Sec 3.4 |
| blocks per Transformer layer | 2 Mamba blocks match 12D^2 of MHA+MLP | Sec 3.4 |
| state N | 16 (ablations, scan benchmark); README example `d_state=16` | Sec 4.5; Table 9 cap.; README (code-only) |
| conv width | 4 | README `d_conv=4` (code-only) |
| activation | SiLU / Swish | Sec 3.4, Fig. 3 |
| A init (real, default) | S4D-Real, n-th element -(n+1) | Sec 3.6; Table 8 row 3 |
| A init (complex) | S4D-Lin, -1/2 + n i (used only for audio) | Sec 3.6; Sec 4.4.1 |
| Delta bias init | tau_Delta^{-1}(Uniform([0.001, 0.1])) | Sec 3.6 |
| Delta rank R | "small fraction of D"; ablated 1..64 at D=1024 | Sec 3.6; Table 9 |
| field | real by default | Sec 3.6 |

Pretrained LM sizes (README table, code-only; paper Table 12 has only the Transformer-layer view up to 1.3B):
| params | layers | d_model |
|---|---|---|
| 130M | 24 | 768 |
| 370M | 48 | 1024 |
| 790M | 48 | 1536 |
| 1.4B | 48 | 2048 |
| 2.8B | 64 | 2560 |

Scaling-law sizes (Table 12, App. E.2.1; n_layers and heads are Transformer-only):
| params | n_layers | d_model | heads/d_head | steps | LR | batch | tokens |
|---|---|---|---|---|---|---|---|
| 125M | 12 | 768 | 12 / 64 | 4800 | 6e-4 | 0.5M | 2.5B |
| 350M | 24 | 1024 | 16 / 64 | 13500 | 3e-4 | 0.5M | 7B |
| 760M | 24 | 1536 | 16 / 96 | 29000 | 2.5e-4 | 0.5M | 15B |
| 1.3B | 24 | 2048 | 32 / 64 | 50000 | 2e-4 | 0.5M | 26B |

Training recipe for the released LMs (Sec 4.2.2; App. E.2.1, E.2.3):
- Data: the Pile, 300B tokens, GPT-NeoX tokenizer; context length 2048 (same tokenizer/data/length as Pythia).
- Scaling-law runs instead: GPT2 tokenizer, Chinchilla token counts (Table 12).
- AdamW; gradient clip 1.0; weight decay 0.1; no dropout; linear warmup then cosine decay.
- "Improved recipe" (Mamba uses it): cosine decay to 1e-5, peak LR 5x the GPT3 value, no linear bias,
  RMSNorm instead of LayerNorm, AdamW beta = (.9, .95).
- Batch: 0.5M tokens for scaling laws; 1M tokens for the 1.3B in the 300B-token run. Peak LR and batch for 2.8B:
  not stated. GPU type/hours for LM training: not stated.
- README: `mamba-2.8b-slimpj` trained on 600B tokens of SlimPajama (code-only). PyTorch AMP, fp32 params.

Synthetic setups (App. E.1; Sec 4.1.2):
- Selective Copying: length 4096, vocab 16 (incl. noise token), 16 data tokens to memorize, 2 layers, D = 64,
  400K steps, constant LR 0.0001, batch 64.
- Induction heads: 2-layer models, train length 256, vocab 16, test 2^6 = 64 .. 2^20 = 1048576; D = 64 for Mamba,
  128 for others; MHA has 8 heads; batch 8; Adam, no weight decay; LR 2e-4 (all but Mamba) or 1e-3 (Mamba).

Benchmarks (App. E.5): A100 80GB PCIe. Scan: batch 1, L from 2^9 to 2^19, D = 1024, N = 16, BF16; baselines
PyTorch scan (no fusion), PyTorch FFT conv, FlashAttention-2 causal. Inference: Mamba 1.4B and untrained Mamba
6.9B vs GPT3-architecture Transformers 1.3B and 6.7B (HF transformers); prompt 2048, generate 128, batch
1..128; throughput = batch x 128 / time; mean of 3.

## 8. Results worth remembering
Synthetics:
- Selective Copying acc. (Table 1, p. 11): S4 no-gate 18.3; S6 no-gate 97.0; H3 57.0; Hyena 30.1; H3-arch+S6 99.7;
  Mamba-arch+S4 56.4; Mamba-arch+Hyena 28.4; Mamba 99.8.
- Induction heads (PDF caption "Table 2", a plot, p. 11; numbers in Table 11, p. 29): Mamba (74K params) perfect
  at every length 2^6..2^20, "4000x longer than it saw during training"; "no other method goes beyond 2x".
  At 2^9 = 512: MHA-Abs 58.6, MHA-RoPE 83.6, MHA-xPos 99.6, H3 80.9, Hyena perfect. At 2^10: MHA-Abs 26.6,
  MHA-RoPE 31.3, MHA-xPos 67.6, H3 39.5, Hyena 44.1. At 2^20: H3 7.4, Hyena 9.8. MHA OOM beyond 2^14 = 16384.
  Params: MHA 137K, H3 153K, Hyena 69M (mostly positional encodings), Mamba 74K.
Language:
- Scaling laws (Fig. 4, p. 11): ~125M to ~1.3B, Chinchilla protocol, context 2048 and 8192; Mamba is "the first
  attention-free model to match ... Transformer++", "particularly as the sequence length grows". No readable
  numbers in the plot; keep qualitative.
- Zero-shot (Table 3, p. 12), Mamba-2.8B vs Pythia-2.8B vs Pythia-6.9B:
  Pile ppl 6.22 / 6.73 / 6.51; LAMBADA ppl 4.23 / 5.04 / 4.45; LAMBADA acc 69.2 / 64.7 / 67.1; HellaSwag 66.1 /
  59.3 / 64.0; PIQA 75.2 / 74.0 / 75.2; Arc-E 69.7 / 64.1 / 67.3; Arc-C 36.3 / 32.9 / 35.5; WinoGrande 63.5 /
  59.7 / 61.3; Average 63.3 / 59.1 / 61.7. Also RWKV-3B avg 59.6; Mamba-1.4B avg 59.7 vs Pythia-1.4B 55.2.
  Abstract calls it "Mamba-3B"; the table row is "Mamba-2.8B".
- Ablations at ~350M (Tables 6-10, pp. 16-17): Mamba+S6 ppl 8.69 vs Mamba+S4(real) 10.56; selective Delta alone
  9.81, none 10.93, all three 8.71 (Table 7); A init S4D-Real 8.71 vs S4D-Lin complex 9.16 (Table 8); N from 1
  to 16 with selective B, C: 9.73 -> 8.71, params 367.1M -> 371.5M (Table 10) = "over a 1.0 perplexity
  improvement for a cost of only 1% additional parameters".
Efficiency:
- Scan (Fig. 8 left, p. 15; Sec 4.5): faster than FlashAttention-2 beyond length 2K; "up to 20-40x faster than a
  standard scan implementation in PyTorch" (Sec 4.5); caption says "40x". App. D: "up to 7x times faster than
  attention at sequence length 32K". Intro: "up to 3x faster on A100 GPUs" (vs previous methods). Plot: PyTorch
  scan OOM at 128k, convolution OOM at 256k; our scan and FA2 run to 512k.
- Inference (Fig. 8 right, bar labels, tokens/s): batch 1: Mamba-1.4B 140, Transformer-1.3B 79, Mamba-6.9B 58,
  Transformer-6.7B 46. Batch 16: 1089 / 323 / 364 / 120. Batch 32: 1445 / 364 / 443 / OOM. Batch 64: 1688 / OOM /
  490 / OOM. Batch 128: 1814 / OOM / 515 / OOM. Peak 1814 vs 364 = 4.98x (our reading of "5x"). Text: "4-5x
  higher inference throughput ... since without the KV cache it can use much higher batch sizes"; "Mamba-6.9B
  (untrained) would have higher inference throughput than a 5x smaller Transformer-1.3B" (515 vs 364).
- Memory (Table 15, p. 36; 125M models, length 2048, A100 80GB): Transformer+FA2 vs Mamba: bs1 4.6GB vs 4.8GB;
  bs8 11.5GB vs 12.3GB; bs32 34.5GB vs 38.2GB. Claim is "comparable", Mamba slightly higher.
- Activation bytes (App. D "Memory"): attention (FlashAttention) ~12 bytes/token, MLP ~20, total 32; one selective
  SSM ~16, so two SSM layers ~ one attention + one MLP layer (mixed precision).
DNA:
- Model-size scaling (Fig. 5 left): HG38, length 1024, ~200K to ~40M params; Mamba matches Transformer++ and
  HyenaDNA "with roughly 3x to 4x fewer parameters".
- Length scaling (Fig. 5 right): 6 layers x width 128 (~1.3M-1.4M params), 2^10..2^20; Mamba perplexity improves
  with length; HyenaDNA gets worse.
- Great Apes (Table 13 / Fig. 6, 5 classes, random 20%): at 2^20, Mamba 1.4M 71.67, Mamba 7M 81.31, HyenaDNA 1.4M 54.87.
Audio:
- YouTubeMix (Fig. 7): lengths 2^13..2^20 at fixed compute; Mamba better than SaShiMi throughout, gap widens.
  Uses the complex parameterization (only experiment that does).
- SC09 (Table 4): Mamba 6.1M FID 0.94, IS 6.26; Mamba 24.3M FID 0.67, IS 7.33, mIS 144.9; SaShiMi 5.8M FID 1.99;
  DiffWave+SaShiMi 23.0M FID 1.42. Intro: "reducing FID ... by more than half".

## 9. Limitations the paper states
- Continuous-discrete spectrum (Sec 5): selection helps text/DNA but "can impede" LTI-friendly data. Fig. 10 /
  App. E.4.1: on raw audio waveforms S6 "significantly hampers performance"; real-valued SSM still trails complex.
- Scale (Sec 5): evaluated only at small sizes, below 7B; scaling "may involve further engineering challenges".
- Downstream affordances (fine-tuning, prompting, ICL, RLHF, quantization) untested (Sec 5).
- Training memory comparable to, not below, an optimized Transformer (Table 15).
- Later work (not read here; see those papers' notes): Mamba-2 (2405.21060), NVIDIA 8B study (2406.07887),
  Jamba (2403.19887). Only sourced detail: the README's Mamba-2 example uses `d_state=64` ("typically 64 or 128").

## 10. Outline feedback
- ch 0 / interlude: the 130M..2.8B config table exists only in the README (code-only); paper Table 12 stops at
  1.3B and lists Transformer layers. Label the Mamba-2.8B config code-only. N = 16 for the checkpoints is our
  reading (README example default). State-bytes number is our reading: 11.875 MiB at 2 bytes, 23.75 MiB at 4.
- ch 0: "5x" sources are the abstract and Fig. 8 caption; Sec 4.5 text says "4-5x". It is peak throughput at
  large batch (Mamba-1.4B 1814 vs Transformer-1.3B 364 tokens/s), prompt 2048, A100. Say "at large batch".
- ch 5: "Table 2" is the induction-heads plot (captioned Table 2 in the PDF); exact accuracies are in Table 11.
  Selective Copying is Table 1. Extrapolation: trained at 256, perfect to 2^20 = 1048576 ("4000x").
- ch 6: "conv1d k=4" is code-only (README). The outline's "A_log, D, dt bias init" line: A_log and D appear in
  neither the paper nor the README. Only the Delta bias init (Uniform [0.001, 0.1] through softplus inverse) and
  S4D-Real A init are sourced. Either fetch `mamba_simple.py` or drop A_log/D from the walkthrough.
  dt_rank is "a small fraction of D" in the paper; no default value in the sources.
- ch 6: Algorithm 2 discretizes with ZOH per the paper; do not claim what the CUDA kernel does for B_bar.
- ch 7: the scan operator is not written in Mamba; cite Martin and Cundy / Blelloch for it. Mamba speed claims:
  "20-40x" (Sec 4.5, App. D) vs "40x" (Fig. 8 caption); faster than FA2 beyond 2K; "7x" at 32K (App. D).
  Memory: same as FlashAttention Transformer (not less).
- ch 6 sourced results: add Table 10 (N = 1 -> 16 helps only when B, C are selective) as the cleanest "selection
  makes a bigger state useful" number; it fits ch 6 or ch 7.
- ch 10 may want Mamba's own audio limitation (Fig. 10) as the counter-case to selection.
