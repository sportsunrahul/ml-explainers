# Griffin (and Hawk): Mixing Gated Linear Recurrences with Local Attention

Branch extraction only (RG-LRU recurrence, Hawk vs Griffin, one key result, the local-attention mix).
Lens: architecture; the method leads, results are scaling curves and one downstream table.
Sources read: `sources/griffin/src/main.tex` (full), `main.bbl` (names), `paper.pdf` (pages, Fig 2),
`abs.html`. Serves ch 1 (LRU lineage, family tree) and possibly ch 11 (hybrids) and ch 7 (scan).

## 1. Identity
- Title: Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.
- arXiv 2402.19427, version read: v1 (abs.html "Submitted on 29 Feb 2024"; only v1 listed).
- Published: 2024-02-29 (abs.html citation_date).
- Authors: Soham De, Samuel L. Smith, ... Caglar Gulcehre (17 authors; De, Smith, Fernando, Botev,
  Cristian-Muraru equal contribution). Google DeepMind.
- Venue: not stated (no journal-ref; comments "25 pages, 11 figures"). Repo: not stated in the paper.

## 2. Core problem
RNNs "have fast inference and scale efficiently on long sequences, but they are difficult to train and
hard to scale" (Abstract). Transformers pay quadratic attention cost and a KV cache that grows linearly
with length, even with MQA (Sec 1, p.1). A replacement must match Transformers both in quality at scale
and in hardware efficiency (Sec 1, p.1).

## 3. What prior methods did and why that hurt (per the paper)
- "Due to their sequential processing structure, classical RNNs suffer from slow training speeds during
  both forward and backward propagation" (Sec 7, p.13; cites Werbos 1990). "Transformers achieve better
  performance than RNNs in practice and are also very efficient at utilizing modern hardware" (Sec 1,
  p.1; cites Kaplan et al. 2020).
- LRU (Orvieto et al. 2023b) "systematically analyzed and ablated multiple modifications to standard
  RNNs"; through "better parameterization and initialization simplified linear RNNs (the LRU), perform
  just as well as other SSMs variants on various long-range tasks" (Sec 7, p.13).
- Fixed-state RNNs are weak at exact retrieval: pre-trained Hawk "fails to memorize and retrieve the
  correct phone number when the phonebook length grows" (Sec 6.2, p.13; cites Jelassi et al. 2024).

## 4. The one key idea
Take the LRU's stable diagonal linear recurrence, make it real-valued, and give it two input-dependent
gates (none depends on h_{t-1}, so it stays a linear recurrence); stack it with gated MLPs (Hawk), and
interleave it with local sliding-window attention so a fixed-size cache covers both recent detail and
long-range carry (Griffin) (Sec 2.4, p.4; Sec 3, p.5).

## 5. Mechanism, step by step
1. Residual block (pre-norm): RMSNorm -> temporal-mixing block -> add skip; RMSNorm -> gated MLP -> add
   skip. Stacked N times, final RMSNorm, tied output embedding (Sec 2, 2.1, p.2-3; Fig 2a).
2. Gated MLP: two branches of width M D (M = 3), GeLU on one, multiply, linear back to D (Sec 2.2, p.3).
3. Recurrent block (replaces MQA): two parallel linears D -> D_RNN. Branch 1: Conv1D (temporal width 4,
   4 D_RNN params) -> RG-LRU. Branch 2: GeLU. Multiply, linear D_RNN -> D (Sec 2.3, p.3; Fig 2c). Same
   shape as Mamba's and GSS's block (p.3).
4. RG-LRU (Eqs. 1-4, p.4): sigmoid gates r_t, i_t from x_t; per-channel decay a_t = a^{c r_t} with
   a = sigma(Lambda), c = 8; h_t = a_t h_{t-1} + sqrt(1 - a_t^2) (i_t x_t); y_t = h_t. Computed in log
   space (Eq. 6, p.19). No complex numbers (complex "not beneficial for language modelling in practice",
   p.4; complex CG-LRU in App B).
5. Hawk = residual blocks with the recurrent block as temporal mixer everywhere, D_RNN about 4D/3 to
   roughly match MHA parameter count (Sec 3 "Hawk", p.5).
6. Griffin = the same backbone, with the temporal mixers in a repeating pattern: two residual blocks with
   the recurrent block, then one residual block with local MQA. Default local window 1024 tokens (Sec 3
   "Griffin", p.5). So recurrence : local attention = 2 : 1 (our reading of "alternating two ... followed
   by one").
7. Training kernel: a sequential "linear scan" in Pallas that keeps h in VMEM, not an associative scan
   (Sec 4.2, p.7-8). Gate weights W_a, W_x are block-diagonal with 16 blocks for sharding (Sec 4.1, p.7).

Gate behaviour (Sec 2.4 p.4; App A p.19): the input gate is LSTM-like. The recurrence gate is "biased
towards retaining information": it "can approximately interpolate between the standard LRU update ...
and the previous hidden state", so it can "discard the input and preserve all information from the
previous history"; unlike Mamba/GRU gates it "does not allow to fully discard the contribution of
$h_{t-1}$ (this depends, however, on the value of $\Lambda$)".

## 6. Equations
Notation: Griffin already uses page notation (input x_t, state h_t, output y_t). Its a is a decay in
(0, 1); Fig 7 caption (p.19): "in the Mamba's notations this is $-A$". Its Lambda is a real logit (not the
LRU's complex Lambda). Custom macro: `\softplus` (operator).

G1-G4. RG-LRU, Eqs. (1)-(4), labels `eq:recurrent_gate`, `eq:input_gate`, `eq:a_rglru`, `eq:RG_LRU`, Sec 2.4,
p.4 (verbatim, eqnarray):
```
 r_t &=& \sigma\left(W_a x_t + b_a\right), \quad \textcolor{gray}{\text{ \emph{recurrence gate}}} \label{eq:recurrent_gate} \\
 i_t & = & \sigma(W_x x_t + b_x), \quad \textcolor{gray}{\text{\emph{ input gate}}} \label{eq:input_gate}\\
a_t &=& a^{c r_t}, \label{eq:a_rglru} \\
h_{t} &=& a_t \odot h_{t-1} + \sqrt{1-a_t^2} \odot (i_t \odot x_t).
```
Paper: "$\sigma$ ... is the sigmoid function. The recurrent weight $a$ ... is diagonal. Hence all
operations are element-wise." "$a = \sigma(\Lambda)$ ... guarantees that $0 \leq a \leq 1$, ensuring that
the recurrence is stable." "$c$ is a scalar-valued constant set to $8$." "neither gate depends on the
recurrent state $h_{t-1}$, which ensures that the computation can be executed efficiently on device."
Gloss: r_t in (0,1) is an input-dependent exponent on the base decay a: r_t near 0 gives a_t near 1
(keep the state, ignore the input); r_t = 1 gives a_t = a^8 (the strongest decay). i_t scales the input.
sqrt(1 - a_t^2) plays the role of the LRU's gamma = sqrt(1 - |lambda|^2) (our reading; the paper says only
"inspired by" the LRU), now recomputed per token. W_a, W_x: LeCun init. Lambda init so that a^c is
uniform in [0.9, 0.999] (p.4).

G6. Log-space implementation, Eq. (6), App A, p.19:
```
\log a_t = \log a^{c r_t} = \log \sigma(\Lambda) ^ {cr_t} = - c \softplus(\Lambda) \odot r_t.
```
Gloss: the log of the decay is minus c times softplus(Lambda) times the gate, so a_t is always in (0, 1].

G7. General gated update used to compare gates, Eq. (7), App A, p.19:
```
h_t = \alpha(r_t) h_{t-1} + \beta(r_t) x_t.
```
RG-LRU: $\alpha(r_t) = a_t = a^{c r_t}$, $\beta(r_t) = \sqrt{1 - \alpha(r_t)^2}$. GRU: $\alpha(r_t) = 1 - r_t$,
$\beta(r_t) = r_t$. Mamba (B = 1, C = 1): $\alpha(r_t) = (1 - r_t)^{-A}$, $\beta(r_t) = (1 - \alpha) / A$.

G5. Decode-time cost model, Eq. (5), Sec 5.1, p.9:
```
\textit{Time to sample next token} \approx \frac{\textit{param size} + \textit{batch size} \times \textit{cache size}}{\textit{memory bandwidth}}.
```
Cache sizes (App F.4, p.23-24): KV cache $2NTh_k d_{head}$ ($d_{head} = 128$; MQA $h_k = 1$); one RG-LRU
layer's state $D_{RNN}$, whole Hawk "$ND_{RNN} \approx 4BND/3$" (as printed, with B although B = 1 is
assumed); conv state $(4-1)D_{RNN}=3D_{RNN}=4D$ per block; local MQA cache "upper bounded by
$2T_{WS}d_{head}$".
Compute scaling (Sec 4.3, p.8): linear layers "$O(T D^2)$", RG-LRU "$O(T D)$", global attention
"$O(T^2 D)$".

## 7. Config / setup
Model scales, shared by MQA baseline, Hawk and Griffin (Table 2, p.20):
| Size | D | D_RNN | Depth N | MLP M | Heads | Tokens (optimal scaling) |
|---|---|---|---|---|---|---|
| 100M | 768 | 1024 | 12 | 3 | 6 | 1.9B |
| 200M | 1024 | 1536 | 12 | 3 | 8 | 3.9B |
| 400M | 1536 | 2048 | 12 | 3 | 12 | 7.8B |
| 1.3B | 2048 | 2560 | 24 | 3 | 16 | 25B |
| 3B | 3072 | 4096 | 24 | 3 | 24 | 60B |
| 7B | 4096 | 5632 | 32 | 3 | 32 | 132.5B |
| 14B | 5120 | 8192 | 40 | 3 | 40 | 300B |
- Head dim 128, MQA, RoPE, no absolute positions (Sec 2.3, p.3).
- Data: MassiveText (slightly different subset mix); training sequence length 2048; AdamW; LR, weight
  decay and beta_2 tuned on small models and extrapolated (Sec 3.1, p.5). Downstream models trained on
  300B tokens (Sec 3.2, p.6).
- Hardware: TPU-v3 (Sec 4.2, p.7; Table 3 p.21: HBM 32 GB, 900 GB/s, MXU 123 TFLOPs/s bf16, VPU FLOPs/byte
  4.2). bfloat16 params and activations, ZeRO, Megatron-style sharding (Sec 4.1, p.7). GPU/TPU hours: not
  stated.
- Loss: next-token softmax over a tied output layer (Sec 2, p.2); exact loss formula not written.
- Local attention window: 1024 for all 2048-length runs (Sec 3 p.5; App E p.21).

## 8. Results worth remembering
A. Key result, downstream (Table 1, p.6, column Average; character-normalized accuracy, WinoGrande
unnormalized with partial scoring):
| Model | Size | Tokens | MMLU | HellaSwag | PIQA | WinoGrande | ARC-E | ARC-C | Average |
|---|---|---|---|---|---|---|---|---|---|
| Mamba | 3B | 600B | 26.2 | 71.0 | 78.1 | 65.9 | 68.2 | 41.7 | 58.5 |
| Llama-2 | 7B | 2T | 45.3 | 77.2 | 78.8 | 69.2 | 75.2 | 45.9 | 65.3 |
| Llama-2 | 13B | 2T | 54.8 | 80.7 | 80.5 | 72.8 | 77.3 | 49.4 | 69.3 |
| MQA Transformer | 1B | 300B | 28.9 | 64.8 | 75.0 | 62.0 | 60.2 | 35.4 | 54.4 |
| MQA Transformer | 3B | 300B | 31.7 | 71.0 | 77.6 | 66.1 | 68.1 | 39.2 | 59.0 |
| MQA Transformer | 6B | 300B | 38.9 | 77.0 | 79.5 | 70.4 | 74.1 | 45.2 | 64.2 |
| Hawk | 1B | 300B | 29.7 | 63.3 | 76.1 | 57.2 | 60.6 | 34.6 | 53.6 |
| Hawk | 3B | 300B | 31.3 | 71.7 | 78.8 | 66.5 | 68.4 | 40.2 | 59.5 |
| Hawk | 7B | 300B | 35.0 | 77.6 | 80.0 | 69.9 | 74.4 | 45.9 | 63.8 |
| Griffin | 1B | 300B | 29.5 | 67.2 | 77.4 | 65.2 | 67.0 | 36.9 | 57.2 |
| Griffin | 3B | 300B | 32.6 | 73.5 | 78.1 | 67.2 | 71.5 | 41.4 | 60.7 |
| Griffin | 7B | 300B | 39.3 | 78.6 | 81.0 | 72.6 | 75.4 | 47.9 | 65.8 |
| Griffin | 14B | 300B | 49.5 | 81.4 | 81.8 | 74.1 | 79.1 | 50.8 | 69.5 |
One-line takeaway: Hawk-3B (59.5) beats reported Mamba-3B (58.5) on half the tokens; Griffin-7B (65.8)
and Griffin-14B (69.5) match Llama-2-7B (65.3) and -13B (69.3) on 300B vs 2T tokens. Token-ratio wording
differs in the source: Abstract and Sec 8 say "over 6 times fewer tokens"; Sec 1, Sec 3.2 and the Table 1
caption say "roughly 7 times" / "nearly seven times". Do not reconcile; quote with location.
Caveat stated: Mamba and Llama-2 used different data and tuning (Sec 3.2, p.6).

B. Scaling (Fig 1a, p.2; Sec 3.1, p.5): all three families show power-law loss vs FLOPs from 100M to 7B,
plus Griffin 14B; "Griffin achieves lower validation loss than the Transformer baseline across all FLOPs
budgets despite not using any global attention layers"; Hawk slightly higher, gap "appears to close".
Numbers only in the plot.

C. Local-attention mix (App E, Fig 9, p.21-22; 400M models, train lengths 2048 / 4096 / 8192): Griffin
with a fixed 1K window "outperforms all global attention and local attention MQA Transformer baselines
across all training sequence lengths"; the gap to global MQA "reduces as the sequence length grows";
pure local-attention MQA is "significantly worse". Recommendation: grow the window slowly with length.

D. Retrieval (Sec 6.2, Fig 6, p.12-13): synthetic tasks use 5-block, D = 64, about 250K-param models, Griffin
with "a single local attention in the middle of the network, in the third block" (windows 512 for
selective copying at length 1024, 128 for induction heads at length 256). Griffin learns selective
copying about as fast as the Transformer; Hawk is "significantly slower". Phonebook (pre-trained 7B Hawk
and Griffin vs 6B MQA): Griffin "can perfectly solve this task up to a context length that matches its
local attention window size of 1024", then degrades; Hawk fails as the phonebook grows.

E. Hardware (Sec 4.2, p.7-8; App D.2 p.21): RG-LRU update loads 6 bytes, writes 2, does 6 FLOPs per
element: FLOPs-to-byte "0.75" vs TPU-v3 VPU "4.2", so memory bound. Pallas linear scan: "almost 3x speed
up over the native Jax implementation", "10-20% lower training times per step" of full Hawk; associative
scan "significantly slower", Hawk step "up to 50% slower" with it. Inference (Sec 5.2, p.10): 1B models,
batch 16; lower latency at long lengths; higher max throughput because of smaller cache; Hawk > Griffin
in throughput at large batch (local cache).

## 9. Limitations the paper states
- Pre-trained Hawk and Griffin "perform less well than Transformers when we evaluate pre-trained models
  on copying and exact-retrieval tasks without fine-tuning" (Sec 1 item 6, p.2); Griffin degrades past
  its 1024 window on phonebook (Sec 6.2, p.13); "more work is needed".
- Local window should likely grow with sequence length (App E, p.22).
- Kernel conclusions are TPU-v3 specific: "do not necessarily apply to other accelerators" (footnote 3,
  p.7). The gate is incompatible with the convolutional view (Sec 4.2, p.8).
- Training with 8k context makes short-context loss slightly worse (Sec 6.1, p.11).

## 10. Outline feedback
- ch 1: Griffin is the natural "what happened next" for LRU: same group, the LRU recurrence made real and
  input-gated, scaled to 14B language models. One sentence plus the Table 1 takeaway fits the LRU box.
  Griffin's sqrt(1 - a_t^2) factor visibly reuses LRU's gamma = sqrt(1 - |lambda|^2) (our reading).
- ch 1 "why dropped": Griffin Sec 7 (p.13) gives a second, citable sentence on sequential training
  (Werbos 1990) and Sec 1 (p.1) the hardware-efficiency framing (Kaplan et al. 2020).
- ch 6 (Mamba): RG-LRU's gate never depends on h_{t-1} and decays only toward "keep", while Mamba's Delta
  can fully reset (App A, Fig 7). Useful contrast if ch 6 discusses gates.
- ch 7 (scan): Griffin did NOT use an associative scan; on TPU-v3 a memory-aware sequential linear scan
  was about 3x faster than native Jax and the associative scan was slower (Sec 4.2, App D.2). This is a
  counterpoint to "scan = log-depth tree" worth one line.
- ch 11 (hybrids): Griffin is a hybrid with LOCAL attention (2 recurrent : 1 local MQA, window 1024), so its
  KV cache is bounded; Jamba uses global attention (our reading, confirm with the jamba group). The outline's hybrid chapter should say which kind.
  App E (Fig 9) and the phonebook-vs-window result are sourced points for "a little attention".
- ch 12 tree: LRU (2023-03-11) -> Hawk/Griffin (2024-02-29); "concurrent to" Mamba per Griffin Sec 7.
