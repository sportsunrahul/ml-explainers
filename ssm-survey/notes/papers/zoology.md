# Zoology (recall group, serves ch 10; ch 11 hybrid evidence)

## 1. Identity
- Title: "Zoology: Measuring and Improving Recall in Efficient Language Models".
- arXiv 2312.04927, version read: v1 (submitted 2023-12-08, abs.html; only version listed).
- Authors: Simran Arora (first, equal contribution with Sabri Eyuboglu, "random ordering by coin toss") ...
  Christopher Re (last). Stanford, University at Buffalo, Purdue.
- Venue: ICLR 2024 style file (`iclr2024_conference.sty` in src); Zoology README lists it as ICLR. Not
  stated on the arXiv title page.
- Repo: https://github.com/HazyResearch/zoology (README at `sources/zoology/README.md`).
- Sources read: `src/main.tex`, `src/Sections/sec_1..6`, `src/Tables/*`, appendix files
  `experiment.tex`, `mqar_framework.tex`, `scaling.tex`, `general_ar/setup_iclr.tex`,
  `circuit/retnet_equivalency.tex`; `paper.pdf` pp. 2-10, 29-31, 48-50 for numbering and figures.
- Macros: `\Coyote` = `\textsc{BaseConv}`, `\Task` = `\textsc{Mqar}` (`math_commands.tex` l.510-511);
  `\hyenaInput` = `\bm{u}` (`hyena.sty` l.22).

## 2. Core problem
Gated-convolution LMs (H3, Hyena, RWKV) were claimed to match attention, and they solve the classic
synthetic associative-recall (AR) task perfectly. Yet at matched data and recipe they still trail attention
in Pile perplexity. The paper asks which skill explains the gap, and why.

## 3. What prior methods did and why that hurt (per the paper)
- Prior synthetic AR (H3, Hyena, Olsson et al.): one query per input, at a fixed position, tiny vocabulary
  ("|V| < 50, less than model dimension", Sec 1; "Prior work uses vocab sizes <= 40 tokens", App E.1).
  Gated convs pass it, so it hid the gap.
- Real text needs many recalls per forward pass, at varying distances, from a large vocabulary.
- Gated convolutions mix the sequence with fixed (input-independent) filters, so they must cover every
  token-interaction distance with weights, which costs model dimension (Sec 1, Sec 4).

## 4. The one key idea
Measure recall where it matters: split Pile tokens into "AR Hits" (repeated bigrams that are rare in
training) and the rest. AR Hits are 6.4% of tokens but carry 82% of the gap. Then replace the toy AR task
by multi-query AR (MQAR), which reproduces the gap synthetically and makes it analyzable.

## 5. Mechanism, step by step
1. Pretrain 17 LMs (abstract; Sec 1: "4 scales (70M - 1.4Bn) and 5 architectures") on the Pile with the
   GPT-NeoX stack and a Llama-style recipe (RoPE, SwiGLU for attention). Main grid: 10B tokens at
   70M/125M(150M)/360M; plus 1.4B attention and Hyena at 50B tokens (Sec 3.1, App G). NOTE: App C.2 says
   "for a total of 14 training runs". Count mismatch with "17"; not reconciled.
2. Define AR Hit: final token of a bigram that appeared earlier in the same context and was seen
   <= 1,250 times in training (Sec 3.1; App C.1). 6.4% of tokens; other tokens 93.6%.
3. Compute share of the gap due to AR Hits (Sec 3.1 formula, App C.1 formula with min(., 1.0)).
4. Define MQAR (Definition 3.1) and a synthetic generator (Procedure 1, App E.1).
5. Train 2-layer models on synthetic MQAR, sweep model dimension d and sequence length N
   (Fig 2, p.8), and number of KV pairs (Fig 6, p.29).
6. Theory: BaseConv (Definition 4.1) simulates any gated-conv model (Theorem 4.2); attention solves MQAR
   with parameters independent of N (Proposition 4.3); data-independent BaseConv needs ~N log c
   parameters (Theorem 4.4); input-dependent kernels need O(t N c) with t distinct distances
   (Theorem 4.5). For recurrences: RetNet needs Omega(N) bits of state (Corollary H.24).
7. Fix: BaseConv + 3 attention layers, with full, random, programmatic or learned token selection
   (Sec 5, Eq. selective, Table 2 p.10).

## 6. Equations

**MQAR definition (Definition 3.1, Sec 3.2, PDF p.6).** Verbatim:
```
\begin{definition} [Multi-Query-{AR} $\paren{\Task}$]
    We are given an input sequence
    $\bm{x} = \{x_0, \ldots, x_{N-1}\}$ where each $x_i \in C$ is a token drawn from a vocabulary of size $c = |C|$.
    The task
    is to check, for every query
    $1\le i<N$, whether there exists a
    $0 \le j < i$ such that $\bm{u}_i \equiv \bm{u}_j$. If so, output $\bm{u}_{j+1}$.
\end{definition}
```
Gloss: N = sequence length (here N is length, not state size). Every position is a potential query; if the
same token appeared before at j, the answer is the token right after it (its "value"). The paper's example:
input `A 4 B 3 C 6 F 1 E 2 -> A ? C ? F ? E ? B ?`, answers `4, 6, 1, 2, 3`.

**Formal (theory) MQAR layout (App H.7.1, p.49; `general_ar/setup_iclr.tex`).** Verbatim:
```
\bm{u}[0 \cdots N-1] \triangleq \left\{\paren{\bm{k}_0, \bm{v}_0, \bm{q}_0}, \ldots, \paren{\bm{k}_{\frac{N}{3}-1}, \bm{v}_{\frac{N}{3}-1}, \bm{q}_{\frac{N}{3}-1}}\right\}
```
Gloss: triples (key, value, query); for each query q_i, if some earlier key k_j (j < i) equals it, output
v_j. Keys sit at positions i = 0 mod 3, values at 1 mod 3, queries at 2 mod 3 (Remark, eq. kqv-indices).

**Synthetic MQAR generator (Procedure 1 / Algorithm "Mqar Synthetic Procedure", App E.1, p.27).** Steps,
verbatim in substance: (1) first half of vocabulary C = keys K, second half = values V; (2) pair each key
with a random value; (3) sub-select D random KV pairs; (4) place the D pairs at the start, consuming the
first 2D positions; (5) place a second occurrence of each key at a distance drawn from positions [2D..N]
with a power law of parameter alpha; (6) output. Gloss: D = number of KV pairs per example (the "phone book
size"); alpha = 0.1 in all experiments; |C| = 8,192.

**AR share of the gap (Sec 3.1, PDF p.5).** Verbatim:
```
$\frac{\Delta \log(\phi_\text{AR}) \cdot |T_\text{AR}|}{\Delta \log(\phi) \cdot |T|}$
```
Paper's words: "the percentage of the difference in perplexity between attention and each model that is due
to AR tokens ... the fraction of the overall gap that would close if a model matched attention on the AR
slice." Gloss: phi = perplexity, Delta log = gap in log-perplexity to attention, T = test tokens,
T_AR = AR-Hit tokens.

**Same, as implemented (App C.1, p.20).** Verbatim:
```
$$\min (\frac{(l_{H}^m - l_{H}^M) p_{H}}{l^m - l^M}, 1.0)$$
```
Gloss: l_H^m, l_H^M = mean log-prob on AR Hits for model m and attention M; l^m, l^M = mean log-prob on all
tokens; p_H = fraction of AR-Hit tokens (about 6.4%). Capped at 1.0, which is why several cells read 100%:
"if m is better than attention (M) overall and M is better than m at AR, we ascribe 100% of the gap to AR."

**BaseConv (Definition 4.1, Eq. coyote-recursion, p.7).** Verbatim:
```
\bm{y}
&:=
\underbrace{\paren{\bm{u} \cdot \bm{W}^\ell+\bm{b}_1^{\ell}}}_{\mathclap{\textbf{Linear Projection}}}
\odot
\underbrace{\paren{\bm{h}^{\ell} \ast \bm{u}+\bm{b}_2^{\ell}}}_{\mathclap{\textbf{Convolution}}}
```
Gloss: u in R^{N x d} input; W (d x d) projection; h (N x d) a long filter fixed by the weights (not by the
input); * = per-channel convolution; odot = elementwise gate.

**Theorem 4.4 (Data-Independent Filters, p.7).** Plain words: with tokens in O(log c) bits, some BaseConv
solves MQAR with O~(N log c) parameters and O~(1) layers. Parameters grow with N.

**Proposition 4.3 (Attention, p.7).** Plain words: attention, even without softmax, solves MQAR on one-hot
inputs (d = 3c) with O(c^2) parameters, O(N c^2 + N^2 c) time, O(1) layers. No N in the parameter count.

**Theorem 4.5 (Input-Dependent Filters, p.8).** Plain words: with at most t distinct key-query distances,
BaseConv with input-dependent kernels solves MQAR with O(t N c) parameters and O(1) layers.

**RetNet state recurrence (App E.2 / H.6, eq. retnet-recurrence, p.48).** Verbatim:
```
\bm{z}^{n} \equiv \gamma \bm{z}^{n-1} + \paren{\tbf{A}[n,:]}^{\top}\tbf{V}[n,:],
```
Gloss: z^n is a d x d matrix state (page notation: h_t, size d x d); gamma a scalar decay; the write is an
outer product of a key-like vector A[n,:] and a value V[n,:]. This is exactly the linear-attention style
outer-product write the page's `viz-phonebook` uses. App E.2 says RetNet "is a special case of the recently
proposed Mamba ... and GateLoop", which "replace the gamma term ... with yet another input-dependent matrix".

**Corollary H.24 (App H.6, p.48).** Verbatim: "The RetNet model requires $\Omega(N)$-bits to solve AR for
$d \le \sqrt{N}$." Proof idea: Alice encodes a bit string x as N pairs (i, x_i), runs the model, sends the
d x d state (O(d^2) numbers) to Bob, who answers any index query; the one-way index problem needs Omega(N)
bits (Theorem H.23, Jayram et al. 2008). Gloss: a d x d state cannot hold N independent key-value facts when
d^2 is much smaller than N.

## 7. Measurement setup (analysis lens)
Downstream (Sec 3.1, App C):
- Data: the Pile, GPT2BPETokenizer, same data order for all models; A100 80GB; GPT-NeoX stack.
- Held fixed: data, tokens (10B main grid; 50B for 1.4B), Llama recipe; only the sequence mixer changes.
- Models: attention (RoPE, SwiGLU), Long Conv (no gating), H3, Hyena, RWKV (v4); RetNet and Hyena +
  sliding window attention in App F (Table 4). Mamba is NOT trained in this paper.
- Metric: validation perplexity overall and on two slices; AR-gap share formula above.
- Falsifier: if AR-Hit perplexity matched attention while the overall gap stayed, the share would be near 0.

Synthetic MQAR (Sec 4.3, App E.2):
- Vocabulary |C| = 8,192; alpha = 0.1; exactly 2 layers (sequence mixer + MLP, with layer norm); attention
  has 1 head and position embeddings; pure convolution runs use none.
- Fig 2 (Experiment 1): N in {64, 128, 256, 512}, d in {64, 128, 256, 512}. D for this experiment is "held
  constant" but its value is not stated in the paper.
- Fig 6 (Experiments 2-3): N = 256, D in {N/32, N/16, N/8, N/4} = {8, 16, 32, 64} KV pairs; d in
  {64, 128, 256}.
- Training: AdamW, weight decay 0.1, 10% linear warmup, LR sweep np.logspace(-4, -2, 4), 64 epochs; batch 8
  (N or d >= 512), 16 (>= 256), else 64; 100,000 train / 3,000 test examples; report max test accuracy.
- Falsifier for Claim 1: a gated conv reaching > 0.9 accuracy with d < N at N = 512.

## 8. Results worth remembering
- AR share: "the AR slice accounts for 82% of the average quality gap between the gated convolutions and
  attention" (abstract; Sec 3.1). The paper does not show the averaging. Our reading: the mean of the
  "% of gap due to AR Hits" column for H3, Hyena, RWKV at 70M, 125M-class and 360M in Table 4 (p.30),
  Long Conv excluded, is 81.6 (values 63.2, 60.8, 57.9, 88.4, 100, 100, 65.8, 98.2, 100).
- AR Hits are 6.4% of Pile tokens (Sec 1, Sec 3.1).
- Max perplexity gap: "up to 2.1 perplexity points" (abstract); minimum gaps +2.14, +0.59, +0.35 PPL at 70M,
  160M, 360M (Sec 3.1).
- Table 1 (p.3), 360M rows, overall ppl: attention 9.44, H3 10.38, Hyena 10.07, RWKV 9.79; AR Hits ppl:
  attention 1.98, H3 4.81, Hyena 3.83, RWKV 3.82; % of gap from AR: H3 65.8%, Hyena 98.2%, RWKV 100.0%.
- Table 1, 125M-class rows: attention 11.01 overall / 2.16 AR; Hyena 11.60 / 5.00, 100.0%.
- 70M attention beats 1.4B Hyena on AR Hits: 2.41 vs 3.43 ppl (Sec 3.1; 70M row Table 4 p.30, 1.4B row
  Table 5 p.30). 1.4B Hyena AR share is 40.3% (Table 5).
- Fig 2 top (p.8): "attention solves MQAR perfectly at all sequence lengths using a constant model dimension of
  64. In contrast, MQAR does not achieve accuracy > 0.9 unless d >= N" (Sec 4.3, Claim 1). The sentence's
  subject should be the gated convolutions (typo in source). No per-point numbers in the text. Visual
  read: at N = 512, H3/Hyena/RWKV stay below about 0.4 even at d = 512; BaseConv reaches about 0.7 at d = 512.
- Fig 2 bottom: attention hybrid and programmatic input-dependent filters near-flat in d; autocorrelation
  filters improve over input-independent BaseConv.
- Fig 6 top (p.29): RetNet (chunk 8 and 32) needs more d as KV pairs grow; at 64 KVs it reaches 1.0 only at
  d = 256; attention is near 1.0 everywhere. Fig 6 bottom: sliding window 32 works up to 16 KVs then degrades.
- Fig 7 (p.31): RWKV-Raven 7B vs Llama 2 7B on synthetic MQAR, seq len 1024, P in {16, 32, 64, 128, 256} KV
  pairs: RWKV AR perplexity degrades with P; Llama 2 flat (no numbers in text).
- Rare bigrams (seen 1x in training) at 350M scale: gap to attention 12.0 (Hyena) / 13.2 (RWKV) ppl (App G).
- Hybrids (Table 2, p.10): BaseConv + 3 attention layers (6.3% of layers at 354M, 10% at 168M). At 360M,
  full-attention hybrid 8.59 ppl vs attention 9.44 ("outperform attention only models by 0.85 perplexity
  points while enabling an 18% reduction in total FLOPs", Sec 5); programmatic selection closes 85% of the
  AR-slice gap; learned selection 72% with k = 256.
- Abstract: hybrids with input-dependent sparse attention "close 97.4% of the gap to attention". The 97.4 is
  not derived in the main text; source of the number not located.

## 9. Limitations the paper states
- AR Hits heuristic catches only exact repeated bigrams; fuzzy or semantic recall is missed (App C.1).
- MQAR is "useful but simplistic" (README note 3, code-only).
- Input-dependent filters are hard to make causal and may be expensive (Sec 4.3).
- Later fixes in our sources: Based (2402.18668) extends the state-size vs recall trade-off to Mamba and
  linear attention and gives a lower bound for any recurrent model; Mamba-2 reports MQAR results (see
  mamba2 notes).

## 10. Outline feedback
- Ch 10 must not say "Zoology shows SSMs need d >= N". Fig 2 and Theorem 4.4 cover gated convolutions
  (H3, Hyena, RWKV, BaseConv), not Mamba. The fixed-state recall claim is the RetNet result
  (Fig 6, Corollary H.24), which the paper extends to Mamba only by the remark that RetNet is a special case.
- "Share of perplexity gap from recall" = 82%, measured on gated convolutions vs attention, not SSMs in the
  Mamba sense. Say so on the claim card.
- RetNet is compared only in App F (Fig 6, Table 4), not in main Fig 2.
- Outline item "MQAR" in ch 8 sources: Zoology defines the task; quote the definition here once and reuse.
- For `viz-phonebook`: the RetNet recurrence above is the exact outer-product write; Corollary H.24 is the
  closest capacity statement (no K-vs-N formula; see recall.json phonebook entries).
