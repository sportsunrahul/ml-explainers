# Branch papers for the family tree (ch 12)

One paragraph per branch paper. Tree data: `notes/family-tree.json` (28 branches + 12 spine nodes).
Claims: `notes/claims/branches.json`; facts already claimed in another claims file are not repeated, and a
tree headline points to its claim with `claim_id` + `claim_file`. "posted" is always the arXiv v1 date from abs.html; all 28 dates in
`branches.tsv` match abs.html. Abstract-only papers (no downloaded source) were read from
`sources/<name>/abs.html` only; anything else about them is marked as our reading. Where a paper has a full
note in `notes/papers/`, the facts come from that note and the note is cited.

Parent rule: a parent is a paper the child states it builds on, or the model it analyses. Nodes with
`"our_reading": true` have a parent link the source does not state (hyena, retnet, hymba, falconh1).
xlstm and minrnn have no parent in the tree: they descend from LSTM/GRU, which are not nodes. Griffin's tsv
family "rnn+hybrid" is not in the enum; the tree uses `rnn`, to sit with LRU as the outline's "LRU/Griffin" lineage.

## SSM line

**lssl** (2110.13985v1, posted 2021-10-26). Parent: hippo. Puts HiPPO-LegS as A into an LTI state space
with step Delta t, and shows one layer is continuous, recurrent and convolutional. Headline: sCIFAR 84.65
(Table 1 p8, row LSSL). Source: `notes/papers/lssl.md` (App B.2 init, Table 1). The note flags a sign
mismatch in App B.2 (A without the minus).

**dss** (2203.14343v3, posted 2022-03-27; abstract only). Parent: s4 (stated: "match the performance of S4
even without the low rank correction"). A purely diagonal state matrix matches S4 on LRA and Speech Commands.
No number in the abstract, so no headline.

**s4d** (2206.11893v2, posted 2022-06-23). Parents: s4, dss (stated in Sec 1: DSS found diagonal A nearly
matches S4 without an explanation). Diagonal A with S4's other choices; the kernel is a Vandermonde product;
the HiPPO-derived init (S4D-LegS/Inv/Lin) is what makes it work, random diagonal inits fail. Headline: S4D-Inv
LRA average 85.50 (Table 5 p14). Source: `notes/papers/s4d.md`.

**hippo2** (2206.12037v2, posted 2022-06-24; abstract only). Parents: hippo, s4 (stated: generalizes the
HiPPO framework and explains S4's matrix). S4 as projection onto exponentially-warped Legendre polynomials;
Fourier-basis variants; explains the timescale init. Headline: S4 to 86% on LRA, 96% on Path-X (abstract).

**s5** (2208.04933v3, posted 2022-08-09; abstract only). Parent: s4 (stated: "We build on the design of the
S4 layer"). One MIMO SSM instead of many SISO SSMs, computed with parallel scans. Headline: LRA average 87.4%,
Path-X 98.5% (abstract). The abstract does not cite the parallel-scan paper, so no parscan link.

**h3** (2212.14052v3, posted 2022-12-28). Parents: s4d (its SSM_diag), linattn (Sec 2: linear attention
"inspired our model design", `src/sections/background.tex`). Shift SSM x V gate, diagonal SSM store, Q
read-out. Headline: associative recall 99.8 vs S4D 86.0 (Table 2). Source: `notes/papers/h3.md`.

**mamba3** (2603.15569v1, posted 2026-03-16, ICLR 2026). Parent: mamba2 (Sec 1: "three core methodological
changes on top of Mamba-2"). (1) Exponential-trapezoidal discretization: Prop `prop:trap` gives
h_t = e^{Delta_t A_t} h_{t-1} + (1-lambda_t) Delta_t e^{Delta_t A_t} B_{t-1} x_{t-1} + lambda_t Delta_t B_t x_t; the paper
calls Mamba-1/2's rule "exponential-Euler" (Sec 3.1, `structure/methods_trap.tex`). With B, C biases it
replaces the short causal conv (Sec 4.1; Sec 4.2, Table 5a). (2) Complex-valued state, equivalent to a data-dependent
RoPE (Sec 3.2); it solves parity and modular arithmetic where Mamba-2 fails (Table 5b); Sec 3.2 cites the
Illusion of State as motivation. (3) MIMO: matrix-multiply state update, rank R=4 in the main runs (Sec 3.3).
1.5B, 100B FineWeb-Edu tokens (Table 3, column Average): Transformer 55.4, GDN 55.8, Mamba-2 55.7, Mamba-3 SISO
56.4, Mamba-3 MIMO 57.6; so +0.6 SISO vs GDN, +1.2 MIMO vs SISO, +1.8 MIMO vs GDN (abstract and Sec 1 state
these). Half-state claim has three wordings: abstract "comparable perplexity to Mamba-2 despite using half of
its predecessor's state size" (variant unstated); Sec 1 bullet "Mamba-3 (MIMO) with state size 64 matches the
perplexity of Mamba-2 with state size 128"; Sec 4.3 "Mamba-3 with 2x smaller state size achieves better
pretraining perplexity than its Mamba-2 counterpart". Evidence is Fig 3 only (440M models, 2x Chinchilla
tokens, d_state 16/32/64/128); no table gives the perplexities. MIMO raises decode FLOPs up to 4x at similar
latency (Sec 1). Sources: `sources/mamba3/src/structure/*.tex`, `tables/lm_harness.tex`, PDF table numbers.

## Linear attention line

**retnet** (2307.08621v4, posted 2023-07-17; abstract only). Parent: linattn, our reading (the abstract
derives "the connection between recurrence and attention" but names no prior model). Retention with parallel,
recurrent (O(1) per token) and chunkwise-recurrent forms. No number in the abstract.

**gla** (2312.06635v6, posted 2023-12-11; abstract only). Parent: linattn (stated: generalizes linear
attention "with data-dependent gates"). I/O-aware FlashLinearAttention kernel, faster than FlashAttention-2
even at 1K. Headline: trained on 2K, generalizes to "longer than 20K" (abstract). RetNet and Mamba are
baselines, not parents.

**based** (2402.18668v2, posted 2024-02-28). Parents: linattn, zoology (Sec 1: "Building on this analysis"
of Arora et al. 2023a = Zoology). Linear attention plus sliding-window attention; window and feature
dimension set the state size. Headline: 24x generation throughput vs FlashAttention-2 at 1024 tokens, 1.3b
(abstract; Sec 1). Source conflict: the recall gain over Mamba is "6.22" on the arXiv abstract page, "10.36"
in the v2 PDF abstract, Sec 1 and `src/sections/results.tex` (1.3b, 50b tokens), and "9.03" in the unused
`src/arxiv_main.tex`. Use 10.36 with its PDF location; the 6.22 claim is marked uncertain.

**deltanet** (2406.06484v6, posted 2024-06-10; abstract only). Parent: linattn (stated: replaces "the additive
update in linear transformers with the delta rule"). The paper parallelizes an existing model (DeltaNet,
Schlag et al., not a node) with a Householder-product representation. Headline: 1.3B model, 100B tokens,
beats Mamba and GLA in perplexity and zero-shot (abstract).

**gdn** (2412.06464v3, posted 2024-12-09, ICLR 2025). Parents: mamba2, deltanet (title; Sec 1). Gated delta
rule, Eq `eq:gated_delta_rule`: S_t = S_{t-1}(alpha_t (I - beta_t k_t k_t^T)) + beta_t v_t k_t^T. alpha_t -> 0
clears memory (Mamba-2's gate); alpha_t -> 1 is the pure delta rule, which replaces one key-value pair. Chunkwise
training extends DeltaNet's WY algorithm. Headline: average 55.32 vs Mamba2 54.89 vs DeltaNet 52.14 at 1.3B /
100B FineWeb-Edu tokens (Table 3, column Avg.); hybrid H1 (with SWA) 56.40. Source: `sources/gdn/src/main.tex`.

## RNN line

**parscan** (1709.04057v2, posted 2017-09-12, ICLR 2018). Root node. A linear recurrence is a scan, so
Blelloch's parallel scan computes it over length. Headline: "up to 9x" model speed-up (abstract; Table 2, SRU
9.21 at length 65,536). Mamba Sec 3.3 cites it for its scan, so mamba lists it as a parent. Source:
`notes/papers/parscan.md`.

**rwkv** (2305.13048v2, posted 2023-05-22; abstract only). Parent: linattn (stated: "leverages a linear
attention mechanism"). Trains like a Transformer, runs as an RNN. Headline: "14 billion" parameters (abstract).

**griffin** (2402.19427v1, posted 2024-02-29). Parent: lru (the RG-LRU is the LRU recurrence made real and
input-gated). Hawk = RG-LRU + gated MLP; Griffin adds local (sliding-window) MQA attention. Headline: Griffin-7B
average 65.8 on 300B tokens vs Llama-2-7B 65.3 on 2T (Table 1 p6). The token-ratio wording varies in the source
("over 6 times" in the abstract, "roughly 7 times" in Sec 1). Source: `notes/papers/griffin.md`.

**xlstm** (2405.04517v2, posted 2024-05-07; abstract only). No parent in the tree (LSTM lineage). Exponential
gating; sLSTM (scalar memory, memory mixing) and mLSTM (parallelizable matrix memory, covariance update). Our
reading, not drawn: mLSTM's matrix memory is close to gated linear attention. No number in the abstract.

**minrnn** (2410.01201v3, posted 2024-10-02; abstract only). No parent in the tree (LSTM/GRU lineage).
minLSTM and minGRU: fewer parameters, fully parallel training, competitive results. Our reading, not drawn:
the parallel training is a scan over a linear recurrence (parscan). No number in the abstract.

## Conv line

**hyena** (2302.10866v3, posted 2023-02-21; abstract only). Parent: h3, our reading (the abstract names no
parent model). Implicit long convolutions interleaved with data-controlled gating. Headline: 100x faster than
optimized attention at 64K (abstract); also "more than 50 points" over state-space operators on recall and
reasoning, and 20% less training compute to reach Transformer quality at 2K. Tension: Zoology (spine), which
names H3, Hyena, RWKV and RetNet as gated convolutions, finds them far behind attention on real-language recall.

## Vision line

**mambaout** (2405.07992v3, posted 2024-05-13). Parent: mamba (Gated CNN blocks, i.e. Mamba blocks without
the SSM, Sec 1). Headline: MambaOut-Small 84.1 top-1 vs LocalVMamba-S 83.7 (Table 1). The abstract says it
"surpasses all visual Mamba models", but MambaOut-Tiny 82.7 ties LocalVMamba-T 82.7 (Table 1). Detection and
segmentation trail VMamba (Tables 2-3). Source: `notes/papers/mambaout.md`.

**mlla** (2405.16605v2, posted 2024-05-26, NeurIPS 2024). Parents: mamba, linattn. Mamba as single-head linear
attention with six differences (input gate, forget gate, shortcut, no normalization, single head, block
design); forget gate and block design matter most (Table 1). The v2 text renames the model MILA (README:
formerly MLLA). Headline: MILA-B 85.3 top-1 (Table 3), trained with MESA; 85.0 without (Table 7). Source:
`notes/papers/mlla.md`; `sources/mlla/src/sec/5_exp.tex`.

**mambavision** (2407.08083v2, posted 2024-07-10, CVPR'25; abstract only). Parent: mamba (stated:
"redesigning the Mamba formulation"). Hierarchical backbone with self-attention blocks in the final layers.
The abstract claims SOTA top-1 and throughput but gives no number.

## Hybrid line

**zamba** (2405.16712v1, posted 2024-05-26; abstract only). Parent: mamba (stated: "a Mamba backbone with a
single shared attention module"). 7B, trained on 1T tokens of open data (abstract).

**samba** (2406.07522v3, posted 2024-06-11, ICLR 2025; abstract only). Parent: mamba (stated: "layer-wise
combines Mamba ... with Sliding Window Attention"). Up to 3.8B params, 3.2T tokens; perfect passkey recall at
256K after 4K fine-tuning. Headline: 3.73x throughput vs GQA Transformers on 128K prompts (abstract).

**jamba15** (2408.12570v1, posted 2024-08-22). Parent: jamba. Same block, scaled to 94B active / 398B total;
ExpertsInt8 fits 256K context on "a single machine with 8 80GB GPUs" (Sec 1). Headline: Large KV cache 9GB
at 256K, 16-bit (Table 1 p2). Keeps Mamba-1, not Mamba-2 (Fig 1). Source: `notes/papers/jamba15.md`;
`sources/jamba15/src/main.tex`.

**hymba** (2411.13676v1, posted 2024-11-20; abstract only). Parent: mamba, our reading (the abstract says
"SSMs" and never names Mamba). Attention heads and SSM heads in parallel, learnable meta tokens, cross-layer
KV sharing, partial SWA. Headline: Hymba-1.5B-Base vs Llama-3.2-3B: 11.67x smaller cache, 3.49x throughput,
+1.32% average accuracy (abstract).

**nemotronh** (2504.03624v4, posted 2025-04-04). Parents: mamba2 (its layers), nvidia-mamba (Sec 2.1: the 8%
attention share is "As suggested by prior work"). 8B: 4 attention layers of 52; 56B: 10 of 118 (Sec 2.1).
The "roughly 8%" counts FFN layers in the total; by sequence mixers it is 4 of 28 (8B). Headline: "up to 3x"
faster (abstract); 3x vs Llama-3.1-8B on long contexts (Sec 2.6). Source: `notes/papers/nemotronh.md`.

**falconh1** (2507.22448v1, posted 2025-07-30; abstract only). Parent: mamba2, our reading (the abstract says
"SSMs" and names Mamba only for earlier Falcon models). Parallel hybrid; 0.5B to 34B. Headline: Falcon-H1-34B
"matches or outperforms models up to 70B scale" (abstract). The abs.html title is longer than the tsv title
("... Redefining Efficiency and Performance").

## Analysis

**illusion** (2404.08819v3, posted 2024-04-12, ICML 2024). Parents: s4, mamba (the models it bounds).
Log-precision non-gated SSMs (S4, Theorem 4.2) and diagonal SSMs (S6/Mamba, Theorem 4.4) are in TC^0, like
transformers, so (if TC^0 != NC^1) they cannot express permutation composition (S_5 word problems). Fig 3:
S4 and Mamba need depth growing with length on group multiplication; RNNs and IDS4 do not. Mamba-3 later answers
this with complex-valued states. No headline number. Source: `sources/illusion/src/SSM-NoStateTracking.tex`,
PDF theorem and figure numbers.

## Outline feedback (ch 12)
- The tree line "linear attention -> RetNet/GLA/DeltaNet -> Gated DeltaNet" holds for GLA and DeltaNet
  (stated); the RetNet link is our reading. Gated DeltaNet's parents are Mamba-2 and DeltaNet, not GLA.
- "What came next" should quote Mamba-3's 1.5B numbers from Table 3. Its half-state claim is figure-only and
  worded three ways; quote the Sec 1 MIMO version (d_state 64 vs 128) with "Fig 3".
- Mamba-3 compares against GDN as the strongest baseline, and its intro names Mamba-2 and GDN as the layers
  inside production hybrids (Nemotron-H, Qwen3, Kimi Linear). The two "next" papers form a pair, not a sequence.
- Based: do not quote the arXiv-page 6.22; the paper says 10.36.
- MLLA: label the node "MLLA (MILA)"; v2 uses MILA throughout.
