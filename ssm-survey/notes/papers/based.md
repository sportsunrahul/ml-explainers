# Based: recall-throughput tradeoff (recall group branch, serves ch 10 and the phonebook toy)

## 1. Identity
- Title: "Simple linear attention language models balance the recall-throughput tradeoff".
- arXiv 2402.18668, version read: v2 (2025-03-07, abs.html; PDF stamp "arXiv:2402.18668v2 7 Mar 2025").
  v1 2024-02-28.
- Authors: Simran Arora (first; equal contribution with Sabri Eyuboglu, Michael Zhang) ... Christopher Re
  (last). Stanford, University at Buffalo.
- Venue: ICML 2024 (`\usepackage[accepted]{icml2024}`, `src/main.tex` l.34).
- Code: https://github.com/HazyResearch/based; MQAR scripts in zoology repo.
- Sources read: `src/main.tex` (matches v2 PDF: "10.36"); `src/arxiv_main.tex` is the v1 text ("9.03") and
  was NOT used for numbers. `sections/intro.tex`, `related_work.tex` (Sec 3), `appendix/mqar_lower/space.tex`,
  `appendix/experiment-details.tex` (App E.1), `tables/lm-quality.tex`; `paper.pdf` pp. 1, 4, 8.

## 2. Core problem
Attention recalls well but its KV cache grows with length. Fixed-state models (H3, Mamba, RWKV) are
fast but recall worse. Is there a fundamental trade-off between state size and recall, and can a simple
model sit on its Pareto frontier?

## 3. What prior methods did and why that hurt (per the paper)
- Linear attention alone lacks precision for local shifts and comparisons; sliding window attention (SWA)
  limits recall range to the window (Sec 1).
- Gated convolutions (H3, Hyena) fall well below the frontier (Sec 3.1).

## 4. The one key idea
Plot MQAR accuracy against recurrent state size in bytes for many architectures and sizes: recall rises with
state size within every class, and attention (state grows with N) is the only one at full recall. Based
mixes Taylor-feature linear attention (global, fixed state) with small-window SWA (local precision) to
move along that frontier.

## 5. Mechanism, step by step (only what ch 10 needs)
1. MQAR training on length 256 with 4-64 KV pairs; evaluation on length 1,024 with 4-256 KV pairs (Sec 3.1).
2. Six mixers: attention, SWA, Mamba, H3, Hyena, Based; vary state-size hyperparameters (d, d_state, window,
   feature dim).
3. State size computed per App E.1 formulas (below).
4. Theorem 3.1: any causal recurrent model needs Omega(N) bits of state to solve MQAR.

## 6. Equations
**Theorem 3.1 (Sec 3.2, p.4), verbatim:**
```
Any recurrent model\footnote{For Mamba~\cite{gu2023mamba}, see \Cref{cor: mamba-ar}.} depending causally on input $\vu \in \{0,1\}^{N \times d}$ requires $\Omega(N)$-bits\footnote{Here, we need the entries of the state to be bounded.} in state size to solve $\Task$.
```
Gloss: N = sequence length (number of stored pairs scales with it); state entries bounded. Same index-problem
argument as Zoology Cor H.24, now for any recurrent model (Theorem F.12 in the appendix).

**Corollary F.13 (App F, `space.tex`), verbatim:**
```
Given an input $\vu \in \R^{\inputLength \times \modelDim}$ to the AR problem, a causal \Mamba\ model with all entries in its computation taking $O(1)$ bits needs $\modelDim + \stateDim \ge \Omega(N)$ to solve AR.
```
Gloss: d = model dim, d_state = N_state in page notation. The proof counts the carried state as
`3 d_state + d` numbers (h, plus carried Delta, B, C parts). Our reading: this per-channel accounting is
smaller than the d x N state of a real Mamba layer; treat it as the paper's bound, not as Mamba's state size.

**Recurrent state sizes used for Fig 2 (App E.1), verbatim:**
- Based: `\text{sizeof}(\bm{s}_i) = (d + 1) \times \tilde{d}`; with Taylor map after projecting to d':
  `(d + 1) \times (1 + \frac{3d'}{2} + \frac{d'^2}{2})`. The +1 is the K-state for the denominator.
- Attention: `\text{sizeof}(\bm{s}_i) = 2 \times d \times N` (keys and values).
- Sliding window: `2 \times d \times \min(N, k_\text{sliding})`.
- Mamba: `2 \times d \times d_\text{state}` ("The 2 ... accounts for the expansion in the Mamba block";
  the preceding sentence says "number of heads h", a copy-paste slip in the source).
- H3: `d \times d_\text{state}`; Hyena: `d \times N`.
Gloss for the toy: linear attention's state is (d + 1) x feature-dim numbers per layer; attention's KV
cache is 2 d N.

## 7. Measurement setup (analysis lens)
- Synthetic sweeps: Based d in {48, 64, 128}, d' in {8, 16, 24}; attention d in {64, 128}; SWA d = 128,
  window in {8, ..., 1024}; Mamba d in {64, 128, 256}, d_state in {8, 16, 24}; H3 d in {64, 128, 256},
  d_state = d/4; Hyena d in {64, 128, 256} (App E.1).
- Falsifier: an architecture at small fixed state reaching attention-level MQAR accuracy at length 1,024
  with 256 pairs.

## 8. Results worth remembering
- Fig 2 (p.4): x = recurrent state size (bytes, log scale about 2^14 to 2^22), y = MQAR accuracy. "Within
  each architecture class ... increasing the recurrent state size almost always leads to an improvement in
  accuracy"; "Mamba makes the best use of a limited memory budget" among prior work; Hyena and H3 "fall
  well below the Pareto frontier"; Based "expands the Pareto-frontier beyond Mamba" (Sec 3.1).
- Recall-intensive tasks: Based beats Mamba by 10.36 accuracy points (v2 abstract, Sec 1, Sec 6). Our
  arithmetic on Table 1's three recall columns (SWDE, FDA, SQuAD) at 1.3b/50b gives 9.76, so 10.36 may come
  from a wider task set; source not located. v1 abstract said 9.03 (matches 1.3b/10b in Table 1).
- Attention vs Mamba on recall-intensive tasks: "32.2 accuracy points (Table 1)" (Sec 1); not reproduced by
  simple averaging of Table 1 either.
- Throughput: "24x higher throughput on language generation than FlashAttention-2, when generating 1024
  tokens using 1.3b parameter models" (abstract); batch size 128, single H100 (Sec 6).
- Table 1 example (1.3b/50b): SWDE 76.50 (Transformer++), 64.45 (Based), 52.75 (Mamba); FDA 80.47, 30.40,
  18.51 (p.8).

## 9. Limitations the paper states
- Not stated as a limitation, but our reading of Table 1: Based (~20% linear attention, ~20% SWA, ~60% gated
  convolution layers, Sec 6) still trails Transformer++ on recall-intensive tasks (FDA 30.40 vs 80.47 at
  1.3b/50b).

## 10. Outline feedback
- Fig 2 is the best single picture of "recall grows with state size" for ch 10; it includes Mamba, which
  Zoology's figures do not.
- No paper in this group gives "K pairs storable in a state of size N". Use Theorem 3.1 / Cor F.13 and
  Repeat's Cor 2.8 as the closest statements beside `viz-phonebook`; the App E.1 formulas give the
  state-size axis.
