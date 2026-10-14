# An Empirical Study of Mamba-based Language Models (recall group, serves ch 10 and ch 11)

## 1. Identity
- Title: "An Empirical Study of Mamba-based Language Models".
- arXiv 2406.07887, version read: v1 (submitted 2024-06-12, abs.html; only version listed).
- Authors: Roger Waleffe (first) ... Bryan Catanzaro (last); 16 authors (abs.html, PDF p.1). NVIDIA, plus
  Tri Dao (Princeton, Together AI) and Albert Gu (CMU, Cartesia AI).
- Venue: technical report (none stated).
- Code: NVIDIA Megatron-LM; frozen snapshot https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba
  (abstract footnote). README at `sources/nvidia-mamba/README.md` (current `examples/mamba` README).
- Weights: Mamba-2 8B and Mamba-2-Hybrid 8B on Hugging Face (Sec 1; README).
- Sources read: `src/sections/{abstract, introduction, preliminaries, pure_models, hybrid_models,
  conclusion (holds Related Work + Conclusion), appendix_a}.tex`; `paper.pdf` pp. 8-15 for figures.

## 2. Core problem
Earlier SSM-vs-Transformer comparisons at matched data were small (< 3B params, < 1T tokens). The report
trains 8B Mamba, Mamba-2, Transformer and a Mamba-2-Hybrid on the same data with the same hyperparameters,
up to 3.5T tokens, and asks where pure SSMs match and where they fall short.

## 3. What prior methods did and why that hurt (per the paper)
- Mamba / Mamba-2 papers: match Transformers up to 2.8B params; small-scale ICL probes (Park et al.).
- Jamba, Zamba: hybrids beyond 7B, but no controlled comparison with an equivalent Transformer (same data,
  params). Jamba's controlled ablations were 1.3B/250B tokens or 7B/50B tokens (Related Work).
- Griffin: local (windowed) attention degrades beyond the window (Sec 1).

## 4. The one key idea
Hold everything fixed except the sequence mixer, at 8B scale and trillions of tokens. Pure SSMs match on
most tasks but lose on in-context learning and copying (5-shot MMLU, Phonebook). Putting back a few
attention layers (4 of 56) recovers this and beats the Transformer on all 12 standard tasks.

## 5. Mechanism, step by step
1. Implement Mamba and Mamba-2 in Megatron-LM with tensor, sequence and (Mamba-2 only) pipeline
   parallelism. Mamba-2 needs one all-reduce per layer (Mamba needs two), using GroupNorm with group size
   > 256 (Sec 2.1, Fig 1).
2. Train Transformer, Mamba, Mamba-2 at 8B on 1.1T tokens; Transformer and Mamba-2 on 3.5T tokens
   (pure Mamba skipped at 3.5T: "almost 3x slower than the pure Mamba-2 model due to the large state
   dimension", Sec 3.3.1).
3. Diagnose MMLU (three prompt formats, Table 4) and Phonebook (standard, reversed; Fig 2b, Fig 3).
4. Ablate hybrids at 130M (24 layers) and 840M: attention fraction, MLP fraction, RoPE, ordering, GQA
   (Sec 4.1, Fig 4, Table 5).
5. Train Mamba-2-Hybrid 8B (24 Mamba-2 + 4 attention + 28 MLP layers) on the same data and hyperparameters
   (Sec 4.2, Table 6).
6. Extend hybrid and Transformer to 16K and 32K (and hybrid to 128K) with 50B tokens of continued
   pretraining; evaluate standard tasks, 9 natural long-context tasks, 13 RULER tasks, Phonebook.

## 6. Equations
The report has no model equations. Its one algorithm is the layer-allocation procedure (Algorithm 1,
App A): place round(L x attention_ratio) attention layers so the Mamba runs between them are as equal as
possible (starting and ending with Mamba), then spread round(L x mlp_ratio) MLP layers over the remaining
Mamba slots, biased away from the start. Gloss: it generates the hybrid pattern string below from
(56, 0.08, 0.50).

## 7. Config / setup table (controlled setup; architecture lens for the hybrid)

| | Transformer | Mamba | Mamba-2 | Mamba-2-Hybrid |
|---|---|---|---|---|
| Params (B) | 8.53 | 8.15 | 8.24 | 8.66 |
| Layers | 32 (attn + MLP each) | 56 | 56 | 56 (24 M2, 4 attn, 28 MLP) |
| Model dim | 4096 | 4096 | 4096 | 4096 |
| Attention | 32 heads, 128 KV channels, MHA | - | - | 32 heads, GQA 8 groups, 128 KV channels |
| State dim N | - | 128 | 128 | 128 |
| Groups | - | - | 8 | 8 |
| Mamba-2 head dim / expand / conv | - | - | 64 / 2 / 4 | 64 / 2 / 4 |
| MLP | 4x, SwiGLU | - | - | 4x, GELU |
| Norm | LayerNorm | RMSNorm | RMSNorm (GroupNorm inside block) | RMSNorm pre-norm |
| Pos. emb. | RoPE | None | None | None |
| Seq. len (base) | 4096 | 4096 | 4096 | 4096 |
Sources: Table 1 (p.5), Sec 3.1, Table 6 (p.10), Sec 4.2. All: no biases, no dropout, untied embeddings.

Hybrid layer pattern (Table 6, M = Mamba-2, * = self-attention, + = MLP), verbatim:
`M+M+M++M+M*+M+M+M+M++M*+M+M+M+M+M*++M+M+M+M+M*+M++M+M+M+`
- 56 symbols: 24 M, 4 *, 28 + (our count; matches Sec 4.2 "4 (7.1%) self-attention layers, 24 (42.9%)
  Mamba-2 layers, and 28 (50%) MLP layers"; abstract rounds to 43% / 7% / 50%).
- Attention at 0-indexed positions 10, 22, 33, 45 (our reading).
- README writes the same pattern with `-` for MLP and says "56-layer model with 4 attention layers, 28 MLP
  layers, and 24 Mamba layers" (code-only confirmation).

Training (Sec 3.2): data 1.1T and 3.5T token sets (predecessors of the Nemotron-4 data; 70% English, 15%
non-English, 15% code); vocab 256K SentencePiece. 1.1T: batch 256, LR 1e-4 peak / 1e-5 min. 3.5T: batch
1024, LR 3e-4 / 3e-5. Warmup 122K samples, cosine, weight decay 0.1, Adam beta1 0.9 / beta2 0.95, BF16. Same
hyperparameters for all models. MFU on 1024 H100 (TP 4, DP 256, micro batch 4, global 1024): hybrid 29.9%,
Transformer 30.7% (Sec 4.2).

Long-context extension (Sec 4.3.2): continued pretraining 50B tokens on the same 3.5T data (packed
sequences), LR linear warmup over 1.7B tokens then cosine, max 3e-5 / min 3e-6; full global attention in the
4 attention layers; Transformer uses dynamic NTK RoPE scaling. Note the source writes "16386 (16K)"
(sic; 16384 elsewhere).

Measurement (Sec 2.3): LM Evaluation Harness commit 94cc1850 for 12 short tasks (WinoGrande, PIQA,
HellaSwag, ARC-E, ARC-C, MMLU, OpenBookQA, TruthfulQA, PubMedQA, RACE, NQ, SQuADv2); LongBench commit
48798083; RULER 13 tasks at 400 samples each; Phonebook from Jelassi et al. (Repeat After Me), accuracy
averaged over 20 random initializations per length.

Falsifier for the headline: a matched-data pure Mamba-2 reaching Transformer Phonebook accuracy to 4K, or
the hybrid losing to the Transformer on any of the 12 tasks at 3.5T.

## 8. Results worth remembering
Pure SSMs, 1.1T tokens (Table 2, p.6):
- MMLU 5-shot: Transformer 46.28, Mamba 28.00, Mamba-2 29.19. MMLU 0-shot: 38.32, 28.63, 28.94.
- Avg w/o MMLU: Transformer 67.87, Mamba 68.07, Mamba-2 68.56 (SSMs ahead on the 5 commonsense tasks).
- Gap wording conflicts: Sec 3.3.1 says "Mamba-2 five-shot accuracy is 17 points lower" (46.28 - 29.19 =
  17.09); Sec 1 says "both Mamba and Mamba-2 models produce nearly 15 points lower accuracy". Not reconciled.
- Table 4 (p.7): MMLU cloze 0-shot: Transformer 37.26, Mamba 38.26, Mamba-2 37.68 (SSMs match or beat), so
  "pure SSM models contain the same knowledge as the Transformer, but ... require substantially more
  training to understand the format of the multiple choice questions" (Sec 3.3.2).
Pure SSMs, 3.5T tokens (Table 3, p.6):
- MMLU 5-shot: Transformer 50.07, Mamba-2 48.7 (gap 1.37, stated "closing the gap ... to just 1.37 points").
- MMLU 0-shot: Transformer 45.69, Mamba-2 47.25 (Mamba-2 ahead).
- 0-to-5-shot gain: 1.45 vs 4.38 for the Transformer (Sec 3.3.2). The text attributes 1.45 to "the Mamba-2
  hybrid model"; the numbers match pure Mamba-2 in Table 3 (47.25 -> 48.7). Our reading: wording slip.
Phonebook, pure SSMs (Fig 3, p.8; Fig 7a, p.14):
- 1.1T: Transformer near 100% up to its 4096 context; Mamba and Mamba-2 "begin to respond incorrectly for
  input sequence lengths beyond approximately 500 tokens" (Sec 3.3.3).
- Fuzzy memory (Fig 3b): SSMs output numbers sharing several digits with the right one.
- Reversed prompt (Fig 3c): SSMs improve but still degrade below 4096.
- 3.5T Mamba-2: "unable to accurately predict the required phone numbers for sequences >1000 tokens"
  (Sec 4.3.2).
Hybrid design (Sec 4.1):
- Fig 4 (p.9): 130M, 24 layers: validation loss minimized "when roughly 8% of the layers are self-attention
  layers"; confirmed at 840M. 30%-50% MLP layers without loss increase; 50% MLP trains 20% faster than 5%.
- Table 5: 840M hybrid, 16K extension, no RoPE avg 53.43 vs RoPE 52.61 (10K base) and 51.52 (500K base).
- GQA instead of MHA: validation perplexity up about 0.04%.
Hybrid 8B, 3.5T (Table 7, p.11):
- Avg over the 12 tasks: hybrid 55.82, Mamba-2 54.69, Transformer 53.17; "+2.65 points on average"
  (abstract, Sec 4.3.1). Our check: the Avg column is the mean of 13 columns (MMLU counted twice, 0- and
  5-shot); 55.8177 - 53.1692 = 2.65. Hybrid beats the Transformer on every column.
- MMLU 5-shot 53.60 vs 50.07 ("3.5 points higher", Sec 1); 0-shot 51.46 vs 45.69.
- Fig 6 (p.11): hybrid matches Transformer 5-shot MMLU at 1.5T tokens, then leads.
Inference (Fig 5, p.11):
- "Predicted speedup to generate one token" for 8B hybrid vs 8B Transformer, batch size 32, x-axis input
  context 0-128K. "For long context lengths ... generates the content nearly 8x faster" (Sec 4.2); abstract
  "predicted to be up to 8x faster". Visual read: the last plotted point (~128K) is a bit above 7x.
  Predicted, not measured.
- Separate 8x: "the SSM scan used by Mamba-2 is up to 8x faster than that of Mamba" (Sec 4.2), cited from the
  Mamba-2 paper. Do not merge the two.
Long context (Tables 8-10, pp.12-14; Figs 7-8):
- Standard tasks avg: 16K hybrid 56.27 vs Transformer 53.84; 32K 56.4 vs 53.50 (Table 8).
- Natural long-context avg (9 tasks, Table 9): 4K hybrid 40.83 vs Transformer 40.66 vs Mamba-2 34.31; 16K
  hybrid 44.6 vs 45.24; 32K hybrid 44.16 vs 45.57 (Transformer ~1 point ahead, from multi-doc QA, e.g.
  HotpotQA 16K 42.17 vs 48.61).
- RULER avg (Table 10): 4K hybrid 81.99 vs Transformer 77.62 vs Mamba-2 52.14; 16K 74.19 vs 62.33; 32K
  69.93 vs 56.49. Prose says "The 16K hybrid model achieves 13 points higher average accuracy"; Table 10 gives
  11.86 at 16K (13.44 at 32K). Discrepancy, not reconciled.
- Mamba-2 (4K) on RULER multi-key NIAH: MK-NIAH-2 13.75, MK-NIAH-3 5.5 vs Transformer 94, 58 (Table 10).
- Phonebook: 4K hybrid "100 percent accuracy on Phonebook up to 5.5K tokens" (Fig 7a); 16K and 32K hybrids
  perfect beyond trained length, Transformers degrade as length nears their context (Fig 7b, 7c); 128K
  hybrid (Fig 8, p.15) perfect "up to and beyond the sequence length it was trained on"; Sec 1: "even when
  the phone book contains more than 150K tokens".

## 9. Limitations the paper states
- Hybrids trail on LongBench multi-document QA; hypothesis: SSM states confused by irrelevant packed
  documents; hybrids more prompt-sensitive (Musique 4K hybrid range [10.63, 16.16] vs Transformer
  [15.25, 17.68]).
- Inference speedup is predicted, not measured; KV-cache batch benefits only "expected".
- No instruction-tuned comparison.
- Later in our sources: Nemotron-H (2504.03624) scales the same hybrid recipe (see hybrid notes).

## 10. Outline feedback
- Ch 10 claim card: the recall loss is at matched data and hyperparameters, 8B, up to 3.5T tokens. Two
  clean numbers: pure Mamba-2 fails Phonebook beyond ~1000 tokens even at 3.5T; MMLU 5-shot gap shrinks from
  17.09 (1.1T) to 1.37 (3.5T), so MMLU is a training-length effect, Phonebook is not.
- Ch 10 intuition "phone number seen 10K tokens ago": replace with the sourced ~500 tokens (1.1T) /
  > 1000 tokens (3.5T) for pure SSMs.
- Ch 11: the "a little attention" number is Fig 4 (loss minimum near 8% attention at 130M) and the 8B
  hybrid's 4 of 56 layers (7.1%). The 8x inference figure must carry "predicted, batch 32".
- Ch 11 `viz-stack` could also use this config. Our reading (not in the paper): KV cache per token, BF16,
  Transformer 32 layers x 2 x 4096 x 2 B = 512 KiB; hybrid 4 layers x 2 x (8 x 128) x 2 B = 16 KiB (GQA 8
  groups, 128 channels). A 32x reduction; label as computed.
- Mamba-2 here uses state size N = 128 with 8 groups (head dim 64). The ch 8 walkthrough's "N 64-256" range
  can cite this as one production config inside that range.
