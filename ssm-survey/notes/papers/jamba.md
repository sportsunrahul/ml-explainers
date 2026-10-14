# Jamba (hybrid group, serves ch 11)

## 1. Identity
- Title: "Jamba: A Hybrid Transformer-Mamba Language Model".
- arXiv 2403.19887, version read: v2 (2024-07-03). v1 submitted 2024-03-28 (abs.html submission history).
- Authors: Opher Lieber (first, equal contribution with Barak Lenz) ... Yoav Shoham (last). AI21 Labs (logo on title page).
- Venue: none stated (preprint, NeurIPS 2023 style file with `[preprint]`).
- Weights: https://huggingface.co/ai21labs/Jamba-v0.1 (Apache 2.0). Model card at `sources/jamba/README.md`.
- Sources read: `src/main.tex` in full, figure PDFs/PNGs in `src/figures/` (rendered), `paper.pdf` for numbering, `README.md`.

## 2. Core problem
Transformers have a KV cache that grows with context and recompute over the whole context per generated
token, so long contexts are memory- and throughput-bound. Pure SSMs (Mamba) are efficient but "still lag
behind the performance of comparably sized Transformer language models" (Sec 1). Goal: a production-scale
model that keeps Transformer quality with a small KV cache and high long-context throughput.

## 3. Prior methods and why they hurt (per the paper, Sec 1)
- Transformer: KV cache limits long context; no single summary state -> slow inference, low throughput.
- RNNs: cannot parallelize training over time; weak on long-distance relationships.
- Mamba-style SSMs: efficient, but lag same-size Transformers.
- Earlier attention/SSM mixes: small scale or simple tasks (S4 + local attention; block-state; Mamba's own
  interleaving "only slightly better than pure Mamba" up to 1.3B; H3 hybrid up to 2.7B/400B tokens lags pure
  Mamba; StripedHyena 7B lags Mistral-7B). Jamba claims "the first production-grade Attention-SSM hybrid".

## 4. The one key idea
Interleave a few attention layers among many Mamba layers (1 attention : 7 Mamba), and put MoE in every
other MLP. Attention supplies in-context learning (induction-like heads); Mamba supplies cheap sequence
mixing and implicit position; MoE adds capacity without adding active parameters. Result: KV cache 8x
smaller than a vanilla Transformer of the same depth.

## 5. Mechanism, step by step
1. A "layer" = (RMSNorm -> Attention or Mamba -> residual) then (RMSNorm -> MLP or MoE -> residual)
   (Fig 1b). Four layer types drawn: Transformer, Attention-MoE (drawn but NOT used, footnote Sec 2),
   Mamba, Mamba-MoE.
2. A Jamba block = l layers mixed at a:m (a attention per m Mamba); MoE replaces the MLP every e layers;
   MoE has n experts, router picks top K per token (Sec 2).
3. Released config (Sec 3.1): 4 Jamba blocks; l = 8, a:m = 1:7, e = 2, n = 16, K = 2.
4. Layer order inside one block, Fig 1a read bottom (input) to top: Mamba, Mamba-MoE, Mamba, Mamba-MoE,
   Transformer, Mamba-MoE, Mamba, Mamba-MoE. So attention is the 5th layer of each block (1-indexed),
   with a plain MLP; MoE sits on the 2nd/4th/6th/8th layers (1-indexed). Sec 6.2 names the hybrid
   1.3B ablation model's attention layers "4, 12, 20", which matches 0-indexed position 4 per 8-layer
   block (our reading).
5. Counts for the released model (our reading from 4 x 8 and Fig 1a): 32 layers; 4 attention layers
   (stated explicitly in Sec 5.2.1: "uses only 4 attention layers"); 28 Mamba layers; 16 MoE layers and
   16 dense-MLP layers. Sec 3.1: n and e "balanced to have an average of ~8 experts per layer".
6. Mamba layers: the paper cites Mamba (Gu and Dao, arXiv 2312.00752) only. It does not name a version;
   Mamba-2 (2405.21060) postdates Jamba v1. Read as Mamba (v1) by citation.
7. Added normalization: RMSNorm on internal activations of the Mamba layers (Sec 2, Sec 6.4). Why: at the
   7B-based (12B/52B) scale, "inner parts of the Mamba layers suffer from large activation values, leading
   to the spikes"; up to 1.3B training was stable (Sec 6.4, Fig 9). The paper does not say exactly where
   inside the Mamba layer the norms go.
8. No positional encoding at all (no RoPE): Mamba layers placed before attention "provide implicit position
   information" (Sec 6.5, Table 8).
9. Other details (Sec 2): GQA in attention, SwiGLU, MoE load balancing, vocab 64K, BPE tokenizer with each
   digit a separate token, dummy space removed.
10. What the paper does NOT give: d_model, number of attention heads, number of KV heads, head dim, MLP
    width, Mamba state size N / expansion E / conv width, training token count for the released model,
    optimizer, LR schedule, batch. README does not give them either. "7B-based" (Sec 1) is the only size
    hint.

## 6. Equations
The paper has no numbered equations. The design space is a list (Sec 2, verbatim from main.tex):
```
\item $l$: The number of layers.
\item $a:m$: ratio of attention-to-Mamba layers.
\item $e$: how often to use MoE instead of a single MLP.
\item $n$: total number of experts per layer.
\item $K$: number of top experts used at each token.
```
Gloss: l = layers per Jamba block (8); a:m = attention layers per Mamba layers (1:7); e = MoE period in
layers (2 = every other layer); n = experts per MoE layer (16); K = experts active per token (2).
Note the list says "number of layers" for l; Sec 3.1 and Fig 1 make clear it is per block.

KV-cache formula (our reading; not in the paper):
`KV_bytes = 2 (K and V) x n_attn_layers x d_kv x L x bytes_per_value`,
where d_kv = n_kv_heads x head_dim (channels per K or V per attention layer), L = context tokens,
bytes_per_value = 2 for 16-bit. The Mamba layers add a fixed-size state independent of L; the paper gives
no state size, so that term cannot be computed from this source.

### KV-cache check against Table 1 (our reading)
The paper gives no head config, so d_kv must be back-solved, then cross-checked.
- Table 1 Jamba: 4GB at 256K context, 16-bit. Take 256K = 262144 tokens and GB = GiB (2^30 bytes):
  4 x 2^30 / 262144 = 16384 bytes/token = 16 KiB/token.
  Per attention layer: 16384 / 4 layers = 4096 bytes/token = 4 KiB/token.
  d_kv = 4096 / (2 for K,V x 2 bytes) = 1024 channels per attention layer.
- The fit is exact only with binary units. With GB = 10^9 bytes, 4e9 / 262144 / 16 = 953.7, not an integer
  number of channels. So Table 1's "GB" is read as GiB.
- Cross-check 1 (sourced): Sec 2 says the architecture aims at "an 8x smaller KV cache compared to a vanilla
  Transformer". 1:7 means 4 attention layers in 32, i.e. 1/8 of the layers. Table 1: Mixtral 32GB vs Jamba
  4GB = 8x. Consistent.
- Cross-check 2: 4 GiB / 4 attention layers = 1 GiB per layer = Mixtral's 32 GiB / 32 layers (Mixtral's
  32 layers is not stated in this paper). So per attention layer, Jamba's footprint equals Mixtral's/
  Mistral's: 4 KiB/token. 32 such layers = 128 KiB/token, the figure the outline's transformers-page
  config uses.
- Llama-2 row is Llama-2 7B (6.7B params, MHA). 128GB = 4x Mixtral's 32GB. Its config is not in this
  source, so this row is a listed value, not a check.
- Any split of d_kv = 1024 into heads x head_dim (e.g. 8 KV heads x 128) is an inference beyond the paper.
- Worked values (our reading, bf16, 16 KiB/token): 128K = 131072 tokens -> 2 GiB; 256K -> 4 GiB.
  Attention-fraction slider (our reading): KV bytes/token = n_attn_layers x 4 KiB, for 32 layers total.

## 7. Config / setup table
| Item | Value | Location |
|---|---|---|
| Jamba blocks | 4 | Sec 3.1 |
| Layers per block l | 8 | Sec 3.1, Fig 1 caption |
| Total layers | 32 | our reading (4 x 8) |
| Attention : Mamba a:m | 1:7 | Sec 3.1 |
| Attention layers | 4 | Sec 5.2.1 ("only 4 attention layers") |
| Mamba layers | 28 | our reading |
| MoE every e layers | 2 | Sec 3.1 |
| Experts n / top K | 16 / 2 | Sec 3.1 |
| MoE layers | 16 | our reading (Fig 1a: 4 per block) |
| Avg experts per layer | ~8 | Sec 3.1 |
| Total / active params | 52B / 12B | Table 1, abstract, Sec 1 |
| "7B-based" | stated | Sec 1, Sec 6.4 |
| Attention | GQA (heads not given) | Sec 2 |
| MLP activation | SwiGLU | Sec 2 |
| Positional encoding | none (no RoPE) | Sec 2, Sec 6.5 |
| Norm | RMSNorm pre-norm (Fig 1b) + RMSNorm inside Mamba | Fig 1b, Sec 2, Sec 6.4 |
| Mamba version | Mamba (2312.00752), version not named | Sec 2 citation |
| Vocab | 64K, BPE, digits split | Sec 2 |
| Context supported / trained | 256K / up to 1M | Sec 3.1, Sec 5.2 |
| KV channels per attention layer | 1024 | our reading (back-solved from Table 1) |
| d_model, heads, KV heads, N, E | not stated | -- |
| Training hardware | NVIDIA H100 | Sec 4 |
| Training framework | in-house; FSDP, tensor, sequence, expert parallelism | Sec 4 |
| Data | in-house web, books, code; last update March 2024 | Sec 4 |
| Released-model training tokens | not stated | -- |
| Ablation scales | up to 7B params, up to 250B tokens | Sec 1 |
| Knowledge cutoff | March 5, 2024 | README (code-only) |

## 8. Results worth remembering
KV cache and size (Table 1, p4; "KV cache (256K context, 16bit)"):
| Model | Available | Active | KV cache |
|---|---|---|---|
| LLAMA-2 (7B) | 6.7B | 6.7B | 128GB |
| Mistral | 7.2B | 7.2B | 32GB |
| Mixtral | 46.7B | 12.9B | 32GB |
| Jamba | 52B | 12B | 4GB |

Single 80GB GPU (four phrasings; do not merge):
- Abstract: "fits in a single 80GB GPU" (no length).
- Sec 1: "fits in a single GPU (with 8bit weights) even with contexts of over 128K tokens, which is
  impossible with similar-size attention-only models such as Mixtral-8x7B".
- Sec 3.1: experts chosen "to fit in a single 80GB GPU (with int8 weights)"; Fig 2 (single A100 80GB):
  "Jamba enables 2x the context length of Mixtral and 7x that of Llama-2-70B". Fig 2 bars (read off,
  not stated in text): Jamba 140K, Mixtral 8x7B ~64K, Llama-2 70B ~16K.
- Sec 7 (Conclusion): "fitting in a single 80GB GPU even when processing 140K-token texts".
- README: "Using 8-bit precision, it is possible to fit up to 140K sequence lengths on a single 80GB GPU";
  recommends excluding Mamba blocks from int8 quantization; in BF16 the model is "to[o] big to fit on a
  single 80GB GPU" (code-only).

Throughput (Sec 3.2, Fig 3; footnote: end-to-end encoding+decoding, "should be taken relatively rather than
absolutely, as they are without possible optimizations"):
- Fig 3a: varying batch, single A100 80GB, int8, 8K context, 512 output tokens. Jamba 3x Mixtral throughput;
  Mixtral "does not fit with a batch of 16". From the plot (read off): Llama-2 70B only at batch 1-2,
  Llama-2 13B up to 4, Mixtral up to 8, Jamba up to 16.
- Fig 3b: batch 1, 4 A100 GPUs, no quantization, context 1K..128K, 512 output tokens. "at small context
  lengths all models have a similar throughput"; at 128K Jamba is 3x Mixtral; Llama-2-70B does not fit at
  128K (its line stops at 64K in the plot).
- Abstract/Sec 5.1: "up to 3x improvement" over Llama-2 70B and Mixtral.

Needle-in-a-haystack (Sec 5.2.1, Fig 4): contexts 2K, 4K, 8K, 32K, 128K, 256K x depth 0-100%. Text:
"excellent performance ... noteworthy especially given that our implementation of Jamba uses only 4
attention layers". Read off the heatmap (not stated): near 1.0 almost everywhere; weakest cells at 256K,
depth 0-10%, roughly 0.6-0.7.

Long-context ICL classification (Sec 5.2.2, Fig 5): Trec-Fine (50 labels), NLU Intent (68), Banking77 (77),
CLINC150 (150), few-shot examples up to 128K context. Jamba > Mixtral on Trec-Fine and Banking77; on par on
NLU Intent and CLINC150.

Long-context QA, L-Eval in 3-shot format, F1 (Table 3, p9): Mixtral 0.42/0.46/0.29/0.58/0.42 avg 0.43;
Jamba 0.44/0.44/0.30/0.60/0.40 avg 0.44 (LongFQA/CUAD/NarrativeQA/NQ/SFiction). Avg input 6K-62K tokens.

Standard benchmarks (Table 2, p7):
| Model | HellaSwag | WinoGrande | ARC-E | ARC-C | PIQA | NQ | TruthfulQA | BoolQ | QuAC | GSM8K | HumanEval | MMLU | BBH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-2 13B | 80.7 | 72.8 | 77.3 | 59.4 | 80.5 | 37.7 | 37.4 | 81.7 | 42.7 | 34.7 | 18.3 | 54.8 | 39.4 |
| Llama-2 70B | 85.3 | 80.2 | 80.2 | 67.3 | 82.8 | 46.9 | 44.9 | 85 | 42.4 | 55.3 | 29.9 | 69.8 | 51.2 |
| Gemma | 81.2 | 72.3 | 81.5 | 53.2 | 81.2 | 32.6 | 44.8 | 87.2 | 39.2 | 54.5 | 32.3 | 64.3 | 55.1 |
| Mixtral | 86.7 | 81.2 | 77.6 | 66 | 83 | 44.8 | 46.8 | 88.4 | 40.9 | 60.4 | 34.8 | 70.6 | 50.3 |
| Jamba | 87.1 | 82.5 | 73.5 | 64.4 | 83.2 | 45.9 | 46.4 | 88.2 | 40.9 | 59.9 | 29.3 | 67.4 | 45.4 |
Shots (Sec 5.1): HellaSwag 10, WinoGrande 5, ARC-E 0, ARC-C 25, PIQA 0, BoolQ 10, QuAC 0, GSM8K 3 CoT,
HumanEval pass@1, NQ 5, TruthfulQA 0, MMLU 5, BBH 3. README repeats the Jamba row.

Ablations (Sec 6). Footnote: "pure Mamba" = Mamba layers interleaved with MLP layers. Metrics: OLLM
(their reproduction), HellaSwag, WinoGrande, NQ, log-prob per byte on C4/Books/Code (higher = better).
- Ratio, 1.3B params, 250B tokens (Table 4, p9; Fig 6 loss curves):
  | | OLLM | HellaSwag | WinoGrande | NQ | C4 | Books | Code |
  |---|---|---|---|---|---|---|---|
  | Attention | 36.4 | 62.4 | 59.6 | 14.5 | -0.543 | -0.659 | -0.331 |
  | Mamba | 36.1 | 62.6 | 59.4 | 14.5 | -0.543 | -0.661 | -0.334 |
  | Jamba 1:3, no MoE | 37.2 | 65.1 | 61.7 | 16.5 | -0.533 | -0.649 | -0.321 |
  | Jamba 1:7, no MoE | 37.2 | 65.1 | 61.7 | 16.0 | -0.533 | -0.650 | -0.321 |
  "1:3 or 1:7 with virtually no performance difference"; 1:7 chosen as "more compute-efficient". Fig 6:
  both hybrids lower loss throughout; pure attention and pure Mamba curves overlap.
- Pure vs hybrid, 7B params, 50B tokens (Table 5, p10; Fig 7):
  | | OLLM | HellaSwag | WinoGrande | NQ | C4 | Books | Code |
  |---|---|---|---|---|---|---|---|
  | Attention | 36.1 | 60.4 | 59.7 | 13.7 | -0.555 | -0.666 | -0.347 |
  | Mamba | 35.3 | 60.2 | 55.8 | 14.0 | -0.554 | -0.667 | -0.355 |
  | Jamba 1:7, no MoE | 36.6 | 62.5 | 58.8 | 15.4 | -0.547 | -0.658 | -0.340 |
  Pure Mamba "quite competitive, but lags slightly behind pure Attention"; hybrid best.
- ICL / format adherence, 1.3B, 250B tokens (Table 6, p11): IMDB / QuAC / NarrativeQA:
  Attention 84.1 / 27.9 / 45.8; Mamba 48.8 / 20.2 / 27.7; Attention-Mamba 90.9 / 26.6 / 43.7.
  Pure Mamba "often does not follow the correct format": on IMDB (answers "Positive"/"Negative") it outputs
  "Very Good", "Very Positive", "Funny", "Bad", "Poor", "3/10". Hypothesis: SSMs have difficulty with
  emergent ICL; ICL is linked to induction heads (Olsson et al.); hybrid does ICL "even when only 1 out of
  8 layers is an Attention one". Evidence: Fig 8 (p12), head H3 of the first attention layer of a 1.3B
  hybrid, on an IMDB example where pure Mamba failed; attention from the last token ":" lands on the
  few-shot label tokens. "12 such heads" found, in all three attention layers (layers 4, 12, 20).
  Anecdotal, by the paper's own word.
- MoE, 7B params, 50B tokens, n=16, K=2, e=2 (Table 7, p12):
  Jamba no MoE 36.6 / 62.5 / 58.8 / 15.4 / -0.547 / -0.658 / -0.340;
  Jamba+MoE 38.1 / 66.0 / 61.2 / 18.9 / -0.534 / -0.645 / -0.326.
- RMSNorm in Mamba (Fig 9, p13): "Jamba 7B" loss curve with spikes without RMSNorm, none after adding it.
  No numbers on the axes.
- No positional encoding, 1.3B, 250B tokens, Jamba with MoE (Table 8, p13): OLLM 39.6 vs Jamba+RoPE 40.1;
  NarrativeQA 50.5 vs 46.2; log-probs identical (-0.516/-0.623/-0.299). "results are similar".

## 9. Limitations the paper states
- Throughput numbers are unoptimized and relative (Sec 3.2 footnote); Jamba lacks Transformer-style
  inference optimizations.
- Induction-head evidence is anecdotal; emergence of ICL in hybrids at scale left to future work (Sec 6.2).
- Mamba needs extra normalization at scale (Sec 6.4).
- Released model is a base model without alignment or moderation (Sec 1 notice).
- Later in this survey's sources: Jamba-1.5 (2408.12570) scales the same design; Nemotron-H (2504.03624)
  is a later hybrid. Their files had not arrived when this note was written (see report).

## 10. Outline feedback
- ch 11 `viz-stack` and `viz-jambaflow` say "shapes from Jamba config". The paper gives no d_model, head
  count, KV-head count, head dim, MLP width, or Mamba N/E/conv. Either add the HF `config.json` to
  `sources/jamba/` (values then `code-only`) or build viz-stack on the back-solved 4 KiB/token per
  attention layer (our reading, exact against Table 1 and the 8x claim).
- ch 11 "Sourced: Jamba KV-cache and throughput table": there is no throughput table. Sec 3.2 gives only
  the two "3x" ratios in text; every bar and line height in Fig 3 is a read-off. Use the ratios.
- ch 11 attribution: Table 1's Llama row is Llama-2 7B (6.7B), not 70B. Llama-2 70B appears only in Fig 2,
  Fig 3, Table 2.
- viz-jambaflow: draw the attention layer in position 5 of 8 (1-indexed) with a dense MLP; MoE on the even
  1-indexed layers; no positional encoding anywhere; RMSNorm both pre-norm and inside Mamba.
- The ICL/format finding (Table 6, Fig 8) is the hook between ch 10 (recall) and ch 11; suggest opening
  ch 11 with it.
- Mamba state memory cannot be computed from this paper; if viz-stack shows "SSM state bytes" next to the
  KV cache, it needs the Mamba config from the mamba sources plus an assumed Jamba d_model (label it).
