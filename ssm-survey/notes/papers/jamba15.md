# Jamba-1.5 (hybrid group, branch, serves ch 11)

## 1. Identity
- Title: "Jamba-1.5: Hybrid Transformer-Mamba Models at Scale".
- arXiv 2408.12570, version read: v1 (2024-08-22; only version in abs.html).
- Authors: "Jamba Team" (abs.html lists Barak Lenz first by name ... Yoav Shoham last). AI21.
- Venue: none stated (preprint). Weights: huggingface.co/ai21labs/AI21-Jamba-1.5-Mini and
  .../AI21-Jamba-1.5-Large (Jamba Open Model License). ExpertsInt8 kernel: vLLM PR 7415.
- Read: `src/main.tex` in full; figures rendered from `src/figures/`.

## 2. Core problem
Scale the Jamba hybrid to a large model (398B total) that is instruction-tuned, keeps a 256K effective
context, and still serves on one 8-GPU machine.

## 3. Prior methods and why that hurt (per the paper)
Transformer models of similar size need roughly an order of magnitude more KV cache at 256K (Table 1).
Standard quantization in vLLM relies on calibration (slow, can be unstable); FP8 runs only on H100
(Sec 3.1).

## 4. The one key idea
Same Jamba block, scaled (9 blocks, d_model 8192), plus ExpertsInt8: store MoE and MLP weights in INT8 and
dequantize to BF16 inside the fused MoE kernel, so the 94B-active / 398B-total model fits 8 x 80GB GPUs at
256K context.

## 5. Mechanism, step by step
1. Jamba-1.5-Large: 9 blocks; l = 8 layers per block; a:m = 1:7; MoE every e = 2 layers; n = 16 experts,
   top K = 2; hidden dimensionality 8192; 64 query heads, 8 KV heads (Sec 2 list).
2. Jamba-1.5-Mini: "an updated and instruction-tuned version of our earlier Jamba release" (Sec 1);
   Table 1 gives the same 52B / 12B / 4GB as Jamba v0.1. Its d_model and heads are not stated here.
3. Mamba version: Mamba-1. They tried Mamba-2 in the hybrid and found Mamba-1-Attention better than
   Mamba-2-Attention (Sec 2, Fig 1).
4. ExpertsInt8 (Sec 3.1): over 85% of weights are in MoE layers, over 90% in MoE or MLP layers. These
   are quantized to INT8, stored INT8, and dequantized to BF16 inside vLLM's `fused_moe` kernel just
   before compute. No calibration; quantization takes seconds at load; activations stay BF16; works on
   A100.
5. Activation loss (Sec 3.2): some expert outputs and the last Mamba layers' outputs grew to 4 x 10^6
   during pre-training (BF16 was fine, but FP16 inference tops out at 64K). An auxiliary loss proportional
   to the mean square of activations, weight alpha; alpha = 10^-5 for Large brought max activations to
   2K-3K; no effect on training for alpha up to at least 10^-3.
6. Training (Sec 5): H100 GPUs, in-house framework (FSDP, tensor, sequence, expert parallelism; MegaBlocks
   adapted). Three stages: pre-training (data to March 2024, multilingual), short mid-training with long
   documents, post-training (SFT with heavy synthetic data). Token counts not stated.

## 6. Equations
None in the paper. KV check (our reading), formula as in `jamba.md`:
- Large: 9 attention layers (9 blocks x 1, from a:m = 1:7), 8 KV heads, head_dim = 8192 / 64 = 128 (our
  reading; head_dim not stated). Bytes/token = 2 x 9 x 8 x 128 x 2 = 36864 = 36 KiB.
  At 256K = 262144 tokens: 36 KiB x 262144 = 9 GiB. Table 1: "9GB". Exact match with GB read as GiB.
- Mini: 4GB, same as Jamba v0.1 (see `jamba.md`, back-solved 1024 KV channels per attention layer).
- Cross-check of comparators against the Llama 3 paper config (sources/llama3, 2407.21783v3, Table 3:
  8B 32 layers, 70B 80 layers, 405B 126 layers; all 8 KV heads; head_dim = model dim / heads = 128):
  8B -> 128 KiB/token -> 32 GiB at 256K (Table 1: 32GB, match); 70B -> 320 KiB/token -> 80 GiB (Table 1:
  80GB, match); 405B -> 504 KiB/token -> 126 GiB, but Table 1 lists 252GB, 2x our computation. We do not
  reconcile it; flag only. Mixtral 8x22B and Mistral-Large-2 configs are not in our sources.

## 7. Config / setup table
| Item | Jamba-1.5-Large | Jamba-1.5-Mini | Location |
|---|---|---|---|
| Total / active params | 398B / 94B | 52B / 12B | Table 1; abstract; Sec 1 |
| Blocks | 9 | not stated (Jamba v0.1 has 4) | Sec 2 |
| l, a:m, e, n, K | 8, 1:7, 2, 16, 2 | (Jamba v0.1 values) | Sec 2 |
| Attention layers | 9 (our reading) | -- | from Sec 2 |
| Hidden dim | 8192 | not stated | Sec 2 |
| Q heads / KV heads | 64 / 8 | not stated | Sec 2 |
| Mamba | Mamba-1 | -- | Sec 2, Fig 1 |
| KV cache, 256K, 16-bit | 9GB | 4GB | Table 1 |
| Effective context (RULER) | 256K | 256K | Table 4 |
| Serving | 8 x 80GB GPUs at 256K with ExpertsInt8 | -- | abstract; Sec 1; Sec 2 |
| Training GPUs | H100 | -- | Sec 5.1 |

## 8. Results worth remembering
- Table 1 (p2), KV cache at 256K, 16-bit: Mistral 7.2B/7.2B 32GB; Mixtral 8x7B 46.7B/12.9B 32GB;
  LLaMA-3.1 8B 8B/8B 32GB; Mixtral 8x22B 141B/39B 56GB; Mistral-Large-2 123B/123B 88GB; LLaMA-3.1 70B
  70B/70B 80GB; LLaMA-3.1 405B 405B/405B 252GB; Jamba-1.5-Mini 52B/12B 4GB; Jamba-1.5-Large 398B/94B 9GB.
  Sec 1: "10x reduction in KV cache memory"; Sec 2: "roughly an order of magnitude reduction".
- Mamba-1 vs Mamba-2 in-hybrid (Sec 2, Fig 1 p3): 350M and 1.3B models, 100B tokens each. "While Mamba-2
  outperforms Mamba-1 without attention, the hybrid Mamba-1-Attention performs better." Hybrid also beats
  pure Mamba-2. Hypothesis: Mamba-2's advantages, "in particular the ability to use a much larger state
  size, are less significant when we have full attention layers interleaved", since attention "can pool
  information from the entire context". Fig 1a (350M) shows all four curves; Fig 1b (1.3B) only the two
  hybrids. Loss values are read-off only; no table. The ablation's a:m ratio is not stated.
- ExpertsInt8 latency (Fig 2 p4): end-to-end latency, 1024-token context, 128-token decoding. On H100
  matches FP8; on A100 (no FP8) beats GPTQ "by a large margin". Panels: Jamba-1.5-Mini 2xH100 and 2xA100;
  Large 8xH100; Mixtral-8x7B 2xH100; Mixtral-8x22B 8xH100.
- Throughput/latency (Sec 4): Fig 3 Mini on 2xA100 80GB, batch 1, output 512; Fig 4 Large on 8xA100 80GB,
  batch 1, output 512. Better latency "especially at large contexts, with only a slight reduction in output
  tokens throughput". LLaMA-3.1-405B truncates at 64K, cannot fit beyond about 100K tokens on 8 80GB GPUs.
  Axes run to 262144 total context. No numbers in text.
- RULER (Table 4 p8): Large claimed 256K, effective 256K, 96.7/96.6/96.4/96.0/95.4/95.1/93.9 at 4k..256k,
  avg 95.7. Mini: 95.7/95.2/94.7/93.8/92.7/89.8/86.1, avg 92.6. Comparators: LLaMA 3.1 70B claimed 128K,
  effective 64K; Mixtral 8x7B 32K/32K; Gemini-1.5-pro 1M/>128K (256k = 65.1, run by them).
- Infinite-Bench (Table 5 p9): EN.MC Mini 76.9, Large 80.4; EN.QA Mini 40.6, Large 34.9.
- Academic (Table 2 p7): Mini MMLU 69.7, Large 80.0 (vs LLaMA-3.1 70B 83.6); others in claims file.

## 9. Limitations stated
Observations "not fully explored" (Sec 5.4). Mamba-1 vs Mamba-2 evidence is loss curves at 350M/1.3B only.
Throughput curves trade slightly lower output-token throughput for lower latency at batch 1.

## 10. Outline feedback
- ch 11 "Mamba-1 vs Mamba-2 in a hybrid" is a good one-line delta: Jamba-1.5 keeps Mamba-1 because, with
  attention present, Mamba-2's bigger state buys little (Fig 1). Nemotron-H, by contrast, uses Mamba-2.
  Worth a sentence: the two production hybrids disagree.
- Jamba-1.5 calls 1:7 "found optimal"; the Jamba paper says 1:3 and 1:7 tie and 1:7 was picked for
  compute efficiency. Quote the Jamba paper.
- viz-stack can add Jamba-1.5-Large as a second sourced point (9 attention layers, 36 KiB/token, 9GB at
  256K, check exact). Do not show the LLaMA-3.1 405B 252GB row without the 2x caveat.
