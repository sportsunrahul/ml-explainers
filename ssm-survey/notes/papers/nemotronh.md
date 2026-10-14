# Nemotron-H (hybrid group, branch, serves ch 11)

## 1. Identity
- Title: "Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models".
- arXiv 2504.03624, version read: v4 (2025-09-05). v1 2025-04-04; v2 2025-04-10; v3 2025-04-15 (abs.html).
- Authors: "NVIDIA" (contributor list in `sections/contributors.tex`). Venue: NVIDIA technical report.
- Checkpoints: huggingface.co/nvidia/Nemotron-H-{8B,47B,56B}-Base-8K, plus 8B/47B Reasoning-128K.
- Read: `src/sections/{abstract,introduction,model_arch,base_model_evals}.tex` in full; speed, training,
  MiniPuzzle results and alignment sections in the parts that bear on ch 11; Figs 1, 2, 7 rendered.

## 2. Core problem
Reasoning models scale inference-time tokens, and attention's per-token compute and KV cache grow with
tokens generated. Build models with Transformer-level accuracy at lower inference cost.

## 3. Prior methods and why that hurt (per the paper, Sec 1)
Self-attention per-token compute is linear in tokens seen and the KV cache is linear in sequence length,
so "attention layers can fundamentally limit overall model intelligence" under inference-time scaling.

## 4. The one key idea
Replace most self-attention layers with Mamba-2: keep roughly 8% of all layers as self-attention,
evenly spaced, and make the rest an even split of Mamba-2 and FFN layers (Sec 2.1).

## 5. Mechanism, step by step
1. Layer accounting differs from Jamba: here Mamba-2, self-attention and FFN are each a separate
   "layer", each with its own residual skip (Sec 2.1). Jamba counts mixer + MLP as one layer.
2. Rules (Sec 2.1): attention about 8% of total layers, evenly dispersed; first layer Mamba-2; last layer
   FFN; attention always followed by FFN.
3. Pattern (Fig 2 p3): [Mamba-2, FFN] x3, then [Mamba-2, Attention, FFN, (Mamba-2, FFN) x4] x4 (8B) or
   x10 (56B), then [Mamba-2, FFN] x1.
4. Counts: 8B = 52 layers with 4 attention (stated); Mamba-2 24 and FFN 24 (our reading from "even
   split" and Fig 2). 56B = 118 layers with 10 attention (stated); Mamba-2 54 and FFN 54 (stated in
   Sec 3.5: 47B has "44 Mamba-2 layers (down from 54) and 49 FFN layers (down from 54)").
5. 47B (MiniPuzzle from 56B, Sec 3.5): 5 attention, 44 Mamba-2, 49 FFN (98 layers, our reading); FFN
   width 32768 -> 30720; distilled with 63B tokens, "roughly 300x fewer tokens" than from scratch.
6. Other details (Sec 2.1): GQA with 8 KV heads; 8 Mamba-2 groups; squared ReLU FFN; no position
   embeddings; RMSNorm; untied embeddings; no dropout; no biases. Mamba-2 defaults: head dim 64,
   expansion 2, conv window 4.
7. Baseline Nemotron-T-8B: 32 Transformer layers, same width/heads/FFN/data as Nemotron-H-8B, with RoPE.
8. Pre-training (Sec 2.4): 8B on 15T tokens, 56B on 20T tokens; sequence length 8192; global batch 768
   (6291456 tokens); peak LR 8e-4 (8B) / 4e-4 (56B); cosine decay to 1% of peak; weight decay 0.1; Adam
   betas 0.9, 0.95. 56B trained with FP8 per-tensor current scaling, first and last 4 layers' GEMMs BF16
   (Sec 1, Sec 2.3).

## 6. Equations
None needed for ch 11. Derived quantities (our reading, not in the paper):
- Attention head_dim = d_model / Q heads = 4096/32 = 8192/64 = 128.
- KV bytes/token (16-bit) = 2 x n_attn x 8 x 128 x 2 = n_attn x 4 KiB:
  8B: 4 x 4 KiB = 16 KiB/token (same as Jamba); 56B: 40 KiB/token; 47B: 20 KiB/token;
  Nemotron-T-8B: 32 x 4 KiB = 128 KiB/token, so the hybrid cuts the KV cache 8x at matched width.
- Attention share by sequence mixers only (attention / (attention + Mamba-2)): 8B 4/28 = 1:6; 56B 10/64;
  vs Jamba 4/32 = 1:7. By all layers: 8B 4/52, 56B 10/118 (the paper's "roughly 8%").
- Mamba-2 state per layer = (E x d_model) x N = 8192 x 128 = 1048576 values (8B); 16384 x 256 = 4194304
  (56B). Fixed in sequence length. Precision of the cached state not stated.

## 7. Config / setup table (Table 1 p4 unless noted)
| Item | Nemotron-H-8B | Nemotron-H-56B | Nemotron-H-47B |
|---|---|---|---|
| Layers | 52 | 118 | 98 (our reading) |
| Attention layers | 4 | 10 | 5 (Sec 3.5) |
| Mamba-2 / FFN layers | 24 / 24 (our reading) | 54 / 54 (Sec 3.5) | 44 / 49 (Sec 3.5) |
| Model dim | 4096 | 8192 | -- |
| FFN dim | 21504 | 32768 | 30720 (Sec 3.5) |
| Q heads / KV heads | 32 / 8 | 64 / 8 | -- |
| Mamba-2 state dim N | 128 | 256 | -- |
| Mamba groups | 8 | 8 | -- |
| Mamba-2 head dim / expand / conv | 64 / 2 / 4 | 64 / 2 / 4 | -- |
| Position embeddings | none | none | -- |
| Training tokens | 15T | 20T | +63B distillation |
| Seq len / batch | 8192 / 768 | 8192 / 768 | -- |

## 8. Results worth remembering
- Speed claim, setup (Sec 2.6, Fig 1 p2 and Fig 7 p14): input 65536 tokens, 1024 output tokens; NVIDIA
  H100; batch as large as possible to maximize per-GPU throughput; reported as output tokens/s/GPU;
  Nemotron-H on "an initial Megatron-LM implementation", baselines on vLLM v0.7.3 (different stacks).
  - 56B: 2.4x Qwen-2.5-72B and Llama-3.1-70B; 47B: 2.9x; vs Llama-3.1-405B: 19.6x.
  - 8B: 1.8x Qwen-2.5-7B and 3x Llama-3.1-8B "on longer contexts".
  - Abstract/Conclusion: "up to 3x faster at inference". 47B is "20% faster" (abstract) / "1.2x" (Sec 3.5)
    than 56B.
  - Stated causes: constant compute in Mamba-2 vs linear in attention; lower memory lets batch grow.
  - Fig 7 read-off (not stated): Nemotron-T-8B about 97 vs Nemotron-H-8B about 211 tokens/s/GPU. Uncertain.
- Accuracy: 56B-Base beats Llama-3.1-70B-Base on 16 of 17 tasks (all but Winogrande) (Sec 1, Sec 2.5,
  Table 4 p12). MMLU-Pro 60.5 (56B), 61.8 (47B), 58.8 (Qwen-2.5-72B), 51.3 (Llama-3.1-70B).
- Hybrid vs Transformer on identical data (Table 5 p13): Nemotron-H-8B beats Nemotron-T-8B on 7 of 15 tasks
  and is within one point on 4 more. MMLU 72.8 vs 73.2; MMLU-Pro 44.0 vs 41.4; GSM8k 87.1 vs 89.0.
- Post-training parity (Sec 5.1, Tables 9-10): Exp-Base T vs H MMLU 70.3 vs 69.9; Exp-Instruct MT-Bench
  7.9 vs 8.0, MMLU 68.0 vs 67.4.
- Long context (Sec 5.2.2): after 512k-token extended blends, 83% RULER at 128k (non-reasoning mode).
- MiniPuzzle (Sec 1, Sec 3): 47B fits FP4 on one RTX 5090 (32GiB); 56B weights alone would need about
  29.5GiB, leaving 2.5GiB.

## 9. Limitations stated
Inference implementation is initial and unoptimized relative to Transformers (Sec 2.6). Speed comparisons
use different serving stacks (Megatron-LM vs vLLM). Base models released at 8K context.

## 10. Outline feedback
- ch 11 "Nemotron-H layer mix": the "roughly 8%" counts FFN layers as layers. By sequence mixers it is
  4 of 28 (8B) and 10 of 64 (56B), i.e. a higher attention share than Jamba's 1:7. viz-stack's slider must
  state its denominator, or the two papers look more different than they are.
- Nemotron-H uses Mamba-2 (Jamba-1.5 chose Mamba-1). It gives a full config (d_model, heads, KV heads, N),
  so it is the hybrid whose KV cache and SSM state the page can compute from sourced numbers; consider it
  for viz-stack's computed point, with Jamba Table 1 as the reported point.
- The apples-to-apples Nemotron-H-8B vs Nemotron-T-8B result (Table 5) fits ch 11's "NVIDIA hybrid vs
  Transformer" line; the ch 10 NVIDIA 8B study is a separate, earlier paper (2406.07887).
