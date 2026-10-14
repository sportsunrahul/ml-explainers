# Qwen3-VL (2511.21631v2) -- extraction for ch-qwen3 (12)

## Family (Sec. 1, Sec. 2)
- Dense: 2B / 4B / 8B / 32B; MoE: 30B-A3B and 235B-A22B (235B total, 22B active). All on Qwen3 backbones.
  (Text says "three dense variants" but lists four; count is four.)
- Three modules like Qwen2.5-VL: vision encoder, MLP merger, LLM. Instruct and Thinking variants.
- Native context 256K (S3 sequence length 262,144).

## Vision encoder (Sec. 2)
- SigLIP-2 architecture, initialised from the official SigLIP-2 checkpoints, then continued training at dynamic resolution.
  2D-RoPE plus interpolated absolute position embeddings (following CoMP).
- Default SigLIP2-SO-400M; SigLIP2-Large (300M) for the 2B and 4B LLMs.
- Merger: two-layer MLP compressing 2x2 visual features into one token (as Qwen2.5-VL). Extra "specialized mergers" for DeepStack.
- Patch size: not stated in the LaTeX text (SigLIP-2 uses 16; NOT verified from this paper, do not print without a source).

## Interleaved MRoPE (Sec. 2.2)
- Old MRoPE (Qwen2-VL): the embedding dimensions are PARTITIONED into contiguous t / h / w subspaces, each with its own rotary frequencies.
  Because RoPE frequencies run from high to low across the dimensions, each axis then only sees a slice of the spectrum
  ("imbalanced frequency spectrum"), which hurts long-video understanding.
- New: interleave t, h, w components across the embedding dimensions so each axis is represented in both low- and high-frequency
  bands ("uniformly ... balanced spectrum"). No equation; the exact repeating pattern (t,h,w,t,h,w,...) is in the cited
  interleave-mrope work, not in this paper (mark pattern as uncertain).
- No ablation table for interleaved-MRoPE in this paper.

## DeepStack (Sec. 2.3; Fig. arc, qwen3vl_arc.jpg)
- Idea from DeepStack (Meng et al. 2024), which stacked tokens from multi-scale visual inputs. Qwen3-VL instead takes features from
  INTERMEDIATE LAYERS of the single ViT.
- "we select features from three distinct levels of the vision encoder"; dedicated vision-language merger modules project each level
  into visual tokens, "which are then added directly to the corresponding hidden states of the first three LLM layers".
  Figure: three arrows from the vision encoder to LLM Block 1, Block 2, Block 3, adding vision tokens (residual add, no extra sequence length).
  Level 1 -> LLM block 1, level 2 -> block 2, level 3 -> block 3 (mapping order per figure/"corresponding"). Which ViT layer indices: NOT
  given in the text or figure (uncertain).
- Intro: "lightweight residual connections ... without introducing extra context length".
- Ablation (Table deepstack_results; internal 15B-A2B LLM, 200B pretraining tokens, no post-training; validation sets):
  | | AVG | AI2D | OCRB | TVQA | InfoVQA | ChartQA | DocVQA | MMMU | MMStar | RLWDQA | MMB-EN | MMB-CN |
  | Baseline | 74.7 | 81.8 | 81.0 | 80.6 | 71.9 | 81.5 | 89.5 | 52.9 | 55.5 | 67.7 | 81.0 | 78.1 |
  | DeepStack | 76.0 | 83.2 | 83.6 | 80.5 | 74.2 | 83.3 | 91.1 | 54.1 | 57.7 | 68.1 | 81.2 | 78.5 |
  Gains largest on fine-grained InfoVQA (+2.3) and DocVQA (+1.6); TextVQA is -0.1.

## Text-based video timestamps (Sec. 2.4)
- Qwen2.5-VL tied temporal position ids to absolute time -> (1) very large, sparse temporal ids for long videos hurting long-context
  understanding; (2) needs extensive, uniform fps sampling in training data.
- Qwen3-VL: each video temporal patch (a group of frames) is prefixed with a plain text timestamp, e.g. `<3.0 seconds>`
  (Fig. arc shows `<0.0 seconds>` and `<4.0 seconds>` before frame-group embeddings). Timestamps generated in both seconds and HMS
  (hh:mm:ss) formats in training. Costs a modest increase in context length; helps video grounding and dense captioning.

## Training (Table training_setup)
S0 alignment: merger only trained (ViT and LLM frozen), 67B tokens, seq 8,192. S1 multimodal pretrain: all unfrozen, ~1T tokens, 8,192.
S2 long context: all, ~1T tokens, 32,768. S3 ultra-long: all, 100B tokens, 262,144.
Loss: square-root-normalised per-token loss (Sec. 1) to balance text vs multimodal.

## Headline numbers (flagship / medium tables)
DocVQA test: 235B-A22B thinking 96.5, instruct 97.1 (Gemini 2.5 Pro thinking 92.6 / budget-128 94.0). 32B instruct 96.9,
32B thinking 96.1 (medium table order: 30B-A3B thinking, instruct, 32B thinking, instruct = 95.5, 95.0, 96.1, 96.9; check column order before printing).
MMMU: 235B thinking 80.6, instruct 78.7.
Figure arc example token counts: a 1248x9376 page = 11427 tokens; 256x32 = 8 tokens; 1440x800 = 1125 tokens; video 736x448, 8 s.

## Outline assertions I could not confirm or that are wrong
- "SigLIP-2 init?": confirmed (SigLIP2-SO-400M, SigLIP2-Large 300M for 2B/4B).
- "dense 2B/4B/8B/32B": confirmed (text miscounts "three").
- "MLP merger": confirmed (2-layer MLP, 2x2).
- "DeepStack (which ViT layers -> which LLM layers)": ViT layer indices are NOT stated; only "three levels" -> first three LLM layers.
- "M-RoPE (now interleaved)": confirmed; explanation is about spectrum balance, not new positions. Timestamps additionally replace Qwen2.5-VL's absolute-time temporal ids.
- Note Qwen3-VL compares to Qwen2.5-VL (not Qwen2-VL) for the MRoPE and timestamp changes; Qwen2.5-VL is not in the outline.
