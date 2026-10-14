# LLaVA-1.5 (Improved Baselines with Visual Instruction Tuning) 2310.03744v2 -- chapter 7 delta

Source: sources/llava15/src/sec/*.tex. Base LLM: Vicuna v1.5 (7B and 13B).

## What changed (three edits, plus data)
1. Connector: "a two-layer MLP" replaces the single linear projection. Motivation quoted: inspired by
   linear->MLP improvements in self-supervised learning (SimCLR, MoCo v2). NOT stated in the paper:
   activation (GELU), hidden width. (GELU/hidden=LLM dim are in the released code, not the text; treat
   as uncertain.) Pretrain lr halved (2e-3 -> 1e-3) "due to the usage of the MLP projection layer".
2. Resolution: swap encoder to CLIP-ViT-L-336px ("the highest resolution available for CLIP"),
   input 336^2. Tokens per image = (336/14)^2 = 24^2 = 576 (COMPUTED; 224 -> 256). Original: 224^2.
3. Academic-task VQA data + a response formatting prompt appended to short-answer questions.
   Also: GQA, ShareGPT (language-only, 40K), region-level VQA (VG, RefCOCO), 13B LLM.

## Response formatting prompts (verbatim, Table "Instruction-following Data Mixture" / Sec 3.1)
- Main: "Answer the question using a single word or phrase."
  (appended at the end of VQAv2, GQA, OKVQA, OCRVQA questions)
- Multiple choice (A-OKVQA): "Answer with the option's letter from the given choices directly."
- TextCaps: "Provide a one-sentence caption for the provided image."
- RefCOCO (randomly one of two): "Provide a short description for this region." /
  VG: "Provide the bounding box coordinate of the region this sentence describes."
- Eval only, VizWiz: "When the provided information is insufficient, respond with `Unanswerable'.
  Answer the question using a single word or phrase."
- Why (Sec 3.1): InstructBLIP fails to balance short/long answers because (1) ambiguous format
  prompts like "Q: {Question} A: {Answer}" overfit the LLM to short answers, and (2) it only
  finetunes the Q-Former (like prefix tuning), which lacks capacity to control output length.
  Fix: one clear format prompt + finetune the LLM. Original LLaVA "tends to answer yes for yes/no
  questions" (no such data). Example (Table 1b, after stage-1 only, zero-shot): normal prompt ->
  "The man is wearing a yellow shirt."; formatting prompt -> "Yellow."

## Data mixture (Appendix Table, sums to 665K)
LLaVA 158K + ShareGPT 40K + VQAv2 83K + GQA 72K + OKVQA 9K + OCRVQA 80K + A-OKVQA 66K +
TextCaps 22K + RefCOCO 48K + VG 86K = 664K listed, paper prints Total 665K (rounding of "K").
Stage-1 pretrain: 558K image-text pairs (table column "Sample Size Pretrain 558K"; LLaVA-1.5 rows).
Total: 558K + 665K = 1.223M ~ "merely 1.2M publicly available data" (abstract). The paper states
1.2M but does not show the sum; the 558K/665K split is in the comparison table.
Data handling: QA pairs of one image merged into one conversation; A-OKVQA question augmented k times
(k = choices); OCRVQA subsampled to 80K; VG 10 annotations per image; per-batch single modality
(speeds training 25%); all splits concatenated, sampled with equal probability.

## Training
- Stage 1 hyperparameters: batch 256, lr 1e-3, 1 epoch, DeepSpeed 2. Stage 2: batch 128, lr 2e-5,
  1 epoch, DeepSpeed 3. Cosine decay, warmup 0.03, wd 0, AdamW. Same frozen/trainable pattern as
  LLaVA (stage 1: projector only; stage 2: projector + LLM; encoder frozen throughout -- inherited
  from LLaVA; the 1.5 paper does not restate it).
- Time: ~2x LLaVA: "~6 hours of pretraining and ~20 hours of visual instruction tuning, using 8x A100s"
  (~26 h = "~1 day on a single 8-A100 node", abstract).

## Ablation table (Table "Scaling results"; GQA / MME / MM-Vet), all rows
InstructBLIP 14B, 224: 49.5 / 1212.8 / 25.6
0 LLaVA 7B 224:              -- / 809.6 / 25.5
1 +VQA-v2:                 47.0 / 1197.0 / 27.7
2 +Format prompt:          46.8 / 1323.8 / 26.3
3 +MLP VL connector:       47.3 / 1355.2 / 27.8
4 +OKVQA/OCR:              50.0 / 1377.6 / 29.6
5 +Region-level VQA:       50.3 / 1426.5 / 30.8
6 +Scale up resolution 336: 51.4 / 1450 / 30.3
7 +GQA:                    62.0* / 1469.2 / 30.7   (* GQA train images seen in training)
8 +ShareGPT:                62.0* / 1510.7 / 31.1
9 +Scale up LLM 13B:       63.3* / 1531.3 / 36.1
Text: "by merely including VQAv2 ... MME improves (1323.8 vs 809.6) and outperforms InstructBLIP by
111 points" (1323.8 - 1212.8 = 111.0; that text attaches it to the +format-prompt row). Largest MM-Vet
jump is the 13B step (+5.0). MLP step: +31.4 MME, +0.5 GQA, +1.5 MM-Vet over row 2.

## Benchmark results
- Abstract/Fig 1: "state-of-the-art across 11 benchmarks". Sec 4: "a collection of ... totalling 12
  benchmarks" ; "best overall performance on 12 benchmarks". An "11/12 best, second on the other"
  caption exists only COMMENTED OUT in the tex. Published statements: 11 (abstract, Fig 1) and
  12 evaluated. Academic table caption: "best performance on 4/5 benchmarks, ranks second on the other."
- Academic table (VQAv2 / GQA / VizWiz / SciQA-IMG / TextVQA), LLaVA-1.5 13B 336: 80.0 / 63.3 / 53.6 / 71.6 / 61.3;
  7B: 78.5 / 62.0 / 50.0 / 66.8 / 58.2; 13B-HD (448): 81.8 / 64.7 / 57.5 / 71.0 / 62.5.
- Instruction-LMM table 13B: POPE adv 84.5, MME 1531.3, MMBench en 67.7 / cn 63.6, SEED all 61.6,
  LLaVA-Wild 72.5, MM-Vet 36.1. LLaVA-7B (v1): MME 809.6, MM-Vet 25.5, LLaVA-Wild 62.8.
- Comparison of data: BLIP-2 pretrain 129M; InstructBLIP 129M + 1.2M; Qwen-VL 1.4B + 50M
  (in-house); LLaVA-1.5 558K + 665K, public only. 7B beats IDEFICS-80B.

## Connector token counts for the visual (COMPUTED from grids unless noted)
- LLaVA: 224/14 = 16 -> 256 tokens (linear). LLaVA-1.5: 336/14 = 24 -> 576 (2-layer MLP).
- Flamingo 64 latents (Perceiver Resampler) and BLIP-2 32 queries come from those papers, not here.
- Paper says Q-Former/resamplers "reduce the number of visual patches".

## LLaVA-1.5-HD (Sec 3.3 + Appendix A)
- Reason: open CLIP encoders top out at 336^2, and ViT positional-embedding interpolation needs
  large-scale finetuning and fixes the resolution. Instead: split image into grid cells of the
  encoder's native size, encode each independently, merge into one large feature map, and concatenate
  the features of a downsampled (global) image "to provide the LLM with global context and to reduce
  split-encode-merge artifacts". No positional-embedding interpolation.
- Appendix details: base encoder CLIP-ViT-L-14 (224^2) for HD (not the 336 one); target resolution
  chosen from a predefined set of up to six grids (1x1,1x2,1x3,1x4,1x5,1x6,2x2,2x3 and transposes),
  max 672x448 (or 448x672); criteria: detail preservation and resource efficiency. Postprocess:
  (1) drop padding-only features, (2) append a special row-end token per row so the LLM knows the
  shape, (3) flatten. No extra pretraining; instruction tuning directly on higher-res images.
- Tables list HD as "448^2" (13B-HD). HD academic gains vs 13B: VQAv2 +1.8, GQA +1.4, VizWiz +3.9,
  TextVQA +1.2; MM-Vet 39.4 vs 36.1; MME 1500.1 (LOWER than 1531.3); SciQA-IMG 71.0 (lower than 71.6).
- Hallucination finding: hallucination "significantly reduced" at 448^2 (Sec 5.2).
- Data efficiency: with 50% random subsample keeps >98% of full performance (Sec 5.1).
- A commented-out global-context ablation in the tex (7B): patches only GQA 62.9/MME 1425.8/MM-Vet 31.9;
  +global 63.8/1497.5/35.1. It is commented out of the published version: mark uncertain.

## Outline assertions I could not confirm or that are wrong
- "MLP 2-layer GELU": the paper only says "two-layer MLP"; GELU and widths are not in the text
  (they are in the released code). Mark uncertain / do not show as sourced.
- "11/12 SOTA": abstract and Fig 1 say 11 benchmarks/tasks; Sec 4 says 12 benchmarks evaluated;
  the explicit "best on 11/12" caption is commented out. Safe phrasing: "SoTA across 11 benchmarks
  (12 evaluated)". Academic table alone: best on 4/5.
- "665K instruction mix, 1.2M total": 665K is exact; 1.2M is the abstract's figure = 558K pretrain +
  665K finetune (1.223M).
- Pretraining set: 1.5 text says "same pretraining dataset" as LLaVA, but the tables give 558K vs
  LLaVA's 595K (a LAION-CC-SBU 558K subset per a commented footnote). Do not say 595K for 1.5.
- "~1 day 8xA100": abstract; breakdown ~6 h + ~20 h.
- Stage steps/epochs for 1.5: both stages 1 epoch (LLaVA used 3 epochs in stage 2).
- HD resolution: tables say 448^2; appendix grid set goes up to 672x448.
