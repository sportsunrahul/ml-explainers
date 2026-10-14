# Qwen2-VL (2409.12191v2) -- extraction for ch-resolution (8)

## Naive dynamic resolution (Sec. 2.1)
- ViT ~675M params (same 675M for 2B/7B/72B LLMs, so ViT compute is constant). Init from DFN ViT, with
  its fixed absolute position embedding REPLACED by 2D-RoPE (Sec. 2.2 pretraining para). Absolute pos. embeddings removed.
- patch_size = 14. After the ViT, a simple MLP merges adjacent 2x2 patch tokens into ONE token.
  => one LLM token covers 28 x 28 = 784 px (my arithmetic from 14 * 2; paper states patch 14 and 2x2 merge).
- Paper's worked example: 224x224 image, patch 14 -> (224/14)^2 = 256 patches -> /4 = 64 tokens, +
  <|vision_start|> and <|vision_end|> = "66 tokens before entering LLM" (verbatim: "compressed to 66 tokens").
- At inference, images of different resolutions are packed into one sequence (packed length capped for GPU memory).
- min/max pixels: the paper gives them ONLY in the ablation: "min_pixels = 100 x 28 x 28 and max_pixels = 16384 x 28 x 28"
  i.e. 100 to 16384 tokens per image = 78,400 to 12,845,056 px. These are the ablation settings, not stated as global defaults.
  Small images are upscaled to exceed min_pixels (Fig. min_pixels ablation).
- Fixed vs dynamic (Table fix_dyn_resolution, Qwen2-VL-7B): tokens 64 / 576 / 1600 / 3136 fixed vs dynamic avg 1924:
  InfoVQA val 28.85 / 65.72 / 74.99 / 77.27 vs 75.89; RealWorldQA 56.47/65.88/69.54/70.59 vs 70.07;
  OCRBench 572/828/824/786 vs 866; MMMU 53.33/52.78/52.89/53.44 vs 53.44.
  (Dynamic beats fixed-1600 with ~1924 tokens; fixed-3136 is best on InfoVQA but uses more tokens.)

## M-RoPE (Sec. 2.1; Fig. mrope.png; no numbered equation)
- Rotary embedding split into three components: temporal, height, width (position id triple (t,h,w)).
- Text: all three components get the SAME id -> equivalent to 1D-RoPE.
- Image: t constant for all visual tokens; h = row index, w = column index (of the merged-token grid).
- Video: t increments per frame (temporal id per frame/tube); h, w as for images.
- Multiple modalities: "position numbering for each modality is initialized by incrementing the maximum position ID of
  the preceding modality by one."
- Figure example (read from mrope.png): video of 3 temporal ids, 3 rows (h 0..2) x 4 columns (w 0..3):
  first frame ids (0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,1,0)...(0,2,3); second frame (1,*,*); third (2,*,*).
  Following text "This video features a dog, specifically a Shiba ..." gets (4,4,4),(5,5,5),(6,6,6),... .
  So next text id = max over ALL components of the visual block + 1 = max(2,2,3)+1 = 4, and text is then diagonal (n,n,n).
  (The prose says "maximum position ID of the preceding modality"; the figure shows it is the max across t/h/w, i.e. the width max 3.)
- Benefit claimed: image/video position ids grow slowly (max(h,w,t) rather than token count), enabling length extrapolation:
  trained to 16K tokens/video, good up to 80K tokens inference on Video-MME medium (Fig. mrope_extropolate).
- Ablation (Table abla_mrope; Qwen2-1.5B + ViT-L, pretrained models), 1D-RoPE -> M-RoPE:
  MathVista 39.2->43.4, MMB 58.6->60.6, MMStar 36.7->36.7, RWQ 54.5->53.7, DocVQA 82.5->82.8, ChartQA 68.0->68.4,
  InfoVQA 50.8->50.3, TextVQA 71.3->71.8, PerceptionTest 46.6->47.4, NextQA 43.9->46.0, STAR 55.5->57.9.

## Video / unified (Sec. 2.1 "Unified Image and Video Understanding")
- 2 frames per second sampling. 3D convolution with depth two (3D tubes instead of 2D patches, i.e. 2 frames -> one
  temporal slice; more frames without longer sequence). Each image treated as two identical frames.
- Per-video token cap 16384 during training (frame resolution reduced dynamically). Video-MME eval capped at 768 frames.

## Sizes (Table model_description)
- Qwen2-VL-2B: ViT 675M + LLM 1.5B; 7B: 675M + 7.6B; 72B: 675M + 72B. (Abstract/conclusion call them 2B, 8B, 72B: "2, 8, and 72 billion".)

## Training (Sec. 2.2) -- 3 stages following Qwen-VL
- Stage 1: "focus exclusively on training the ViT" (~600B tokens); LLM init Qwen2, ViT init DFN.
- Stage 2: unfreeze all, additional ~800B image-related tokens. Stage 3: lock ViT, fine-tune LLM only on instruction data.
- Total pretraining 1.4T tokens; supervision only on text tokens.
- Grounding tokens: <|box_start|>(701,531),(869,675)<|box_end|>, <|object_ref_start|>...<|object_ref_end|> (example in appendix) -- same [0,1000)-style coords as Qwen-VL (range not restated; uncertain).

## Headline numbers (Table comparison)
DocVQA test: 72B 96.5, 7B 94.5, 2B 90.1 (GPT-4o 92.8, Claude-3.5-Sonnet 95.2, previous SoTA 94.1).
InfoVQA test 84.5/76.5/65.5; ChartQA 88.3/83.0/73.5; TextVQA val 85.5/84.3/79.7; OCRBench 877/866/809;
MathVista testmini 70.5/58.2/43.0; MMMU val 64.5/54.1/41.1; RealWorldQA 77.8/70.1/62.9.
Compare Qwen-VL DocVQA 65.1.

## Outline assertions I could not confirm or that are wrong
- "2B/8B/72B": the conclusion says 2, 8, 72 billion but Table 1 names 2B / 7B / 72B (7.6B LLM + 0.675B ViT ~ 8.3B). Use "7B/8B" carefully.
- "min/max pixels": not global defaults; values only in the ablation (100*28*28 and 16384*28*28). HF-processor defaults are not in the paper.
- "video 2 fps / 3D conv over 2 frames": confirmed (2 fps, 3D conv depth 2).
- "1 token per 28x28 px": arithmetic from patch 14 and 2x2 merge; the paper never writes "28x28" outside the min_pixels formula.
- "M-RoPE id rule": the prose (preceding modality's max + 1) is ambiguous; the figure shows max over t,h,w components +1 (4 after a (3 x 3 x 4) block).
