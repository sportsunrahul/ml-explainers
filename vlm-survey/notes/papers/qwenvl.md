# Qwen-VL (2308.12966v3) -- extraction for ch-resolution (8) and ch-grounding (9, box format)

## Architecture (Sec. 3.1, Table 1)
- LLM: Qwen-7B init. Visual encoder: ViT init from OpenCLIP ViT-bigG, patch stride 14.
- Params (Table "model_parameter"): vision encoder 1.9B, VL adapter 0.08B, LLM 7.7B, total 9.6B.
- Position-aware VL adapter: ONE randomly-initialised cross-attention layer. A group of trainable
  vectors (learned embeddings) are the QUERIES; the ViT patch features are the KEYS (and values).
  Output length = number of queries = fixed 256 tokens, regardless of input length.
  2D absolute positional encodings are added into the cross-attention query-key pairs, to keep
  positional information lost by compression. The 256 tokens go to the LLM.
- Query-count ablation (App. D.2): tested with ViT-L/14 at 224px (256 patches). Fewer queries =>
  lower initial loss, but too many or too few slows convergence. At 448px there are (448/14)^2 = 1024
  ViT tokens, too few queries lose info; chose 256.
- Input resolution per stage (Hyperparameter table): stage 1 224^2 (ViT seq 256, LLM seq 512);
  stage 2 and 3: 448^2 (ViT seq 1024, LLM seq 2048). Queries 256 in all stages.
  So at 448px the adapter compresses 1024 -> 256 (4x); at 224px it is 256 -> 256 (1x).
- Window vs global attention ablation (App. D.3): 448 global ~10 s/iter, 448 window ~9 s/iter;
  896 window not used: ~2.5x slower than 448.

## Image/box I/O (Sec. 3.2)
- Image: <img> ... </img> wrap the 256 image features.
- Box normalised to [0, 1000); string form "(X_topleft,Y_topleft),(X_bottomright,Y_bottomright)".
  Tokenised as ordinary text (no extra location vocabulary). Wrapped in <box> ... </box>.
  <ref> ... </ref> wrap the phrase the box refers to.
- Appendix data-format examples (verbatim):
  Referring grounding: `<img>VG_100K_2/3.jpg</img><ref>the ear on a giraffe</ref><box>(176,106),(232,160)</box><eos>`
  Caption w/ grounding: `Beautiful shot of <ref>bees</ref><box>(661,612),(833,812)</box><box>(120,555),(265,770)</box> gathering nectars from <ref>an apricot flower</ref><box>(224,13),(399,313)</box><eos>`
  Grounded captioning: `<ref>This</ref><box>(360,542),(476,705)</box> is Yellow cross country ski racing gloves`
  OCR with grounding uses a 4-point form: `<ref>It is managed</ref><quad>(568,121),(625,131),(624,182),(567,172)</quad>`

## 3-stage pipeline (Sec. 4; Hyperparameter table)
| stage | data | frozen | trained | res | steps | batch |
| 1 pre-training | 1.4B cleaned pairs (of 5B raw; 77.3% EN / 22.7% ZH) | LLM | ViT + adapter | 224 | 50k | 30720 |
| 2 multi-task pre-training | 7 tasks, fine-grained VL, interleaved | nothing (all unlocked) | all | 448 | 19k | 4096 |
| 3 SFT (Qwen-VL-Chat) | 350k instruction data | ViT | LLM + adapter | 448 | 8k | 128 |
Stage 1 ~1.5B samples seen (text: "approximately 1.5 billion image-text samples").
Peak LR 2e-4 / 5e-5 / 1e-5. Stage 1 LLM seq 512, stage 2/3 2048.

## Headline numbers
- Flickr30K zero-shot caption CIDEr 85.8 (beats Flamingo-80B).
- RefCOCO val/testA/testB: Qwen-VL-7B 89.36 / 92.26 / 85.34; Shikra-13B 87.83/91.11/81.81.
- Text-VQA table: Qwen-VL TextVQA 63.8, DocVQA 65.1, ChartQA 65.7, AI2D 62.3, OCR-VQA 75.7.
  (Specialist PALI-X-55B DocVQA 80.0; Pix2Struct-Large 76.6.) DocVQA 65.1 is the "336px cannot read a page" baseline.

## Outline assertions I could not confirm or that are wrong
- Outline "cross-attn adapter to 256 tokens": confirmed (fixed 256 queries, single layer).
- Outline ch-resolution title says 336 px: that is not Qwen-VL (448) -- it is LLaVA-1.5's 336. Fine as a hook but do not attribute 336 to Qwen.
- Qwen-VL box normalisation is [0, 1000) (half-open), not 0-1000 inclusive; outline example format matches.
- The paper's 2D positional encoding in the adapter is described only in one sentence: "absolute", added to query-key pairs; no equation. Do not draw more detail than that.
- No <box> location-token vocabulary: coordinates are plain decimal text digits.
