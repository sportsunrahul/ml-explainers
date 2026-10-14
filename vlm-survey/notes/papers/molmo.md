# Molmo / PixMo (2409.17146v2) -- outline ch 9 (pointing) and ch 10 (ch-recipes)

## Architecture (Sec. 2, Fig. 2, appendix "Image Encoding")
Four parts: (1) pre-processor -> multiscale multi-crop images, (2) ViT computing per-patch features per crop independently, (3) connector that pools and projects to LLM space, (4) decoder-only LLM.
- Encoder: OpenAI ViT-L/14 336px CLIP (default; SigLIP-So400m 384px, MetaCLIP, DINOv2 also tested). LLMs: OLMo-7B-1024-preview (fully open), OLMoE-1B-7B (MolmoE-1B, most efficient), Qwen2 7B (Molmo-7B-D), Qwen2 72B (Molmo-72B, best).
- Multi-crop: image tiled by a rectangular grid of ViT-sized squares (2x2, 3x1, ...) plus one low-res full-image view. Grid chosen for least up-scaling (ties: smallest); max crop count, else down-scaled; padded with black borders (learned embedding marks no/some/all padding per patch). Typically 12 high-res crops, 12x12=144 pooled patches per crop.
- Overlap: crops move closer so they overlap by a margin of 4 patches (56 px). Overlap patch features are NOT passed to the connector/LLM, so passed features exactly tile the image; overlap only supplies context to border patches.
- Connector: concat features from third-to-last and tenth-from-last ViT layers; each 2x2 patch window pooled to ONE vector by multi-head attention whose query is the mean of the 4 patches; then MLP to LLM embedding space. (576 ViT patches/crop -> 144 tokens/crop.)
- Token order: low-res full image patches first, then crops row-major; special start/end tokens for both low- and high-res sequences, row-end (column) tokens between rows.
- Dropout: residual dropout in LLM only; in pre-training on dense captions, dropout only on text tokens (forces reliance on image).
- Multi-annotated images: all annotations of an image packed in one sequence with attention masks so annotations do not see each other; cuts images processed by two-thirds, training time by over half, +25% sequence length.
- Training: two stages only (no connector-only stage). Pre-train all parameters on PixMo-Cap 4 epochs; LR 2e-4 connector, 6e-6 ViT, 2e-5 LM; warmup 200 steps connector, 2000 ViT/LM; AdamW cosine to 10%. Then fine-tune on PixMo + academic mix (sqrt-size sampling, pointing up-weighted, style tags like "vqa2:").

## PixMo datasets (Sec. 3)
- PixMo-Cap: 712k distinct images, 1.3M transcripts and captions; ~70 topics; annotators SPEAK the description: initially 3 annotators per image, >=60 s; later 1 annotator per image, 90 s minimum. 7 guiding questions (first glance, objects and counts, text, positions, subtle details, background, style/colour). Speech-to-text -> language-only LLM merges/cleans transcripts. Audio recording is an "audio receipt" proving no VLM was used. Average 196 words (COCO 11, Localized Narratives 37). Pre-training prompts include a length hint 90% of the time.
- PixMo-AskModelAnything (AMA): 162k QA pairs on 73k images; annotator writes question; a non-VLM OCR + PixMo-Cap-trained captioner feeds a language-only LLM to answer; annotator accepts or requests revision.
- PixMo-Points: 2.3M question-points pairs from 223k images (main text); plus 79k point-explanation annotations on 14k images. Annotators point at an item, describe it, then point to each instance; "not present" data included. Appendix: 229k unique images, 1.98M referring expressions, 8.7 expressions/image, 5.5 points/expression, 47.7 points/image, 359k no-target instances. Training excludes counts above 40.
- PixMo-CapQA: 214k QA from 165k images (LLM sees only ground-truth caption). PixMo-Docs: 255k code-generated charts/docs/tables/diagrams, 2.3M QA. PixMo-Clocks: 826k synthetic clock examples (~50 watch bodies, ~160k faces). PixMo-Count: 36k train images with points + QA (val/test 540 images each, verified, counts 2-10).
- No-distillation statement (abstract): "The strongest open-weight models rely heavily on synthetic data from proprietary VLMs ... effectively distilling these closed VLMs into open ones." Molmo datasets "all collected without the use of external VLMs". Intro: "without any reliance on synthetic data from other VLMs, including proprietary ones". Synthetic PixMo sets use LLMs/detectors but not VLMs ("no or minimal human annotations, but still not using a VLM").

## Point output format (Sec. 4 Fine-tuning; Appendix "Pointing")
- "the model outputs points as plain-text coordinates normalized between 0 and 100." (x,y) scaled to 0-100. Multiple items ordered top-down, left-to-right, each point numbered.
- Single: `<point x="10.0" y="10.0" alt="alt text">Inline text</point>`
- Multiple: `<points x1="10.0" y1="10.0" x2="20.0" y1="20.0" ... alt="alt text">Inline text</points>` (the paper's own example repeats "y1" for the second point; it is a typo for y2). Numbering makes the count = index of the last point.
- Counting = chain of thought by pointing, model then states total count.
- Ablation: plain text (one decimal) vs adding 1000 special point tokens at the same spatial precision: plain text wins.

## Counting results
- Ablation Table 6 (CBQA = CountBenchQA test, PCQA = PixMo-Count val; models fine-tuned on PixMo-Points + PixMo-Count only): count only 87.9 / 80.2; point then count (default) 89.4 / 86.3; count then point 81.5 / 77.6; pointing + regex 88.4 / 85.4. Point order ordered 89.4 / 86.3 vs unordered 85.4 / 74.1. Correct points+length vs random points, correct length 89.4/86.3 vs 85.9/76.3; random,random 76.3/75.7. Plain-text 89.4/86.3 vs special tokens 85.8/80.9.
- Main results (Table 1): CountBenchQA / PixMo-Count test: Molmo-72B 91.2 / 85.2; Molmo-7B-D 88.5 / 84.8; Molmo-7B-O 89.0 / 83.3; MolmoE-1B 87.2 / 79.6; GPT-4o 87.9 / 59.6; Gemini 1.5 Pro 85.8 / 64.3; Claude 3.5 Sonnet 89.7 / 58.3; GPT-4V 69.9 / 45.0; Qwen2-VL-72B 80.4 / 55.7.
- Pointing eval (493 image-question pairs, SAM masks, appendix Table): Molmo-72B P/R/F1 75.8/75.4/75.2; 7B-D 75.0/74.6/74.3; 7B-D with 36 test crops (train 12) collapses to 58.4/58.7/58.1. Counting/pointing require equal train and test crop counts.

## Human preference (Sec. 5, Table 1, Appendix)
- 15k image-text prompt pairs, ~870 annotators, pairwise preference, Elo via Bradley-Terry (Chatbot Arena method); ~450 feedback responses per model pair in first study; "I don't know" removed.
- Elo (Table 1; rank in parens of 27 models): GPT-4o-0513 1079 (1); Molmo-72B 1077 (2); Gemini 1.5 Pro 1074 (3); Claude-3.5 Sonnet 1069 (4); Llama-3.2V-90B 1063 (5); Molmo-7B-D 1056 (6); Gemini 1.5 Flash 1054 (7); LLaVA-OV-72B 1051 (8); Molmo-7B-O 1051 (9); GPT-4V 1041 (10); MolmoE-1B 1032 (13); Qwen2-VL-72B 1037 (12); Qwen2-VL-7B 1025 (14).
- Academic average over 11 benchmarks: Molmo-72B 81.2 (highest), GPT-4o 78.5, Gemini 1.5 Pro 78.3, Claude-3.5 Sonnet 76.7, Qwen2-VL-72B 79.4, Molmo-7B-D 77.3, MolmoE-1B 68.6, GPT-4V 71.1.
- Claim in abstract: 72B "outperforms ... Claude 3.5 Sonnet, and Gemini 1.5 Pro and Flash, second only to GPT-4o based on both academic benchmarks and on a large human evaluation". Metric = Elo from the authors' own human eval (and average of 11 academic benchmarks). Independent Chatbot Arena vision: Molmo-72B beats all open models but ranks below several proprietary (e.g. GPT-4o, Claude 3.5 Sonnet). Authors note Qwen2-VL is strong on benchmarks but weaker in Elo.
- Ablation Elo (Table 4, second study: 500 fixed questions, 50 per category): Claude-3.5 Sonnet 1047 (65% win vs default); PixMo-Cap w/ GPT-4o captions 1018 (55%); Molmo-7B-D default 1014; GPT-4V 1010 (47%); DINOv2 encoder 999 (45%); no PixMo-Cap 990 (35%); remove AMA 995 (40%); academic only 897 (17%). Win% excludes ties. Breakdown vs Claude 3.5 Sonnet: 45.5% tie-good, 14.1% tie-bad, Claude 26.1%, Molmo-7B-D 14.3%.

## Ablations (all on Molmo-7B-D; metrics: cap F1 after pre-training, 11-avg)
- Cropping (Tab 3d): single 46.7 / 62.8; multi no overlap 53.4 / 75.7; multi overlap (default) 54.1 / 76.9.
- Resolution (Tab 3b), train,test crops: 4,4 -> 52.0 / 71.0; 12,12 -> 54.1 / 74.9; 12,36 -> 54.1 / 76.9 (default); 36,36 -> 54.0 / 77.2.
- Pooling (Tab 3f): stacking 53.7 / 76.1; attention 54.1 / 76.9.
- Length conditioning (Tab 3e): off 53.0 / 76.2; on 54.1 / 76.9.
- Dropout (Tab 3c): off,off 53.1 / 74.6; on,on 53.7 / 77.0; text-only in pre-train, on in FT 54.1 / 76.9.
- Vision encoder (Tab 3a): OpenAI CLIP 336 54.1 / 76.9; MetaCLIP 54.1 / 77.2; SigLIP-So400m 384 54.4 / 77.1; DINOv2 53.2 / 75.6 (encoders roughly equal).
- PixMo-Cap scaling (Tab 4a): 0 -> 11-avg 74.9; 89k 49.6 / 75.5; 178k 51.6 / 76.3; 356k 52.6 / 76.2; 712k 54.1 / 76.9.
- Pre-training data (Tab 4b): LAION stage-0.5 connector-only 53.9 / 76.9 (no gain); ShareGPT4V+o 158k images 36.3 / 74.9; PixMo raw transcripts only 45.2 / 76.4; cleaned only 53.0 / 76.5; raw+cleaned 54.1 / 76.9; PixMo images captioned by GPT-4o 52.9 / 77.5 (human-spoken roughly equals distilling GPT-4o on the same images; note GPT-4o gives higher 11-avg).
- Fine-tuning data (Tab 4c, 11-avg): academic only 72.5; + PixMo-Docs 74.0; full 76.9; remove AMA 76.8; remove CapQA 77.0; remove Docs 75.8; remove Clocks 76.9; remove pointing 76.2.

## Benchmarks headline (Table 1, Molmo-72B): AI2D 96.3, ChartQA 87.3, VQA v2.0 86.5, DocVQA 93.5, InfoQA 81.9, TextVQA 83.1, RealWorldQA 75.2, MMMU 54.1, MathVista 58.6, CountBenchQA 91.2, PixMo-Count 85.2, Average 81.2.

## Outline assertions I could not confirm or that are wrong
- "spoken 60-90s": partially right. First 3 annotators per image spoke for >=60 s; later 1 annotator per image with 90 s minimum. Not a 60-90 s window.
- "712k images?": confirmed 712k distinct images (1.3M transcripts/captions).
- "coords 0-100?": confirmed, plain-text, normalised 0-100 (one decimal in the examples).
- "72B beats Claude 3.5 Sonnet/Gemini 1.5 Pro (which metric)": both the 11-benchmark average (81.2 vs 76.7 and 78.3) and the authors' own human-preference Elo (1077 vs 1069 and 1074); second only to GPT-4o (Elo 1079, avg 78.5). Independent Chatbot Arena ranks Molmo-72B below GPT-4o and Claude 3.5 Sonnet, so do not claim it beat Claude there.
- PixMo-Points counts differ inside the paper: main text 2.3M question-points pairs / 223k images; appendix 1.98M referring expressions / 229k images. Cite the main-text figure with the appendix caveat.
- The paper's own multi-point format example repeats "y1" (typo for "y2").
- Data ablation text-vs-table mismatch: appendix text says academic-only 72.2 vs 76.8 full; Table 4c shows 72.5 vs 76.9. Use the table.
- Molmo says the GPT-4o-caption variant is roughly on par (Elo 1018 vs 1014), so "human-spoken captions beat distilling a closed VLM" (outline ch 10 intuition) is NOT what the paper shows: it shows parity, and beats ShareGPT4V (158k) at similar scale. Restate as "matches" not "beats".
