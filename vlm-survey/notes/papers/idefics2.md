# Idefics2 (2405.02246v1) -- chapter 10 ch-recipes, Idefics2 half

Source: sources/idefics2/src/sections/{model_architecture_choice,pretraining,finetuning,appendix}.tex.
Paper title question: "What matters when building vision-language models?"

## Ablation protocol (Sec. 3 intro)
Unless stated: 6,000 steps; metric = average of 4-shot scores on VQAv2, TextVQA, OKVQA, COCO (COCO is CIDEr,
the rest VQA acc.). OCR ablation is different: 5,500 pretrain steps + 500 steps fine-tune on DocVQA (lr 1e-5).
Which architecture each table used: LM backbone, vision backbone, captions = cross-attention arch.
Pooling, aspect-ratio = fully autoregressive arch. Architecture table = both.
Tables in the paper are unnumbered in the LaTeX (only labels); the arXiv PDF numbers them, cite by label/caption.

## Tables copied in full (avg score, higher is better)

### LM backbone swap (label ablations_archi_lm_backbone; cross-attention)
| LM backbone | Avg. score |
| Llama-1-7B | 62.5 |
| Mistral-7B | 67.6 |
Text: Llama-1-7B (35.1% MMLU) -> Mistral-7B (60.1% MMLU) "yields a boost of 5.1".

### Vision encoder swap (ablations_archi_vision_encode_backbone; cross-attention)
| VE backbone | Res. | Avg. score |
| CLIP-ViT-H | 224 | 57.4 |
| EVA-CLIP-5B | 224 | 60.2 |
| SigLIP-SO400M | 384 | 60.7 |
Text: CLIP-ViT-H (78.0% ImageNet) -> SigLIP-SO400M (83.2% ImageNet) "yields a 3.3 increase".
Detailed appendix table (adds size + per-benchmark):
| VE | Size | Res | Avg | VQAv2 | OKVQA | TextVQA | COCO |
| CLIP-ViT-H | 600M | 224 | 57.4 | 52.4 | 41.7 | 28.2 | 107.5 |
| EVA-CLIP-5B | 4.4B | 224 | 60.2 | 53.4 | 43.3 | 30.4 | 113.7 |
| SigLIP-SO400M | 400M | 384 | 60.6 | 53.6 | 43.4 | 33.8 | 111.6 |
(SigLIP avg is 60.7 in main text table and 60.6 in the appendix table; see discrepancies.)
Confound stated by the paper: appendix says TextVQA "is the most sensitive benchmark to image resolution,
which accounts for the performance increase" (SigLIP ran at 384, the others at 224). Appendix: EVA-CLIP-5B
"has 11 times more parameters" (main text says "ten times bigger"). Paper hypothesis: EVA-CLIP-5B "could be
heavily under-trained".
Finding 1 (verbatim): "For a fixed number of parameters, the quality of the language model backbone has a
higher impact on the performance of the final VLM than the quality of the vision backbone."
NB the two swaps are not equal-effort: LM swap crosses a 25-point MMLU gap; vision swap crosses a 5.2-point
ImageNet gap. Do not present as "LM matters 1.5x more" without that caveat.

### Architecture x training method (ablations_archi_type_archi_method_training)
| Architecture | Backbones training | Avg. score |
| Fully autoreg. no Perceiver | Frozen | 51.8 |
| Fully autoreg. | Frozen | 60.3 |
| Cross-attention | Frozen | 66.7 |
| Cross-attention | LoRA | 67.3 |
| Fully autoreg. | LoRA | 69.5 |
Setup: cross-attention has 1.3B more trainable params (2B total trainable) than fully autoregressive; ~15%
(FA) vs ~25% (CA) of parameters trained when frozen; CA uses 10% more inference flops. CA feeds 1 token per
image to the LM, FA feeds 64 (appendix).
Text claims: frozen: "cross-attention architecture performs 7 points better" (66.7 vs 60.3 is 6.4; 66.7 vs
51.8 is 14.9; the paper's "7" is loose). Unfreezing via LoRA: "12.9 points increase under the fully
autoregressive architecture, and 0.6 point under the cross-attention" (69.5-60.3 = 9.2, NOT 12.9; see
discrepancies; 67.3-66.7 = 0.6 ok).
Instability: "training the fully autoregressive architecture would yield loss divergences [when all params are
unfrozen], and we were not successful in stabilizing the training even by aggressively lowering the learning
rate or gradually unfreezing various components." Fix: LoRA on the pre-trained parameters, standard full
fine-tuning for newly initialised ones. LoRA layers merge into linear layers, no inference cost.
Applied: LoRA on the LLM for FA; on the cross-attention modules and the LLM for CA (appendix).
Finding 2 (verbatim): "The cross-attention architecture performs better than the fully autoregressive one
when unimodal pre-trained backbones are kept frozen. However, when training the unimodal backbones, the fully
autoregressive architecture outperforms the cross-attention one, even though the latter has more parameters."
Finding 3 (verbatim): "Unfreezing the pre-trained backbones under the fully autoregressive architecture can
lead to training divergences. Leveraging LoRA still adds expressivity to the training and stabilizes it."
Paper says this contradicts Prismatic (which found unfreezing the visual backbone hurts); hypothesis: PEFT.
Appendix Fig (autoregressive_lora_vs_flamingo_lora): FA better "across the board" vs steps, images, text tokens.

### Pooling / number of visual tokens (ablations_archi_pooling; fully autoregressive)
| Pooling | # vis. tok. | Avg. score |
| Perceiver | 128 | 71.2 |
| Perceiver | 64 | 71.7 |
Text: learned pooling "increases the performance by 8.5 points on average and reduces the number of visual
tokens necessary for each image from 729 to 64" (8.5 = 60.3 - 51.8, the FA rows above; the "no Perceiver" row
is the 729-token no-pooling variant; 729 = 27x27 from SigLIP 384px / patch 14 -- 27x27 is my arithmetic, the
paper only says 729). "we observe no gains when using more than 64 visual tokens" (contrasts DePALM, MM1).
Other Perceiver variants (MAPL, register tokens, DePALM) "resulted in decreased performance".
Finding 4 (verbatim): "Reducing the number of visual tokens with learned pooling significantly improves
compute efficiency at training and inference while improving performance on downstream tasks."
Appendix connector ablation (training setup not stated (values match frozen FA rows); table label vision_language_adaptor_ablation, only in a
commented-out block so it is NOT rendered in the paper):
| Vision-language Connector | Avg. score |
| Linear Projection | 44.5 |
| MapNet (MAPL) | 51.8 |
| Perceiver | 60.3 |
Rendered appendix text: "The perceiver resampler significantly outperforms these two options."
Also: number of perceiver layers (rendered): 1 -> 69.3; 3 -> 68.6; 12 -> 69.0 ("no statistically
significant differences"; they settle on 3). 2-layer MLP modality projection before the resampler:
W/ 71.4, W/o 69.6.
Prior context: other VLMs use 576 (DeepSeek-VL) to 2890 (SPHINX-2k) visual tokens per image.

### Aspect-ratio preserving (ablations_archi_aspect_ratio_preserving; FA)
| Images | Res. | Avg. score |
| Square images | 768 | 73.1 |
| AR preserving | 378-768 | 72.1 |
Method: patches passed to encoder without resizing / changing aspect ratio (following Pix2Struct, Patch n' Pack);
SigLIP positional embeddings interpolated to higher resolution; encoder trained with LoRA.
Finding 5 (verbatim): "Adapting a vision encoder pre-trained on fixed-size square images to preserve
images' original aspect ratio and resolution does not degrade performance while speeding up training and
inference and reducing memory." NB it does score 1.0 lower (72.1 vs 73.1); the paper says "maintains performance".

### Image splitting (no ablation table; evidence in the main results table + prose)
Method: image -> 4 equal sub-images (crops) + the original = 5 images. 64 tokens/image standalone,
320 tokens total with splitting (5 x 64). Applied in the instruction fine-tuning stage.
Only 50% of training samples need splitting: "does not impair the performance increase".
Evaluation: raising sub-image resolution to max only gives a minor boost vs splitting alone: TextVQA val
73.6% (max res) vs 73.0%; DocVQA val 72.7 vs 72.9 ANLS.
Same-weights evidence (Table perf_sft): Idefics2 8B, 64 tok: MMMU 43.5/37.9, MathVista 51.6, TextVQA 70.4,
MMBench 76.8; Idefics2 8B, 320 tok: MMMU 43.0/37.7, MathVista 51.4, TextVQA 73.0, MMBench 76.7.
Appendix full table (perf_sft_full) adds DocVQA and VQAv2: 64 tok -> DocVQA 67.3, VQAv2 80.8; 320 tok ->
DocVQA 74.0, VQAv2 81.2 (DocVQA +6.7 from splitting, the biggest gain).
Finding 6 (verbatim): "Splitting images into sub-images during training allow trading compute efficiency
for more performance during inference. The increase in performance is particularly noticeable in tasks
involving reading text in an image."

### Synthetic captions vs alt-text (ablations_pretraining_type_captions; cross-attention)
| Captions | Avg. score |
| Alt-texts | 49.8 |
| Synthetic | 52.9 |
Synthetic = LAION COCO (images captioned by a model trained on COCO). Text: "This improves the quality of
the training samples and thus the quality of the resulting model." (+3.1). Commented-out Finding 7: "Replacing
original alt-text with synthetic captions ... significantly boosts performance." NSFW filter removed 7% of
LAION COCO.

### OCR data x resolution (ablations_finetuning_ocr; 5,500 pretrain + 500 DocVQA ft steps)
| OCR data | Res. | DocVQA |
| W/o | 384 | 22.6 |
| W/o | 768 | 42.9 |
| W/ | 768 | 49.9 |
Text: PDF/OCR integrations "significantly boost the performance on benchmarks that require reading text
without decreasing the performance on other benchmarks". Appendix: zero/few-shot showed "no statistically
significant differences"; only after DocVQA fine-tune did the gap appear (so it is a fine-tune-probe metric).
Resolution effect +20.3 (22.6 -> 42.9); OCR data effect at 768 +7.0 (42.9 -> 49.9).

## Final model spec
- Vision encoder: SigLIP-SO400M (400M params; ablation table lists 400M), init from pre-trained,
  aspect-ratio preserving, trained with LoRA; positional embeddings interpolated.
- LLM: Mistral-7B-v0.1 (Sec. 4.1).
- Connector: 2-layer MLP modality projection then Perceiver resampler, 3 layers, 64 latents -> 64 visual
  tokens per image (320 with 4-crop splitting). Architecture: fully autoregressive (visual tokens concatenated
  with text embeddings; Fig architecture_idefics2).
- Total: "8 billion parameters". Resolution: max 384 px in stage 1; up to 980 px in stage 2 ("a maximum of 980
  pixels"). Images randomly scaled up during training.
- Pre-training: two stages. Stage 1 (max res 384, global batch 2,048 = 17k images + 2.5M text tokens):
  OBELICS 70% (max seq 2,048), image-text pairs 30% (max seq 1,536). Stage 2 (max res 980): OBELICS 45%,
  image-text 35%, PDF 20% (max seq 1,024). lr 1e-4, ~2 epochs, ~1.5 billion images and 225 billion text
  tokens. Whole model trained in both stages (some params via LoRA), not selective freezing.
- Data: OBELICS (350M images, 115B text tokens); image-text pairs from PMD (human-annotated) + LAION COCO
  synthetic captions; PDFs: OCR-IDL 19M industry documents, PDFA 18M pages, Rendered Text.
- Instruction tuning: The Cauldron, 50 vision-language datasets; DoRA; NEFTune; loss on answers only; plus
  text-only instruction data. Then Idefics2-chatty via a few hundred steps on LLaVA-Conv + ShareGPT4V.
- Results: Idefics2-base (8B, FA, 64 tok) 8-shot: VQAv2 70.3, TextVQA 57.9, OKVQA 54.6, COCO 116.0 vs MM1
  7B (144 tok) 63.6 / 46.3 / 51.4 / 116.3. Instruct vs LLaVA-NeXT 13B (2880 tok), DeepSeek-VL 7B (576),
  MM1-Chat 7B (720): see Table perf_sft numbers above and paper claims "on par with models 4x its size".

## Outline assertions I could not confirm or that are wrong
- Outline "Idefics2 sizes (8B, SigLIP-SO400M + Mistral-7B, 64 tokens, 980px)": all confirmed. 980 px is the
  MAX resolution of stage 2 (and inference upper bound), not a fixed input size; 384 in stage 1.
- Outline "connector style matters less": the paper does NOT say connector style matters little. Its own
  numbers: Linear 44.5 vs MapNet 51.8 vs Perceiver 60.3 (16-point spread, appendix, rendered text only says
  Perceiver "significantly outperforms"). What is "less important" is the CA-vs-FA choice once backbones are
  trained (+2.2, 67.3 vs 69.5) and token count above 64. Reword.
- Text says cross-attention is "7 points better" when frozen; table gap is 6.4 (66.7 vs 60.3).
- Text says LoRA unfreezing gives "12.9 points increase under the fully autoregressive architecture"; table
  gives 60.3 -> 69.5 = 9.2. 12.9 does not match any pair of rows in the LaTeX table (69.5-51.8 = 17.7).
  Show table values, not 12.9.
- SigLIP-SO400M avg: 60.7 in the main-text table, 60.6 in the appendix detailed table.
- Vision-encoder swap is confounded with resolution (224 vs 384); LM swap with a 25-point MMLU gap.
- "8.5 points" from pooling compares FA-frozen with vs without Perceiver (51.8 -> 60.3), not the 128 vs 64
  table.
- The pooling table (71.2/71.7) and the AR table (73.1/72.1) use a different, stronger training setup than the
  architecture table, so absolute numbers are not comparable across tables.
- Image-splitting has no ablation table in the paper (evidence is benchmark rows at 64 vs 320 tokens plus
  prose); the viz tab for splitting must use the perf_sft rows.
- Appendix connector table (44.5/51.8/60.3) is inside a \begin{comment} block in the LaTeX; verify it is
  visible in the PDF before showing.
