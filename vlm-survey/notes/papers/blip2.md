# BLIP-2 (2301.12597v3) -- ch-qformer (outline chapter 6)

Source: sources/blip2/src/3-method.tex, 4-experiment.tex, table/zeroshot.tex, table/vqa_zeroshot.tex, figure/stage1.tex, figure/stage2.tex.

## Q-Former architecture (Sec. 3.1)
- Two transformer submodules that SHARE the same self-attention layers: (1) image transformer (interacts with frozen image encoder), (2) text transformer (works as text encoder and text decoder).
- Learnable query embeddings feed the image transformer: 32 queries, each 768-d (= Q-Former hidden dim = BERT-base). Output Z is 32 x 768. Queries are counted as model parameters.
- Queries interact with each other via self-attention, with frozen image features via cross-attention (inserted every other transformer block), and with text via the same self-attention layers, controlled by a per-objective self-attention mask.
- Init: BERT-base weights; cross-attention layers randomly initialised. Total 188M parameters (Sec. 3.1; Table 1 "#Trainable params" column).
- Bottleneck: Z is 32x768 vs frozen image features 257x1024 for ViT-L/14 (independent of input resolution).
- Frozen image feature source: last layer of ViT removed, second-to-last layer output used ("slightly better").

## Stage 1: representation learning (Sec. 3.2), three objectives sharing input format and params
Text side notation: text tokens T; Q = 32 queries. No numbered equations; all described in prose.
1. ITC: Z from the image transformer vs t = output embedding of text [CLS]. Similarity of image and text = MAX over the 32 query-output similarities with t (pairwise similarity per query, take highest). Mask: UNI-MODAL self-attention mask: queries and text cannot see each other (queries attend to queries; text tokens attend to text tokens). Negatives: in-batch negatives (no momentum queue, unlike BLIP), because the frozen encoder lets more samples fit per GPU.
2. ITG (image-grounded text generation): Mask: MULTIMODAL CAUSAL mask (UniLM style): queries attend to each other but NOT to text tokens; each text token attends to ALL queries and its PREVIOUS text tokens (causal in text). [CLS] replaced by [DEC] as first token. Text can only get image information via the queries (no direct path from frozen encoder to text tokens).
3. ITM: BI-DIRECTIONAL mask: all queries and all text tokens see each other. Each output query embedding goes into a two-class linear classifier; logits averaged across the 32 queries = matching score. Hard negative mining from ALBEF/BLIP.
Summary of who sees whom (queries Q, text T):
| | Q->Q | Q->T | T->Q | T->T |
|---|---|---|---|---|
| ITC | yes | no | no | yes (bidirectional, text-only encoder) |
| ITG | yes | no | yes (all queries) | causal (previous tokens only) |
| ITM | yes | yes | yes | yes (bidirectional) |
(Q->Q full bidirectional in all three is my reading of "queries can attend to each other"; ITC's Q->Q and T->T entries are implied by "unimodal", not spelled out in prose. Fig. 2 right panel of the paper shows the mask matrices; check the PDF figure if the visual needs exact cell shading.)

## Stage 2: generative learning (Sec. 3.3)
- FC layer linearly projects Z (32 x 768) to the LLM's text-embedding dimension; projected queries are PREPENDED to the input text embeddings = soft visual prompts. LLM and image encoder frozen.
- Decoder-only LLM (OPT): standard language modeling loss; the frozen LLM generates the text conditioned on the visual prompt.
- Encoder-decoder LLM (FlanT5): prefix language modeling loss; text is split into prefix and suffix; prefix text is concatenated with the projected visual representation as ENCODER input; suffix is the DECODER's generation target.
- (In stage 2 only the Q-Former's image-transformer path + FC are trained; the paper does not spell this out but Table 2 lists 103-108M trainable params vs 188M Q-Former total.)

## Frozen models (Sec. 3.4)
- Image encoders: ViT-L/14 from CLIP; ViT-g/14 from EVA-CLIP.
- LLMs: OPT family (decoder, unsupervised; OPT-2.7B, OPT-6.7B in tables); FlanT5 family (encoder-decoder, instruction-trained; FlanT5-XL, FlanT5-XXL in tables; text mentions 11B FlanT5-XXL).
- fp16 for frozen ViT and LLM, bfloat16 for FlanT5.

## Data and compute (Sec. 3.4)
- 129M images (COCO, VG, CC3M, CC12M, SBU, 115M LAION400M) = same as BLIP. CapFilt: 10 captions per web image from BLIP-large captioner; rank the 10 synthetic + original web caption by CLIP ViT-L/14 image-text similarity; keep top-2 per image; randomly sample one each step.
- 250k steps stage 1, 80k steps stage 2. Batch sizes: stage 1 2320 (ViT-L) / 1680 (ViT-g); stage 2 1920 (OPT) / 1520 (FlanT5). AdamW, peak LR 1e-4, 2k warmup, min LR in stage 2 5e-5, images 224x224.
- Largest model (ViT-g + FlanT5-XXL) on one 16-A100(40G) machine: < 6 days stage 1, < 3 days stage 2.

## Trainable-params table and Flamingo80B comparison
Table 2 (zero-shot VQA), columns trainable / total params; VQAv2 val, test-dev; OK-VQA test; GQA test-dev:
- Frozen: 40M / 7.1B; VQAv2 val 29.6; OK-VQA 5.9
- FewVLM: 740M / 785M; 47.7; -; 16.5; 29.3
- VLKD: 406M / 832M; 42.6; 44.5; 13.3
- Flamingo3B: 1.4B / 3.2B; test-dev 49.2; OK-VQA 41.2
- Flamingo9B: 1.8B / 9.3B; 51.8; 44.7
- Flamingo80B: 10.2B / 80B; test-dev 56.3; OK-VQA 50.6 (best)
- BLIP-2 ViT-L OPT2.7B: 104M / 3.1B; 50.1 / 49.7; 30.2; 33.9
- BLIP-2 ViT-g OPT2.7B: 107M / 3.8B; 53.5 / 52.3; 31.7; 34.6
- BLIP-2 ViT-g OPT6.7B: 108M / 7.8B; 54.3 / 52.6; 36.4; 36.4
- BLIP-2 ViT-L FlanT5XL: 103M / 3.4B; 62.6 / 62.3; 39.4; 44.4
- BLIP-2 ViT-g FlanT5XL: 107M / 4.1B; 63.1 / 63.0; 40.7; 44.2
- BLIP-2 ViT-g FlanT5XXL: 108M / 12.1B; 65.2 / 65.0; 45.9; 44.7
Table 1 (overview, zero-shot): BLIP-2 188M trainable; VQAv2 test-dev 65.0; NoCaps CIDEr 121.6 / SPICE 15.8; Flickr TR@1 97.6 / IR@1 89.7. Flamingo 10.2B, VQAv2 56.3. BLIP 583M, NoCaps 113.2/14.8.
Headline: "outperforms Flamingo80B by 8.7% on zero-shot VQAv2 with 54x fewer trainable parameters" (abstract, Sec. 4.1). Arithmetic: 65.0 - 56.3 = 8.7 (ABSOLUTE points on VQAv2 test-dev, not relative); 10.2B / 188M = 54.3, so the 54x uses the 188M Q-Former total. Using the 108M actually trained for the ViT-g FlanT5-XXL row in Table 2, the ratio is 10.2B / 108M = ~94x.
On OK-VQA BLIP-2 (45.9) is second to Flamingo80B (50.6); paper hypothesis: 70B Chinchilla holds more knowledge than 11B FlanT5-XXL.
A commented-out Table (param_count, NOT in the compiled paper) lists BLIP-2 199M / 14M images: do not use.

## Outline assertions I could not confirm or that are wrong
- "+8.7% zero-shot VQAv2": it is +8.7 absolute points (65.0 vs 56.3), stated as "8.7%" in the paper. Page should say "8.7 points".
- "54x fewer trainable parameters": paper's 54x = 10.2B / 188M; Table 2 reports ~104-108M trainable for the BLIP-2 rows (ratio ~94x for the XXL row). The paper does not reconcile these; my inference is 188M counts the full Q-Former incl. text transformer, stage 2 trains fewer. Mark inference as uncertain.
- "129M images": correct, but the images are captioned via BLIP-large CapFilt (10 captions, CLIP-ranked top-2), not BLIP's one-caption CapFilt.
- "compute": paper only gives an upper bound (<6 days + <3 days, one 16xA100-40G machine), for the largest config.
- Outline says "FC projection into LLM": correct (single fully-connected layer, linear).
- Q-Former "188M params" is listed as the total Q-Former size; queries counted inside it.
