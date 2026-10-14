# Florence-2 (2311.06242v1) -- outline ch 9 ch-grounding

## Core idea in plain words
Every vision task is cast as "image + text prompt -> text out", with one seq2seq model and one cross-entropy loss. Boxes/polygons are text: the tokenizer gets extra location tokens (quantised coordinates), so detection, grounding, region captioning, OCR-with-regions and referring segmentation are all next-token prediction. No task-specific heads (Sec. 3, "Task formulation").

## Location tokens (Sec. 3 "Task formulation", Region bullet)
- "we add location tokens to the tokenizer's vocabulary list, representing quantized coordinates. We create 1,000 bins" (similar to Pix2Seq, Unified-IO, OFA).
- Box (x0,y0,x1,y1): 4 location tokens, top-left then bottom-right corner. Used for object detection, dense region captioning.
- Quad box (x0,y0,...,x3,y3): 8 location tokens, corners "starting from the top-left and going clockwise". Used for text detection and recognition (OCR with region).
- Polygon (x0,y0,...,xn,yn): 2n location tokens, vertices "in clockwise order". Used for referring segmentation.
- Exact token spelling: the main text never writes it, but the appendix does. Fig. 16 (Appendix E.6, Region to segmentation, PDF p. 29) prompts read "What is the polygon mask of region <loc_586><loc_294><loc_929><loc_814>": the LaTeX source (sec/results/region_to_seg.tex) has $\langle$loc\_586$\rangle$, i.e. angle brackets and an underscore, and the rendered page agrees. pdftotext drops the underscore (its text layer shows "loc 586"). Values seen: 7, 53, 66, 102, 266, 294, 314, 317, 375, 386, 541, 583, 586, 627, 648, 692, 759, 794, 814, 893, 904, 929, 998. So the paper form is <loc_N> with N an integer bin index. The paper does not state whether bins are 0-999 or 1-1000; observed max in figures is 998. Bins are of normalised coordinates (image-relative); the paper does not say the normalisation explicitly, but 1,000 bins over the image extent is implied.
- Region as prompt input: "Region to text" task takes region as input (Table 13 task table: Prompt Input = Image, text, region).
- Quantisation error for a viz: with 1000 bins over an image side of S px, one bin = S/1000 px (derived, not in paper).

## Architecture (Sec. 3, Fig. 2; appendix Table 15)
- Vision encoder: DaViT. Image I in R^{H x W x 3} -> flattened visual tokens V in R^{N_v x D_v}.
- Multi-modality encoder-decoder: standard transformer enc-dec (BART-style word embedding layer + extended tokenizer). Prompt embeddings T_prompt in R^{N_t x D}; V' = LayerNorm(Linear(V)) in R^{N_v x D}; encoder input X = [V', T_prompt] (concatenate). Decoder generates the answer text.
- Init: image encoder from UniCL, enc-dec from BART (Sec. 4.1 Setup).
- Loss (unnumbered, Sec. 3 "Optimization objective"):
  `\mathcal{L} = - \sum_{i=1}^{|y|} \log P_\theta(y_i | y_{<i}, x)`
  x = image + prompt combined; y = target tokens; theta = all network parameters; |y| = number of target tokens; i indexes target position.
- Table 15: Florence-2-B: DaViT dims [128,256,512,1024], blocks [1,1,9,1], heads/groups [4,8,16,32], 90M; enc-dec 6 enc layers, 6 dec layers, dim 768, 140M. Florence-2-L: DaViT dims [256,512,1024,2048], blocks [1,1,9,1], heads [8,16,32,64], 360M; enc-dec 12+12 layers, dim 1024, 410M. Totals stated in text: 232M and 771M (Table 1 headline: 0.23B / 0.77B).
- Training: AdamW, cosine LR, max LR 1e-4 (B) / 1e-5 (L), 5,000 warm-up steps; batch 2048/3072 (B/L); 384x384 until 3 billion effective samples; then high-res tuning at 768x768 for 0.5B (B) / 0.1B (L) samples.

## Tasks (appendix Table 13)
12 tasks: caption, detailed caption, more detailed caption, region proposal, object detection, dense region caption, phrase grounding, referring expression comprehension, open vocabulary detection, referring segmentation, region to text, text detection and recognition. Annotation types: Text, Region, Region-Text, Text-Phrase-Region.

## FLD-5B (Sec. 4-5)
- 126M images from 5 sources: ImageNet-22k, Object 365, Open Images, Conceptual Captions, LAION filtered by BLIP.
- Annotations: 500M text + 1.3B region-text + 3.6B text-phrase-region = 5.4B (intro says 5.4B; Fig. 3 caption and Table 1 round to "5B").
- Table 3 (annotation stats): Text: brief 235M image annotations, 7.95 avg tokens; detailed 126M, 31.65; more detailed 126M, 70.53. Region-text: 681M regions, 5.42 avg regions/image (phrase avg 1.19 tokens, brief 2.55). Text-phrase-region: brief 1007M regions (4.27/img), detailed 1289M (10.25), more detailed 1278M (10.17). Note 681M regions in Table 3 vs "1.3B region-text annotations" in text = phrase 681M + brief 681M (sum 1.36B), consistent.
- Region-text is 30x larger than OpenImages/Objects365 (text claim).
## Data engine loop (Sec. 4.2, Fig. 3): three phases
1. Initial annotation by specialist models (offline models on public data + cloud services; merged with existing human labels e.g. Objects365 boxes). Text: Florence-2-style captioner for brief; LLM/LMM prompted with brief text + region-text for detailed/more detailed (small set, then used to fine-tune a detailed-caption specialist). Region-text: Azure OCR API for text regions, DINO detector for objects; cropped-region captions; each region gets phrase (category), brief text, noun-phrase chunks; Florence-1 picks the most similar. Text-phrase-region: Grounding DINO for noun phrases -> boxes; SAM makes masks.
2. Filtering: SpaCy parser (from DiHT) drops texts with excessive objects, keeps texts with minimum action/object complexity (dependency-tree node degree); boxes under confidence threshold removed, NMS for overlaps; noun-phrase blacklist (pronouns, abstract concepts).
3. Iterative refinement: train a multitask model on filtered annotations, re-predict on training images (it corrects noisy alt-text etc.), merge with originals, retrain; repeat. Tasks skipped initially (detailed descriptions) are then annotated by the iteratively trained model fine-tuned on the sparse data.

## Headline numbers
- Zero-shot (Table 4, Florence-2-B / -L): COCO caption Karpathy test CIDEr 133.0 / 135.6 (Flamingo 80B: 84.3); COCO det val2017 mAP 34.7 / 37.5; Flickr30k R@1 83.6 / 84.4 (Kosmos-2 1.6B: 78.7); RefCOCO val acc 53.9 / 56.3 (Kosmos-2 52.3); RefCOCO+ val 51.5 / 53.6 (45.5); RefCOCOg val 66.3 / 68.0 (60.6), test 65.1 / 67.0 (61.7); RefCOCO RES val mIoU 34.6 / 35.8 (not supported by prior foundation models).
- Fine-tuned generalist (Table 5, region tasks; -B / -L): COCO det val2017 mAP 41.4 / 43.4; Flickr30k R@1 84.0 / 85.2; RefCOCO val 92.6 / 93.4; RefCOCO+ val 86.8 / 88.3; RefCOCOg val 89.8 / 91.2; RES val mIoU 78.0 / 80.5. Sota comparisons: L beats PolyFormer on RefCOCO REC by 3.0 Accuracy@0.5 and RES by 3.54 mIoU; beats UNINEXT by 0.8 on RefCOCO. TextVQA generalist L 73.5 in Table 4 table, but text says "81.5 without external OCR" (see wrong assertions).
- Model scaling (Table 8, zero-shot; different training setting): Base vs Large caption CIDEr 118.7 vs 124.4, det AP 19.7 vs 22.6, grounding R@1 76.3 vs 78.2, RES mIoU 18.6 vs 21.5.
- Data scaling (Table 9, 0.12M/0.36M/1.2M/12M images, same 72M samples): CIDEr 102.8/114.3/118.1/118.7; det AP 16.1/18.7/18.9/19.7; grounding 74.0/75.8/76.3/76.3; RES mIoU 15.9/16.6/19.3/18.6.
- Basic components (Table 7): freezing the vision encoder collapses detection (6.9 AP vs 19.7 unfrozen); vision-pretrained + LM-pretrained unfrozen = best overall (CIDEr 118.7, AP 19.7).

## Outline assertions I could not confirm or that are wrong
- "Florence-2 <loc_k> tokens": confirmed. The main text never spells a token out; the Appendix E.6 prompts (Fig. 16, PDF p. 29) print <loc_586> etc. (angle brackets and an underscore; pdftotext drops the underscore).
- "5.4B annotations": correct per intro and abstract-figure caption, but Fig. 3 caption / Table 1 say "5B" (rounded).
- 1000 bins: confirmed. 0.23B/0.77B: confirmed (232M/771M).
- Bin range (0-999 vs 1-1000) and the exact normalisation are not stated in the paper.
- TextVQA inconsistency inside the paper: text says generalist Florence-2-L gets 81.5 TextVQA; Table 4 lists 73.5. Not needed for the chapter; use neither unless flagged.
- Table 5 lists Florence-2-B RefCOCOg test 82.2 (lower than val 89.8 and than L 91.7); likely a typo in the paper; avoid.
