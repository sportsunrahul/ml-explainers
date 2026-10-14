# BLIP (2201.12086v2) -- ch-bootstrap (outline chapter 4)

Source: sources/blip/src/3-method.tex, 4-experiment.tex, appendix.tex, table/*.tex.

## MED architecture (Sec. 3.1)
- Image encoder: ViT (ViT-B/16 default; ViT-L/16 variant), [CLS] token = global image feature. Init from ImageNet-pretrained ViT (DeiT/ViT). Text side init from BERT-base.
- MED = one multi-task model that runs in one of three modes:
  1. Unimodal encoder: image and text encoded separately. Text encoder = BERT, [CLS] prepended.
  2. Image-grounded text encoder: one extra cross-attention (CA) layer inserted between self-attention (SA) and FFN in every transformer block of the text encoder. Task token [Encode] appended; its output embedding = multimodal representation.
  3. Image-grounded text decoder: the bidirectional SA of (2) replaced by causal SA. [Decode] token = start of sequence, an end-of-sequence token = end.
- Parameter sharing (Sec. 3.2, Sec. 4.4, Table "med"): text encoder and text decoder SHARE ALL PARAMETERS EXCEPT THE SELF-ATTENTION LAYERS. So shared = embeddings, CA layers, FFN. Separate = SA (bidirectional in the encoder, causal in the decoder). Rationale in paper: encoding vs decoding differences are best captured by SA; embeddings/CA/FFN function similarly.
- The ViT is shared by all three objectives (one ViT forward pass per pair; three passes through the text transformer).
- Table (pre-train on 14M web-text, no CapFilt): layers shared -> #params, COCO TR@1 / IR@1, Flickr ZS TR@1 / IR@1, COCO cap B@4 / CIDEr, NoCaps CIDEr / SPICE:
  - All: 224M; 77.3/59.5; 93.1/81.0; 37.2/125.9; 100.9/13.1
  - All except CA: 252M; 77.5/59.9; 93.1/81.3; 37.4/126.1; 101.2/13.1
  - All except SA (the chosen setting): 252M; 78.4/60.7; 93.9/82.1; 38.0/127.8; 102.2/13.9
  - None: 361M; 78.3/60.5; 93.6/81.9; 37.8/127.4; 101.8/13.9
  Sharing SA degrades performance ("conflict between encoding and decoding"); sharing all but SA beats no sharing while being 252M vs 361M.

## Three objectives and which module each activates (Sec. 3.2)
| Objective | Module activated | Head / details |
|---|---|---|
| ITC (image-text contrastive) | unimodal encoders (ViT + text encoder) | ALBEF-style: momentum encoder produces features; soft labels from momentum encoder as targets |
| ITM (image-text matching) | image-grounded text encoder | binary classification, ITM head = a linear layer; hard negative mining (ALBEF): in-batch negatives with higher contrastive similarity more likely sampled |
| LM (language modeling) | image-grounded text decoder | autoregressive cross-entropy on the text; label smoothing 0.1 |
Each image-text pair: 1 ViT forward pass + 3 text-transformer forward passes. Losses are jointly optimised.
No numbered equations in the BLIP method section (objectives are described in prose only).

## CapFilt (Sec. 3.3), step by step
Data: human-annotated pairs {(I_h,T_h)} (COCO), web pairs {(I_w,T_w)} (alt-text, noisy).
1. Pre-train a MED (on the original noisy corpus).
2. Initialise BOTH captioner and filter from that SAME pre-trained MED; fine-tune each INDIVIDUALLY (end-to-end, decoupled parameters) on COCO (human-annotated). Lightweight finetune.
   - Captioner = image-grounded text decoder, finetuned with the LM objective.
   - Filter = image-grounded text encoder, finetuned with ITC and ITM objectives.
3. Captioner produces one synthetic caption T_s per web image I_w (nucleus sampling, p = 0.9).
4. Filter judges BOTH the original web text T_w and the synthetic text T_s; a text is removed if the ITM head predicts "unmatched" for the image.
5. Keep: filtered web pairs + filtered synthetic pairs + the human-annotated pairs = new dataset. Pre-train a NEW model on it.
Ablations (Table "capfilt", 14M, ViT-B; C = captioner, F = filter): TR@1/IR@1 COCO FT, TR@1/IR@1 Flickr ZS, B@4/CIDEr COCO, CIDEr/SPICE NoCaps
- neither: 78.4/60.7; 93.9/82.1; 38.0/127.8; 102.2/13.9
- F only: 79.1/61.5; 94.1/82.8; 38.1/128.2; 102.7/14.0
- C only: 79.7/62.0; 94.4/83.6; 38.4/128.9; 103.4/14.2
- C+F: 80.6/63.1; 94.8/84.9; 38.6/129.7; 105.1/14.4
- 129M, ViT-B: none 79.6/62.0; 94.3/83.6; 38.8/130.1; 105.4/14.2 ; C+F (B) 81.9/64.3; 96.0/85.0; 39.4/131.4; 106.3/14.3 ; C+F (L captioner/filter) 81.2/64.1; 96.0/85.5; 39.7/133.3; 109.6/14.7
- 129M, ViT-L: none 80.6/64.1; 95.1/85.5; 40.3/135.5; 112.5/14.7 ; C+F (L) 82.4/65.1; 96.7/86.7; 40.4/136.7; 113.2/14.8
Sharing parameters between captioner and filter hurts (confirmation bias): noise ratio 8% (shared) vs 25% (decoupled); COCO IR@1 62.2 vs 63.1; NoCaps CIDEr 103.5 vs 105.1.

## Nucleus vs beam (Sec. 4.3, Table "generate"; 14M images)
- Nucleus sampling: each token sampled from the smallest set whose cumulative probability mass exceeds p (p = 0.9). Stochastic.
- Beam search: deterministic, highest-probability caption.
- Filter-measured noise ratio: beam 19%, nucleus 25% (nucleus is noisier), yet nucleus is better downstream:
  - COCO retrieval FT TR@1/IR@1: none 78.4/60.7, beam 79.6/61.9, nucleus 80.6/63.1
  - Flickr ZS TR@1/IR@1: 93.9/82.1, 94.1/83.1, 94.8/84.9
  - COCO caption B@4/CIDEr: 38.0/127.8, 38.4/128.9, 38.6/129.7
  - NoCaps CIDEr/SPICE: 102.2/13.9, 103.5/14.2, 105.1/14.4
- Paper's explanation (hypothesis): nucleus captions are more diverse and surprising = more new information; beam gives "safe" common captions. Section title: "Diversity is Key for Synthetic Captions".
- Note: the beam-vs-nucleus comparison is on the NEW pre-training run; at inference for COCO captioning finetune the paper uses beam size 3, max length 20 (appendix).

## Pre-training data (Sec. 4.1, appendix Table "pretrain_data")
- 14M images (same as ALBEF): COCO 113K img / 567K text; Visual Genome 100K / 769K; SBU 860K / 860K; CC3M 3M / 3M; CC12M 10M / 10M. (Sum of images = ~14.07M.) COCO and VG are human-annotated; CC3M, CC12M, SBU are web.
- 129M images = 14M + LAION 115M (115M img / 115M text; only images with shorter edge > 256 px from LAION400M; only 1/5 of LAION used per epoch).
- Training: ViT init ImageNet, BERT-base text; 20 epochs, batch 2880 (ViT-B) / 2400 (ViT-L); AdamW wd 0.05; LR warm to 3e-4 (B) / 2e-4 (L), decay rate 0.85; 224x224 random crops (384 in finetune); two 16-GPU nodes.

## Headline gains (abstract)
+2.7% average recall@1 (image-text retrieval), +2.8% CIDEr (captioning), +1.6% VQA score; zero-shot transfer to video-language tasks.

## Outline assertions I could not confirm or that are wrong
- "dataset sizes (14M, 129M)": correct. But note 129M = 14M + 115M LAION; the pre-training dataset the outline calls 129M is where CapFilt is scaled, and the paper's CapFilt uses one caption per web image (not 10). The BLIP-2 paper's "10 captions ranked by CLIP" is BLIP-2's variant, not BLIP's.
- Outline says "objectives ITC/ITM/LM sharing one model": correct, but strictly the ViT is shared by all; the text encoder/decoder share all params except SA. ITC uses the unimodal encoder (not the image-grounded encoder).
- Headline "+2.7/+2.8/+1.6" are from the abstract versus prior SOTA, not CapFilt ablation gains; CapFilt's own gain is Table capfilt (e.g. 14M: COCO IR@1 60.7 -> 63.1).
