# OWL-ViT: Simple Open-Vocabulary Object Detection with Vision Transformers

## 1. Identity
- Title: "Simple Open-Vocabulary Object Detection with Vision Transformers". Method name: OWL-ViT
  ("Vision Transformer for Open-World Localization").
- arXiv 2205.06230, read v2. v1 2022-05-12, v2 2022-07-20 (abs.html submission history).
- Authors: Matthias Minderer, Alexey Gritsenko (equal contribution) ... Neil Houlsby (14 authors), Google Research.
- Venue: abs.html comment "ECCV 2022 camera-ready version".
- Repo: github.com/google-research/scenic/tree/main/scenic/projects/owl_vit (JAX + Scenic). No README in sources.
- Sources read: src/main.tex, appendix.tex, all table_*.tex; paper.pdf for page numbers (PDF pages below).

## 2. Core problem
Transfer image-level contrastive image-text pre-training to open-vocabulary detection with a simple, scalable
recipe. Detection data is small (about 2M images) compared with billions of image-text pairs, so the recipe must
transfer without destroying pre-trained semantics.

## 3. What prior methods did and why that hurt (per the paper, sec 2)
- ViLD distills CLIP/ALIGN crop embeddings over a class-agnostic RPN; "The RPN, however, limits generalization
  performance on novel objects", made worse by the two-step distillation-training process.
- RegionCLIP: multistage (pseudo-labels on captions -> region-text pre-training -> detection transfer).
- MDETR and GLIP put a single text query for the whole image and frame detection as phrase grounding; "This
  limits the number of object categories that can be processed per forward pass." Early fusion "dramatically
  reduces inference efficiency" because each image/query combination needs a full image-model pass (sec 3.1).
- OVR-CNN: closest prior, but ResNet + Faster R-CNN + PixelBERT-style pre-training.

## 4. The one key idea
Take a contrastively pre-trained ViT, remove the final token pooling and projection, and attach a linear class
embedding head and a small box MLP to every output token; classify each token by its dot product with
text-encoder embeddings of the query strings. No decoder, no fusion, so queries (text or image-derived) are
computed independently of the image.

## 5. Mechanism, step by step
1. Pre-train image ViT and text Transformer contrastively from scratch on LiT data (3.6 billion image-text
   pairs, LiT "uu" setting), image representation by MAP pooling, text by the final EOS token. Alternatively
   use public CLIP checkpoints (sec 3.2, 4.1).
2. Remove token pooling and final projection from the image encoder (sec 3.1). Pre-training at 224 (288 for
   some hybrids); position embeddings linearly interpolated to the detection resolution (sec 4.1).
3. Each output token -> linear projection -> per-object image embedding (class embedding) (sec 3.1).
4. Each output token -> small MLP -> box. A bias makes each box by default centred on the image patch of its
   token when the token sequence is laid out as a 2D grid; the model predicts the difference from that default
   location, like RPN anchor offsets (sec 4.1). Box size default: not stated in the paper.
5. CLIP backbones only: the CLIP class token is multiplied into the other feature-map tokens, followed by layer
   norm (Appendix A1.3 "CLIP-based models"). So predictions = patch tokens (our reading).
6. Text queries: each query string is a separate token sequence (max 16 text tokens, A1.3), encoded alone; no
   concatenation of queries (sec 3.1). Query embedding = text-encoder output (EOS token).
7. Scores: logits of every token against every query; sigmoid probabilities. Max detections = number of tokens
   (at least 576; LVIS max instances per image is 294) (sec 3.1).
8. Training (sec 3.2, A1.3, A1.5, A1.9): DETR bipartite matching loss; classification = focal sigmoid
   cross-entropy (alpha 0.3, gamma 2.0) instead of softmax, because federated datasets have non-disjoint labels;
   box L1 + gIoU + classification with equal weights. Queries per image = all its positive and negative
   (known-absent) categories, plus pseudo-negatives sampled in proportion to category frequency until there are
   at least 50 negatives (positives of the image excluded). Loss normalized per example, gradient norm clipped
   to 1.0 per example before accumulation.
9. Prompts: training samples from the 80 CLIP templates (same template for all instances of a category within
   an image); appendix says train-time random prompting is used only for OI. Evaluation averages predicted
   probabilities over the "7 best" CLIP templates (sec 4.6, A1.9 "Prompting").
10. One-shot (image-conditioned): run the model on the query image, take predicted boxes with IoU > 0.65 to the
    query box, pick the class embedding most dissimilar to the others (argmin of f), use it as the query on
    target images. About 10% of cases have no box with IoU > 0.65 -> fall back to text query
    "an image of an object" (A1.7). Few-shot: average image embeddings of k examples (sec 4.4).

Trainable vs frozen: nothing is frozen. "Both the image and the text model are fine-tuned end-to-end" (sec 1).
Text encoder LR 2e-6 = 100x smaller than the image encoder LR in the ablation baseline (image LR 2e-4, Table A5
header). "freezing the text encoder completely yields poor results" (sec 4.6), but Table A5 row "Do not
fine-tune text enc." is -1.5 AP / -2.3 APr (R26+B/32), which is less bad than "same LR" (-3.0 / -8.5). The
text LR for the main Table 1 models (image LR 5e-5 or 2e-5) is not printed per model; OWLv2 A.7 says text LR
2e-6 for its models. Detection heads are at most 1.1% of parameters (sec 3.2).

Tensor trace, CLIP L/14 at 840 (our reading; widths are not printed in this paper):
`image 840x840x3 -> 60x60 = 3600 patch tokens x d_img -> [ViT-L/14 encoder] -> 3600 tokens`
`-> class head: 3600 x d_q embeddings; box MLP: 3600 x 4 (+ grid-cell bias)`
`text: Q query strings, each <= 16 tokens -> text encoder -> Q x d_q`
`logits = 3600 x Q (dot products) -> sigmoid; LVIS eval Q = 1203 per image, x 7 prompts averaged`

## 6. Equations
No displayed or numbered equations in main.tex or appendix.tex (grep for equation/align/\[ found none). The
loss is described in words only (see step 8). Inline formulas:
- A1.7 (one-shot query selection), verbatim: `$ f(z_{i}) = \sum_{j=0}^{N-1} z_{i} \cdot z_{j}^{T}$`, then
  "we use the most dissimilar class embedding $\mathrm{argmin}_{z_{i}} f(z_{i})$ as our query feature".
  Gloss: z_i = class-head embedding of predicted box i among the N boxes with IoU > 0.65 to query box Q;
  f(z_i) = summed similarity of z_i to all of them; the foreground embedding is the outlier (lowest f), because
  DETR matching makes one foreground and many similar background embeddings.
- A1.9 "Mosaics" (sampling heuristic), verbatim: `$\frac{2\cdot(M - k + 1)}{M\cdot(1 + M)}$` = probability of a
  k x k grid when grids from 1x1 to MxM are used. With M = 3 this gives 0.5 / 0.33 / 0.17 (our reading; matches
  sec 4.6).

## 7. Config / setup
Common (sec 4.1, A1.3): batch 256; at most 140k steps; Adam (beta1 0.9, beta2 0.999); cosine LR decay; weight
decay 0 in fine-tuning; focal alpha 0.3, gamma 2.0; equal box/gIoU/class weights; text max 16 tokens; LiT text
model 12 layers, hidden D = 512, MLP 2048, 8 heads ("smaller than B"). Random crop: aspect 0.75-1.33, area
33-100%, keep boxes with >= 60% of area inside; pad to square with gray at bottom/right (A1.6). Instance
merging at IoU >= 0.9 (Table A5 header). Stochastic depth ("droplayer") per Table A3.

Table A3 (PDF p21), detection fine-tuning columns, plus token counts:

| Model (pre-train) | Det. res | Patch/stride | Tokens | FT steps | LR | Droplayer img/txt | FT data (ratio) | Mosaic 1/2/3 |
|---|---|---|---|---|---|---|---|---|
| CLIP B/32 | 768 | 32 | 576 (stated sec 3.1) | 140k | 5e-5 | .2/.1 | O365, VG (.8/.2) | .4/.3/.3 |
| CLIP B/16 | 768 | 16 | 2304 (our reading) | 140k | 5e-5 | .2/.1 | O365, VG (.8/.2) | .4/.3/.3 |
| CLIP L/14 | 840 | 14 | 3600 (our reading) | 70k | 2e-5 | .2/.1 | O365, VG (.8/.2) | .4/.3/.3 |
| LiT B/32 (16B seen) | 768 | 32 | 576 | 140k | 2e-4 | 0.0 | O365, VG (.8/.2) | .4/.3/.3 |
| LiT B/16 (8B) | 768 | 16 | 2304 (our reading) | 140k | 2e-4 | 0.0 | O365, VG | .4/.3/.3 |
| LiT R26+B/32 (16B, pre-train 288) | 768 | 32 | 576 (our reading) | 140k | 2e-4 | 0.0 | O365, VG | .4/.3/.3 |
| LiT L/16 (16B) | 768 | 16 | 2304 (our reading) | 70k | 5e-5 | 0.0 | O365, VG | .4/.3/.3 |
| LiT H/14 (12B) | 840 | 14 | 3600 (our reading) | 70k | 5e-5 | .1/.0 | O365, VG | .4/.3/.3 |
| R50+H/32 one-shot (24B, GAP, batch 12k, LR 7e-4) | 960 | 32 | 900 (our reading) | 28k | 2e-4 | 0.1 | OI, O365, VG (.4/.4/.2) | .5/.33/.17 |
| Ablation baselines B/32 (2B), R26+B/32 (8B) | 768 | 32 | 576 | 70k | 2e-4 | 0.0 | OI, VG (.7/.3) | .5/.33/.17 |

LiT pre-training (Table A3): batch 16k, LR 3e-4, weight decay 1e-5, image 224 (288 for R26+B/32), MAP pooling.
"Tokens" = (res / stride)^2; for CLIP models the class token is merged into patch tokens (A1.3).
Widths are not in the paper; standard ViT sizes (B 768, L 1024, H 1280) are external knowledge (uncertain).
Inconsistency to note: sec 4.6 and A1.6 give mosaic probabilities 0.5 / 0.33 / 0.17, but Table A3 lists
.4/.3/.3 for all Table 1 models; .5/.33/.17 appears only for the one-shot, ablation and scaling models.
Data (A1.2, Table A1): O365 608.5k train images, VG 84.5k, OI V4 1.7M; total "around 2 million images".
Rare-category boxes matching LVIS rare names are removed from all training data (sec 4.3). Strict dedup of
COCO/LVIS val/test images (Table A2). LVIS eval: 1203 queries per image (sec 4.3).
Compute reference: ViT-B/32 vs ResNet50 inference 139.6 vs 141.5 GFLOPs (Table 1 caption). TPU/GPU type and
hours are not stated.

## 8. Results worth remembering
- Table 1 (PDF p8), LVIS v1.0 val, AP^LVIS and AP^LVIS_rare. OWL-ViT models never see LVIS-rare box labels, so
  APr is zero-shot. Mean of three fine-tuning runs. Unrestricted training (O365+VG):
  LiT B/32 768: 23.3 / 19.7; LiT R26+B/32 768: 25.7 / 21.6 (printed row 11); LiT B/16 768: 26.7 / 23.6 (printed
  row 10); LiT L/16 768: 30.9 / 28.8; LiT H/14 840: 33.6 / 30.6; CLIP B/32 768: 22.1 / 18.9; CLIP B/16 768:
  27.2 / 20.6; CLIP L/14 840: 34.6 / 31.2 (best APr).
  LVIS base training: LiT H/14 840: 35.3 (gray) / 23.3; CLIP L/14 840: 34.7 (gray) / 25.6.
  Prior rows as quoted: ViLD-ens R50 25.5 / 16.6; ViLD-ens EffNet-b7 29.3 / 26.3; RegionCLIP R50x4-C4
  32.3 / 22.0; GLIP Swin-T 17.2 / 10.1; GLIP Swin-L 26.9 / 17.1.
  Metric: OWL-ViT is box-only, so its AP is box AP (our reading). The prior rows are copied from the other
  papers (LaTeX comments cite ViLD Table 3, RegionCLIP Table 2, GLIP Table 4); their metric (mask vs box) is not
  stated in this table.
- Table 2 (PDF p10), one-shot image-conditioned COCO AP50 (R50+H/32; mean of 3 runs; splits as CoAE):
  Unseen: OWL-ViT 43.6 / 41.3 / 40.2 / 41.9, mean 41.8; k=10: 49.3 / 51.1 / 42.4 / 44.5, mean 46.8;
  AIT 26.0 / 26.4 / 22.3 / 22.6, mean 24.3; CoAE mean 22.0.
  Seen: OWL-ViT 49.9 / 49.1 / 49.2 / 48.2, mean 49.1; k=10 mean 55.1; AIT mean 47.5.
  "72%" (sec 4.4) = 41.8 / 24.3 (means; our reading). Intro says "from 26.0 to 41.8 AP50": 26.0 is AIT split 1,
  41.8 is the OWL-ViT mean, so that sentence mixes a split with a mean.
- Table 3 (PDF p13), ablations, R26+B/32, 70k steps, OI+VG, mean of 2 replicates, deltas vs baseline
  (AP^LVIS 21.0, APr 18.9, AP^COCO 30.9, AP^OI 54.1):
  (1) only VG -14.5 / -14.0; (2) only OI -6.9 / -5.7; (3) same LR image and text -3.0 / -8.5;
  (4) no prompt ensembling at inference -2.8 / -5.5; (5) no prompts -1.2 / -1.3; (6) no random negatives
  -1.0 / -2.8; (7) no mosaics -2.3 / -1.5; (9) no mosaics, 3x longer -3.4 / -3.6; (11) no location bias
  -1.2 / -1.1 (B/32 in Table A5: -2.8 / -2.9); (14) keep OI crowd: 0.0 / +0.7, OI AP +3.0;
  (15) keep LVIS rare labels +0.1 / +0.2 (so removing rare labels costs almost nothing).
- Table A5 (PDF p28), text encoder LR (R26+B/32, AP / APr): 2e-3 -7.1 / -14.1; 2e-4 -3.0 / -8.5; 2e-5
  -1.2 / -3.2; frozen -1.5 / -2.3. B/32 baseline 15.7 / 14.1.
- Table A4 (PDF p24), COCO (not zero-shot): CLIP L/14 (O365, VG, 840) AP^COCO 43.5, AP50 64.7. Text (sec 4.3)
  says a model trained without O365 reaches 15.8 AP^O365; Table A4 best O365 AP is 15.5 (LiT H/14, OI+VG).
  Mismatch, not reconciled.
- Fig. 3 caption: Pearson r = 0.73 between ImageNet zero-shot accuracy and LVIS APr (r = 0.98 for image-level
  transfer vs pre-training task). "High image-level performance is necessary, but not sufficient".

## 9. Limitations and failure cases stated
- No limitations section. Stated weaknesses: detection performance peaks and then drops as small models are
  pre-trained longer (sec 4.5, Fig. 3 right); best zero-shot recipe does not maximize in-distribution OI AP
  (sec 4.6, A1.9).
- Fig. 2 (butterflies): text query works for "swallowtail butterfly" but not for "luna moth"; the image query
  finds the right species in both cases. Directly relevant to ch 11 and ch 13 item 1.
- One-shot: about 10% of query boxes have no predicted box with IoU > 0.65 (fallback to a generic text query).
- COCO and O365 results are not zero-shot (categories present in training data).
- Hybrids vs ViTs: hybrids better at small sizes; pure ViTs better at zero-shot and at large sizes (Fig. 4).

## 10. Outline feedback (ch 5, ch 11)
- ch 5 archflow shapes: B/32 at 768 = 576 tokens (paper), B/16 at 768 = 2304, L/14 at 840 = 3600 (our reading).
  For CLIP backbones say the class token is multiplied into patch tokens, so token count = patch count.
- "trainable vs frozen per the paper": nothing is frozen; the text encoder trains at LR 2e-6. Freezing it was
  tested and is worse (Table A5). Do not show a brown "frozen" text tower.
- The loss has no equation in the paper. Show it as a labelled diagram (matching -> focal sigmoid CE +
  L1 + gIoU, equal weights), not as an equation.
- One-shot numbers (Table 2) come from an R50+H/32 hybrid at 960, not from CLIP L/14. Say so in the chapter.
- Use 24.3 -> 41.8 (Table 2 means) for the one-shot gain, not the intro's "26.0 to 41.8".
- Table 1 mixes metrics from other papers (ViLD rows from ViLD Table 3). Before the page compares OWL-ViT APr
  31.2 with ViLD 16.6 / 26.3, the ViLD group must confirm whether those are mask or box AP.
- ch 11: Fig. 2 (luna moth) is a sourced example of "text fails, exemplar works". OWL-ViT queries are
  per-image embeddings averaged for k-shot (k = 10 gives unseen mean 46.8).
- ch 13 item 3 (tiny objects): pixels per token are 32 (B/32 at 768), 16 (B/16), 14 (L/14 at 840); the
  max number of detections equals the token count.
- Mosaic proportions conflict between text (0.5/0.33/0.17) and Table A3 (.4/.3/.3 for the Table 1 models).
