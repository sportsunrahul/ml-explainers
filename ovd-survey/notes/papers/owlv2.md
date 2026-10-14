# OWLv2 / OWL-ST: Scaling Open-Vocabulary Object Detection

## 1. Identity
- Title: "Scaling Open-Vocabulary Object Detection". Model: OWLv2. Self-training recipe: OWL-ST.
- arXiv 2306.09683, read v3. v1 2023-06-16, v2 2023-07-20, v3 2024-05-22 (abs.html submission history).
  src/main_table.tex is dated 2024-05-22 (v3 change); other .tex files are dated 2023-07-20.
- Authors: Matthias Minderer, Alexey Gritsenko, Neil Houlsby (Google DeepMind per PDF).
- Venue: abs.html has no comment field. src uses neurips_2023.sty, so NeurIPS 2023 is inferred only (uncertain).
- Repo: none named in the sources read (no README). Teacher is "the publicly available OWL-ViT CLIP L/14 model".
- Sources read: src/3_main.tex, appendix.tex, main_table.tex, main_table_fixed_ap.tex, hparams_table.tex,
  odinw_table.tex; paper.pdf (22 pages) for page numbers.

## 2. Core problem
Open-vocabulary detectors are limited by the amount of human box annotations. Self-training on Web image-text
pairs can create unlimited pseudo-box data, but nobody had scaled it near image-level pre-training sizes. The
paper asks how to pick the label space, how to filter pseudo-boxes, and how to train efficiently at billions of
examples.

## 3. What prior methods did and why that hurt (per the paper)
- Detection-training stages have been brief "due to the scarcity of detection data and the fragility of
  pretrained representations", limiting performance and scaling (sec 1).
- Prior self-training (RegionCLIP, Detic, 3Ways, DetCLIP) runs at sizes comparable to human-labelled datasets;
  "prior methods rarely go much beyond 10M examples" (sec 2.3).
- They used human-curated label spaces or complex concept mining, and strict filtering: Detic keeps the single
  largest box, 3Ways the single highest-scoring box and uses the whole caption as one prompt (sec 1, 2.3).
- Grammatical parsing for nouns "may add biases that reduce the diversity of extracted queries" (sec 3.1).

## 4. The one key idea
"Let the data do the work": query an existing OWL-ViT with every word N-gram of each Web image's alt-text,
keep all boxes above a weak confidence threshold, and self-train a new, cheaper-to-train OWL model (OWLv2) on
billions of these pseudo-labelled images; optionally fine-tune briefly on human boxes.

## 5. Mechanism, step by step
1. Data: WebLI, approximately 10B images with alt-text (non-English alt-text machine-translated) (sec 3.1).
   WebLI was deduplicated against 68 common CV datasets incl. COCO/LVIS, O365, VG, but NOT against ODinW
   (sec 4.1).
2. Annotator (teacher): public OWL-ViT CLIP L/14 annotates all 10B images (sec 3.1, 4.1).
3. Label space, one of: (a) human-curated: union of LVIS, Objects365, OpenImagesV4 and Visual Genome labels,
   de-duplicated and de-pluralized, 2520 categories, same for every image; (b) machine-generated: all word
   N-grams up to length 10 of the image's text, minus generic words (e.g. "image", "png") and all-stop-word
   N-grams, at most 300 queries per image (A.2 code: `max_num_queries=300, max_ngram_len=10`); (c) both, with
   curated-query scores multiplied by 0.3 before thresholding (A.3). Main open-vocabulary results use N-grams.
   Note: the A.1 code defines a VG list but only concatenates LVIS + O365 + OI into `queries`.
4. Pseudo-labels: predictions ensembled over seven prompt templates; each box takes its highest-scoring query
   as a hard label (sec 3.1).
5. Filtering: Fig. 3 study (B/16): images remaining at threshold 0.1: 5B, 0.3: 2B, 0.5: 782M, 0.7: 224M;
   "a moderate threshold of 0.3 works well, while strict thresholds lead to poor results". Main results:
   keep all boxes above 0.1, but only images with at least one box above 0.3 (sec 4.4). Self-training set is
   approximately 2B images (sec 4.5). A.7 bullet lists "confidence score threshold: 0.3".
6. Student init: image and text encoders from CLIP (B/16, L/14) or SigLIP (G/14); detection heads random
   (sec 3.2, 4.1). Same losses as OWL-ViT; pseudo-negatives sampled from other images' queries (sec 3.2).
   No random prompt templates, no flips, no random crops during self-training (A.4).
7. Efficiency (sec 3.2), together called OWLv2:
   a. Token dropping: drop the lower 50% of patches by pixel variance (uniform noise in [0, 0.01] added first,
      A.5); training only, "No patches are dropped during inference."
   b. Objectness head: scalar per encoder token, predicts "the future classification score of a token",
      supervised by the actual classification score of the selected tokens; boxes, class scores and losses only
      for the top k tokens by objectness, "approximately 10%" during training; all tokens at inference.
      Table A2: k = 256 (B/16), 512 (L/14, G/14).
   c. Mosaics: 1x1, 2x2, 3x3, 4x4, 6x6 grids in equal proportions -> average 13.2 raw images per example
      ((1+4+9+16+36)/5 = 13.2, our reading). Each raw image randomly resized to 0.5-1.0x the tile width (A.4).
   d. Adafactor variant, inverse square-root LR schedule (timescale 10'000 steps), linear cool-down of
      checkpoints for evaluation (A.6).
   Result: training FLOPs about 50% lower, throughput 2x. L/14 at 840x840 on TPUv3: 11'945.4 vs 5357.9
   GFLOPs/example, 1.0 vs 2.2 examples/s/core. "at inference, the model is identical to the original OWL-ViT."
8. Optional fine-tuning on LVIS_base (LVIS minus rare annotations): start from the self-trained LR and optimizer
   state, linearly cool the LR to zero; mosaics up to 3x3; until 256'000 mosaics seen (1.1M images, about 100
   LVIS epochs); same augmentations as OWL-ViT (sec 3.3, 4.1, A.4).
9. Weight-space ensemble: average self-trained and fine-tuned weights (Wortsman et al.) to recover OOD
   performance; co-training on WebLI + LVIS was worse (sec 4.6).

Trainable vs frozen: nothing frozen. Text encoder LR 2e-6 for all models (A.7); image LR per Table A2.
Token counts (our reading except 5184): B/16 at 960 -> 60x60 = 3600 tokens, 1800 kept after 50% drop,
top k 256 (7.1% of 3600). L/14 and G/14 at 1008 -> 72x72 = 5184 tokens (stated, sec 3.2), 2592 kept,
top k 512 (9.9% of 5184). L/14 at 840 (FLOPs comparison) -> 3600.

## 6. Equations
None. No displayed or numbered equations in any .tex file (grep for equation/align/\[ found none). The loss is
"the same losses" as OWL-ViT (sec 3.2). The objectness supervision is described in words only.

## 7. Config / setup
Table A2 (PDF p15) plus A.7 constants:

| Row | Model | Backbone | Image size | LR | Dropout img/txt | Droplayer img/txt | Top k | Batch ST | Batch FT | Examples seen |
|---|---|---|---|---|---|---|---|---|---|---|
| 11 | OWL-ST | CLIP B/16 | 960 | 5e-5 | .0/.0 | .2/.1 | 256 | 256 | -- | 365260288 (3.7e8) |
| 12 | OWL-ST | CLIP L/14 | 1008 | 2e-5 | .0/.0 | .2/.1 | 512 | 256 | -- | 225893376 (2.3e8) |
| 13 | OWL-ST | SigLIP G/14 | 1008 | 2e-5 | .0/.1 | .2/.4 | 512 | 128 | -- | 160396032 (1.6e8) |
| 14 | OWL-ST+FT | CLIP B/16 | 960 | 5e-5 | .0/.0 | .2/.1 | 256 | 256 | 256 | 361573632 (3.6e8) |
| 15 | OWL-ST+FT | CLIP L/14 | 1008 | 2e-5 | .0/.0 | .2/.1 | 512 | 256 | 128 | 228197248 (2.3e8) |
| 16 | OWL-ST+FT | SigLIP G/14 | 1008 | 2e-5 | .0/.1 | .2/.4 | 512 | 128 | 128 | 162904704 (1.6e8) |
| 20 | OWL-ST+FT curated | CLIP B/16 | 960 | 5e-5 | .0/.0 | .2/.1 | 256 | 256 | 256 | 816945920 (8.2e8) |
| 21 | OWL-ST+FT curated | CLIP L/14 | 1008 | 2e-5 | .0/.0 | .2/.1 | 512 | 256 | 128 | 360215936 (3.6e8) |

"Examples seen" counts mosaics (ST + FT); raw images = 13.2x for self-training (Fig. 2/3 captions). The intro
says the L/14 model was "trained on 2.3B image-text pairs"; Table A2 prints 2.3 x 10^8 examples for rows
12/15. Not reconciled (13.2 x 2.3e8 is about 3.0B raw images, our reading).
G/14 (A.7): started at LR 5e-5, droplayer .1/.0, no dropout; switched to LR 2e-5, droplayer .2/.4, dropout
.0/.1 after 740'000 self-training steps because it overfit in fine-tuning. G/14 has 5.2x the parameters and
4.3x the inference FLOPs of L/14 (sec 4.5). Accelerator: TPUv3 named only for the throughput numbers.

## 8. Results worth remembering
- Table 1 (PDF p5), LVIS standard AP ("AP^old"), ODinW13 zero-shot mean AP; mini = MDETR minival.
  Columns: ODinW13 | AP^mini_all | AP^mini_rare | AP^val_all | AP^val_rare.
  11 OWL-ST B/16: 48.8 | 31.8 | 35.4 | 27.0 | 29.6
  12 OWL-ST L/14: 53.0 | 38.1 | 39.0 | 33.5 | 34.9
  13 OWL-ST G/14: 49.9 | 37.8 | 40.9 | 33.7 | 37.5
  14 OWL-ST+FT B/16: 48.6 | 47.2 | 37.8 | 41.8 | 36.2
  15 OWL-ST+FT L/14: 50.1 | 54.1 | 46.1 | 49.4 | 44.6
  16 OWL-ST+FT G/14: 50.1 | 51.3 | 50.9 | 47.0 | 47.2
  20 OWL-ST+FT B/16 N-grm+curated: 48.9 | 51.1 | 41.9 | 45.6 | 40.5
  21 OWL-ST+FT L/14 N-grm+curated: 48.7 | 55.8 | 50.0 | 50.4 | 45.9
  Reference rows: 2 OWL B/16 val 27.2 / 20.6; 3 OWL L/14 ODinW13 48.4, val 34.6 / 31.2; 8 F-VLM R50x64 val
  34.9 / 32.8 (prior SOTA, open vocab); 19 DetCLIPv2 Swin-L val 36.6 / 33.3 (prior SOTA, curated); GLIPv2
  ODinW13 48.5 (Swin-T), 54.2 (Swin-B, gray), 55.5 (Swin-H, gray; gray = trained on OpenImages, overlaps ODinW).
  Text: OWL-ST B/16 29.6 is "9 points more" than OWL B/16; G/14 37.5 is 4.7 above F-VLM; G/14 ST+FT 47.2 is a
  14.4-point improvement over F-VLM. Abstract: L/14 APr 31.2 -> 44.6 (43% relative). Intro: 44.6 is a 36%
  relative improvement over 32.8. After self-training only, APr > AP_all (e.g. 29.6 vs 27.0).
- Table A3 (PDF p17), fixed AP, LVIS val rare: 30.3 (row 11), 36.2 (12), 39.0 (14), 47.4 (15), 42.5 (20),
  48.5 (21); G/14 not reported. Fixed AP = AP on top 10'000 predictions per class.
- Table A1 (PDF p14), token dropping, standard OWL-ViT L/14 trained on O365+VG, LVIS val AP_all / AP_rare:
  drop 0.00: 33.3 +-0.33 / 31.8 +-1.16; 0.25: 33.1 / 31.0; 0.33: 33.6 / 32.6; 0.50: 32.9 / 30.8;
  0.70: 30.4 / 28.2. Sourced data for the ch 6 drop-rate slider.
- Table A4 (PDF p17), all 35 ODinW datasets: see notes/data/odinw35-owl.json. Aggregates (mean13 / mean35 /
  median35): row 11: 48.8 / 22.1 / 11.6; 12: 53.0 / 24.4 / 16.2; 13: 49.9 / 22.9 / 17.5; 14: 48.6 / 20.8 / 6.0;
  15: 50.1 / 22.3 / 6.3; 16: 50.1 / 22.5 / 9.5; 20: 48.9 / 21.7 / 6.8; 21: 48.7 / 21.9 / 7.0;
  OWL-ST/FT ens (L/14, of row 21, fine-tuned weight 0.4): 56.3 / 25.6 / 10.6, "our best model by ODinW13".
  Recomputed from per-dataset values: all aggregates reproduce (our reading). Hard Hat Workers: 1.9 (row 12),
  2.1 (row 15).
- Fine-tuning trade-off from tables only: LVIS val APr rises with FT at every size (29.6 -> 36.2, 34.9 -> 44.6,
  37.5 -> 47.2). ODinW13 falls for L/14 (53.0 -> 50.1) and B/16 (48.8 -> 48.6) but not for G/14 (49.9 -> 50.1).
  Median35 falls at every size (11.6 -> 6.0, 16.2 -> 6.3, 17.5 -> 9.5). The ensemble row recovers ODinW13
  (56.3 vs 48.7 for row 21) but no table gives its LVIS AP. The B/16 trade-off curves (Fig. 5) and L/14
  (Fig. A1) are plots only; do not read numbers off them.
- Text (sec 4.6): "LVIS APrare therefore only measures a narrow concept of open-vocabulary performance, and does
  not reveal the fact that fine-tuning significantly reduces generalization to broader distribution shifts."
- COCO fine-tuning (sec 4.2): B/16 54.3, L/14 56.0 COCO mAP, matching ViTDet with Cascade Mask-RCNN head.
- Scaling (sec 4.5, Fig. 4): self-training helps already below the annotator's compute; Pareto-optimal model
  size grows with compute; further out of distribution, L/14 needs more compute to overtake B/16.
- Label space (sec 4.3, Fig. 2): curated labels best on fine-tuned classes (APfrequent); N-grams better on
  unseen classes (APrare) and in-the-wild datasets.

## 9. Limitations the paper states (sec 5, PDF p9)
- "The main limitation of our method is the amount of compute and data needed for self-training." Gains keep
  coming with compute, but "cost likely increases faster than resources can realistically be grown in
  practice."
- Trade-off between fine-tuned and open-vocabulary performance: "For out-of-distribution queries, predictions
  of fine-tuned models may be poorly calibrated and may depend on the precise wording of the query." Weight
  ensembling mitigates this; "more research is needed".
- Also stated elsewhere: G/14 hyperparameters "may still not be optimal" (A.7); significant ODinW13 variability
  between checkpoints late in self-training (Fig. 5 caption); curated-label models are not fully
  open-vocabulary on LVIS (sec 3.1).

## 10. Outline feedback (ch 6, 12, 13)
- ch 6 stepper is right: WebLI ~10B -> N-grams (up to 10 words, max 300 per image) -> OWL-ViT CLIP L/14
  teacher (7 prompts, top query per box) -> threshold (boxes > 0.1, image needs one > 0.3; ~2B images kept) ->
  student. Add the objectness top-k and mosaics (13.2 images per example) to the efficiency panel.
- ch 6 token-drop slider can be sourced, not toy: Table A1 gives LVIS AP/APr at drop rates 0, .25, .33, .5, .7.
  Fig. A3 (PDF p19) shows a 4x4 mosaic before and after dropping 50% of patches (1008, patch 14): most dropped
  patches are padding or uniform background.
- ch 13 item 2 needs a fix: token dropping happens only during training; at inference no patches are dropped
  and the model equals OWL-ViT. Do not say OWLv2 discards low-variance regions at test time. A narrower claim
  that holds: the student is never trained on the low-variance half of each image (our reading).
- ch 13 item 4 is supported verbatim by sec 5 (poor calibration, wording dependence for OOD queries, for
  fine-tuned models).
- ch 12 card 4 (fine-tuning forgets): use median35 (all sizes drop) or ODinW13 for L/14 and B/16. G/14 ODinW13
  does not drop (49.9 -> 50.1). The ensemble (56.3 ODinW13) is of the curated-vocabulary row 21, not of the
  N-gram row 15, and has no LVIS number in a table.
- ch 12 card 2: OWL-ST L/14 mean13 53.0 vs median35 16.2; 5 of 35 datasets below 1.0 AP and 15 below 5.0 AP
  (our reading, counted from Table A4). Caveat: WebLI was not filtered for ODinW, and the annotator was trained
  on O365+VG; the paper grays GLIPv2 ODinW numbers for OpenImages overlap.
- Metric note for the page: OWLv2 Table 1 LVIS values are standard ("old") box AP on LVIS val; the fixed-AP
  variant in Table A3 is 1-3 points higher. Compare to other papers only with this said.
- The OWL L/14 ODinW13 value 48.4 first appears in OWLv2 Table 1 (row 3), not in the OWL-ViT paper.
