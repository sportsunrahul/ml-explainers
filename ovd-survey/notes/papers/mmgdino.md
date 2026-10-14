# MM-Grounding-DINO -- An Open and Comprehensive Pipeline for Unified Object Grounding and Detection (BRANCH, short note)

## 1. Identity
- Title: "An Open and Comprehensive Pipeline for Unified Object Grounding and Detection" (model name MM-Grounding-DINO, "MM-G").
- arXiv 2401.02361, read v2. v1 2024-01-04 (Thu, 4 Jan 2024 17:00:49 UTC); v2 2024-01-05 (abs.html).
- Authors: Xiangyu Zhao (first) ... Haian Huang (last); 7 authors; affiliations Shanghai AI Lab and SenseTime Research (src/main.tex author block).
- Venue: none stated (abs.html Comments "10 pages, 6 figures"; template is CVPR style, not a venue claim).
- Repo: https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino (code and all weights released).
- Source read: `src/sec/tex/0_abstract.tex`, `1_intro.tex`, `2_baseline.tex`, `3_experiment.tex`; PDF for table numbers.

## 2. Core problem
The original Grounding DINO released only test and demo code, so its training could not be reproduced (abstract; Table 5 caption: "The training details of Grounding-DINO have not been released"). MM-G rebuilds training in MMDetection, open-sources it, and scales the open data.

## 4. The one key idea
Keep the Grounding DINO architecture "almost unchanged" and win on training: a CLIP-style scaled-and-biased contrastive head at initialization, random negative prompts, and more open data (GRIT, V3Det with 13,029 categories) instead of the closed Cap4M.

## 5. Mechanism (at most 5 steps)
1. Architecture = Grounding DINO, read from the official test code. Feature enhancer layer as written in Algorithm 1: `BiAttentionBlock(image, text)` first (text-to-image and image-to-text cross-attention), then text self-attention + FFN and image deformable self-attention + FFN. Content queries are "an all-zero learnable query"; position part = anchors from language-guided query selection; 900 queries (Sec. 2.1).
2. The only model change (Algorithm 2): logits = visual_feat @ text_feat^T, then divided by sqrt(d) and a learned bias added at initialization, "motivated by CLIP". "This can significantly reduce the initial loss value and accelerate the convergence" (Sec. 2.1 "Difference").
3. Text input: OVD categories concatenated, like "People. Ball. Racket. Cat."; for grounding data every referred phrase is a target (MDETR style) (Sec. 2.3).
4. Negatives: "we concatenate the categories or text descriptions, which are randomly sampled from other images as negative examples, with ground-truth descriptions serving as positive examples", to "suppress the hallucination" (Sec. 2.3, Data Augmentation).
5. Training: BERT-base-uncased + Swin-T; 32 NVIDIA 3090 GPUs, total batch 128, 30 epochs (Sec. 2.3). Data variants in Table 1: (b) O365+GoldG; (c3) O365+GoldG+GRIT+V3Det (61.8M images seen). GRIT (9M images) is used in 500K-image chunks per epoch.

## 8. Key table (zero-shot, Swin-T)
| Model | LVIS minival AP | LVIS minival APr | ODinW13 | ODinW35 |
|---|---|---|---|---|
| G-DINO-T(c), re-evaluated by MM-G | 28.8 | - | - | 22.7 |
| MM-G-T(c3) (O365, GoldG, GRIT, V3Det) | 41.4 | 34.2 | 53.3 | 28.4 |
Locs: Table 4 p6 (LVIS MiniVal cols AP, APr) and Table 6 p6 (ODinW13 / ODinW35). COCO zero-shot best is MM-G-T(c2) 50.6 (Table 3). Fixed-AP protocol not stated.
- Same table, other values: G-DINO-T(c) LVIS minival APr 18.8; G-DINO-T(c) ODinW13 51.4.

## 10. Outline feedback
- Interlude `ch-training` (negatives): this is the clearest primary statement of how negatives enter a Grounding-DINO-style prompt. Negatives are category names or descriptions sampled from other images and concatenated with the positives (Sec. 2.3). Grounding DINO itself states only random sampling of category names. Label it "MM-G's open re-implementation".
- Interlude: MM-G's Algorithm 1 puts the bi-directional cross-attention before the self-attention in each enhancer layer. Grounding DINO's Fig. 3 block 2 draws self-attention first. MM-G follows the released test code. The stepper should pick one order and cite it.
- ch 9 numbers: MM-G's re-evaluation of the same G-DINO-T(c) checkpoint gives LVIS minival 28.8 / APr 18.8 and ODinW35 22.7. The original paper prints 27.4 / 18.1 and 22.3. Do not mix sources in one chart.
- ch 9 ODinW-13 for G-DINO-T(c): MM-G prints 51.4, Grounding DINO 1.5 prints 49.8. Two primary sources disagree; cite one and say which.
- ch 12/13 strong example (Table 11 p8, box AP, MM-GDINO Swin-T): Brain Tumor zero-shot 0.4 -> fine-tuned 47.5. The dataset "only utilizes numerical identifiers without providing descriptive label information" (Sec. 3, paragraph "Object Detection for Brain Tumor"). Also RUOD underwater 29.8 -> 65.5 and RTTS haze 49.8 -> 69.1. This is direct evidence that the name is the failure, and that fine-tuning fixes it.
- MM-G Table 1 lists G-DINO-Tiny image consumption as "30 Ep x 128 Bs". The Grounding DINO paper states a total batch of 32 for T and no epochs. Treat MM-G's figure as its own accounting, not as Grounding DINO's schedule.
