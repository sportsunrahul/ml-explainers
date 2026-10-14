# Detic -- Detecting Twenty-thousand Classes using Image-level Supervision (BRANCH, short note)

## 1. Identity
- Title: "Detecting Twenty-thousand Classes using Image-level Supervision"
- arXiv 2201.02605, read v3. v1 2022-01-07 (Fri, 7 Jan 2022 18:57:19 UTC). v2 2022-01-10. v3 2022-07-29 (abs.html).
- Authors: Xingyi Zhou (first), Rohit Girdhar, Armand Joulin, Philipp Krahenbuhl, Ishan Misra (last).
- Venue: ECCV 2022 (abs.html Comments: "ECCV 2022 camera ready").
- Repo: https://github.com/facebookresearch/Detic
- Read: `src/method.tex`, `src/experiments.tex`, `src/supplement.tex`, README.md.

## 2. Core problem
Detection datasets are small in vocabulary; classification datasets (ImageNet-21K, 14M images, 21K classes) are large but have no boxes. Prior weakly supervised methods assign image labels to boxes using the model's own predictions, which are poor for rare or novel classes (Sec. 4, "Relation to prediction-based assignments").

## 4. The one key idea
Do not predict which box an image label belongs to: apply the label to a fixed box, the largest RPN proposal ("max-size"), and train only the classifier on it. The classifier rows are CLIP text embeddings (following ViLD), shared between detection and classification data.

## 5. Mechanism (5 steps)
1. Detector: CenterNet2 with ResNet-50 (main LVIS setting), Federated Loss, repeat factor sampling, large-scale jittering at 640x640; classifier W = CLIP text embeddings of class names (Sec. 5.1, Sec. 3).
2. Mini-batches mix detection and image-labeled data 1:1; image-labeled images use 320x320 (Sec. 5.1).
3. Detection images: the standard losses L_rpn + L_reg + L_cls (Sec. 4).
4. Image-labeled images: take the RPN proposal with the largest size, compute BCE(W f_j, c) for every image label c, weight lambda = 0.1. No box or RPN loss on these images (Sec. 4, "max-size loss").
5. Schedule: train a base-class-only model (4x), then fine-tune with image-labeled data for another 4x; R50 takes about 22 hours on 8 V100 GPUs (Sec. 5.1). Data: IN-L = 997 ImageNet-21K classes that overlap LVIS (1.2M images); CC = 1.5M Conceptual Captions images with 992 LVIS classes; also full IN-21K (Sec. 5).

## 8. Key table
| Source | Setting | Metric | Value |
|---|---|---|---|
| Table 2 (ViLD's settings: Mask R-CNN R50, from scratch) | Detic (IN-L) | LVIS mask mAP_novel (APr) | 17.8 |
| Table 2 | Detic (IN-L) | LVIS mask mAP | 26.8 |
| Table 2 | ViLD-ensemble (quoted) | LVIS mask mAP_novel | 16.6 |
| Table 1 (CenterNet2 R50 baseline, IN-L, mean of 3 runs) | Box-Supervised -> Detic max-size | LVIS mask mAP_novel | 16.3 -> 24.6 |
| Table 3 (OVR-CNN setting, R50-C4) | Detic | COCO box mAP50_novel | 27.8 |
- Table 2 and Table 1 are mask AP; Table 3 is box AP50. Table 1 uses Detic's own stronger baseline (CenterNet2), so 24.6 is not comparable with ViLD's 16.1; Table 2 is the like-for-like one.
- Paper text: "Detic provides a gain of 7.7 points on mAP_novel" over ViLD-text in Table 2 (17.8 vs 10.1).
- Max-size proposal covers 92.8% of target objects on an annotated IN-L subset vs 69.0% for the prediction-based method (Sec. 5.2).

## 10. Outline feedback
- Ch. 4 branch paragraph is right ("image-level labels train the classifier on the largest proposal, ImageNet-21k classes"). Add that W is CLIP text embeddings (the ViLD-text classifier) and that the fixed box choice is the point: no prediction-based assignment.
- Ch. 3 extra source: Detic App. A, Table 8(a), proposal recall on LVIS rare classes at IoU 0.5 (CenterNet2 baseline): LVIS-all ARr50@1k 79.7 vs LVIS-base 78.5. It agrees with ViLD Table 1 (different detector and AR definition; do not mix the two).
- Ch. 1 LVIS bar: Detic Sec. 5 (LVIS paragraph) says only "The classes are divided into three groups - frequent, common, rare based on the number of training images." No thresholds.
