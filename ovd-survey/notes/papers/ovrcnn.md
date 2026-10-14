# OVR-CNN -- Open-Vocabulary Object Detection Using Captions (BRANCH, short note)

## 1. Identity
- Title: "Open-Vocabulary Object Detection Using Captions"
- arXiv 2011.10678, read v2. v1 2020-11-20 (Fri, 20 Nov 2020 23:05:46 UTC). v2 2021-03-14 (abs.html).
- Authors: Alireza Zareian (first), Kevin Dela Rosa, Derek Hao Hu, Shih-Fu Chang (last).
- Venue: CVPR 2021, oral (abs.html Comments: "To be presented at CVPR 2021 (oral paper)").
- Repo: https://github.com/alirezazareian/ovr-cnn (src/cvpr.tex, author block).
- Read: `src/3_method.tex`, `src/4_experiments.tex`, `src/tables/main_results.tex`.

## 2. Core problem
Zero-shot detectors replace the classifier with fixed word embeddings of base classes, but the visual-to-word projection learned from a few base classes overfits: "the state-of-the-art mAP on target classes is 9 times lower than base classes" (Sec. 3). The paper names the task "open-vocabulary object detection": boxes for base classes plus cheap image-caption pairs that cover more words.

## 4. The one key idea
Learn the vision-to-language (V2L) projection before detection, on image-caption pairs, by weakly supervised word-region grounding, so the projection covers the whole caption vocabulary and not only base classes. Then initialize a Faster R-CNN with that ResNet + V2L and keep the classifier head (V2L + word embeddings) fixed.

## 5. Mechanism (5 steps)
1. Pretrain (PixelBERT-like): ResNet-50 turns a w x h image into a w/32 x h/32 grid of region features; a linear V2L layer maps each to the language space e_i^I. A frozen BERT-Base gives word embeddings e_j^C. A 6-layer multimodal transformer (trained from scratch) mixes them (Sec. 3.1, 4.2).
2. Pretraining loss: L = L_G(I) + L_G(C) + L_MLM + L_ITM. The grounding score <I,C>_G averages word-region dot products with softmax attention over regions; L_G is contrastive over the batch (Sec. 3.1). Data: COCO Captions (118,287 images, 5 captions each) by default, Conceptual Captions (2,749,293 pairs) as an alternative (Sec. 4.1).
3. Transfer: initialize Faster R-CNN (R50-C4, maskrcnn-benchmark) with the pretrained ResNet and V2L. Each proposal feature r_i -> V2L -> e_i^I; score = exp<e_i^I, e_k^V> over base-class BERT input embeddings, with "1 +" in the denominator for a fixed all-zero background embedding (Sec. 3.2, Eq. 6).
4. What trains: ResNet blocks 3-4 fine-tuned (stem and blocks 1-2 fixed), RPN and box regression from scratch; the classifier head (V2L + word embeddings) is fully fixed. Background loss weight alpha = 0.2 (Sec. 3.2, 4.2).
5. Test: swap the word embeddings for target classes (or base + target in the generalized setting).

Contrast with ViLD (for the ch. 4 delta): OVR-CNN uses an all-zero fixed background ("performs better than a trainable one", Sec. 3.2); ViLD learns e_bg. OVR-CNN text = BERT input word embeddings; ViLD text = CLIP text encoder with prompts.

## 8. Key table
Table 1 (`tables/main_results.tex`), COCO 2017, 48 base / 17 target split, mAP at IoU 0.5 (Sec. 4.1: "mean Average Precision (mAP) at an IoU of 0.5"), box (Faster R-CNN, no masks):
| Setting | Base (48) | Target (17) | Generalized Base | Generalized Target | Generalized All |
|---|---|---|---|---|---|
| OVR-CNN | 46.8 | 27.5 | 46.0 | 22.8 | 39.9 |
ViLD's Table 4 "Novel AP 22.8" for OVR-CNN is the generalized-target number (22.8), not the target-only 27.5.
Compute (Sec. 4.2): pretraining batch 64 on 8 V-100 GPUs, about 10 hours; detection training batch 8 on 8 V-100 GPUs, about 18 hours.

## 10. Outline feedback
- Ch. 4 branch paragraph is right: "caption pretraining of a vision-to-language projection". Add that the classifier head stays frozen during detection training and that background is a fixed zero vector (ViLD changed this to a learned e_bg).
- If the page quotes "OVR-CNN 22.8 novel AP50", say it is the generalized setting (base + target classes scored together).
- Date for the family tree: 2020-11-20 (v1), so OVR-CNN predates ViLD (2021-04-28) and the Act II "2021" label is slightly off for this branch.
