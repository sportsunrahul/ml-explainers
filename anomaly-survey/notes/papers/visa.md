# VisA (from "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation")

## 1. Identity
- Title: "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation". It introduces the Visual Anomaly (VisA) dataset and the SPD pre-training method. This note covers VisA and the paper's baselines; SPD is not on the outline.
- arXiv 2207.14315, version read: v1 (abs.html "[Submitted on 28 Jul 2022]"; only version). Published 2022-07-28.
- Authors: Yang Zou ... Onkar Dabeer (5 authors; AWS AI Labs, plus KAIST for the second author).
- Venue: abs.html Comments "Accepted to European Conference on Computer Vision 2022" (ECCV 2022; LaTeX has `\ECCVSubNumber{2149}`).
- Repo: http://github.com/amazon-research/spot-diff (abstract).
- Sources: `sources/visa/src/spot-diff.tex`, `paper.pdf` (25 pages incl. appendix).

## 2. Core problem
MVTec AD is "saturating with the Area Under the Receiver Operating Characteristic (AU-ROC) approaching ~95%" and is limited to the 1-class setup (Sec. 1). Real inspection has complex structures (PCBs), several objects per image, pose variation, and sometimes a few labelled defects. A larger, harder dataset and 2-class protocols are needed.

## 3. What prior work did and why that hurt (per the paper)
- MVTec AD: single, roughly aligned object per image; 1-class only; AU-ROC inflated by class imbalance (Sec. 1, Sec. 4.2).
- AU-ROC alone: "in imbalanced dataset where performance of minor class is more important, AU-ROC might provide an inflated view of performance" (Sec. 4.2). Example quoted from the MVTec AD IJCV paper: Student-Teacher has 92.2% AU-ROC but only 59.9% AU-PR (Sec. 4.2).
- Contrastive SSL (SimCLR, MoCo, SimSiam) learns invariance to global augmentations and so misses local defects (Sec. 1). (Method motivation; not used on the page.)

## 4. The one key idea (dataset part)
A dataset twice the size of MVTec AD with harder scenes (complex PCBs, multiple instances, pose and location variation), plus three protocols (1-class, 2-class high-shot, 2-class 5/10-shot) and AU-PR reported next to AU-ROC.

## 5. Mechanism (dataset and protocol), step by step
1. 12 subsets = 12 objects, in 3 domains: complex structure (PCB1-PCB4), multiple instances (Capsules, Candle, Macaroni1, Macaroni2), single instance (Cashew, Chewing gum, Fryum, Pipe fryum) (Sec. 4.1, Table 1).
2. 10,821 images: 9,621 normal and 1,200 anomalous (abstract, Sec. 1, Sec. 4.1).
3. Sensor: "All images were acquired using a 4,000x6,000 high-resolution RGB sensor" (Sec. 4.1). The paper gives no per-subset stored resolution.
4. Defects: surface (scratches, dents, color spots, cracks) and structural (misplacement, missing parts); manually generated; "5-20 images per defect type"; an image may contain several defects (Sec. 4.1). "Multiple anomaly classes (up to 9) for each object" (Sec. 1).
5. Labels: image-level and pixel-level (Sec. 4.1).
6. In Capsules and Macaroni2 the instances "largely differ in locations and poses"; Cashew, Chewing gum, Fryum, Pipe fryum are "roughly aligned" (Sec. 4.1).
7. Protocols (Sec. 4.2), applied per object:
   - 1-class: "assigning 90% normal images to train set while 10% normal images and all anomalous samples are grouped as test set".
   - 2-class high-shot: 60% / 40% of normal and of anomalous images to train / test.
   - 2-class low-shot: 20% / 80% split, then k = 5 or 10 images sampled per class from the train part; mean over 5 random runs.
8. Metrics: AU-PR and AU-ROC, each for classification (image level) and segmentation (pixel level), averaged over the 12 subsets (Sec. 4.2, Sec. 5). No PRO / AU-PRO in this paper.
9. Baseline algorithms (Sec. 5): 1-class = PaDiM (hyperparameters "exactly the same" as the PaDiM paper, App. C) with a ResNet-50 backbone from different ImageNet pre-trainings; also PatchCore with Wide ResNet-50 (Table 6). 2-class = binary ResNet-50 classifier and a U-Net (ResNet-50 encoder) with focal loss.

## 6. Equations
App. B (p. 19), verbatim from `spot-diff.tex`:
- Eq. (`eqn:recall`): `Recall=\frac{TP}{P}=\frac{TP}{TP+FN}`
- Eq. (`eqn:FPR`): `FPR=\frac{FP}{N}=\frac{FP}{TN+FP}=\frac{N-TN}{N}`
- Eq. (`eqn:prec`): `Precision=\frac{TP}{TP+FP}`
- Eq. (`eqn:F1`): `F_1=\frac{2*Precision*Recall}{Precision+Recall}=\frac{TP}{TP+0.5*(FP+FN)}`
Paper's wording: "Varying the operating thresholds, ROC curve measures the trade-off between recall and FPR and AU-ROC is the Area Under the ROC curve. PR curve measures trade-off between precision and recall and AU-PR is the Area Under the PR curve."
Gloss: P / N = ground-truth positives (anomalous) / negatives (normal); at image level a sample is an image, at pixel level a sample is a pixel (our reading from "Classification" vs "Segmentation" columns; the paper does not spell out the unit). Recall = TPR.

Toy example (App. B, p. 19-20): 100 positives, 100,000 negatives (neg/pos = 1,000).
- Model A: each new TP costs 10 FPs. AU-ROC = 99.5%, AU-PR = 10.5%, max F1 = 18.2%.
- Model B: first 90 TPs with no FP, then 3,000 FPs per TP. AU-ROC = 98.5%, AU-PR = 90.6%, max F1 = 94.7%.
- Point: the model with the lower AU-ROC is far better on the anomaly class.

## 7. Measurement setup
Table 1 (p. 8), "Overview of VisA dataset":

| Domain | Object | # normal | # anomaly | # anomaly classes |
|---|---|---|---|---|
| Complex structure | PCB1 | 1,004 | 100 | 4 |
| Complex structure | PCB2 | 1,001 | 100 | 4 |
| Complex structure | PCB3 | 1,006 | 100 | 4 |
| Complex structure | PCB4 | 1,005 | 100 | 7 |
| Multiple instances | Capsules | 602 | 100 | 5 |
| Multiple instances | Candle | 1,000 | 100 | 8 |
| Multiple instances | Macaroni1 | 1,000 | 100 | 7 |
| Multiple instances | Macaroni2 | 1,000 | 100 | 7 |
| Single instance | Cashew | 500 | 100 | 9 |
| Single instance | Chewing gum | 503 | 100 | 6 |
| Single instance | Fryum | 500 | 100 | 8 |
| Single instance | Pipe fryum | 500 | 100 | 6 |

Sums (our reading): normal 9,621; anomalous 12 x 100 = 1,200; anomaly classes 75 (not printed).
Train / test counts under the 1-class protocol are not printed; they follow from 90% / 10% of the normal counts (our reading; rounding not stated).

## 8. Results worth remembering (all means over subsets, %)
- Table 2 (p. 11), PaDiM, ResNet-50, supervised ImageNet pre-training ("Sup. pre-train" row), 1-class:
  VisA: classification AU-PR 88.2, AU-ROC 87.8; segmentation AU-PR 11.4, AU-ROC 93.1.
  MVTec AD: classification AU-PR 97.4, AU-ROC 94.5; segmentation AU-PR 35.2, AU-ROC 94.4.
  Best VisA row "Sup. pre-train+SPD": 88.6 / 87.8 / 12.0 / 93.8.
- Table 6 (p. 13), PatchCore, Wide ResNet-50, supervised pre-training, 1-class:
  VisA: classification AU-PR 93.3, AU-ROC 92.4; segmentation AU-PR 38.4, AU-ROC 98.4.
  MVTec AD: classification AU-PR 99.2, AU-ROC 99.8; segmentation AU-PR 48.8, AU-ROC 97.6.
  "+SPD": VisA 93.8 / 92.5 / 39.3 / 98.1.
- Text (Sec. 5.1): "PaDiM is one of the SOTA methods with >97% AU-ROC in MVTec. But it just achieves <90% AU-PR and AU-ROC in VisA - classification. For VisA - segmentation, PaDiM only achieves about 10% AU-PR."
- Table 3 (p. 12), 2-class high-shot, supervised pre-training: classification AU-PR 97.5, AU-ROC 99.5; segmentation AU-PR 65.1, AU-ROC 97.3.
- Difficulty order (App. D): multiple-instance subsets hardest, then complex-structure PCBs, single-instance easiest.
- Protocol caveat for the scoreboard: these are re-runs inside the VisA paper (PaDiM with ResNet-50, not PaDiM's own WRN-50 / R18 configs; image resolution not stated in the paper), on the VisA paper's own 1-class split. Later papers' VisA numbers (EfficientAD, MVTec AD 2 Table 1) report AU-PRO, which VisA never reports.

## 9. Limitations the paper states
- Large gap between 1-class and 2-class performance on VisA; "bridging the gap is an open challenge" (Sec. 5.1).
- Later: MVTec AD 2 (Sec. 2.1) says VisA is "close to being solved"; EfficientAD-M reaches 98.1 mean image AU-ROC on VisA (EfficientAD Table 2).

## 10. Outline feedback
- Outline says VisA has "multiple instances and complex structures": correct; add the third domain (single instance, roughly aligned) and the 12 subset names.
- VisA's main metric argument is AU-PR, not PRO. The outline's viz-metrics covers image AUROC, pixel AUROC and PRO; the AU-PR toy example (99.5% AU-ROC vs 10.5% AU-PR) is a strong, sourced illustration of the same "ocean of normal pixels" point and is missing from ch-benchmark.
- The "split protocol it proposes": 1-class 90/10 normal split with all anomalies in test (Sec. 4.2). The sources here do not say which split later papers use; check each method paper before mixing VisA numbers.
- VisA's own baseline numbers (PaDiM R50, PatchCore WRN-50) are not the numbers the scoreboard should use for PaDiM / PatchCore; use them only as "VisA paper re-run" cells with protocol labels.
