# ADer: A Comprehensive Library for Benchmarking Multi-class Visual Anomaly Detection

## 1. Identity
- Title: "A Comprehensive Library for Benchmarking Multi-class Visual Anomaly Detection" (library ADer, eval package ADEval).
- arXiv 2406.03262, read v6 (2025-08-19). v1 submitted 2024-06-05 (abs.html). Six versions; v1/v2 June 2024, v3 2024-09-30, v4 2025-07-15, v5 2025-07-20, v6 2025-08-19.
- Authors: Jiangning Zhang (first, equal contribution with Haoyang He) ... Yong Liu (last, corresponding). YouTu Lab Tencent, Zhejiang University, HUST.
- Venue: abs.html has no Comments or journal-ref field. main.tex uses the ICCV 2025 author-kit style in camera-ready mode (`\usepackage{iccv}`, `\confName{ICCV}`, `\confYear{2025}`). This is a template observation, not a stated acceptance.
- Repo: https://github.com/zhangzjn/ADer (title block).
- Sources: `sources/ader/src/` (secs/*.tex, appendix/*.tex), `paper.pdf` (104 pages; main tables on pp. 4-7).

## 2. Core problem (benchmark lens)
Multi-class AD papers use different training epochs, resolutions and metric code, and select the best epoch, so their numbers are not comparable. Most methods were never tested on the new large datasets (Real-IAD, COCO-AD) or metrics (mAD, mIoU-max). ADer re-runs 15 methods in one codebase under one multi-class protocol.

## 3. What prior benchmarks did and why that hurt (per the paper)
- "The absence of standardized experimental setups can lead to potential biases in training epochs, resolution, and metric results, resulting in erroneous conclusions." (abstract)
- open-iad and Anomalib cover general AD models "only up to 2022" and "have not yet implemented the latest and practical multi-class anomaly detection methods" (Sec. 3.5).
- "most authors select the best epoch's results ... this method of epoch selection is unscientific" (Sec. 4.2, Stability analysis).
- Naive sklearn/skimage pixel metrics (mAU-PRO) take over an hour on Real-IAD / COCO-AD (Sec. 3.6).

## 4. The one key idea
Fix the factors that move scores most (resolution 256^2, training length 100 and 300 epochs, report the FINAL epoch, one GPU type, one metric implementation) and run every method in the multi-class setting (one model for all classes of a dataset).

## 5. Measurement setup
1. Setting: multi-class. "Given a training dataset ... with C categories ... learn a unified AD model M" (Sec. 2.1). One model per dataset, all classes.
2. Fixed: resolution 256^2; epochs 100 and 300 (Real-IAD and COCO-AD: 100 only, Table 2); results "corresponding to the final epoch at the end of training"; "All experiments are conducted on one L40S GPU" (Sec. 4.1).
3. Kept from each original paper: batch size, optimizer, learning rate, data augmentation (Sec. 4.1). Table 1 lists them.
4. Metrics (9, Sec. 3.3, following ViTAD): image mAU-ROC, mAP, mF1-max; region mAU-PRO; pixel mAU-ROC, mAP, mF1-max; mAD (average of the seven); mIoU-max (from InvAD). "m" = mean over categories. The AU-PRO FPR integration limit is NOT stated in ADer.
5. Datasets (11): MVTec AD, MVTec AD 3D, MVTec LOCO-AD, VisA, BTAD, MPDD, MAD_Real, MAD_Sim, Real-IAD, Uni-Medical, COCO-AD (Table 2). MVTec AD: 15 classes, 3,629 train normal, 1,258 test anomaly, 467 test normal. VisA: 12 classes, 8,659 train normal, 962 test anomaly, 1,200 test normal.
6. Methods (15 named; 16 table rows because InvAD-lite has its own row), Sec. 3.1:
   - Augmentation-based: DRAEM, SimpleNet, RealNet.
   - Embedding-based: CFA, PatchCore, CFLOW-AD, PyramidalFlow.
   - Reconstruction-based: RD, DiAD, ViTAD, InvAD, InvAD-lite, MambaAD.
   - Hybrid: UniAD, RD++, DesTSeg.
7. Efficiency (Table 1): Params, FLOPs, backbone, train memory and test memory (MB, "tested under the standard setting with a batch size of 8"), and total train time and total test time per dataset on one L40S. Test time is for the whole test set, not per-image latency. The caption does not say whether train time is for 100 or 300 epochs.
8. ADEval: GPU histogram-binned, iterative accumulation of TPR/FPR for AUROC/AUPR/AUPRO, O(N) memory in the number of bins; "more than 1000-fold" faster evaluation (abstract; Sec. 3.6). Equations (1)-(4) define the binned TPR/FPR; not needed for ch-uniad.

Spine coverage asked for this chapter:
- Present in ADer: PatchCore, DRAEM, UniAD, DiAD, MambaAD.
- ABSENT from ADer: PaDiM, CutPaste, FastFlow, EfficientAD, Dinomaly (grep over all sources: none appear in any table; CutPaste is cited only in related work).
- Branch methods present: CFA (PatchCore branch), CFLOW-AD (flow branch), SimpleNet (synthetic branch), RD and RD++ (student-teacher branch).

What would falsify the benchmark's claims: a method's ranking flipping under a different fixed epoch count or resolution (Table 3 vs Table 4 already shows UniAD and DiAD move a lot from 100 to 300 epochs), or rankings that differ on another GPU (the paper says "the overall relative trends remain largely unchanged", Sec. 3.1, but shows no second GPU).

## 6. Equations
None needed for the chapter. ADEval Eqs. (1)-(4) (Sec. 3.6) are histogram forms of TPR/FPR; skip.

## 7. Results worth remembering
Protocol on every number below: "multi-class, ADer protocol, 256x256, final epoch, one L40S GPU". "100e/300e" = the two epoch settings. Values in %.

Table 3 (p. 6), MVTec AD, 100e / 300e:
| method | img mAU-ROC | pix mAU-ROC | mAU-PRO |
|---|---|---|---|
| DRAEM | 54.5 / 55.2 | 47.6 / 48.7 | 14.3 / 15.8 |
| SimpleNet | 95.4 / 79.2 | 96.8 / 82.4 | 86.9 / 62.0 |
| RealNet | 84.8 / 82.9 | 72.6 / 69.8 | 56.8 / 51.2 |
| CFA | 57.6 / 55.8 | 54.8 / 43.9 | 25.3 / 19.3 |
| PatchCore | 98.8 / - | 98.3 / - | 94.2 / - |
| CFLOW-AD | 91.6 / 92.7 | 95.7 / 95.8 | 88.3 / 89.0 |
| PyramidalFlow | 70.2 / 66.2 | 80.0 / 74.2 | 47.5 / 40.0 |
| RD | 93.6 / 90.5 | 95.8 / 95.9 | 91.2 / 91.2 |
| DiAD | 88.9 / 92.0 | 89.3 / 89.3 | 63.9 / 64.4 |
| ViTAD | 98.3 / 98.4 | 97.6 / 97.5 | 92.0 / 91.7 |
| InvAD | 98.1 / 98.9 | 98.0 / 98.1 | 94.4 / 94.4 |
| InvAD-lite | 97.9 / 98.1 | 97.3 / 97.3 | 93.3 / 93.1 |
| MambaAD | 97.8 / 98.5 | 97.4 / 97.6 | 93.4 / 93.6 |
| UniAD | 92.5 / 96.8 | 95.8 / 96.8 | 89.3 / 91.0 |
| RD++ | 97.9 / 95.8 | 97.3 / 97.3 | 93.2 / 92.9 |
| DesTSeg | 96.4 / 96.3 | 92.0 / 92.6 | 83.4 / 82.6 |
PatchCore: "requires no training that shares results under different epoch settings" (caption), so one value; the 300e cell is "-".

Table 4 (p. 6), VisA, 100e / 300e (no PatchCore row: Table 1 lists PatchCore VisA test as OOM):
| method | img mAU-ROC | pix mAU-ROC | mAU-PRO |
|---|---|---|---|
| DRAEM | 55.1 / 56.2 | 37.5 / 45.0 | 10.0 / 16.0 |
| SimpleNet | 86.4 / 80.7 | 96.6 / 94.4 | 79.2 / 74.2 |
| RealNet | 71.4 / 79.2 | 61.0 / 65.4 | 27.4 / 33.9 |
| CFA | 66.3 / 67.1 | 81.3 / 83.0 | 50.8 / 48.7 |
| CFLOW-AD | 86.5 / 87.2 | 97.7 / 97.8 | 86.8 / 87.3 |
| PyramidalFlow | 58.2 / 69.0 | 77.0 / 79.1 | 42.8 / 52.6 |
| RD | 90.6 / 93.9 | 98.0 / 98.1 | 91.9 / 91.9 |
| DiAD | 84.8 / 90.5 | 82.5 / 83.4 | 44.5 / 44.3 |
| ViTAD | 90.4 / 90.3 | 98.2 / 98.2 | 85.7 / 85.8 |
| InvAD | 95.4 / 95.6 | 98.9 / 99.0 | 93.1 / 93.0 |
| InvAD-lite | 94.9 / 95.3 | 98.6 / 98.7 | 93.1 / 93.2 |
| MambaAD | 94.5 / 93.6 | 98.4 / 98.2 | 92.1 / 90.5 |
| UniAD | 89.0 / 91.4 | 98.3 / 98.5 | 86.5 / 89.0 |
| RD++ | 93.9 / 93.1 | 98.4 / 98.4 | 91.9 / 91.4 |
| DesTSeg | 89.9 / 89.0 | 86.7 / 84.8 | 61.1 / 57.5 |
Appendix Tables A1/A4 (100e) and A12/A15 (300e) repeat these averages.

Table 1 (p. 4), efficiency, one L40S (memory at batch 8; times are totals per dataset):
| method | BS | optim | LR | params | FLOPs | backbone | train mem (M) | test mem (M) | MVTec train / test | VisA train / test |
|---|---|---|---|---|---|---|---|---|---|---|
| DRAEM | 8 | Adam | 1e-4 | 97.4M | 198G | UNet | 12,602 | 2,858 | 3.4h / 35s | 8.0h / 36s |
| SimpleNet | 32 | AdamW | 1e-4 | 72.8M | 17.715G | WRN50 | 2,266 | 4,946 | 1.7h / 5m50s | 3.9h / 7m21s |
| CFA | 4 | AdamW | 1e-3 | 38.6M | 55.3G | WRN50 | 4,364 | 2,826 | 1.2h / 18s | 2.7h / 17s |
| PatchCore | 8 | - | - | - | - | WRN50 | - | - | 0.6h / 9h22m | - / OOM |
| CFLOW-AD | 32 | Adam | 2e-4 | 237M | 28.7G | WRN50 | 3,048 | 1,892 | 5.2h / 56s | 10.4h / 1m15s |
| RD | 16 | Adam | 5e-3 | 80.6M | 28.4G | WRN50 | 3,286 | 1,464 | 0.8h / 13s | 1.1h / 18s |
| DiAD | 12 | Adam | 1e-5 | 1331M | 451.5G | RN50 | 26,146 | 20,306 | 9.1h / 16m | 21.6h / 19m |
| ViTAD | 8 | AdamW | 1e-4 | 39.0M | 9.7G | ViT-S | 1,470 | 800 | 0.8h / 15s | 1.1h / 15s |
| InvAD | 32 | Adam | 1e-3 | 95.6M | 45.4G | WRN50 | 5,920 | 3,398 | 1.0h / 31s | 2.3h / 33s |
| MambaAD | 16 | AdamW | 5e-3 | 25.7M | 8.3G | RN34 | 6,542 | 1,484 | 2.4h / 34s | 5.6h / 23s |
| UniAD | 8 | AdamW | 1e-4 | 24.5M | 3.4G | EN-b4 | 1,856 | 968 | 0.8h / 22s | 1.0h / 18s |
| RD++ | 16 | Adam | 1e-3 | 96.1M | 37.5G | WRN50 | 4,772 | 1,480 | 3.3h / 28s | 7.8h / 33s |
Real-IAD: PatchCore test OOM; DRAEM train cell printed as "39.2" with no unit.

Text claims:
- UniAD and DiAD "require more epochs to converge and do not perform well under the 100/300 epoch standard we set" (Sec. 4.2).
- "single-class methods like DRAEM, SimpleNet, CFA, and PyramidFlow show significant performance gaps in multi-class anomaly detection" while RD, RD++, CFLOW-AD, RealNet "also perform well in multi-class settings" (Sec. 4.2).
- PatchCore: "only feasible for multi-class anomaly detection tasks on small-scale datasets"; "its testing time is nearly a thousand times longer than other methods" (Sec. 4.2, Training-free PatchCore).
- InvAD "consistently shows excellent performance across all datasets" (Sec. 4.2).
- Slow convergers (big gain 100 -> 300): CFLOW-AD, DiAD, InvAD, MambaAD, UniAD. Unstable: SimpleNet, PyramidFlow (Sec. 4.2).

Cross-paper discrepancies (record, do not reconcile):
- UniAD MVTec image AUROC: UniAD paper 96.5 (224 px, 1000 epochs, batch 64, 8x V100, mean of 5 seeds) vs ADer 92.5 (100e) / 96.8 (300e) (256 px, batch 8, final epoch, one L40S). Pixel: 96.8 vs 95.8 / 96.8.
- DRAEM multi-class MVTec image AUROC: UniAD's re-run 88.1 (UniAD Table 1) vs ADer 54.5 / 55.2. Pixel: 87.2 vs 47.6 / 48.7.
- UniAD params / FLOPs: UniAD Table A7 7.48 M learnable, 6.46 G at 224 vs ADer 24.5M, 3.4G. `our reading`: ADer's params likely include the frozen backbone; counting rules are not stated.

## 8. Limitations stated
- "Immature method": all methods are weak on MVTec LOCO, MAD and COCO-AD in multi-class (Sec. 4.3).
- Efficiency: "Most methods do not consider model complexity during design" (Sec. 4.3).
- Dataset scale, AD-specific metrics, augmentation/tricks, interpretability listed as open (Sec. 4.3).
- Only one GPU type; hardware dependence acknowledged (Sec. 3.1).
- A commented-out Limitations paragraph exists in the source; not part of the paper.

## 9. Outline feedback
1. ADer has no PaDiM, CutPaste, FastFlow, EfficientAD or Dinomaly. The ch-uniad multi-class comparison must use PatchCore, DRAEM, SimpleNet, CFA, CFLOW-AD, RD/RD++ for "one-class methods run multi-class", and UniAD, DiAD, MambaAD (plus ViTAD, InvAD) for multi-class designs. PaDiM/CutPaste multi-class numbers exist only in UniAD's own Table 1/2 (UniAD protocol).
2. ch-dinov2 wants Dinomaly "on the ch-uniad multi-class axis". ADer has no Dinomaly row, so that tab must use Dinomaly's own paper and say the protocol differs.
3. Key twist for ch-uniad: PatchCore does NOT collapse in multi-class: 98.8 image AUROC on MVTec AD, best in Table 3. Its problem is cost: 9h22m test time on MVTec AD and OOM on VisA and Real-IAD. The "one-class methods drop" story holds for DRAEM, CFA, PyramidalFlow and (in UniAD's runs) PaDiM and CutPaste, not for memory banks.
4. Speed: ADer Table 1 times are whole-test-set totals on an L40S, not per-image latency. ch-speed should not plot them as latency without dividing by test-set size (`our reading`: MVTec AD test = 1,725 images; the paper does not do this division).
5. Real-IAD (Table 5) is a large multi-class dataset where UniAD gets 83.1 image AUROC; worth one line if ch-benchmark covers "after saturation".
