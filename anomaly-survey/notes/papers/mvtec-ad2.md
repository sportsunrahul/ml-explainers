# MVTec AD 2: Advanced Scenarios for Unsupervised Anomaly Detection

## 1. Identity
- Title: "The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection".
- arXiv 2503.21622, version read: v1 (abs.html "[Submitted on 27 Mar 2025]"; only version). Published 2025-03-27.
- Authors: Lars Heckler-Kram ... Carsten Steger (5 authors, MVTec Software GmbH; first author also TUM).
- Venue: abs.html Comments: "paper under review; dataset first released for the VAND3.0 challenge @ CVPR 2025". abs.html `citation_doi` = 10.1007/s11263-026-02743-0 (an IJCV-style DOI). We record both and assert neither.
- Data: https://www.mvtec.com/company/research/datasets/mvtec-ad-2 (CC BY-NC-SA 4.0). Evaluation server: https://benchmark.mvtec.com/.
- Sources: `sources/mvtec-ad2/src/MAD2_main.tex` (macros: `\auprothirty` = AU-PRO_0.30, `\auprofive` = AU-PRO_0.05, `\testprivate` = TEST_priv, `\testprivatemixed` = TEST_priv,mix, `\testpublic` = TEST_pub), `paper.pdf` (21 pages).

## 2. Core problem
MVTec AD and VisA saturate: state-of-the-art AU-PRO values differ by less than one percentage point, which is inside run-to-run noise. Test sets are also misused for hyperparameter tuning. A harder, standardized benchmark is needed (Sec. 1, abstract).

## 3. What prior datasets did and why that hurt (Sec. 2.1, Table 2)
- MVTec AD: 5,354 images, 15 categories; "nearly saturates with the best model achieving 97.8% mean AU-PRO" (cites CPR).
- VisA: 10,821 images, 12 categories; "close to being solved" (cites EfficientAD, DiffusionAD).
- MVTec LOCO AD: logical anomalies; best localization 79.8% (cites EfficientAD; this equals EfficientAD-M LOCO mean AU-sPRO up to FPR 0.05, EfficientAD Table 13, p. 20).
- Real-IAD: >150,000 images, 30 objects; already 94.4% pixel AU-PRO_0.30.
- Table 2: none of MVTec AD, LOCO, VisA, MVTec 3D AD, Eyecandies, Real-IAD has defects at image borders or a varying number of objects per image; only Eyecandies (synthetic) has lighting changes and transparent objects.

## 4. The one key idea
Eight hard real-world scenarios (transparent, reflective, bulk/overlapping objects, high normal variability, tiny defects, defects at borders), test images under unseen lighting, a private test set graded only on a server, and a stricter localization metric (AU-PRO up to FPR 0.05).

## 5. Mechanism (dataset and protocol), step by step
1. Eight scenarios, 8,004 high-resolution images (Sec. 3.1). Images 2.6 to 5 megapixels, varying aspect ratios (Sec. 3.1, Table 4).
2. Bulk goods: Wall Plugs, Walnuts, Rice. High-variability textures: Fabric, Sheet Metal. Reflective: Sheet Metal, Can. Transparent: Vial, Fruit Jelly (Sec. 3.1).
3. Gray-value scenarios: Sheet Metal, Vial, Wall Plugs. Back light: Vial, Fruit Jelly. Dark field: Sheet Metal (Sec. 3.1).
4. Defects placed across the whole image, including at the borders (Sec. 3.1, Fig. 2).
5. Splits (Sec. 3.2): train and validation are defect-free; train uses regular lighting only. Test = TEST_pub (small, with ground truth, all lighting conditions), TEST_priv (same lighting as train; ground truth on server only), TEST_priv,mix (same scenes as TEST_priv, seen and unseen lighting chosen at random per image).
6. Every scene captured under at least four lighting conditions (Sec. 3.3).
7. Benchmark (Sec. 4.1): 7 methods, official code, MVTec AD default settings, no scenario-specific tuning, inputs resized to 256 x 256 unless stated. EfficientAD-S on MVTec AD 2 (EfficientAD-M in Table 1 for MVTec AD / VisA). PatchCore: ensemble WRN-101 + ResNeXt-101 + DenseNet-201, coreset 0.01%, dim 384, center crop disabled.
8. Thresholds (Sec. 4.2.2): mean + 3 x std of per-pixel scores on defect-free validation images. Image score = max pixel value; same threshold for pixel and image decision; F1 at both levels.
9. Timing (Sec. 4.2.3): NVIDIA RTX 2080 Ti, mean of 1000 forward passes after warm-up, batch size 1, float32, time from before host-to-GPU transfer until the anomaly map is on the CPU.

## 6. Equations
Eq. (1), Sec. 4.2.1 (PDF p. 8), verbatim from `MAD2_main.tex`:
```
\mathrm{PRO}=\frac{1}{k} \sum_{i=1}^{\left|D_{\text {test }}\right|} \sum_{j=1}^{k_i} \frac{\left|P_{\mathrm{ano}} \cap C_{i, j}\right|}{\left|C_{i, j}\right|}
```
Paper's wording: "where C_{i,j} represents the set of pixels classified as anomalous for a connected component j in the ground truth image i and P_ano is the set of pixels predicted as anomalous for a given threshold t." AU-PRO "integrates the PRO over the false positive rate (FPR) for varying thresholds."
Gloss:
- D_test: the test images. i: one test image. k_i: number of ground-truth connected components (defect regions) in image i.
- C_{i,j}: the pixels of ground-truth region j in image i.
- P_ano: pixels predicted anomalous at threshold t.
- |P_ano intersect C_{i,j}| / |C_{i,j}|: fraction of region j that is covered by the prediction.
- k: never defined in the text. Our reading: total number of ground-truth regions, k = sum_i k_i, so PRO is the mean coverage over all regions.
- The FPR on the x-axis is the per-pixel false positive rate on defect-free pixels. The curve is integrated from 0 to the limit (0.30 "typically"; 0.05 for MVTec AD 2).

Integration limit, verbatim (Sec. 4.2.1): "Typically, the PRO curve is integrated up to a false positive rate (FPR) of 0.3 (AU-PRO_0.30). However, ... we decrease the tolerance for false positive pixels and establish an integration limit for the PRO curve of 0.05 for MVTec AD 2 (AU-PRO_0.05)." Reason: "for a defect size of 10 pixels in a 5MP image, even a FPR of 5% as integration limit results in a wrongly segmented area that is more than 25,000 times larger than the defect itself."

Why PRO and not pixel AU-ROC (Sec. 4.2.1): pixel AU-ROC "measures the true positive rate (TPR) against the false positive rate"; "In case of multiple anomalous regions the TPR is dominated by large anomalies and overestimates localization quality". Worked example, Fig. 4 (p. 8), ground truth with three regions of 4, 2 and 20 pixels (26 anomalous pixels):
- i) all found: PRO = (4/4 + 2/2 + 20/20)/3 = 1; TPR = 26/26 = 1.
- ii) half of each: PRO = (2/4 + 1/2 + 10/20)/3 = 0.5; TPR = 13/26 = 0.5.
- iii) large region missed: PRO = (4/4 + 2/2 + 0/20)/3 = 0.67; TPR = 6/26 = 0.23.
- iv) only large region found: PRO = (0/4 + 0/2 + 20/20)/3 = 0.33; TPR = (0 + 0 + 20)/26 = 0.77.

## 7. Measurement setup
Table 4 (p. 6), "Statistical overview". Test cells are total (good/defective).

| Object | Train | Val | TEST_pub | TEST_priv | TEST_priv,mix | Test total | Size (W x H) |
|---|---|---|---|---|---|---|---|
| Can | 412 | 46 | 162 (72/90) | 321 (145/176) | 321 (145/176) | 804 (362/442) | 2232 x 1024 |
| Fabric | 387 | 43 | 156 (66/90) | 314 (133/181) | 314 (133/181) | 784 (332/452) | 2448 x 2048 |
| Fruit Jelly | 263 | 37 | 80 (20/60) | 255 (71/184) | 255 (71/184) | 590 (162/428) | 2100 x 1520 |
| Rice | 313 | 35 | 132 (42/90) | 277 (96/181) | 277 (96/181) | 686 (234/452) | 2448 x 2048 |
| Sheet Metal | 137 | 19 | 114 (24/90) | 142 (36/106) | 142 (36/106) | 398 (96/302) | 4224 x 1056 |
| Vial | 291 | 41 | 140 (35/105) | 276 (78/198) | 276 (78/198) | 692 (191/501) | 1400 x 1900 |
| Wall Plugs | 293 | 33 | 150 (60/90) | 232 (96/136) | 232 (96/136) | 614 (252/362) | 2448 x 2048 |
| Walnuts | 432 | 48 | 150 (60/90) | 228 (93/135) | 228 (93/135) | 606 (246/360) | 2448 x 2048 |

Sums (our reading): train 2528, val 302, test 5174; total 8004 = the printed 8,004. TEST_priv and TEST_priv,mix show the same scenes, so the 8,004 counts each scene twice in the test part.

## 8. Results worth remembering
- Table 1 (p. 2), AU-PRO_0.30 (%), columns MVTec AD / VisA / MVTec AD 2:
  PatchCore 92.7 / 79.7 / 53.8; RD 93.9 / 89.0 / 53.0; RD++ 95.0 / 89.3 / 54.3; EfficientAD 93.5 / 94.0 / 58.7; MSFlow 91.2 / 88.4 / 52.7; SimpleNet 89.6 / 68.9 / 46.4; DSR 90.8 / 68.1 / 49.0; Mean 92.4 / 82.4 / 52.6.
  Provenance: RD and RD++ MVTec AD values are taken from their papers (Sec. 4.1); their VisA values were run by the authors with MVTec AD parameters. MSFlow MVTec AD and VisA were run at 256 x 256 (not the official 512). The PatchCore, EfficientAD, SimpleNet and DSR MVTec AD / VisA values are numerically identical to EfficientAD (2303.14535v3) Table 12 (our reading; the paper does not say where they come from). EfficientAD row mixes variants: EfficientAD-M for MVTec AD / VisA, EfficientAD-S for MVTec AD 2. MVTec AD 2 column = Table 9, TEST_priv, 256 x 256.
- Headline drop (our arithmetic, not printed): mean AU-PRO_0.30 92.4 -> 52.6, i.e. 39.8 points. Text: "Methods that achieve over 90% AU-PRO_0.30 on established datasets like MVTec AD score a maximum of 58.7% on MVTec AD 2" (Sec. 1).
- Table 7 (p. 10), AU-PRO_0.05 (%), mean, TEST_priv / TEST_priv,mix, 256 x 256: PatchCore 28.8 / 26.0; RD 26.4 / 25.0; RD++ 27.1 / 25.2; EfficientAD 30.8 / 19.2; MSFlow 24.3 / 11.9; SimpleNet 21.1 / 13.7; DSR 20.3 / 17.4. "All methods achieve less than 31%" (caption).
- Lighting robustness (Sec. 4.3.1): RD only 1.4 points worse on TEST_priv,mix; MSFlow drops 12.4 points, 24.3% -> 11.9%.
- Hardest object: Can ("especially challenging", Sec. 4.3.1); Table 7 Can row ranges 4.7 to 13.9 on TEST_priv.
- Threshold-dependent (Table 8, p. 10): best mean pixel F1 on TEST_priv is MSFlow 21.8%; PatchCore 3.7%. Image F1 (Table 12, p. 15): DSR 78.0, EfficientAD 76.8, PatchCore 8.0 (TEST_priv). PatchCore's low F1 is blamed on its dataset-wide [0,1] normalization, which inflates the validation-derived threshold (App. B.2).
- Higher resolution (Sec. 4.3.3, Fig. 6, Table 11): PatchCore AU-PRO_0.05 rises from 28.8% at 256 x 256 to 62.3% at half the original width and height (TEST_priv); on Rice PatchCore "inference takes 2 seconds per image" (RTX 2080 Ti); SimpleNet runs out of memory at the largest size.
- AU-PRO_0.30 on MVTec AD 2 (Table 9, p. 15, TEST_priv / TEST_priv,mix, 256 x 256): PatchCore 53.8 / 50.8; RD 53.0 / 50.3; RD++ 54.3 / 50.4; EfficientAD 58.7 / 44.1; MSFlow 52.7 / 37.1; SimpleNet 46.4 / 37.6; DSR 49.0 / 45.2.

## 9. Limitations the paper states (Sec. 6, "Limitations and Future Work")
- Only seven methods benchmarked; more to come on the server.
- The simple mean + 3 std threshold; better threshold estimation might align threshold-free and thresholded metrics.
- Augmentation against lighting shift not studied; tiling / multi-scale for large images not studied.
- Calls for public benchmark servers on embedded hardware (Jetson Nano 4 GB, Hailo-8 int8).
- Out of scope by design (Table 2 caption): logical anomalies and 3D data.

## 10. Outline feedback
- ch-benchmark "how far scores drop": use Table 1 (mean 92.4 -> 52.6 AU-PRO_0.30; best 58.7) and say the metric is AU-PRO_0.30, not image AUROC. MVTec AD 2 reports no image AUROC at all; it uses AU-PRO and F1.
- Fig. 4 (p. 8) is a ready-made, sourced PRO-vs-TPR example (regions of 4, 2, 20 pixels). viz-metrics can reuse those exact numbers instead of a toy.
- The PRO FPR cut-off: 0.3 is the convention (confirmed here, in EfficientAD App. D.2, and in Uninformed Students Sec. 4.2). MVTec AD 2 introduces 0.05. The outline should mention both.
- MVTec AD 2 is also a natural hook for ch-speed (resolution vs runtime vs memory, Fig. 6, RTX 2080 Ti).
