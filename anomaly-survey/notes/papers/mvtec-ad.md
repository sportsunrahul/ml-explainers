# MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection

## 1. Identity
- Title: "MVTec AD -- A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection" (PDF title; the PDF uses an em dash).
- Not on arXiv. No abs.html, no LaTeX. Source: `sources/mvtec-ad/paper.pdf` (9 pages, CVF open-access print).
- Venue: CVPR 2019 (PDF metadata "IEEE Conference on Computer Vision and Pattern Recognition"; proceedings pages 9592-9600 printed in the page footers). Cite as "MVTec AD, CVPR 2019".
- Page mapping: PDF page N = proceedings page 9591 + N. Locations below give both.
- Authors: Paul Bergmann ... Carsten Steger (4 authors, MVTec Software GmbH).
- Dataset URL in paper: www.mvtec.com/company/research/datasets (footnote 1, p. 2). No code repo.
- A journal extension (IJCV 2021, "bergmann2021_mvtec_ad_ijcv") is cited by EfficientAD and MVTec AD 2 as the source of AU-PRO. It is NOT in `sources/`.

## 2. Core problem
Anomaly detection was benchmarked by relabelling classes of MNIST/CIFAR/ImageNet as outliers, so the anomalies differ a lot from the training data. Industrial inspection needs to find subtle, small, local defects and to segment them. No real-world, multi-object, pixel-annotated dataset existed for this (Sec. 1, Sec. 2.1).

## 3. Prior datasets and why that hurt (per the paper, Sec. 2.1)
- Class-relabelling (MNIST, CIFAR10, ImageNet): outliers "differ significantly" from the training distribution; image-level only.
- Saleh et al. (abnormally shaped objects vs PASCAL VOC): still image-level, no localization.
- NanoTWICE: 45 gray-scale SEM images of one texture (5 train, 40 test); one texture only.
- DAGM 2007: ten classes of artificially generated textures, 1000 defect-free and 150 defective patches per class; coarse ellipse labels, low variance between textures, synthetic.

## 4. The one key idea
A real-world benchmark: train only on defect-free images of 15 industrial categories (5 textures, 10 objects); test on defect-free and defective images with pixel-precise ground truth, so both image classification and pixel segmentation can be scored.

## 5. Mechanism (dataset construction and protocol), step by step
1. Acquire with a 2048 x 2048 industrial RGB sensor and two bilateral telecentric lenses (magnification 1:5 and 1:1), then crop (Sec. 3, p. 5 / 9596).
2. Resolutions 700 x 700 to 1024 x 1024 (text, p. 5); per category "Image side length" in Table 1 (all square).
3. Grid, screw, zipper are single-channel gray images only (p. 5).
4. Train set: defect-free only ("3629 images for training and validation"). Test set: defect-free plus defective (1725 images) (Sec. 3, p. 4 / 9595).
5. Defects were made by hand to look realistic; 73 defect types, "on average five per category" (p. 4).
6. Pixel-precise masks per defect region; 1888 regions in Table 1 ("almost 1900", p. 5).
7. Benchmark protocol (Sec. 4.3): a threshold is estimated from a held-out set of defect-free validation images (taken out of the train set) via a per-category minimum defect area; then report (a) image classification accuracy on good and defective test images, (b) per-region overlap at that threshold, and (c) threshold-free pixel ROC AUC.

## 6. Equations
The paper has no numbered equations. Metric definitions are in prose (Sec. 4.3, p. 6-7 / 9597-9598):
- Pixel ROC AUC: "We define the true positive rate as the percentage of pixels that were correctly classified as anomalous across an evaluated dataset category. The false positive rate is the percentage of pixels that were wrongly classified as anomalous."
- Classification: "we compute the accuracy of correctly classified images for anomalous and anomaly-free test images." (Table 2 reports the two ratios separately, not an AUROC.)
- Segmentation at a threshold: "we evaluate the relative per-region overlap of the segmentation with the ground truth." This is a single-threshold PRO value. The integrated AU-PRO up to FPR 0.3 is NOT in this paper; it first appears in Uninformed Students (arXiv 1911.02357v2, Sec. 4.2, p. 7) and is attributed by later papers to the IJCV 2021 extension. See `mvtec-ad2.md` for the PRO equation.
- Image-level AUROC is NOT used in this paper.

## 7. Measurement setup (dataset lens)
Table 1 (p. 4 / 9595), "Statistical overview of the MVTec AD dataset":

| Group | Category | # Train | # Test (good) | # Test (defective) | # Defect groups | # Defect regions | Image side length |
|---|---|---|---|---|---|---|---|
| Textures | Carpet | 280 | 28 | 89 | 5 | 97 | 1024 |
| Textures | Grid | 264 | 21 | 57 | 5 | 170 | 1024 |
| Textures | Leather | 245 | 32 | 92 | 5 | 99 | 1024 |
| Textures | Tile | 230 | 33 | 84 | 5 | 86 | 840 |
| Textures | Wood | 247 | 19 | 60 | 5 | 168 | 1024 |
| Objects | Bottle | 209 | 20 | 63 | 3 | 68 | 900 |
| Objects | Cable | 224 | 58 | 92 | 8 | 151 | 1024 |
| Objects | Capsule | 219 | 23 | 109 | 5 | 114 | 1000 |
| Objects | Hazelnut | 391 | 40 | 70 | 4 | 136 | 1024 |
| Objects | Metal Nut | 220 | 22 | 93 | 4 | 132 | 700 |
| Objects | Pill | 267 | 26 | 141 | 7 | 245 | 800 |
| Objects | Screw | 320 | 41 | 119 | 5 | 135 | 1024 |
| Objects | Toothbrush | 60 | 12 | 30 | 1 | 66 | 1024 |
| Objects | Transistor | 213 | 60 | 40 | 4 | 44 | 1024 |
| Objects | Zipper | 240 | 32 | 119 | 7 | 177 | 1024 |
| | Total | 3629 | 467 | 1258 | 73 | 1888 | - |

Checks (our reading, arithmetic): 467 + 1258 = 1725 test; 3629 + 1725 = 5354; texture train 1266, object train 2363. Column sums equal the printed totals.

Category facts (Sec. 3, p. 4): regular textures carpet, grid; random textures leather, tile, wood. Rigid objects with fixed appearance (bottle, metal nut); deformable (cable); natural variation (hazelnut). Roughly aligned: toothbrush, capsule, pill. Random rotation: metal nut, screw, hazelnut. Defect kinds: surface (scratches, dents), structural (distorted parts), missing parts. Some object classes had illumination changed on purpose to add variability (p. 5).

Baselines (Sec. 4.1): AnoGAN, L2 and SSIM autoencoders (CAE of Bergmann et al.; 128 x 128 texture patches, objects at 256 x 256, latent dim 100), CNN Feature Dictionary (ResNet-18 avgpool 512-d, PCA 95% variance, K-means 10 centres), GMM texture inspection (HALCON), variation model (HALCON; objects that can be aligned only). Augmentation to 10000 training patches per category (Sec. 4.2).

## 8. Results worth remembering
- No method is consistently good across categories (Sec. 4.4). The paper prints no means.
- Pixel ROC AUC is high while per-region overlap at the estimated threshold is low (Table 3, p. 7 / 9598): e.g. Cable, AE (SSIM) 0.01 overlap / 0.82 ROC AUC; Transistor, AE (SSIM) 0.01 / 0.90; Hazelnut, AE (SSIM) 0.00 / 0.97. The paper: "a high ROC AUC does not necessarily coincide with a high per-region overlap of the segmentation for the estimated threshold" (Sec. 4.4).
- AE (SSIM) pixel ROC AUC per category, Table 3 bottom rows: Carpet 0.87, Grid 0.94, Leather 0.78, Tile 0.59, Wood 0.73, Bottle 0.93, Cable 0.82, Capsule 0.94, Hazelnut 0.97, Metal Nut 0.89, Pill 0.91, Screw 0.96, Toothbrush 0.92, Transistor 0.90, Zipper 0.88. Protocol: one model per category, gray-scale, objects at 256 x 256.
- Later numbers for these baselines on the integrated AU-PRO (FPR 0.3) come from Uninformed Students Table 1 (p. 5): SSIM-AE mean 0.694, AnoGAN 0.443, CNN Feature Dictionary 0.515 (re-run by Bergmann et al. 2020 with the MVTec AD hyperparameters).

## 9. Limitations the paper states
- Threshold estimation from defect-free data alone is hard; ROC AUC can be fine while segmentation fails (Sec. 4.4).
- Defects were manually generated (Sec. 3).
- Later fixes (per sources): saturation is stated by MVTec AD 2 (Sec. 1, Sec. 2.1: best model 97.8% mean AU-PRO, citing CPR) and by VisA (Sec. 1: AU-ROC "approaching ~95%"). EfficientAD Sec. 4 notes that 99.9% of MVTec AD defects lie fully or partly inside PatchCore's center crop of 76.6% (central-defect bias). MVTec AD 2 Fig. 2 shows MVTec AD labels concentrated at the image center.

## 10. Outline feedback
- ch-benchmark says MVTec AD CVPR 2019 defines the grading. It does not: it reports per-class accuracy (Table 2), single-threshold per-region overlap and pixel ROC AUC (Table 3). Image-level AUROC and AU-PRO up to FPR 0.3 come later (Uninformed Students 2020; IJCV 2021 extension, not in sources). Attribute them accordingly.
- "Over 70" (abstract) vs "73" (Sec. 1, Sec. 3, Table 1): use 73 with Table 1 as loc.
- "Almost 1900" regions (Sec. 3) vs 1888 (Sec. 1, Table 1): use 1888.
- Train count wording: the paper says 3629 images "for training and validation"; the CVPR protocol takes the validation images out of this set.
- The memory-bank sizes in ch-patchcore can be computed from the per-category train counts in Table 1 (smallest: Toothbrush 60; largest: Hazelnut 391).
