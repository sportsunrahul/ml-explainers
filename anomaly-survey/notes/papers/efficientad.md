# EfficientAD (arXiv 2303.14535v3)

Lens: method. Read from `sources/efficientad/src/main.tex` (LaTeX, exact) plus `paper.pdf` for page
numbers and figures. Page numbers below are PDF pages of v3. Line numbers are `main.tex` lines.

## 1. Identity
- Title: "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies".
- arXiv 2303.14535, version read: v3 (Thu 8 Feb 2024). v1 submitted 2023-03-25 (abs.html
  `citation_date` 2023/03/25 and submission history). v2 2023-07-28.
- Authors: Kilian Batzner, Lars Heckler, Rebecca Koenig (o-umlaut in the paper), MVTec Software GmbH.
- Venue: WACV 2024, "Accepted as Oral" (abs.html comments); journal-ref: Proc. IEEE/CVF WACV 2024,
  pp. 128-138.
- Repo: the paper names no official code repository. It links only baseline repos and the
  `per_scenario_results.json` file (mydrive.ch), which is NOT in `sources/`.

## 2. Core problem
Recent anomaly detectors gain accuracy through ensembles, large backbones and inputs up to 768x768,
and so lose speed (line 87-88). Industrial lines have strict runtime limits (line 102). The paper wants
state-of-the-art detection and localization at a latency of about 2 ms per image on a GPU.

## 3. What prior methods did and why that hurt (per the paper)
- Deep pretrained backbones (WideResNet-101 for PatchCore) are slow, and their features have
  long-range dependencies: an anomaly in one place can make distant feature vectors anomalous, which
  hurts localization (Sec. 3.1, Fig. 3, lines 239-243).
- Original S-T (Uninformed Students) uses few layers and 1.6 to 2.7 M parameters per network, but no
  downsampling, so one network is slower and uses more memory than a 31 M-parameter U-Net (lines
  225-229). Parameter count is "a misleading proxy" for latency.
- Prior S-T gains come from ensembles of teachers/students, feature pyramids, or architectural
  asymmetry (AST); all cost runtime (line 265).
- More training images can teach the student to imitate the teacher on anomalies too, which hurts
  detection (line 268-269).
- Autoencoders reconstruct fine patterns poorly, so teacher-minus-AE maps give false positives on
  normal images (lines 344-347).
- Baseline protocol issues the paper corrects: SimpleNet tunes training duration on test images;
  PatchCore crops the center 76.6 % of the image; FastFlow (anomalib) uses early stopping on test images
  (lines 392-405).

## 4. The one key idea
A 4-layer patch description network (PDN), distilled from WideResNet-101 features, serves as the
teacher and the student. A "loss-induced asymmetry" (hard feature loss plus an ImageNet penalty)
replaces architectural asymmetry, so the student fails off the normal data at no test-time cost. An
autoencoder that the same student also imitates adds a global map for logical anomalies.

## 5. Mechanism, step by step
1. Stage 1, distillation (once, not per product; Alg. 3, App. A.2, p. 15-16). Frozen feature extractor
   Psi = WideResNet-101 with PatchCore's feature postprocessing ("pooling features from two layers and
   projecting each feature vector to a reduced dimensionality of 384 dimensions", line 873; the layers
   are not named here). Psi takes a 512x512 image and gives 384x64x64. The teacher PDN T takes the same
   image at 256x256 and gives 384x64x64. Psi channels are normalized with mean/std from 10,000 ImageNet
   images. Loss: MSE between normalized Psi features and T output. Data: ImageNet. Only T is trained.
2. Stage 2, per scenario (Alg. 1, p. 12-13). T is frozen. Compute teacher channel mean/std mu, sigma
   (384 each) over the training images. Student S (PDN, 768 output channels) and autoencoder A are
   randomly initialized and trained jointly for 70,000 iterations, batch size 1.
3. Each iteration, structural branch: T and S on the un-augmented training image I. The first 384
   student channels Y^ST imitate the normalized teacher output. Hard feature loss: keep only the
   elements of the squared-difference tensor at or above its 0.999-quantile.
4. Each iteration, penalty: one random ImageNet image P (resize to 512x512, gray scale with
   probability 0.3, center crop 256x256). Penalty = mean squared student output on P, so the student
   outputs near zero off the normal data.
5. Each iteration, logical branch: augment I (one of brightness, contrast, saturation, factor
   lambda ~ U(0.8, 1.2)). A reconstructs the normalized teacher output of I_aug (L_AE). The last 384
   student channels Y^STAE imitate A's output (L_STAE). A gets gradient from L_AE and L_STAE (line
   850). Total loss = L_ST + L_AE + L_STAE.
6. Stage 3, calibration (no training). On validation images (unseen training images), compute local
   map M_ST and global map M_AE, resize to 256x256, and take the 0.9- and 0.995-quantiles of all pixel
   scores for each map type.
7. Inference (Alg. 2, p. 13-14). One pass each of T, S, A. Local map = channel mean of (normalized T -
   S first half)^2. Global map = channel mean of (A - S second half)^2. Resize both to 256x256
   (bilinear), map q_a -> 0 and q_b -> 0.1 linearly, average 0.5/0.5. Image score = max of the
   combined map. The map is resized back to the original image size (line 844).

Which weights get a gradient:
- Stage 1: T only (from L_dist). Psi frozen.
- Stage 2: S and A (union phi, line 716). T frozen. A from L_AE and L_STAE; S from L_ST and L_STAE.
- Stage 3 and inference: nothing.
- Unlike PatchCore, this method trains per scenario: "twenty minutes in our experimental setup" (line
  625).

## 6. Equations
No equation in the paper has a number or `\label`; all are inline TeX. Macros used in the quotes
below: none beyond amsmath. Macros elsewhere in the source: `\ourmethod` = EfficientAD, `\studteach` =
S--T, `\patchcoreens` = PatchCore with subscript "Ens", `\meanwithstd{a}{b}` = "a (+- b)".

E1. Squared difference (Sec. 3.2, line 277, p. 4)
`$D_{c, w, h} = (T(I)_{c, w, h} - S(I)_{c, w, h})^2$` with `$T(I) \in \mathbb{R}^{C \times W \times H}$`.
- T(I), S(I): teacher and student feature maps of training image I. c, w, h: channel and position.
- In Alg. 1 the teacher output is first channel-normalized (E7) and S is the first 384 channels.

E2. Hard feature loss (Sec. 3.2, lines 278-281, p. 4), defined in words:
"Based on a mining factor $p_\mathrm{hard} \in [0, 1]$, we then compute the $p_\mathrm{hard}$-quantile
of the elements of $D$. Given the $p_\mathrm{hard}$-quantile $d_\mathrm{hard}$, we compute the training
loss $L_\mathrm{hard}$ as the mean of all $D_{c, w, h} \geq d_\mathrm{hard}$."
- Paper's meaning: "only uses the output elements with the highest loss for backpropagation";
  p_hard = 0 gives the original S-T loss. Default p_hard = 0.999, "which corresponds to using, on
  average, ten percent of the values in each of the three dimensions of $D$" (0.1^3 = 0.001).
- Our reading: D has 384 x 64 x 64 = 1,572,864 elements, so about 1,573 elements get a gradient per
  image.

E3. Local anomaly map (Sec. 3.2, line 283)
`$M_{w, h} = C^{-1} \sum_c D_{c, w, h}$` -- D averaged across channels; one score per feature vector.

E4. Student loss with ImageNet penalty (Sec. 3.2, line 326, p. 5)
`$L_\mathrm{ST} = L_\mathrm{hard} + (CWH)^{-1}\sum_c \|S(P)_c\|_F^2$`
- P: a random ImageNet image (the teacher's pretraining data). S(P)_c: student channel c on P.
  Frobenius norm squared over the W x H map. Meaning: "hinders the student from generalizing its
  imitation of the teacher to out-of-distribution images." Alg. 1 line 22 sums c = 1..384 (the ST half).

E5. Autoencoder loss (Sec. 3.3, line 339, p. 5)
`$L_\mathrm{AE} = (CWH)^{-1} \sum_c \|T(I)_c - A(I)_c\|_F^2$`
- A(I): AE reconstruction of the teacher's feature map (not of pixels). In Alg. 1 it uses I_aug and the
  normalized teacher output.

E6. Student-vs-autoencoder loss (Sec. 3.3, line 350, p. 5)
`$L_\mathrm{STAE} = (CWH)^{-1} \sum_c \|A(I)_c - S'(I)_c\|_F^2$`
- S'(I): the additional 384 output channels of the student. Total: "the sum of $L_\mathrm{AE}$,
  $L_\mathrm{ST}$, and $L_\mathrm{STAE}$" (line 351); Alg. 1:
  `$L_\mathrm{total} = L_\mathrm{ST} + L_\mathrm{AE} + L_\mathrm{STAE}$`.
- Why it works (lines 353-355): the student learns the AE's systematic errors on normal images (for
  example blur), but not its errors on anomalies.

E7. Teacher channel normalization (Alg. 1, line 689)
`$\hat{Y}_c = (Y'_c - \mu_c) \sigma_c^{-1}$` -- mu_c, sigma_c: mean and std of teacher channel c over the
training images.

E8. Map normalization and combination (Alg. 2, lines 752-755, p. 13)
`$\hat{M}_\mathrm{ST} = 0.1 (M_\mathrm{ST} - q_a^\mathrm{ST}) (q_b^\mathrm{ST} - q_a^\mathrm{ST})^{-1}$`
`$\hat{M}_\mathrm{AE} = 0.1 (M_\mathrm{AE} - q_a^\mathrm{AE}) (q_b^\mathrm{AE} - q_a^\mathrm{AE})^{-1}$`
`$M = 0.5 \hat{M}_\mathrm{ST} + 0.5 \hat{M}_\mathrm{AE}$`, `$m_\mathrm{image} = \max_{i, j} M_{i, j}$`
- q_a = 0.9-quantile, q_b = 0.995-quantile of validation pixel scores (Alg. 1 lines 733-734). Text
  (Sec. 3.4): linear map sends q_a to 0 and q_b to 0.1; the targets do not change AU-ROC, they suit a
  0-to-1 color scale. Robust to how normal scores are distributed between q_a and q_b.

E9. Distillation loss (Alg. 3, lines 906-910, p. 16)
`$D^\mathrm{dist}_{c, w, h} = (\hat{Y}^\Psi_{c, w, h} - Y'_{c, w, h})^2$`; L_dist = mean of D^dist;
`$L_\mathrm{batch} \leftarrow 16^{-1} L_\mathrm{batch}$` (mean over a batch of 16).

## 7. Config / setup table

PDN, EfficientAD-S teacher (Table 6, p. 14). Student: same, Conv-4 has 768 kernels. Shapes for a
3x256x256 input are our reading, computed with the stated padding; Alg. 1 confirms the 384x64x64 output.

| Layer | Stride | Kernel | Kernels | Padding | Act. | Output (our reading) | Params (our reading) |
|---|---|---|---|---|---|---|---|
| Conv-1 | 1 | 4x4 | 128 | 3 | ReLU | 128x259x259 | 6,272 |
| AvgPool-1 | 2 | 2x2 | - | 1 | - | 128x130x130 | 0 |
| Conv-2 | 1 | 4x4 | 256 | 3 | ReLU | 256x133x133 | 524,544 |
| AvgPool-2 | 2 | 2x2 | - | 1 | - | 256x67x67 | 0 |
| Conv-3 | 1 | 3x3 | 256 | 1 | ReLU | 256x67x67 | 590,080 |
| Conv-4 | 1 | 4x4 | 384 (student 768) | 0 | none | 384x64x64 (student 768x64x64) | 1,573,248 (student 3,146,496) |

S teacher total 2,694,144; S student 4,267,392 (our reading).

PDN, EfficientAD-M teacher (Table 7, p. 14). Student: 768 kernels in Conv-5 and Conv-6. Text (line
411-412): double the hidden kernels and insert a 1x1 conv after the second pooling and after the last conv.

| Layer | Stride | Kernel | Kernels | Padding | Act. | Output (our reading) |
|---|---|---|---|---|---|---|
| Conv-1 | 1 | 4x4 | 256 | 3 | ReLU | 256x259x259 |
| AvgPool-1 | 2 | 2x2 | - | 1 | - | 256x130x130 |
| Conv-2 | 1 | 4x4 | 512 | 3 | ReLU | 512x133x133 |
| AvgPool-2 | 2 | 2x2 | - | 1 | - | 512x67x67 |
| Conv-3 | 1 | 1x1 | 512 | 0 | ReLU | 512x67x67 |
| Conv-4 | 1 | 3x3 | 512 | 1 | ReLU | 512x67x67 |
| Conv-5 | 1 | 4x4 | 384 (student 768) | 0 | ReLU | 384x64x64 |
| Conv-6 | 1 | 1x1 | 384 (student 768) | 0 | none | 384x64x64 (student 768x64x64) |

M teacher 8,026,624; M student 11,615,488 (our reading).

- Receptive field: 33x33 for every output vector (line 208; Fig. 2 shows a 33x33x3 patch -> 1x1x384).
  Our RF recomputation gives 33 for both S and M (1x1 convs add nothing).
- PDN feature extraction for 256x256: "less than 800 us" on RTX A6000 (line 232).
- PADDING CAVEAT (App. E, lines 1239-1241, p. 21): "In latency-critical applications, padding in the
  PDN architecture of EfficientAD can be disabled. This speeds up the forward pass ... by 80 us ... We
  time EfficientAD without padding and therefore report the anomaly detection results for this
  setting." So all reported numbers use no padding. The paper does not state the output size then.
  Our reading: 56x56 if all padding (conv and pool) is removed, 58x58 if only conv padding is removed.
  Either way, maps are resized bilinearly to 256x256.

Autoencoder (Table 8, p. 15), same for S and M. Shapes are our reading; the Bilinear rows state sizes.
- Encoder: EncConv-1..5, each 4x4, stride 2, padding 1, ReLU; kernels 32, 32, 64, 64, 64:
  256 -> 128 -> 64 -> 32 -> 16 -> 8. EncConv-6: 8x8, stride 1, padding 0, 64 kernels, no act. -> 64x1x1.
  Bottleneck "of 64 latent dimensions" (line 343).
- Decoder: 6 x [bilinear resize, DecConv 4x4 stride 1 padding 2, 64 kernels, ReLU, Dropout 0.2]:
  1->3->4, 4->8->9, 9->15->16, 16->32->33, 33->63->64, 64->127->128. Then Bilinear-7 128 -> 64,
  DecConv-7 3x3 64 ReLU, DecConv-8 3x3 384 no act. -> 384x64x64.
- AE parameters 1,096,320 (our reading). Totals T+S+A: S 8,057,856 and M 20,738,432 (our reading),
  matching Table 16's printed "8" and "21" [x10^6]. This check confirms the layer reading.

Training and data

| Item | Value | Source |
|---|---|---|
| Input size | 256x256 (all images resized) | Alg. 1; line 844 |
| Image normalization | torchvision ImageNet mean/std | line 846-847 |
| Distillation teacher source | WideResNet-101 + PatchCore postprocessing, 384-d final features (those given to coreset subsampling) | lines 236, 857, 873-875 |
| Distillation data | ImageNet | lines 237, 871 |
| Distillation input | 512x512 to Psi, 256x256 to T; both 384x64x64 out | lines 862-866 |
| Distillation optimizer | Adam, lr 1e-4, weight decay 1e-5 | Alg. 3 |
| Distillation steps / batch | 60,000 iterations, batch 16 | Alg. 3; line 871 |
| Distillation norm stats | 10,000 images; gray scale with prob. 0.1 | Alg. 3 |
| Stage-2 optimizer | Adam, lr 1e-4, weight decay 1e-5, params of S and A | Alg. 1 |
| Stage-2 steps / batch | 70,000 iterations, batch size 1 | Alg. 1; line 845 |
| LR schedule | decay to 1e-5 after iteration 66,500 | Alg. 1 |
| p_hard | 0.999 | line 281; Alg. 1 |
| ImageNet penalty image | random, resize 512x512, gray prob. 0.3, center crop 256x256; one per step | line 851 |
| AE augmentation | one of brightness / contrast / saturation, lambda ~ U(0.8, 1.2) | Alg. 1 |
| AE dropout | 0.2 | Table 8 |
| Map normalization | q_a = 0.9, q_b = 0.995 quantiles on validation images; q_a -> 0, q_b -> 0.1 | Alg. 1; Sec. 3.4 |
| Map combination | 0.5 / 0.5 average; image score = max | Alg. 2 |
| Init | PyTorch 1.12.0 default | line 839 |
| Training time | "twenty minutes in our experimental setup" | line 625 |
| Inference precision | float16 (same detection results as float32) | line 1237-1238 |
| Runs | mean (and std) of five runs | Table 1 caption |

Distillation backbone robustness (Table 9, p. 18): ResNeXt-101 and DenseNet-201 as Psi give similar
results (EfficientAD-M MVTec AD 99.1 / 99.0 / 99.2).

Measurement setup for speed (Sec. 4 p. 7; App. E p. 21-22)
- GPU for Table 1 and Table 16: NVIDIA RTX A6000. Fig. 6 / Table 17 add RTX A5000, Tesla V100,
  RTX 3080, RTX 2080 Ti.
- Latency: batch size 1, mean of 1000 forward passes after 1000 warm-up passes.
- Throughput: batch size 16; "dividing 16000 by the sum of the runtimes of 1000 forward passes with a
  batch size of 16."
- Timing starts at the CPU->GPU transfer of the image and stops when the anomaly map is on the CPU.
  float16 for all networks; loss computation removed. All methods are faster on GPU than CPU,
  including PatchCore's kNN.
- PatchCore memory/params/FLOPs depend on training data; measured on VisA "cashew" (450 training
  images; the mean over the 32 scenarios is 439). kNN FLOPs and memory not measured.

Accuracy averaging protocol (lines 504-509, p. 6-7)
- Per collection: metric per scenario, then mean over scenarios (MVTec AD 15, VisA 12, MVTec LOCO 5;
  32 in total). MVTec LOCO uses the official script, which weights logical and structural anomalies
  equally.
- "Average" in Table 1 = mean of the three collection means. "Thus, an overall average score weights
  logical anomalies and structural anomalies by roughly one-sixth and five-sixths, respectively."
- Detection: image-level AU-ROC. Localization: AU-PRO up to FPR 30 %; on MVTec LOCO, AU-sPRO.
- All methods: one model per scenario, trained on normal images only, full training set.

How baseline numbers were obtained: all re-run by the EfficientAD authors.
Quote (line 387): "We compare EfficientAD to AST, DSR, FastFlow, GCAD, PatchCore, SimpleNet, and S--T,
using official implementations where available." Per method (App. B, p. 17):
- AST: official repo, default config; "not able to fully reproduce": paper 99.2 % vs re-run 98.9 % on
  MVTec AD (five runs).
- DSR: official repo, default config; re-run 98.1 % vs reported 98.2 % on MVTec AD; 50 epochs on VisA.
- FastFlow: anomalib implementation (not the authors' code), WideResNet-50-2, early stopping disabled,
  fixed 200 steps.
- GCAD: own implementation ("We implement GCAD as described by Bergmann et al."); reports the single
  33x33-patch ensemble member, which halves the latency. Single model 83.9 % on LOCO logical vs 86.0 %
  for the authors' ensemble; LOCO overall unchanged.
- PatchCore: official repo; MVTec AD results reproduced; center crop (76.6 %) disabled. Single model:
  1 % coreset, 224x224, WideResNet-101 (the authors' lowest-latency config). PatchCore-Ens:
  WideResNet-101 + ResNeXt-101 + DenseNet-201, 1 % coreset, 320x320.
- SimpleNet: official repo; reproduced; test-set tuning of training duration disabled; final model
  evaluated.
- S-T: own implementation of multi-scale S-T, default hyperparameters; matches MVTec AD IJCV numbers.

## 8. Results worth remembering
Protocol for all rows unless noted: one model per scenario, full-shot, normal-only training,
EfficientAD at 256x256 with PDN padding disabled, EfficientAD mean of 5 runs; baselines re-run by the
EfficientAD authors (configs in Sec. 7). Speed on RTX A6000.

Table 1 (tab:main, p. 7). AU-ROC / AU-PRO = average of the MVTec AD, VisA, MVTec LOCO means.
Latency batch 1, throughput batch 16.

| Method | Detect. AU-ROC | Segment. AU-PRO | Latency [ms] | Throughput [img/s] |
|---|---|---|---|---|
| GCAD | 85.4 | 88.0 | 11 | 121 |
| SimpleNet | 87.9 | 74.4 | 12 | 194 |
| S-T | 88.4 | 89.7 | 75 | 16 |
| FastFlow | 90.0 | 86.5 | 17 | 120 |
| DSR | 90.8 | 78.6 | 17 | 104 |
| PatchCore | 91.1 | 80.9 | 32 | 76 |
| PatchCore-Ens | 92.1 | 80.7 | 148 | 13 |
| AST | 92.4 | 77.2 | 53 | 41 |
| EfficientAD-S | 95.4 (+-0.06) | 92.5 (+-0.05) | 2.2 (+-0.01) | 614 (+-2) |
| EfficientAD-M | 96.0 (+-0.09) | 93.3 (+-0.04) | 4.5 (+-0.01) | 269 (+-1) |

Table 2 (tab:per_dataset, p. 7) = Table 10 (p. 19). Mean image AU-ROC per collection.

| Method | MVTec AD | LOCO | VisA | Mean | LOCO logical | LOCO structural |
|---|---|---|---|---|---|---|
| GCAD | 89.1 | 83.3 | 83.7 | 85.4 | 83.9 | 82.7 |
| SimpleNet | 98.2 | 77.6 | 87.9 | 87.9 | 71.5 | 83.7 |
| S-T | 93.2 | 77.4 | 94.6 | 88.4 | 66.5 | 88.3 |
| FastFlow | 96.9 | 79.2 | 93.9 | 90.0 | 75.5 | 82.9 |
| DSR | 98.1 | 82.6 | 91.8 | 90.8 | 75.0 | 90.2 |
| PatchCore | 98.7 | 80.3 | 94.3 | 91.1 | 75.8 | 84.8 |
| PatchCore-Ens | 99.3 | 79.4 | 97.7 | 92.1 | 71.0 | 87.7 |
| AST | 98.9 | 83.4 | 94.9 | 92.4 | 79.7 | 87.1 |
| EfficientAD-S | 98.8 | 90.0 | 97.5 | 95.4 | 85.8 | 94.1 |
| EfficientAD-M | 99.1 | 90.7 | 98.1 | 96.0 | 86.8 | 94.7 |

Caption: "Performing method development solely on MVTec AD (MAD) becomes prone to overfitting design
choices to the few remaining misclassified test images." PatchCore-Ens is best on MVTec AD (99.3).

Table 12 (tab:localization_pro_03, p. 20). AU-PRO up to FPR 30 % per collection (LOCO = AU-sPRO).

| Method | MVTec AD | VisA | LOCO struct. | LOCO logic. | LOCO mean | Overall |
|---|---|---|---|---|---|---|
| GCAD | 91.0 | 83.7 | 89.5 | 89.4 | 89.5 | 88.0 |
| SimpleNet | 89.6 | 68.9 | 60.6 | 68.6 | 64.6 | 74.4 |
| S-T | 92.4 | 93.0 | 90.8 | 76.4 | 83.6 | 89.7 |
| FastFlow | 92.5 | 86.8 | 84.2 | 76.5 | 80.3 | 86.5 |
| DSR | 90.8 | 68.1 | 81.3 | 72.3 | 76.8 | 78.6 |
| PatchCore | 92.7 | 79.7 | 64.3 | 76.6 | 70.4 | 80.9 |
| PatchCore-Ens | 95.6 | 79.3 | 62.0 | 72.6 | 67.3 | 80.7 |
| AST | 81.2 | 81.5 | 75.4 | 62.6 | 69.0 | 77.2 |
| EfficientAD-S | 93.1 | 93.1 | 92.6 | 90.1 | 91.3 | 92.5 |
| EfficientAD-M | 93.5 | 94.0 | 93.7 | 91.3 | 92.5 | 93.3 |

Pixel AU-ROC: the paper prints it only up to FPR 5 % (Table 14, p. 20): EfficientAD-S MVTec AD 79.7,
VisA 86.3; EfficientAD-M MVTec AD 79.4, VisA 86.9. Full pixel AU-ROC (FPR 100 %) exists only in the
external `per_scenario_results.json`, which is not in `sources/`.

Table 3 (tab:hyperparams, p. 7). Mean AU-ROC of EfficientAD-M (3-collection average). Bold = default.
- a (for q_a): 0.5 -> 95.9, 0.8 -> 95.9, **0.9** -> 96.0, 0.95 -> 95.9, 0.98 -> 95.9, 0.99 -> 95.8.
- b (for q_b): 0.95 -> 95.8, 0.98 -> 95.9, 0.99 -> 96.0, **0.995** -> 96.0, 0.998 -> 95.9, 0.999 -> 95.9.
- p_hard: 0 -> 94.9, 0.9 -> 94.9, 0.99 -> 95.7, **0.999** -> 96.0, 0.9999 -> 95.8, 0.99999 -> 95.7.

Table 4 (cumulative ablation, p. 8), 3-collection mean AU-ROC, latency ms: PDN 93.2 (2.2) -> with map
normalization 94.0 (+0.8) -> with hard feature loss 95.0 (+1.0) -> with pretraining penalty 95.4
(+0.4); EfficientAD-S 95.4 (2.2), EfficientAD-M 96.0 (+0.6, 4.5 ms).
Table 5 (isolated ablation, p. 8), from EfficientAD-S 95.4: without map normalization 94.7 (-0.7);
without hard feature loss 94.7 (-0.7); without pretraining penalty 95.0 (-0.4). The ablation baseline
for "without quantile normalization" is a Gaussian normalization (mean 0, variance 1 on validation).
Text: the +1.0 from the hard loss "alone is greater than or equal to each of the improvement margins
between the consecutive rows of FastFlow, DSR, PatchCore, PatchCore-Ens, and AST in Table 1".

Table 16 (tab:efficiency, p. 21), RTX A6000. Params [x10^6], FLOPs [x10^9], GPU memory [MB] (peak
reserved). PatchCore: feature extraction + kNN shown separately.

| Method | Params | FLOPs | GPU memory |
|---|---|---|---|
| GCAD | 65 | 416 | 555 |
| SimpleNet | 73 | 38 | 508 |
| S-T | 26 | 4468 | 1077 |
| FastFlow | 92 | 85 | 404 |
| DSR | 40 | 267 | 314 |
| PatchCore | 83 + 3 | 41 + kNN | 637 + kNN |
| PatchCore-Ens | 150 + 8 | 159 + kNN | 1335 + kNN |
| AST | 154 | 199 | 618 |
| EfficientAD-S | 8 | 76 | 100 |
| EfficientAD-M | 21 | 235 | 161 |

Table 17 (tab:latency_per_gpu, p. 23): the exact values behind Fig. 6. Latency [ms], batch 1. These
are printed as text, so they are not plot readings.

| Method | RTX A6000 | RTX A5000 | Tesla V100 | RTX 3080 | RTX 2080 Ti |
|---|---|---|---|---|---|
| EfficientAD-S | 2.2 | 2.5 | 3.9 | 3.8 | 4.5 |
| EfficientAD-M | 4.5 | 5.3 | 6.3 | 7.0 | 7.6 |
| GCAD | 10.7 | 11.7 | 12.9 | 13.7 | 18.0 |
| SimpleNet | 12.0 | 13.3 | 19.2 | 18.1 | 21.9 |
| FastFlow | 16.5 | 17.1 | 26.1 | 27.5 | 31.0 |
| DSR | 17.2 | 18.0 | 24.8 | 24.6 | 34.5 |
| PatchCore | 32.0 | 31.5 | 47.1 | 41.1 | 53.2 |
| AST | 53.1 | 53.4 | 75.6 | 82.3 | 87.1 |
| S-T | 74.7 | 81.0 | 82.2 | 99.6 | 121.7 |
| PatchCore-Ens | 147.6 | 145.0 | 229.2 | 189.0 | 216.9 |

- Fig. 6 caption: "The ranking of methods is the same on each GPU, except for two cases in which DSR is
  slightly faster than FastFlow" (V100 and RTX 3080 in Table 17). Fig. 6 is a log-axis bar chart
  (10^0 to 10^2 ms).
- Table 1 prints rounded latencies (11, 12, 75, 17, 17, 32, 148, 53); Table 17 prints one decimal for
  the same A6000 runs. Use one table per figure; do not mix.
- Conclusion (p. 8): vs AST, EfficientAD-S "reduces the latency by a factor of 24 and increases the
  throughput by a factor of 15".
- Line 406: "With early stopping enabled, EfficientAD itself achieves an image-level detection AU-ROC of
  99.8 % on MVTec AD." Not the reported protocol (test-set tuning); do not put it on a scoreboard.
- Headline for the scoreboard (MVTec AD / VisA): image AU-ROC S 98.8 / 97.5, M 99.1 / 98.1 (Table 2);
  AU-PRO S 93.1 / 93.1, M 93.5 / 94.0 (Table 12).

MVTec LOCO as stated in this paper
- Five scenarios (line 151), following MVTec AD's design; anomalies "empirically more challenging than
  those of MVTec AD".
- Contains structural anomalies ("such as stains or scratches") and logical anomalies: "violations of
  logical constraints, for example, a wrong ordering or a wrong combination of normal objects" (lines
  153-154). Line 334: "missing, misplaced, or surplus objects or the violation of geometrical
  constraints, for example, the length of a screw."
- Localization metric: AU-sPRO, "a generalization of the AU-PRO metric for evaluating the localization
  of logical anomalies" (line 419).
- The paper never lists the five scenario names. Four appear incidentally: splicing connectors (Fig. 5
  caption; App. F), breakfast box, "banana juice bottle", "box of pushpins" (App. F, p. 24). The fifth
  is visible in Fig. 9 (a bag of screws, nuts and washers) but is not named in the text; naming it
  "screw bag" here would be our reading.
- Fig. 5 example: normal = one horizontal cable between two splicing connectors at any height; a
  structural anomaly = a small metal washer at the cable end (local map fires); a logical anomaly = a
  second cable (global map fires, because the patch-limited student cannot see the extra cable while
  the AE fails to reconstruct it).

## 9. Limitations the paper states
- Fine-grained logical anomalies stay hard, "for example a screw that is two millimeters too long";
  practitioners would need traditional metrology (p. 8).
- Unlike kNN methods, it needs per-scenario training (20 minutes), "especially for the autoencoder to
  learn the logical constraints" (p. 8).
- Qualitative (App. F): response strength on some logical anomalies (breakfast box, pushpins) "leaves
  room for improvement".
- Later fixes: none stated in this source.

## 10. Outline feedback
1. ch-efficientad, teacher: the distillation backbone is WideResNet-101, using PatchCore's final
   384-d postprocessed features at 512x512 input, distilled on ImageNet (60,000 steps, batch 16). The
   teacher is trained once and frozen; per product only S and A train. Make the backward pass
   three stages, not one.
2. ch-efficientad, viz: Table 3's p_hard row gives sourced readouts for the quantile slider (0 -> 94.9
   ... 0.999 -> 96.0). The 0.999 default means about 0.1 % of the 1,572,864 elements train per image (our
   reading). Fig. 4 (loss masks on screws) is a ready model for the slider's mask view.
3. ch-efficientad, shapes: the stated architecture gives 384x64x64, but every reported number uses
   padding disabled, and the output size then is not stated (56x56 or 58x58, our reading). Show
   64x64 as "with padding, as in Tables 6-7" and add a footnote.
4. ch-efficientad, branches: the paper compares against AST (2210.07829, architectural asymmetry),
   which is the idea EfficientAD replaces with loss-induced asymmetry. Add AST as a branch. STPM and
   Reverse Distillation are not compared in this paper.
5. ch-speed: the desktop-GPU accuracy axis in Table 1 is NOT MVTec AD image AUROC. It is the mean of
   three collection means (MVTec AD, VisA, MVTec LOCO), with LOCO logical anomalies at about 1/6
   weight. For an "image AUROC" axis that matches the Jetson papers, use the MVTec AD column of
   Table 2. Label the axis either way.
6. ch-speed: all baseline latencies are the EfficientAD authors' re-runs with changed configs
   (PatchCore 224px and no center crop, GCAD single member, FastFlow via anomalib). PatchCore's
   MVTec AD 98.7 here is a re-run number, not PatchCore's own. Keep the source label per point.
7. ch-speed: Table 17 gives five desktop GPUs (A6000, A5000, V100, 3080, 2080 Ti) as text. A
   desktop-GPU sub-toggle needs no plot reading. Timing includes the CPU<->GPU transfers, uses float16,
   batch 1 for latency, batch 16 for throughput.
8. ch-summary / ch-benchmark: EfficientAD prints no full-range pixel AUROC (only FPR <= 5 %, Table 14).
   Leave its pixel-AUROC cell empty or mark it; do not use Table 14.
9. ch-benchmark: the AU-PRO FPR limit of 30 % is confirmed here (App. D.2, with the reason: segmenting
   more than 30 % of defect-free pixels is "of limited use"). LOCO's five category names CANNOT come
   from this source (only four are named); fetch the LOCO IJCV paper or MVTec AD 2 for them.
