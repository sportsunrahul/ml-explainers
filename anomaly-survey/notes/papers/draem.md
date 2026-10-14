# DRAEM: A discriminatively trained reconstruction embedding for surface anomaly detection

## 1. Identity
- Title: "DRAEM -- A discriminatively trained reconstruction embedding for surface anomaly detection" (LaTeX title uses `DR{\AE}M`).
- arXiv 2108.07610, version read: v2 (PDF stamp "arXiv:2108.07610v2 [cs.CV] 27 Sep 2021"; LaTeX src `sources/draem/src/iccv.tex`).
- Dates (abs.html): v1 2021-08-17, v2 2021-09-27.
- Authors: Vitjan Zavrtanik (first), Matej Kristan, Danijel Skocaj (last); University of Ljubljana.
- Venue: ICCV 2021 (abs.html comment: "Accepted to ICCV2021").
- Repo: github.com/VitjanZ/DRAEM (first author's account). The link is in a commented-out line of the LaTeX abstract, so it is NOT in the PDF. Code-only values below are from commit 2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef (2021-10-12), files `model_unet.py`, `train_DRAEM.py`, `test_DRAEM.py`, `loss.py`, `data_loader.py`, `README.md`. Read only, not executed.
- PDF: 10 pages, no supplementary in the source.

## 2. Core problem
Localize surface anomalies (small regions close to normal appearance) with no real anomalous training images. Reconstruction methods reconstruct subtle anomalies too well; a plain segmentation network trained on synthetic anomalies over-fits to the synthetic look.

## 3. Prior methods and why that hurt (per the paper, Sec. 1, 2)
- Autoencoders / GANs: anomalies "not substantially different from normal appearance ... are often well reconstructed" (Fig. 2 top-left); need hand-crafted post-processing (e.g. SSIM difference) that is not optimized for detection.
- Generative methods learn only from anomaly-free data and "are not explicitly optimized for discriminative anomaly detection".
- Discriminative segmentation trained on synthetic anomalies: "leads to over-fitting to synthetic appearances and results in a learned decision boundary that generalizes poorly to real anomalies" (Fig. 2 top-right).
- Gaussian-on-pretrained-features (MAD = Rippel et al., PaDiM): needs a unimodal distribution of normal features, "problematic on diverse datasets".

## 4. The one key idea
Make a fake anomaly, train a network to repair it (reconstruct the clean image), and train a second network to segment the fake by looking at the input and the repair side by side. The segmenter learns "how far is this from its own repaired version", a distance that transfers to real defects, rather than "what does a fake look like".

## 5. Mechanism, step by step
1. Simulate (Sec. 3.3, Fig. 4): Perlin noise P -> binarize with a threshold "sampled uniformly at random" -> mask M_a. Sample a texture image A from an unrelated dataset (DTD); apply 3 random augmentations from {posterize, sharpness, solarize, equalize, brightness change, color change, auto-contrast}. Blend inside the mask with opacity beta in [0.1, 1.0] (Eq. 4). Output triplet (I, I_a, M_a).
2. Reconstruct (Sec. 3.1): encoder-decoder maps I_a -> I_r, trained to output the clean I. Loss L_rec = lambda L_SSIM + l2 (Eq. 1-2).
3. Discriminate (Sec. 3.2): U-Net-like network takes I_c = channel-wise concat of I_r and the input image, outputs a map M_o of the same size as I. Loss: focal loss L_seg against M_a.
4. Train both jointly, end to end, total loss Eq. 3. "an additional training signal is acquired from the downstream discriminative network" for the reconstructive subnet (Sec. 3.1).
5. Test (Sec. 3.4): pixel map = M_o. Image score eta = max of M_o after a mean filter of size s_f x s_f (Eq. 5). A trained classification head was tried and "did not observe improvements".
6. Training data augmentation on anomaly-free images: rotation in (-45, 45) deg (Sec. 4.1).

### Which weights get a gradient
- Reconstructive subnet: from L_rec (Eq. 2) AND from L_seg through I_r (Sec. 3.1; code: `gray_rec` is not detached before the concat in `train_DRAEM.py` L66-69).
- Discriminative subnet: from L_seg only.
- Nothing is pretrained or frozen; both subnets start from random init (code: `weights_init`, normal(0, 0.02), `train_DRAEM.py` L14-20). No ImageNet backbone anywhere.

### Tensor walkthrough (all shapes code-only, `model_unet.py` + `train_DRAEM.py` L35-53; input resized to 256 x 256)
Block = 2 x (conv 3x3 pad 1 + BatchNorm + ReLU). Down = MaxPool 2. Up = bilinear x2 + conv 3x3 + BN + ReLU.

Reconstructive subnet (`ReconstructiveSubNetwork`, in 3, out 3, base_width 128), no skip connections:
| Stage | Shape (C x H x W) |
|---|---|
| input I_a | 3 x 256 x 256 |
| enc block1 | 128 x 256 x 256 |
| enc block2 (after pool) | 256 x 128 x 128 |
| enc block3 | 512 x 64 x 64 |
| enc block4 | 1024 x 32 x 32 |
| enc block5 = bottleneck | 1024 x 16 x 16 |
| dec up1 -> db1 | 1024 x 32 x 32 -> 512 x 32 x 32 |
| dec up2 -> db2 | 512 x 64 x 64 -> 256 x 64 x 64 |
| dec up3 -> db3 | 256 x 128 x 128 -> 128 x 128 x 128 |
| dec up4 -> db4 | 128 x 256 x 256 -> 128 x 256 x 256 |
| fin_out conv 3x3 = I_r | 3 x 256 x 256 |

Discriminative subnet (`DiscriminativeSubNetwork`, in 6, out 2, base_channels 64), U-Net skips:
| Stage | Shape |
|---|---|
| input concat(I_r, I_a) | 6 x 256 x 256 |
| enc b1 .. b6 | 64 x 256^2, 128 x 128^2, 256 x 64^2, 512 x 32^2, 512 x 16^2, 512 x 8^2 |
| up_b + cat b5 -> db_b | 512 x 16 x 16 + 512 -> 1024 -> 512 x 16 x 16 |
| up1 + cat b4 -> db1 | 256 + 512 = 768 -> 256 x 32 x 32 |
| up2 + cat b3 -> db2 | 128 + 256 = 384 -> 128 x 64 x 64 |
| up3 + cat b2 -> db3 | 64 + 128 = 192 -> 64 x 128 x 128 |
| up4 + cat b1 -> db4 | 64 + 64 = 128 -> 64 x 256 x 256 |
| fin_out conv 3x3 | 2 x 256 x 256 logits; softmax; channel 1 = M_o |
| image score | avg_pool 21 x 21, stride 1 -> max -> eta (scalar) |

Parameter counts (our reading: arithmetic over `model_unet.py`, conv weights + biases + BN affine; not in the paper): reconstructive 69,048,579 (69.0M); discriminative 28,374,146 (28.4M); total 97,422,725 (97.4M).

## 6. Equations
Macros: none needed beyond standard LaTeX. Paper text writes the focal loss as `L_{seg}` (Eq. 3) and the figure labels it `L_{focal}` (Fig. 3).

Eq. (1), SSIM loss (Sec. 3.1, p3):
```tex
L_{SSIM}(I, I_r) = \frac{1}{N_p} \sum_{i=1}^H \sum_{j=1}^W 1-SSIM\big(I,I_r\big)_{(i,j)},
```
Paper: "$H$ and $W$ are the height and width of image $I$ ... $N_p$ is equal to the number of pixels in $I$. $I_r$ is the reconstructed image output by the network. $SSIM(I, I_r)_{(i,j)}$ is the SSIM value for patches of $I$ and $I_r$, centered at image coordinates $(i,j)$." Gloss: average over pixels of (1 - local structural similarity) between the clean image and the reconstruction; used because l2 "assumes an independence between neighboring pixels".

Eq. (2), reconstruction loss (Sec. 3.1, label `eq:lrec`):
```tex
L_{rec}(I, I_r) = \lambda L_{SSIM}(I,I_r) + l_{2}(I, I_r),
```
Paper: "where $\lambda$ is a loss balancing hyper-parameter." The value of lambda is NOT given in the paper. Code: `loss = l2_loss + ssim_loss + segment_loss` (`train_DRAEM.py` L76), i.e. lambda = 1 (code-only). Gloss: target is the clean I, input to the network was the corrupted I_a.

Eq. (3), total loss (Sec. 3.2):
```tex
L(I,I_r, M_a, M) = L_{rec}(I,I_r) + L_{seg}(M_a, M),
```
Paper: "where $M_a$ and $M$ are the ground truth and the output anomaly segmentation masks, respectively." L_seg is "Focal Loss [Lin et al. 2017]" applied "to increase robustness towards accurate segmentation of hard examples". The focal loss formula itself is NOT written in the paper (a draft form is commented out in the LaTeX). Code: `FocalLoss()` with gamma = 2, alpha = None (uniform), smooth 1e-5, on the 2-channel softmax output (`loss.py` L21, `train_DRAEM.py` L51, L70, L75). Note the symbol clash: Eq. 3 calls the output mask M; Sec. 3.2 and Eq. 5 call it M_o.

Eq. (4), simulated anomalous image (Sec. 3.3):
```tex
I_a = \overline{M}_a \odot I + (1-\beta) (M_a \odot I) + \beta (M_a \odot A),
```
Paper: "where $\overline{M}_a$ is the inverse of $M_a$, $\odot$ is the element-wise multiplication operation and $\beta$ is the opacity parameter in blending. This parameter is sampled uniformly from an interval, i.e., $\beta \in [0.1,1.0]$." Gloss: outside the mask keep I; inside the mask mix (1 - beta) of the original image with beta of the augmented texture A. beta near 1 = opaque foreign texture; small beta = faint, harder anomaly.

Eq. (5), image-level score (Sec. 3.4, label `eq:G`):
```tex
\eta = max\big(M_o \ast  f_{s_f \times s_f}\big)\hspace{0.2cm},
```
Paper: "where $f_{s_f \times s_f}$ is a mean filter of size $s_f \times s_f$ and $\ast$ is the convolution operator." s_f is NOT given in the paper. Code: `avg_pool2d(..., 21, stride=1, padding=21 // 2)` then `np.max` (`test_DRAEM.py` L105-107), i.e. s_f = 21 (code-only). Fig. 3 draws this as "Local avg. pool" then "Global max pool".

## 7. Config / setup table (method lens)
| Item | Paper value (location) | Code-only value (commit 2dbf673) |
|---|---|---|
| Input resolution | not stated | 256 x 256 (`train_DRAEM.py` L53, `test_DRAEM.py` L46) |
| Reconstructive subnet | encoder-decoder (Sec. 3.1, Fig. 3) | base width 128, 5 enc blocks, 4 up stages, no skips; 3 -> 3 ch |
| Discriminative subnet | "U-Net-like", input = concat of I_r and input image (Sec. 3.2) | base 64, 6 enc blocks, 5 up stages with skips; 6 -> 2 ch |
| Pretrained weights | none mentioned | none; normal(0, 0.02) init |
| Loss | L_rec + L_seg, L_rec = lambda L_SSIM + l2 (Eq. 2-3) | lambda = 1; SSIM window 11, Gaussian sigma 1.5 (`loss.py` L93, L150) |
| Segmentation loss | focal loss (Sec. 3.2) | gamma 2, alpha uniform |
| Optimizer | not stated | Adam, both subnets, same lr (`train_DRAEM.py` L43-45) |
| Learning rate | 1e-4 (Sec. 4.1) | 1e-4 via CLI (README example) |
| Schedule | x0.1 after 400 and 600 epochs (Sec. 4.1) | MultiStepLR at 0.8 and 0.9 of epochs, gamma 0.2 (L47) -- differs from paper |
| Epochs | 700 (Sec. 4.1) | 700 in README training example; released checkpoints named "..._0.0001_800_bs8" (README) |
| Batch | not stated | 8 (README example) |
| Anomaly source | DTD (Sec. 4.1) | DTD `images/*/*.jpg` |
| Perlin mask | threshold "sampled uniformly at random" (Sec. 3.3); range not stated | fixed threshold 0.5; Perlin scale 2^k per axis, k in {0..5}; noise rotated (-90, 90) deg (`data_loader.py` L109-121; rotation object L92) -- differs from paper |
| beta | U[0.1, 1.0], weight on texture A (Eq. 4) | beta = U[0, 0.8) weights the ORIGINAL image; texture weight = 1 - beta in (0.2, 1.0] (`data_loader.py` L126-129) -- differs from paper |
| Texture augmentations | 3 of 7 listed functions (Sec. 3.3) | 3 of 10 imgaug augmenters (adds gamma contrast, invert, affine rotate) (`data_loader.py` L80-89) |
| Anomaly-free image aug | rotation (-45, 45) (Sec. 4.1) | rotation (-90, 90) with probability 0.3 (`data_loader.py` L92, L148-150) |
| Fraction of training samples with a fake anomaly | not stated | 0.5 (`no_anomaly > 0.5` returns clean image, `data_loader.py` L131-134) |
| Image score filter s_f | not stated | 21 |
| Protocol | one model per category, normal-only (Sec. 4.1) | one model per object name |
| Latency / hardware | not reported | -- |

## 8. Results worth remembering
Protocol for MVTec AD rows: one model per category, full-shot, normal-only (synthetic anomalies from DTD); resolution not stated in the paper (256 x 256 in code). No PRO, no VisA, no latency.

Table 1 (p5), image AUROC, row "avg":
- DRAEM 98.0. Best in 9 of 15 classes (text).
- Re-reported baselines (columns by citation): GANomaly 78.2, DAGAN 87.3, US (Uninformed Students) 87.7, RIAD 91.7, MAD (Rippel et al., ICPR 2020; Gaussian on pretrained features) 94.4, PaDiM 95.5. Not stated whether re-run or copied.
- Text: "surpasses the previous best state-of-the-art approach by $2.5$ percentage points" (98.0 vs PaDiM 95.5).

Table 2 (p6), pixel AUROC / pixel AP, row "avg":
- DRAEM 97.3 / 68.4.
- US 93.9 / 45.5; RIAD 94.2 / 48.2; PaDiM 97.4 / 55.0. PaDiM has the better pixel AUROC (97.4 bold) in DRAEM's own table; DRAEM wins AP.
- Text (Sec. 4.1): "surpasses the state-of-the-art by $13.4$ percentage points in terms of AP"; Conclusion: "by $13.5$ AP points on the localization task". Printed both ways; 68.4 - 55.0 = 13.4.
- Text: better AP in 11 of 15 classes.
- The paper argues AP over pixel AUROC: false positive rate "is dominated by the a-priori very high number of non-anomalous pixels".

Table 3 (p7), ablation, Det. (image AUROC) and Loc. (pixel AUROC / AP):
| Variant | Det. | Loc. |
|---|---|---|
| Disc. (discriminative subnet only) | 93.9 | 92.7 / 62.5 |
| Recon.-AE (recon subnet + SSIM map) | 83.9 | 89.7 / 47.5 |
| Recon.-AE_MSGMS | 90.7 | 93.4 / 50.9 |
| Bozic et al. (supervised net, trained on simulated anomalies) | 92.8 | 93.9 / 60.7 |
| DRAEM_ImageNet (ImageNet as texture source) | 97.9 | 97.0 / 67.9 |
| DRAEM_color (homogeneous random-color anomalies) | 96.2 | 92.6 / 56.5 |
| DRAEM_rect (rectangles instead of Perlin) | 96.9 | 96.8 / 65.1 |
| DRAEM_no_aug (no texture aug, no beta) | 97.4 | 94.5 / 64.3 |
| DRAEM_img_aug (texture aug only) | 97.4 | 95.0 / 64.5 |
| DRAEM_beta (beta only) | 97.9 | 97.1 / 68.4 |
| DRAEM (full) | 98.0 | 97.3 / 68.4 |
Key reads: removing the reconstructive subnet costs 4.1 image points (98.0 -> 93.9), attributed to "overfitting of the discriminative sub-network to the simulated anomalies"; unrealistic anomalies (color, rectangles) still beat the prior state of the art.

Fig. 9 (p8): with augmentation, "as low as 10 texture source images" suffice (caption); text says "remarkably small number of texture images (less than 10)".

Table 4 (p8), DAGM (10 textured classes; image classification only), DRAEM: AUROC 99.0, TPR 96.5, TNR 99.4, CA 98.5. Unsupervised baselines: RIAD 78.6, US 72.5, MAD 82.4, PaDiM 95.0 (AUROC). Supervised: Bozic et al. 100 on all four; Racki et al. AUROC 99.6; Lin et al. 99.0; CADN CA 89.1 (weakly supervised).

Code-only (README, not the paper): released checkpoints give "98.1 image-level ROC AUC, 97.5 pixel-wise ROC AUC and a 68.9 pixel-wise AP". Do not mix with the paper's 98.0 / 97.3 / 68.4.

## 9. Limitations the paper states
- Missing parts are hard: "The absence of a part of the object is especially difficult to detect" (transistor cut lead, Sec. 4.1).
- Some errors come from ambiguous ground truth (pill with yellow dots, Fig. 6), which also lowers AP.
- Purely discriminative training over-fits to synthetic appearance (the motivation and the "Disc." ablation).
- No later fix named in this source.

## 10. Outline feedback (ch-synthetic)
- viz-synth DRAEM tab: "a Perlin threshold slider" is fine as a teaching control, but say the paper samples the threshold uniformly at random (range not stated) and the official code uses a fixed 0.5 with a random Perlin scale. The beta slider range per the paper is [0.1, 1.0] with beta weighting the foreign texture (Eq. 4); the code's effective texture weight is (0.2, 1.0]. Use the paper's form and cite Eq. 4.
- "Equations: DRAEM's losses (reconstruction L2 + SSIM; focal loss on the mask)": Eq. 1-3 are DRAEM's; the focal loss formula is NOT in the DRAEM paper. If shown, cite Lin et al. 2017 for the formula and DRAEM code for gamma = 2. lambda is not stated in the paper (code uses 1).
- Add Eq. 5 (mean filter then max) for the image score; it ties to the pixel map in one line.
- Architecture walkthrough shapes exist only in code (256 x 256; recon base 128 with a 1024 x 16 x 16 bottleneck and no skips; discriminative U-Net base 64, 6 -> 2 channels). Mark every shape code-only. Backward: both subnets get gradients; L_seg also flows into the reconstructive subnet; no pretrained backbone, nothing frozen.
- The "risk: the model learns your fakes" sentence has direct evidence here: the discriminative-only ablation drops to 93.9 image / 62.5 AP (Table 3). Also worth one line: color blobs (96.2) and rectangles (96.9) still work, so realism matters less than the reconstruct-then-compare design.
- ch-benchmark link: DRAEM itself argues pixel AUROC hides false positives and reports AP; its own table shows PaDiM 97.4 vs DRAEM 97.3 pixel AUROC but 55.0 vs 68.4 AP. Good concrete example for the "pixel AUROC is swamped" intuition.
- DRAEM's PaDiM image AUROC (95.5) differs from FastFlow's PaDiM entry (97.9); cite per paper.
- No PRO, no VisA, no latency in the paper.
