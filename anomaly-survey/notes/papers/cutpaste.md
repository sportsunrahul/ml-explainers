# CutPaste: Self-Supervised Learning for Anomaly Detection and Localization

## 1. Identity
- Title: "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization".
- arXiv 2104.04015, version read: v1 (only version; PDF stamp "arXiv:2104.04015v1 [cs.CV] 8 Apr 2021"; LaTeX src `sources/cutpaste/src/cvpr.tex`).
- Date (abs.html): v1 2021-04-08.
- Authors: Chun-Liang Li (first, equal contribution), Kihyuk Sohn (equal contribution), Jinsung Yoon, Tomas Pfister (last); Google Cloud AI Research.
- Venue: CVPR 2021 (abs.html comment: "Published at CVPR 2021").
- Repo: none given in the paper or abs page. Implementation in TensorFlow + scikit-learn for GDE (App. A.1).
- PDF: 28 pages; main text p1-8, appendix p12-28.

## 2. Core problem
Detect (and localize) local, fine-grained defects in high-resolution images using normal data only. Self-supervised methods built for semantic outliers (rotation prediction, contrastive learning) do not transfer to defect detection.

## 3. Prior methods and why that hurt (per the paper, Sec. 1, 2.1)
- Reconstruction / generative models: pixel-wise reconstruction error or density "lacks to capture a high-level semantic information".
- Geometric-transformation proxy tasks (rotation, translation) learn objectness, "but less of regularity (e.g., continuity, repetition)"; rotation prediction gives 73.1 AUC on MVTec AD (Table 1) and fails on unaligned objects like screw.
- Cutout is too easy a proxy task; a network may learn a naive solution (Sec. 2.1, 5.1).

## 4. The one key idea
Train a classifier to tell normal images from images where a patch of the SAME image was cut and pasted elsewhere. This creates a local "irregularity" that roughly approximates a defect. Then throw the classifier head away, keep the learned representation, and fit a Gaussian to normal representations for scoring.

## 5. Mechanism, step by step
Two-stage framework (Sec. 2, following Sohn et al. / CSI):
1. Augment (Sec. 2.1; App. A.2): cut a rectangle from a normal training image, optionally rotate or color-jitter it, paste it at a random location of the same image.
   - CutPaste (large patch): area ratio sampled from (0.02, 0.15) of the image; aspect ratio from (0.3, 1) U (1, 3.3); cut and paste locations chosen so the whole patch is inside the image.
   - CutPaste-Scar: width [2, 16] px, length [10, 25] px, rotated (-45, 45) deg.
   - Before pasting: color jitter (brightness, contrast, saturation, hue, random order), maximum intensity 0.1.
   - Implementation "closely follows that of RandomErasing".
2. Train (Sec. 4, App. A.1): ResNet-18 from scratch + MLP projection head on top of average pooling, followed by the last linear layer. Binary (normal vs CutPaste) via Eq. 1, or 3-way (normal / CutPaste / CutPaste-Scar), which is the best variant. Standard augmentations (translation, color jitter) are applied before CP.
3. Fit the detector (Sec. 2.3, 4): Gaussian density estimator (GDE, Eq. 2) on representations f of normal training images. Fitting = computing mean and covariance; nothing more is trained.
4. Image score: the GDE log-density of the test image's representation (low density = anomalous; our reading: score = negative log-density, i.e. a Mahalanobis distance).
5. Localization option A (Sec. 4.2): GradCAM on the image-level model; pixel AUC 88.3.
6. Localization option B, patch model (Sec. 2.4, Eq. 3; Sec. 4.2; App. A.5): train the same task on random crops c(x). At test, densely extract patch embeddings with stride 4, score each with GDE, upsample to full resolution with "receptive field upsampling via Gaussian smoothing" (a transposed convolution, stride 4, one Gaussian kernel of the patch size).
7. GDE placement for patch model (App. A.5): for aligned objects (bottle, cable, capsule, metal nut, pill, toothbrush, transistor, zipper) one GDE per location; for hazelnut, screw (randomly rotated) and all textures one GDE shared across all locations.

### Which weights get a gradient
- Stage 1: the whole ResNet-18 + MLP head + linear classifier, from scratch, by the cross-entropy of Eq. 1 (or 3-way CE). Signal: "is this image normal or CutPaste-augmented?"
- Stage 2: no gradient. GDE mean and covariance are statistics of normal features.
- Transfer variant: ImageNet-pretrained EfficientNet-B4; head trained 10 epochs with backbone frozen, then all layers fine-tuned 64 epochs; batchnorm kept frozen (App. A.4).

### Tensor walkthrough (our reading where marked)
| Stage | Shape | Source |
|---|---|---|
| Input image | 3 x 256 x 256 | Sec. 4, App. A.1 |
| ResNet-18 last conv map | 512 x 8 x 8 | our reading (standard ResNet-18 at 256 px) |
| Average pooling -> representation | 512 | patch case states 512-d embeddings (App. A.5) |
| MLP head -> linear | 2 (binary) or 3 (3-way) logits | Sec. 4; head widths not stated |
| GDE | on f(x); mu in R^512, Sigma 512 x 512 for the 512-d case | our reading of Eq. 2 + App. A.5 |
| Patch model, test | 32 x 32 patches, stride 4 -> 57 x 57 x 512 embeddings -> 57 x 57 scores -> 256 x 256 map | App. A.5 |

Which layer f is for GDE: Sec. 4 says GDE is "based on the top pooled features"; the paper does not say whether that is before or after the MLP head (not stated).

## 6. Equations
No custom macros needed.

Eq. (1), CutPaste training objective (Sec. 2.1, label `eq:cutpaste_prediction`):
```tex
\mathcal{L}_{\mathrm{CP}}\,{=}\,\mathbb{E}_{x{\in}\mathcal{X}}\big\{\mathbb{CE}(g(x), 0) + \mathbb{CE}(g(\mathrm{CP}(x)), 1)\big\}
```
Paper: "where $\mathcal{X}$ is the set of normal data, $\mathrm{CP}(\cdot)$ is a CutPaste augmentation and $g$ is a binary classifier parameterized by deep networks. $\mathbb{CE}(\cdot,\cdot)$ refers to a cross-entropy loss." Gloss: for every normal image x, the classifier should output class 0 for x itself and class 1 for its cut-and-pasted version. The 3-way variant adds class 2 for CutPaste-Scar (no separate equation printed).

Eq. (2), Gaussian density estimator (Sec. 2.3, label `eq:gde_logdensity`):
```tex
\log p_{\mathrm{gde}}(x)\,{\propto}\,\big\{{-}\frac{1}{2}(f(x)\,{-}\,\mu)^{\top}\Sigma^{-1}(f(x)\,{-}\,\mu)\big\}
```
Paper: "where $\mu$ and $\Sigma$ are learned from normal training data." Gloss: f(x) = the learned representation of x; mu = mean of normal representations; Sigma = their covariance; the bracket is minus half the squared Mahalanobis distance, so far-from-normal features get low log-density. Footnote: a Gaussian mixture gave no significant gain.

Eq. (3), patch-level objective (Sec. 2.4, label `eq:cutpaste_prediction_patch`):
```tex
\mathbb{E}_{x{\in}\mathcal{X}}\big\{\mathbb{CE}(g(c(x)), 0) + \mathbb{CE}(g(\mathrm{CP}(c(x))), 1)\big\}
```
Paper: "where $c(x)$ crops a patch at random location of $x$." Gloss: the same objective as Eq. 1, applied to random crops, so the representation describes a patch and can be evaluated at every location.

Patch-grid size (App. A.5, inline):
```tex
(\frac{256-32}{4}+1){\times}(\frac{256-32}{4}+1){\times}512\,{=}\,57{\times}57{\times}512
```

## 7. Config / setup table (method lens)
| Item | Value | Location |
|---|---|---|
| Backbone (main) | ResNet-18, from scratch, + MLP projection head + linear | Sec. 4 |
| Backbone (transfer) | EfficientNet-B4, ImageNet-pretrained, fine-tuned | Sec. 4.3, App. A.4 |
| Input | 256 x 256 | Sec. 4, App. A.1 |
| Optimizer | momentum SGD, lr 0.03, momentum 0.9 | App. A.1 |
| Steps | 65k updates; one "epoch" = 256 updates; default 256 epochs | App. A.1, A.3 footnote |
| Batch | 64 (96 for 3-way) | App. A.1 |
| Schedule | single cycle cosine decay | App. A.1 |
| Weight decay | L2 coefficient 0.00003 | App. A.1 |
| CutPaste area ratio | (0.02, 0.15) | App. A.2 |
| CutPaste aspect ratio | (0.3, 1) U (1, 3.3) | App. A.2 |
| CutPaste-Scar | width [2, 16] px, length [10, 25] px, rotation (-45, 45) deg | App. A.2 |
| Patch color jitter | max intensity 0.1 | App. A.2 |
| Detector | GDE (Eq. 2), scikit-learn | Sec. 2.3, App. A.1 |
| Seeds | 5 runs, mean and standard error | Sec. 4.1 |
| EfficientNet fine-tune | batch 24; head 10 epochs lr 0.03 (backbone frozen); all layers 64 epochs lr 0.0001; BN frozen | App. A.4 |
| Patch model, train | "64 x 64 patches from 256 x 256 image" | Sec. 4.2 |
| Patch model, test | "32 x 32 patches with the stride of 4" -> 57 x 57 x 512 | App. A.5 |
| Score map upsampling | transposed conv, stride 4, one Gaussian kernel 32 x 32 | App. A.5 |
| Protocol | one-class, one model per category, normal-only | Sec. 4 |

In-paper discrepancy: Sec. 4.2 says 64 x 64 training patches; App. A.5 says 32 x 32 patches at extraction (and the 57 x 57 arithmetic uses 32). Both printed; not reconciled here.

## 8. Results worth remembering
All MVTec AD, one model per category, normal-only, full-shot, 256 x 256. No PRO, no VisA, no latency reported.

Table 1 (p6), image AUROC, "average" row (all 15 categories), mean +- standard error over 5 seeds, ResNet-18 from scratch:
- Rotation 73.1; Cutout 71.3; Scar 85.0; CutPaste 90.9; CutPaste (scar) 93.5; CutPaste (3-way) 95.2 (+-0.6); Ensemble of 5 CutPaste (3-way) 96.1.
- CutPaste (3-way) texture average 97.0, object average 94.3.
- Re-reported baselines (copied from their papers; DOCC fine-tunes pretrained VGG16, U-Student distills pretrained ResNet18): DOCC 87.9, U-Student 92.5, P-SVDD 92.1.
- Text: "outperforms existing works ... by at least 3.1 AUC" (vs P-SVDD 92.1).

Table 2 (p7), pixel AUROC, patch-level CutPaste (3-way), ResNet-18: average 96.0 (+-0.1); texture 96.3; object 95.8. Baselines: FCDD 92, P-SVDD 95.7. DistAug contrastive: 90.4 (text, Sec. 4.2).
GradCAM localization from the image-level model: pixel AUC 88.3 (Sec. 4.2 text).

Table 3 (p7), EfficientNet-B4 ImageNet-pretrained, image AUROC average:
- Pool feature: pretrained 94.5, fine-tuned with CutPaste (3-way) 96.6 (+-0.1). This 96.6 is the abstract's "new state-of-the-art 96.6 AUC".
- Level-7 feature: pretrained 96.8, fine-tuned 97.1 (+-0.0).
- Example: pill 81.9 -> 91.3 (pool).

Table 4 (p7), Cutout variants, all-category average: Cutout 71.3, Cutout (Mean) 73.9, Cutout (Color) 76.1, Confetti 77.8, CutPaste 90.9.

Table 5 (p8), binary vs 3-way, all average: CutPaste 90.9, CutPaste (scar) 93.5, Binary (Union) 94.3, 3-Way 95.2.

Table 6 (p8), synthetic anomalies (shapes pasted on MVTec images): CutPaste average 98.3, Cutout 81.5.

Semantic outliers (Sec. 5.4, CIFAR-10 one-vs-rest): CutPaste 69.4, Cutout 60.2, rotation prediction 91.3. CutPaste is worse than rotation on semantic outliers.

Hyperparameter robustness (Tables 7-8, p13): all-average stays 94.6-95.5 across lr, epochs, jitter and patch size.

t-SNE (Fig. 3, p4): "the CutPaste examples are almost not overlapped with real defect examples", yet the representation separates normal, augmented and real defects. The paper reads this as "(1) CutPaste is still not a perfect simulation of real defects and (2) learning on it to find irregularity generalizes well on unseen anomalies."

Re-reports of CutPaste elsewhere (do not mix): FastFlow Table 2 prints Cut Paste as (97.1, 96.0) = this paper's EfficientNet level-7 image AUC and ResNet-18 patch pixel AUC.

## 9. Limitations the paper states
- CutPaste is "a crude approximation of a real anomaly distribution" (Sec. 1); t-SNE shows no overlap with real defects (Sec. 2.2).
- Failure cases (App. B.1): cable missing components; metal nut flipped components; screw speckle noise in background; transistor dislocated or missing components; tile dyed tiles.
- Holistic image representation is hard to turn into a heat map; needs GradCAM or a separate patch model (Sec. 4.2).
- Poor on semantic outlier detection (CIFAR-10 69.4 vs rotation 91.3).
- No later fix named in this source. (DRAEM, in its own source, argues purely discriminative training on synthetic anomalies over-fits; that is DRAEM's claim, not CutPaste's.)

## 10. Outline feedback (ch-synthetic)
- Two image-level headlines with different protocols: 95.2 (ResNet-18 from scratch, 3-way, Table 1) and 96.6 (ImageNet-pretrained EfficientNet-B4 fine-tuned, pool feature, Table 3). Ensemble 96.1, level-7 97.1. The scoreboard must pick one and label it. FastFlow's table uses 97.1.
- The scoring equation is GDE on the representation f, not on the classifier output g. The "density-estimation scoring" in the outline is right; make clear that the classifier only teaches features.
- Pixel AUROC 96.0 comes from a SEPARATE patch-level model (Eq. 3), with per-location GDE for aligned objects and one shared GDE for textures, hazelnut and screw. That per-location trick is the same idea as PaDiM (one Gaussian per position); worth a cross-link to ch-padim.
- viz-synth CutPaste tab: use the paper's ranges as slider limits: area 2-15% of a 256 x 256 image, aspect 0.3-3.3; Scar 2-16 px wide, 10-25 px long, rotated +-45 deg; jitter 0.1. The paste location is random and the patch comes from the same image.
- The t-SNE result (Fig. 3) is direct support for the chapter's "the model learns your fakes, not real defects" risk, with a twist: the fakes do not look like real defects, yet the features still generalize. Consider quoting it.
- Training objective has a 3-way form with no separate equation; describe it in words.
- No PRO, no VisA, no latency.
