# SPADE: Sub-Image Anomaly Detection with Deep Pyramid Correspondences

## 1. Identity
- Title: "Sub-Image Anomaly Detection with Deep Pyramid Correspondences". Method name SPADE = "Semantic Pyramid Anomaly Detection" (abstract).
- arXiv 2005.02357, version read: v3 (PDF stamp "arXiv:2005.02357v3 [cs.CV] 3 Feb 2021"; LaTeX src matches v3).
- Dates (abs.html): v1 2020-05-05, v2 2020-12-10, v3 2021-02-03.
- Authors: Niv Cohen, Yedid Hoshen (Hebrew University of Jerusalem).
- Venue: not stated in abs.html. The LaTeX uses the LLNCS template and defines `\ECCVSubNumber{2772}` (a submission id, not an acceptance).
- Repo: none given in the paper.
- Sources: `sources/spade/src/main.tex`, `sources/spade/paper.pdf` (17 pages).

## 2. Core problem
Deep kNN on pretrained features detects anomalous whole images very well, but gives no map of where the anomaly is. SPADE wants a pixel-level anomaly map in the normal-only training setting, with no training stage.

## 3. Prior methods and why that hurt (per the paper, Sec. 2 and Sec. 5)
- Reconstruction (autoencoders, VAE, GAN): sensitive to the reconstruction loss chosen; the right functional basis is hard to pick.
- Distribution-based (GMM, KDE): real image data rarely fits simple parametric forms; non-parametric methods need large training sets.
- Self-supervised / learned features (Geom, GANomaly, ITAE, Uninformed Students): need an expensive training stage; self-supervised features underperform ImageNet features (cites DN2, Bergman et al.).
- Naive image alignment to one normal image: fails when normal parts vary, no single normal image matches everywhere, and the pixel difference depends on the loss used (Sec. 3.3).

## 4. The one key idea
Do not align the whole image to one normal image. Instead, find for every pixel of the test image its nearest pixel feature among the pixel features of the K most similar normal images, using a multi-level ImageNet feature pyramid. A pixel with no close correspondence is anomalous.

## 5. Mechanism, step by step
Fit ("initialization", Sec. 3.1): nothing is trained. The backbone stays frozen and no weight receives a gradient. Fitting = running every normal training image through the backbone once and storing (a) its global-pooled feature f_i and (b) its pixel feature pyramid.

Test:
1. Extract the global feature f_y of the test image y (global-pooled last conv layer, 2048-d, Sec. 5 "Optimizing runtime").
2. Image level (stage 1 = DN2): find the K nearest normal images by Euclidean distance on global features; image score d(y) = mean squared distance to them (Eq. 2). K = 50 on MVTec AD, K = 1 on STC.
3. Image is flagged anomalous if d(y) > tau (Sec. 3.2). The paper says segmentation is meant for flagged images; it does not say whether pixel metrics were computed only on flagged images (our reading: not stated).
4. Build a gallery G = all pixel-location features of the K retrieved normal images (Sec. 3.3).
5. Pixel level: for each pixel p, score d(y, p) = mean squared distance from F(y, p) to its kappa nearest gallery features (Eq. 3). kappa = 1 in all experiments, so it is the squared distance to the single nearest gallery feature.
6. Pixel features: "we concatenate features from the output of the last $M$ blocks" (Sec. 3.4); implementation uses "the end of the first block ($56 \times 56$), second block ($28 \times 28$) and third block ($14 \times 14$), all with equal weights" (Sec. 3.5). How the three resolutions are brought to one grid is not stated (our reading: upsample to 56 x 56 and concatenate).
7. Smooth the pixel score map with a Gaussian filter, sigma = 4 (Sec. 3.5). Metrics are computed at 256 x 256 (cv2.INTERAREA (as printed) resize "when needed").

### Tensor walkthrough (input 224 x 224, Wide-ResNet-50-2)
Spatial sizes 56/28/14 and the 2048-d global vector are stated in the paper. Channel widths are NOT stated; the values below are the standard torchvision WRN-50-2 widths (our reading).

| Stage | Shape (C x H x W) | Source |
|---|---|---|
| input (resize 256, crop 224) | 3 x 224 x 224 | Sec. 3.5 |
| block 1 (layer1) | 256 x 56 x 56 | 56 stated; 256 our reading |
| block 2 (layer2) | 512 x 28 x 28 | 28 stated; 512 our reading |
| block 3 (layer3) | 1024 x 14 x 14 | 14 stated; 1024 our reading |
| block 4 (layer4) | 2048 x 7 x 7 | our reading (needed for the global feature) |
| global pool -> f | 2048 | Sec. 5 ("$2048$ dimensional vectors") |
| pixel feature F(y,p), if concatenated at 56 x 56 | 1792 per position, 3,136 positions | our reading |
| gallery G for K = 50 at 56 x 56 | 50 x 3,136 = 156,800 vectors | our reading |

Backward: none. "Which boxes get a gradient?" None; the paper calls it "virtually no training time" (abstract) and "does not require feature training" (Sec. 5). PaDiM (2011.08785, Sec. V-D) states it the same way: "SPADE requires no training as there are no parameters to learn".

## 6. Equations (TeX verbatim from main.tex)
No custom macros are used.

Eq. (1), `\label{eq:extract}`, Sec. 3.1:
```tex
f_i = F(x_i)
```
Paper: "Let us denote the global feature extractor $F$, for a given image $x_i$, we denote the extracted features $f_i$". Gloss: F = pretrained ResNet with global pooling of the last conv layer; x_i = a training image; f_i = its 2048-d global feature.

Eq. (2), `\label{eq:knn1}`, Sec. 3.2 (image score):
```tex
d(y) = \frac{1}{K} \sum_{f \in N_K(f_y)}{\|f - f_y\|^2}
```
Paper: "For a given test image $y$, we retrieve its $K$ nearest normal images from the training set, $N_k(f_y)$. The distance is measured using the Euclidean metric between the image-level feature representations." Gloss: y = test image; f_y = its global feature; N_K(f_y) = the K training global features closest to f_y; the score is the mean SQUARED Euclidean distance to them. K = 50 (MVTec AD). Note the text writes N_k, the equation N_K.

Gallery (inline, Sec. 3.3, verbatim including its stray braces):
```tex
G = \{F(x_1, p)|p \in P\} \cup \{F(x_2, p)|p \in P\} \}..\cup \{F(x_K, p)|p \in P\} \}
```
Gloss: x_1..x_K = the K retrieved normal images; P = all pixel locations; F(x, p) = the pyramid feature of image x at pixel p.

Eq. (3), `\label{eq:knn2}`, Sec. 3.3 (pixel score):
```tex
d(y, p) = \frac{1}{\kappa} \sum_{{f \in  N_{\kappa}}(F(y, p))}{\|f - F(y, p)\|^2}
```
Paper: "The anomaly score at pixel $p$, is given by the average distance between the features $F(y,p)$ and its $\kappa$ nearest features from the gallery $G$." Gloss: F(y, p) = test pixel feature; N_kappa = its kappa nearest neighbors in G; score = mean squared distance. kappa = 1 in all experiments, so this is the squared distance to the one nearest gallery feature. Decision: anomalous if d(y, p) > theta.

## 7. Config table
| Item | Value | Location |
|---|---|---|
| Backbone | Wide-ResNet50x2, ImageNet-pretrained, frozen | Sec. 3.5 |
| Trainable weights | none | abstract, Sec. 5 |
| MVTec preprocessing | resize 256 x 256, crop 224 x 224 | Sec. 3.5 |
| STC preprocessing | resize 256 x 256; train subsampled to ~5000 images; test subsampled by 5 | Sec. 3.5 |
| Metric resolution | 256 x 256, cv2.INTERAREA (as printed) | Sec. 3.5 |
| Pixel feature layers | end of blocks 1, 2, 3 (56, 28, 14), equal weights | Sec. 3.5 |
| Image feature | global-pooled last conv layer, 2048-d | Sec. 3.1, Sec. 5 |
| K (image kNN) | 50 MVTec; 1 STC ("due to the runtime considerations") | Sec. 3.5 |
| kappa (pixel kNN) | 1 in all experiments | Sec. 3.5 |
| Distance | squared Euclidean | Eq. 2, Eq. 3 |
| Smoothing | Gaussian filter, sigma = 4 | Sec. 3.5 |
| Latency / throughput | not reported in SPADE. PaDiM re-ran SPADE (WR50): 7.10 s per image on a CPU i7-4710HQ, serial (PaDiM Table VII) | PaDiM Table VII |
| Memory | not reported in SPADE. PaDiM's re-run: 1.4 GB MVTec AD, 37.0 GB STC (PaDiM Table VIII) | PaDiM Table VIII |

## 8. Results worth remembering (protocol: one-class, full-shot, one model per category, 256 resize -> 224 crop, metrics at 256)
MVTec AD (PDF pages 9-10):
- Image AUROC (stage 1 = DN2, K = 50): 85.5 average. Table 1, column SPADE.
- Pixel AUROC: 96.0 average. Table 2, row Average, column SPADE. Lowest classes: Tile 87.4, Wood 88.5.
- PRO (integrated to FPR 0.3): 91.7 average. Table 3, row Average, column SPADE. Lowest class: Tile 75.9.
- Baselines in Table 1 (Geom 67.2, GANomaly 76.2, AE_L2 75.4, ITAE 83.9), Table 2 (AE_SSIM 87, AE_L2 82, AnoGAN 74, CNN Dict 78, TI 75, VM 77, CAVGA-R_u 89) and Table 3 (Student 85.7, 1-NN 64, OC-SVM 47.9, l2-AE 79, VAE 63.9, SSIM-AE 69.4, CNN-Dict 51.5) are copied from the cited papers (Table 2: "reported by Bergmann et al. ... as well as newer results by Venkataramanan et al."; Table 3: "the baselines presented in their paper"). Not re-run.

Ablations (PDF page 13):
- Table 6 (PRO average, MVTec): only 14 x 14 layer 89.38; only 28 x 28 89.6; only 56 x 56 87.74; all three (SPADE) 91.7. Text: 56 x 56 alone "significantly hurts performance (due to limited context)".
- Table 7 (10 neighbours, metric not named in the caption; values near the PRO average): 10 random images 89.2 vs 10NN 91.4 average; Grid 73.2 vs 86.3.

STC (PDF pages 10-11): image AUROC 71.9 (Table 4; best is ITAE 72.5); pixel AUROC 89.9 (Table 5; CAVGA-R_u 85).

No VisA numbers (VisA postdates the paper). No latency of its own.

## 9. Limitations the paper states
- kNN cost scales linearly with dataset size and dimension (Sec. 5, "Optimizing runtime performance"); pixel-level kNN is "significantly slower" than image-level. Mitigation: search only the K retrieved images, and only for images flagged at stage 1. Suggests PatchMatch for speed.
- Features are not adapted: "future work should focus on methods for finetuning the deep pre-trained features" (Sec. 5).
- STC needed K = 1 for runtime.
- Fixed by a later source: PaDiM (2011.08785) replaces test-time kNN with one Gaussian per position; it reports SPADE at 7.10 s vs PaDiM-WR50-Rd550 at 0.95 s per image on CPU (PaDiM Table VII), and SPADE memory growing to 37.0 GB on STC (PaDiM Table VIII).

## 10. Outline feedback (ch-spade)
- Backbone confirmed: Wide-ResNet50x2, ImageNet-pretrained (Sec. 3.5). The "confirm" note can go.
- The pixel score is NOT "mean distance to the k nearest" in practice: kappa = 1 in all experiments, so it is the squared Euclidean distance to the single nearest gallery feature. The image score is the mean SQUARED distance to K = 50 neighbors. Fix the viz-spade readout wording (or expose kappa with default 1).
- Equation numbers in v3: image score = Eq. (2), pixel score = Eq. (3); Eq. (1) is just f_i = F(x_i).
- The architecture walkthrough needs two paths: the image path uses layer4 + global pooling (2048-d) for retrieval; the pixel path uses blocks 1-3 (56/28/14). Channel widths are not in the paper (mark as our reading).
- The image AUROC 85.5 comes from the global-feature kNN (stage 1, identical to DN2), not from the max of the pixel map. Worth saying, because PaDiM and PatchCore take the image score from the map.
- The paper does not say how the three pyramid levels are resized before concatenation; do not show a specific upsampling as the paper's.
- Citation slip in the source: Sec. 3.2 cites DN2 with the GOAD key (bergman2020classification); Sec. 4.1 cites DN2 correctly (bergman2020deep = arXiv 2002.10445). The outline's DN2 id is right.
- PaDiM Table III prints SPADE "All classes" pixel AUROC as 96.5; SPADE's own Table 2 says 96.0 (the mean of the per-class values is 96.03). Use 96.0 on the scoreboard (see padim.md).
