# PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization

## 1. Identity
- Title: "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization".
- arXiv 2011.08785, version read: v1 (only version; PDF stamp "arXiv:2011.08785v1 [cs.CV] 17 Nov 2020").
- Published (abs.html): 2020-11-17.
- Authors: Thomas Defard ... Romaric Audigier (with Aleksandr Setkov, Angelique Loesch); Universite Paris-Saclay, CEA, List.
- Venue (abs.html comments): "accepted at the 1st International Workshop on Industrial Machine Learning, ICPR 2020". 7 pages, 8 tables.
- Repo: none given in the paper.
- Sources: `sources/padim/src/bare_conf.tex`, `sources/padim/paper.pdf`. Tables are numbered I-VIII (verified against PDF captions).

## 2. Core problem
Localize anomalies (a score per patch) and detect anomalous images, trained on normal images only. Prior strong localizers either need deep network training or run kNN over the whole training set at test time, so time and memory grow with the training set.

## 3. Prior methods and why that hurt (per the paper, Sec. I-II)
- Reconstruction (AE, VAE, GAN): an AE "can sometimes yield good reconstruction results for anomalous images too".
- Image-level embedding methods (Deep SVDD, Gaussian/Mahalanobis on image features, DN2): no localization, so little interpretability.
- SPADE, Patch SVDD (patch kNN): "it runs a K-NN algorithm on a set of normal embedding vectors at test time, so the inference complexity scales linearly to the dataset training size".
- Image-level Gaussian methods (Lee et al., Rippel et al. = MahalanobisAD) do not model correlations between CNN semantic levels.

## 4. The one key idea
Describe each patch position (i, j) by one multivariate Gaussian, fitted to the N normal embeddings at that position, where the embedding concatenates three CNN levels so that the covariance captures inter-level correlations. Score a test patch by its Mahalanobis distance to its position's Gaussian. Test cost does not depend on N.

## 5. Mechanism, step by step
Fit: nothing is trained. The CNN is pretrained on ImageNet and frozen; "no deep learning training is required" (Sec. V-D). Fitting = one forward pass per normal image plus computing a sample mean and a sample covariance per position. No gradient, no optimizer, no epochs.

1. Preprocess: resize 256 x 256, center crop 224 x 224 (MVTec AD); STC: resize 256 x 256 only (Sec. IV-B).
2. Forward each of the N normal images through the frozen CNN. Take layers 1, 2, 3 for a ResNet; layers 7, 20, 26 (levels 2, 4, 5) for EfficientNet-B5 (Sec. IV-B).
3. Align levels to the grid of the largest activation map, W x H. The paper: "As activation maps have a lower resolution than the input image, many pixels have the same embeddings and then form pixel patches with no overlap in the original image resolution" (Sec. III-A). How layers 2 and 3 are brought to the layer-1 grid is not spelled out (our reading: each coarse vector is shared by the finer positions it covers, i.e. nearest upsampling).
4. Concatenate per position -> embedding x_ij (448-d for R18, Table II caption).
5. Random dimensionality reduction: keep d randomly selected dimensions, chosen "before the training" (Sec. V-A). d = 100 (R18-Rd100), 550 (WR50-Rd550). Our reading: one random index set shared by all positions.
6. Per position: mu_ij = sample mean of X_ij; Sigma_ij = sample covariance + epsilon I, epsilon = 0.01 (Eq. 1, Sec. IV-B). Store (mu_ij, Sigma_ij) for every position (the "matrix of Gaussian parameters", Fig. 2).
7. Test: same forward pass and same dimension selection; M(x_ij) = Mahalanobis distance to the Gaussian at (i, j) (Eq. 2). The W x H matrix M is the anomaly map.
8. Resize maps with bicubic interpolation; Gaussian filter sigma = 4 (Sec. IV-B, "like in [SPADE]").
9. Image score = maximum of the anomaly map M (Sec. III-C, V-B "Detection"). Whether the max is taken before or after resizing and smoothing: not stated.

### Tensor walkthrough (input 224 x 224)
The paper states only: 448 total dims for R18, the reduced sizes (100, 200, 550), and that the grid is the resolution of the largest map used. Per-layer channels and spatial sizes are the standard torchvision values (our reading); SPADE (2005.02357, Sec. 3.5) states 56/28/14 for WRN-50-2 at the same 224 input.

| Stage | ResNet18 | Wide-ResNet-50-2 | Source |
|---|---|---|---|
| input | 3 x 224 x 224 | 3 x 224 x 224 | Sec. IV-B |
| layer1 | 64 x 56 x 56 | 256 x 56 x 56 | our reading |
| layer2 | 128 x 28 x 28 | 512 x 28 x 28 | our reading |
| layer3 | 256 x 14 x 14 | 1024 x 14 x 14 | our reading |
| concat on 56 x 56 grid | 448 x 56 x 56 | 1792 x 56 x 56 | 448 stated (Table II caption); 1792 our reading |
| random selection | 100 x 56 x 56 (Rd100) | 550 x 56 x 56 (Rd550) | Sec. V-B |
| stack over N normal images | N x 100 x 56 x 56 | N x 550 x 56 x 56 | |
| stored per position | mu: 100; Sigma: 100 x 100 | mu: 550; Sigma: 550 x 550 | Eq. 1 |
| anomaly map | 56 x 56 -> resized -> sigma 4 | same | Sec. IV-B |
| image score | max of map | max of map | Sec. III-C |

EfficientNet-B5: layer indices 7, 20, 26 are stated; their channel counts, the resulting dimension, and whether Rd is applied are not stated.

### Storage per position (our reading, float32, mean + full covariance, no symmetry saving)
| Model | d | floats per position | bytes per position | x 3,136 positions (56 x 56) | x 4,096 positions (64 x 64, STC 256 input) |
|---|---|---|---|---|---|
| PaDiM-R18-Rd100 | 100 | 10,100 | 40,400 | 0.127 GB | 0.165 GB |
| PaDiM-R18 (all) | 448 | 201,152 | 804,608 | 2.52 GB | 3.30 GB |
| PaDiM-WR50-Rd550 | 550 | 303,050 | 1,212,200 (about 1.21 MB) | 3.80 GB | 4.97 GB |
| WR50, no reduction | 1792 | 3,213,056 | 12,852,224 | 40.3 GB | 52.6 GB |

Consistency check, not the paper's accounting: Table VIII prints 3.8 GB for PaDiM-WR50-Rd550 on MVTec AD, which matches the Gaussians alone (3.80 GB). R18-Rd100 (0.17 / 0.21) and WR50 on STC (5.2) match only if the backbone weights are added (R18 about 0.047 GB, WR50 about 0.276 GB at float32). The paper only says "PaDiM keeps in the memory only the pretrained CNN and the Gaussian parameters associated with each patch" and "assuming that parameters are encoded in float32".

Why epsilon matters (our reading): with d = 550 and per-category training sets of a few hundred images (the paper says MVTec AD has "15 classes of approximately 240 images"), the sample covariance has rank at most N - 1 < d, so it is singular; the epsilon I term is what makes it invertible. The paper says the term "makes the sample covariance matrix $\Sigma_{ij}$ full rank and invertible".

## 6. Equations (TeX verbatim from bare_conf.tex)
Packages: `amsmath`, `bm` (for `\bm{\epsilon}`). `[\![1,N  ]\!]` is a hand-built integer-interval bracket, not a macro. No custom macros.

Set of embeddings at position (i, j) (inline, Sec. III-B):
```tex
X_{ij}=\{x_{ij}^k,k \in  [\![1,N  ]\!] \}
```
Gloss: x_ij^k = embedding at position (i, j) of normal training image k; N = number of normal training images.

Eq. (1), `\label{eqn:sigma}`, Sec. III-B:
```tex
\Sigma_{ij} = {\frac{1}{N-1}}\sum_{k=1}^N (\mathbf{x_{ij}^k}-\mathbf{\mu_{ij}}) (\mathbf{x_{ij}^k}-\mathbf{\mu_{ij}})^\mathrm{T}  +\bm{\epsilon}I
```
Paper: "we make the assumption that $X_{ij}$ is generated by a multivariate Gaussian distribution $\mathcal{N}({\mu_{ij}},\boldsymbol{\Sigma_{ij}})$ where $\mu_{ij}$ is the sample mean of $X_{ij}$ ... where the regularisation term $\bm{\epsilon}I$ makes the sample covariance matrix $\Sigma_{ij}$ full rank and invertible." Gloss: mu_ij = mean of the N embeddings at (i, j); the sum is the unbiased sample covariance (divide by N - 1); epsilon = 0.01 ("By default we use $\bm{\epsilon} = 0.01$", Sec. IV-B); I = d x d identity.

Eq. (2), no label (number verified in PDF p. 3), Sec. III-C:
```tex
M(x_{ij}) = \sqrt{(x_{ij} - \mu_{ij})^T \Sigma_{ij}^{-1}(x_{ij} - \mu_{ij})}
```
Paper: "$M(x_{ij})$ can be interpreted as the distance between the test patch embedding $x_{ij}$ and learned distribution $\mathcal{N}({\mu_{ij}},\boldsymbol{\Sigma_{ij}})$". Gloss: x_ij = test embedding at (i, j); Sigma_ij^-1 = inverse regularized covariance at (i, j). Directions where normal data barely varies are weighted strongly.

Anomaly map (inline, Sec. III-C, verbatim including the strict inequalities):
```tex
M = (M(x_{ij}))_{1<i<W, 1<j<H}
```
"The final anomaly score of the entire image is the maximum of anomaly map $M$."

## 7. Config table
| Item | Value | Location |
|---|---|---|
| Backbones | ResNet18 (R18), Wide ResNet-50-2 (WR50), EfficientNet-B5; all ImageNet-pretrained, frozen | Sec. IV-B |
| Layers | ResNet: first three layers; EfficientNet-B5: layers 7 (level 2), 20 (level 4), 26 (level 5) | Sec. IV-B |
| Embedding size | R18: 448 | Table II caption, Sec. V-A |
| Random reduction | R18-Rd100 (100 of 448); WR50-Rd550 (550; out of 1792 is our reading) | Sec. IV-B, V-B |
| Grid | W x H = resolution of the largest activation map used (56 x 56 at 224 input is our reading) | Sec. III-A |
| epsilon | 0.01 | Sec. IV-B |
| Preprocessing | MVTec: resize 256, center crop 224; STC: resize 256 | Sec. IV-B |
| Map post-processing | bicubic resize; Gaussian filter sigma = 4 | Sec. IV-B |
| Image score | max of anomaly map | Sec. III-C |
| Trainable weights | none | Sec. V-D |
| Fit time | PaDiM-WR50-Rd550: 150 s average per MVTec class, 1500 s per STC scene; Intel CPU 6154 3GHz 72th, serial | Sec. V-D |
| Inference time (CPU i7-4710HQ @ 2.50GHz, serial; batch not stated) | SPADE (WR50) 7.10 s; VAE (R18) 0.21; PaDiM R18-Rd100 0.23; PaDiM-WR50-Rd550 0.95 | Table VII |
| Memory (GB, float32) | MVTec AD: SPADE 1.4, VAE 0.09, R18-Rd100 0.17, WR50-Rd550 3.8. STC: SPADE 37.0, VAE -, R18-Rd100 0.21, WR50-Rd550 5.2 | Table VIII |
| Own VAE baseline | R18 encoder, 8x8 conv latent, 10 000 augmented images per class, 100 epochs, Adam, lr 1e-4, batch 32, 2h40 per class on one P5000 | Sec. IV-B, V-D |
| GPU latency | not reported | |

## 8. Results worth remembering
Protocol for all MVTec numbers: one-class, full-shot, one model per category, resize 256 -> center crop 224.

PaDiM headline (MVTec AD):
- Pixel AUROC / PRO, PaDiM-WR50-Rd550: 97.5 / 92.1 (Table III, row "All classes"). Textures 96.9 / 93.2; objects 97.8 / 91.6.
- Pixel AUROC / PRO, PaDiM-R18-Rd100: 96.7 / 90.1 (Table III, row "All classes").
- Pixel AUROC / PRO, PaDiM-R18 (all 448 dims): 97.1 / 90.8 (Table I last row; Table II row "all (448)").
- Image AUROC, PaDiM-WR50-Rd550: 95.3; PaDiM EfficientNet-B5: 97.9 (Table V, row "all classes"). Textures 98.8 / 99.0; objects 93.6 / 97.2. No image AUROC for R18. No localization numbers for EfficientNet-B5.
- The text says EfficientNet-B5 "outperforms every model by at least 2.6p.p" (Sec. V-B). Table V gives 97.9 - 95.8 (MahalanobisAD) = 2.1; 2.6 is the margin over PaDiM-WR50-Rd550. Do not quote the 2.6 as a margin over prior work.

Ablations (R18, MVTec, all classes, (pixel AUROC, PRO)):
- Table I (layers): Layer 1 (94.8, 86.8); Layer 2 (95.7, 88.5); Layer 3 (95.7, 88.3); Layer 1+2+3 summed, no cross-level correlation (96.0, 89.0); PaDiM-R18 (97.1, 90.8). Text: +1.1 p.p. AUROC and +1.8 p.p. PRO over Layer 1+2+3.
- Table II (448 -> 100 / 200): Rd 100 (96.7, 90.5); PCA 100 (93.5, 85.7); Rd 200 (97.0, 90.5); PCA 200 (95.7, 89.3); all (448) (97.1, 90.8). Rd is the average of 10 models with different seeds; SEM of AUROC between 10^-4 and 10^-7. Text: Rd beats PCA "by at least 1.3p.p in the AUROC and 1.2p.p in the PRO-score"; Rd100 costs only 0.4 p.p. AUROC and 0.3 p.p. PRO vs all 448. Reason given: PCA keeps highest-variance dimensions, which "may not be the ones that help to discriminate".

Non-aligned Rd-MVTec AD (Table VI; random rotation (-10, +10) and random crop 256 -> 224 on train and test; mean over 5 seeded versions):
- PaDiM-WR50-Rd550 (92.2, 73.1); SPADE (WR50) (87.2, 69.0); VAE (R18) (62.1, 27.8).
- Text: AUROC drops by 5.3 p.p. for PaDiM, 12.2 for VAE, 8.8 for SPADE vs Table III.

STC pixel AUROC (Table IV): PaDiM-R18-Rd100 91.2; SPADE 89.9; CAVGA-RU 85.

Baselines PaDiM prints, and whether copied or re-run (our reading from digit-for-digit comparison with SPADE's tables):
- SPADE in Table III: per-class values and the PRO average 91.7 are identical to SPADE's Tables 2-3, so copied. The printed SPADE "All classes" pixel AUROC is 96.5, but SPADE's own Table 2 says 96.0 and PaDiM's own per-class SPADE values average to 96.03. PaDiM's Sec. V-C drop "8.8p.p" for SPADE equals 96.0 - 87.2, which is consistent with 96.0. The paper's claim that R18-Rd100 beats "all models in the average AUROC ... by at least 0.2p.p" (96.7 vs 96.5) depends on the 96.5.
- SPADE in Table V (85.5), GANomaly (76.2), ITAE (83.9): identical to SPADE Table 1, so copied (via SPADE). Patch SVDD 92.1 and MahalanobisAD (EfficientNet-B4) 95.8: cited from those papers, not re-run (our reading).
- SPADE in Table IV (89.9) = SPADE Table 5, copied. CAVGA-RU 85, copied.
- Re-run by PaDiM ("We reproduce the model SPADE as described in the original publication with a Wide ResNet-50-2"): only Table VI (Rd-MVTec), Table VII (time), Table VIII (memory).
- VAE (74.4, 64.2 all classes in Table III; Table VI; VII; VIII): PaDiM's own implementation.
- Patch SVDD pixel AUROC 95.7 and Student PRO 85.7 in Table III: copied from those papers.

No VisA numbers (VisA postdates the paper). No GPU latency.

## 9. Limitations the paper states
- Fit time is linear in the training set size (Sec. V-D), though no network is trained.
- Memory does not depend on the training set size but grows with image resolution: "The PaDiM space complexity increases from the MVTec AD to the STC only because the input image resolution is higher" (Sec. V-D). With the same backbone, SPADE uses less memory on MVTec AD (1.4 vs 3.8 GB, Table VIII).
- One Gaussian per fixed position assumes positions correspond across images; the paper tests this with Rd-MVTec AD and reports a 5.3 p.p. AUROC drop (Table VI), smaller than SPADE's.
- Later fix per the sources I read: none named (PaDiM is the later paper relative to SPADE).

## 10. Outline feedback (ch-padim)
- "the three backbone layers concatenated": correct for ResNets (layers 1-3). For EfficientNet-B5 it is layers 7, 20, 26. The concatenated size is stated only for R18 (448); the WR50 size 1792 is our reading.
- "Random dimensionality reduction (how many of how many channels)": stated as 448 -> 100 or 200 for R18 (Table II), and 550 for WR50 (Sec. V-B; "out of 1792" is our reading). The ablation is Table II, R18 only.
- Storage per position from the config: 303,050 floats = 1,212,200 bytes for WR50-Rd550; 10,100 floats = 40,400 bytes for R18-Rd100 (our reading). Totals at 56 x 56 are 3.80 GB and 0.127 GB. Label them computed, and say Table VIII's 3.8 GB matches the Gaussians alone.
- Image score is the max of the Mahalanobis map. Put this in the walkthrough; it differs from SPADE, whose image score is global-feature kNN.
- Add a one-line step on Rd-MVTec AD (Table VI). It tests the "aligned part" premise of the intuition directly: PaDiM loses 5.3 p.p. AUROC under rotation and crop, SPADE 8.8. Also PaDiM is strongest on textures (non-aligned), +4.0 p.p. AUROC and +4.8 p.p. PRO over SPADE (Sec. V-B). The intuition should not imply PaDiM needs perfect alignment.
- viz-padim: the epsilon I term is necessary, not cosmetic, when d exceeds the number of normal images (our reading). Worth a sentence near the Gaussian.
- Scoreboard: use SPADE's 96.0 (SPADE Table 2) for SPADE pixel AUROC, not PaDiM's printed 96.5.
- R18-Rd100 differs between Table II (96.7, 90.5; mean of 10 seeds) and Table III (96.7, 90.1). Cite the table you show.
- STC scene count differs: PaDiM says "13 scenes" (Sec. IV-A), SPADE says "$12$ scenes" (Sec. 4.2). Only matters if STC appears.
- ch-speed: PaDiM's Table VII is CPU only (i7-4710HQ, serial, batch size not stated). If used, label it CPU and do not mix it with GPU latency.
- Branch "Gaussian AD 2005.14140": PaDiM cites it as MahalanobisAD (rippel2020modeling) with 95.8 image AUROC (EfficientNet-B4, Table V). That number is copied in PaDiM; take it from 2005.14140 itself if that paper is in sources.
