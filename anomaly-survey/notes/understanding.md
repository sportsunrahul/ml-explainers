# Understanding: industrial visual anomaly detection

Inputs: `notes/papers/*.md` (19 spine notes, each read from the primary source), `notes/branches-a.json`
and `notes/branches-b.json` (one key claim per branch paper), `notes/resume-notes.md` (corrections).
The PDFs were not reopened for this file. Conventions:
- Every number keeps its source as (paper version, table or section, row). Protocol tags: one-class =
  one model per category; multi-class = one model for all categories of a dataset; k-shot = k normal
  images per category; zero-shot = no image of the target category; continual = categories learned in
  sequence; supervised = trained with anomalous labels.
- "our reading" = an inference made in a per-paper note, not printed in the paper. "code-only" = the
  value is only in the official repository. "our arithmetic" = a derived value that the note computed.
- "per the branch summary" = from `branches-*.json`. Only one claim per branch paper was extracted.
- This file adds no new arithmetic. Conflicts are recorded with both sides in Sec. 6, not reconciled.

## 1. Core problem

A factory can photograph thousands of good parts, but it has few or no images of each defect type, and
new defect types appear without warning. So a normal supervised defect classifier cannot be trained.
Every method here answers one question: how do you find defects when you have almost no defect images?
The common answer is to build a model of normal from good images only (or from no images of this
product), then score how far a test image, and each pixel of it, is from that model. The score must
flag small and rare defects without flagging normal variation. The families differ in two choices: what
the model of normal is (stored features, a Gaussian, a density, a student network, a reconstruction, a
text prompt), and how distance is measured. Four training regimes recur: normal images only; normal
images plus synthetic defects; a few normal images (few-shot); no images of the product (zero-shot).

## 2. The lens per paper

Method papers: trace the tensors forward, then ask which weights get a gradient and from what signal.

| Paper (version) | Lens | Frozen | Trained, and from what signal | What the trace found |
|---|---|---|---|---|
| SPADE 2005.02357v3 | method | WRN-50-2 (ImageNet) | nothing; fit = store global and pixel features | two paths: pooled layer4 (2048-d) for image kNN; blocks 1-3 for pixel kNN; kappa = 1 |
| PaDiM 2011.08785v1 | method | R18 / WR50 / EffNet-B5 | nothing; fit = mean and covariance per position | image score = max of the map; eps I is needed when d > N (our reading) |
| PatchCore 2106.08265v2 | method, plus official code | WRN50 | nothing; fit = features plus greedy coreset | psi is used only for selection; each layer is pooled to 1024, the two are joined end to end and pooled again to 1024 (in effect 512 numbers from layer 2 then 512 from layer 3); Eq. 7 is not in the code |
| FastFlow 2111.07677v2 | method | CaiT / DeiT / R18 / WRN50 | flow subnets (and ActNorm) by negative log-likelihood of normal features (our reading; loss not printed) | three separate flows for ResNets; image score rule not stated |
| CutPaste 2104.04015v1 | method | none (ResNet-18 from scratch); EffNet-B4 variant fine-tuned | whole network by cross-entropy "normal vs pasted"; then a Gaussian is fitted | the score is a density on features, not the classifier output |
| DRAEM 2108.07610v2 | method, plus official code | nothing pretrained | reconstructive subnet from L_rec and L_seg; discriminative subnet from L_seg | all shapes are code-only; L_seg also flows into the reconstructive subnet |
| EfficientAD 2303.14535v3 | method leads; its tables also serve as the desktop speed benchmark | WRN-101 (stage 1); teacher (stage 2) | teacher by distillation MSE on ImageNet; student and autoencoder per scenario by L_ST + L_AE + L_STAE | every reported number uses PDN padding disabled |
| UniAD 2206.03687v3 | method leads; Fig. 2, Fig. 4 and Table 5a are analysis of the shortcut | EfficientNet-b4 | projections, encoder, decoder, query and position embeddings by feature MSE | the MSE target is the clean (un-jittered) frozen feature |
| Dinomaly 2405.14325v5 | method | DINOv2-R ViT-B/14 | bottleneck and decoder by a hard-mining global cosine loss | multi-class baselines are copied, mostly from ADer |
| WinCLIP 2303.14814v1 | method | OpenCLIP ViT-B/16+ | nothing | window class tokens are scored; prompts contain the category name |
| AnomalyCLIP 2310.18961v12 | method | every CLIP weight | prompt words and deep text tokens by L_global + lambda L_local; the gradient passes through the frozen text encoder (our reading) | DPAM has no parameters; prompts are trained on another dataset's test split |
| AnomalyDINO 2405.14529v3 | method | DINOv2 ViT-S/14 | nothing | top-1% mean score; a per-category masking test |

Dataset, benchmark and analysis papers: what is claimed, how it was measured, what would falsify it.

| Paper (version) | Claim | How measured | What would falsify it |
|---|---|---|---|
| MVTec AD (CVPR 2019, not on arXiv) | a real-world, pixel-annotated benchmark for unsupervised AD | 15 categories, defect-free training; per-class accuracy, single-threshold per-region overlap, pixel ROC AUC (Sec. 4.3) | saturation: MVTec AD 2 (Sec. 2.1) cites 97.8% mean AU-PRO for the best model; EfficientAD (Sec. 4): 99.9% of defects lie in PatchCore's 76.6% center crop |
| VisA 2207.14315v1 | MVTec AD saturates; harder scenes and AU-PR are needed | 12 subsets; PaDiM (R50) and PatchCore (WRN-50) re-run on its 1-class split; AU-PR and AU-ROC | methods scoring as high on VisA as on MVTec AD; MVTec AD 2 (Sec. 2.1) later calls VisA "close to being solved" |
| MVTec LOCO (secondary sources only) | logical anomalies need global reasoning | structural and logical anomalies scored separately, equal weight; AU-sPRO | local methods scoring as high on logical as on structural; in EfficientAD Table 2 every method except GCAD scores lower on logical (our reading) |
| MVTec AD 2 2503.21622v1 | existing benchmarks are saturated; this one is harder | 7 methods, official code, default settings, 256 x 256, private test set on a server, AU-PRO_0.30 and AU-PRO_0.05 | scores matching MVTec AD; Table 1 shows the mean AU-PRO_0.30 falling from 92.4 to 52.6 |
| ADer 2406.03262v6 | unstandardized setups give wrong conclusions | 15 methods, multi-class, 256^2, 100 and 300 epochs, final epoch, one L40S | rankings that flip under another fixed epoch count or GPU; Tables 3-4 already show UniAD and DiAD moving a lot from 100 to 300 epochs |
| Anomalib 2202.08341v1 | one library can reproduce many methods | Tables 1-2: MVTec AD image AUROC (5 methods) and pixel AUROC (4); settings not stated | reproductions far below the methods' own numbers; PaDiM image mean 0.891 vs 95.3 in PaDiM Table V |
| Edge papers (`edge.md`) | methods can run on edge devices | device, precision, batch and averaging per paper | see Sec. 5 |

## 3. Benchmarks and metrics

### 3.1 MVTec AD (CVPR 2019)
- 15 categories (Table 1): textures carpet, grid, leather, tile, wood; objects bottle, cable, capsule,
  hazelnut, metal nut, pill, screw, toothbrush, transistor, zipper. Grid, screw and zipper are gray.
- 3629 defect-free training images ("for training and validation"); test 1725 = 467 good + 1258
  defective (Sec. 3, Table 1). 73 defect types (Table 1; the abstract says "over 70"). 1888 defect
  regions (Table 1; the text says "almost 1900"). Side lengths 700 to 1024 px. Training counts range
  from toothbrush 60 to hazelnut 391. Defects were made by hand.
- What it measured: per-class accuracy (Table 2), per-region overlap at one estimated threshold and
  pixel ROC AUC (Table 3). It used no image AUROC and no integrated AU-PRO. The AU-PRO up to FPR 0.3
  first appears in Uninformed Students (1911.02357v2, Sec. 4.2); later papers credit the IJCV 2021
  extension, which is not in `sources/`.
- Key finding: pixel ROC AUC can be high while the overlap is near zero. AE (SSIM): cable 0.01 overlap
  vs 0.82 ROC AUC; hazelnut 0.00 vs 0.97 (Table 3).

### 3.2 VisA (2207.14315v1, ECCV 2022)
- 10,821 images = 9,621 normal + 1,200 anomalous (100 per subset); 12 subsets in 3 domains (Sec. 4.1,
  Table 1): complex structure PCB1-PCB4; multiple instances Capsules, Candle, Macaroni1, Macaroni2;
  single instance Cashew, Chewing gum, Fryum, Pipe fryum. Up to 9 anomaly classes per object.
- 1-class protocol: 90% of normal images train; 10% of normal plus all anomalous images test (Sec. 4.2).
  Also 2-class high-shot (60/40) and low-shot (k = 5 or 10) protocols.
- Metrics: AU-PR and AU-ROC at image and pixel level. No PRO.
- Toy example (App. B): 100 positives, 100,000 negatives. Model A: AU-ROC 99.5, AU-PR 10.5. Model B:
  AU-ROC 98.5, AU-PR 90.6. The model with the lower AU-ROC is far better on the anomaly class.
- In-paper re-runs (not the methods' own numbers): PaDiM R50 on VisA, classification AU-ROC 87.8,
  segmentation AU-PR 11.4 and AU-ROC 93.1 (Table 2); PatchCore WRN-50 on VisA, 92.4, 38.4, 98.4 (Table 6).
- Difficulty (App. D): multiple instances hardest, then PCBs, single instance easiest.

### 3.3 MVTec LOCO (secondary facts only; IJCV PDF missing)
- Five scenarios (EfficientAD Sec. 2.1). Four are named in passing: splicing connectors, breakfast box,
  (banana) juice bottle, box of pushpins. The fifth is not named in either secondary source.
- Logical anomalies: "missing, misplaced, or surplus objects or the violation of geometrical
  constraints, for example, the length of a screw" (EfficientAD Sec. 3.3).
- Metric: AU-sPRO, "a generalization of the AU-PRO metric" (EfficientAD Sec. 4). No formula is
  available. The official script weights logical and structural anomalies equally.
- Image counts, splits and resolutions: not available.
- EfficientAD-M on LOCO: image AU-ROC 90.7 (logical 86.8, structural 94.7) (EfficientAD Table 2);
  AU-sPRO to FPR 0.3, 92.5 (Table 12); to FPR 0.05, 79.8 (Table 13). MVTec AD 2 quotes "79.8%" as the
  best LOCO localization; the match to Table 13 is our reading.

### 3.4 MVTec AD 2 (2503.21622v1)
- 8 scenarios, 8,004 images of 2.6 to 5 megapixels (Sec. 3.1, Table 4): Can, Fabric, Fruit Jelly,
  Rice, Sheet Metal, Vial, Wall Plugs, Walnuts. Transparent, reflective and bulk objects; defects
  anywhere, including the borders. Train 2528, val 302, test 5174 (our arithmetic over Table 4).
- Test splits: TEST_pub (small, with ground truth), TEST_priv (graded on a server), TEST_priv,mix (same
  scenes under seen and unseen lighting). TEST_priv and TEST_priv,mix show the same scenes.
- Protocol (Sec. 4.1): 7 methods, official code, MVTec AD defaults, 256 x 256 unless stated.
- Table 1, mean AU-PRO_0.30 (%): MVTec AD 92.4, VisA 82.4, MVTec AD 2 52.6; best on MVTec AD 2 is
  EfficientAD 58.7 (EfficientAD-S on MVTec AD 2, EfficientAD-M on the other two). The 39.8-point drop is
  our arithmetic.
- Table 7, AU-PRO_0.05 on TEST_priv: every method below 31%. Lighting (Sec. 4.3.1): MSFlow falls from
  24.3% to 11.9% on TEST_priv,mix; RD loses only 1.4 points.
- Resolution (Table 11): PatchCore AU-PRO_0.05 rises from 28.8% at 256 x 256 to 62.3% at half the
  original size; on Rice PatchCore takes 2 s per image on an RTX 2080 Ti.
- No image AUROC anywhere; it reports AU-PRO and F1.

### 3.5 Other benchmark branches (per the branch summary)
- MVTec 3D-AD (2112.09045v1): 4147 scans, 10 object categories, xyz plus RGB (Sec. 1, Sec. 3).
- Real-IAD (2403.12580v1, CVPR 2024): 151,050 images, 30 objects, five views (Table 1; row printed
  "Read-IAD"). MVTec AD 2 says Real-IAD already reaches 94.4% pixel AU-PRO_0.30 (Sec. 2.1).
- IM-IAD (2301.13359v5): 19 algorithms re-run on seven datasets under one setting (abstract).
- Survey (2301.11514v5): unsupervised industrial methods split into two families, feature-embedding
  and reconstruction (Sec. 2).

### 3.6 Metrics
- Image AUROC: one score per image; area under the ROC curve. Standard identity, not printed in any
  spine note: it equals the probability that a random anomalous image scores above a random normal
  image. It needs no threshold.
- Pixel AUROC: the same over pixels. MVTec AD (Sec. 4.3): TPR = percentage of anomalous pixels found;
  FPR = percentage of pixels wrongly flagged. Two weaknesses: the TPR is "dominated by large anomalies"
  (MVTec AD 2 Sec. 4.2.1), and the FPR is dominated by the many normal pixels (DRAEM, Sec. 4.1).
- AU-PRO. MVTec AD 2 v1, Eq. (1), Sec. 4.2.1:
  `PRO = (1/k) sum_{i=1..|D_test|} sum_{j=1..k_i} |P_ano intersect C_{i,j}| / |C_{i,j}|`.
  D_test = the test images; k_i = number of ground-truth regions (connected components) in image i;
  C_{i,j} = pixels of region j in image i; P_ano = pixels predicted anomalous at threshold t; the ratio
  = the fraction of one region that is covered. k is not defined; our reading is k = sum of k_i, so PRO
  is the mean coverage over all regions, and every region counts equally, large or small. AU-PRO
  integrates PRO over the pixel FPR on defect-free pixels, from 0 to a limit, and normalizes.
- The limit: 0.3 by convention (Uninformed Students Sec. 4.2; EfficientAD App. D.2, because flagging
  more than 30% of defect-free pixels is "of limited use"). MVTec AD 2 uses 0.05: for a 10-pixel defect
  in a 5 MP image, an FPR of 5% already flags an area "more than 25,000 times larger than the defect".
- Worked example (MVTec AD 2 Fig. 4; regions of 4, 2 and 20 pixels): all found, PRO 1 and TPR 1; half
  of each, 0.5 and 0.5; large region missed, PRO 0.67 and TPR 0.23; only the large region found, PRO
  0.33 and TPR 0.77.
- AU-PR and AP: precision-recall area; VisA's toy example and DRAEM's pixel AP (PaDiM 97.4 pixel AUROC
  but 55.0 AP, DRAEM 97.3 but 68.4, per DRAEM Table 2) show what AUROC hides.
- Reading rule for the page: a pixel AUROC of 96-98 can coexist with missed small defects; read AU-PRO
  or AP next to it. Image AUROC on MVTec AD is near its ceiling for most methods after 2022 (our
  reading; see the 99+ rows in Sec. 4).

## 4. The families, each paper as a delta

Each family starts with the first paper that has a spine note. Earlier papers that exist only as branch
summaries appear as short deltas, so the chain is complete but not overstated.

### 4.1 Feature embedding: DN2 -> SPADE -> PaDiM -> PatchCore (plus CFA, PNI, ReConPatch)

Precursors (per the branch summary; one claim each):
- DN2 (2002.10445v1): plain kNN distance in ImageNet-pretrained ResNet features beats self-supervised
  detectors at image level. CIFAR-10 one-class mean ROCAUC 92.5 (Table 1, row Avg); no MVTec AD result.
  It gives no map. SPADE's image stage is DN2 (SPADE note).
- Patch SVDD (2006.16067v2): Deep SVDD on patches with a self-supervised position task; distance to the
  nearest normal patch embedding localizes. MVTec AD image AUROC 0.921, pixel 0.957 (Table 1), 256 px.
- AE-SSIM (1807.02011v3): autoencoder trained and scored with SSIM; NanoTWICE pixel ROC AUC 0.966
  (Sec. 4.3); no MVTec AD result. It is the reconstruction baseline that MVTec AD shipped with.
- Gaussian AD (2005.14140v2; PaDiM calls it MahalanobisAD): one Gaussian on pooled EfficientNet features
  per level, image-level only. MVTec AD image AUROC 95.8 (Table V, NPCA 1%, EfficientNet-B4).

SPADE (2005.02357v3; venue not stated)
- Prior methods and why that hurt (Sec. 2, 5): reconstruction depends on the chosen loss; simple
  parametric densities do not fit image data; self-supervised features need training and trail
  ImageNet features; aligning a test image to one normal image fails when normal parts vary.
- Key idea: for each test pixel, find the nearest pixel feature among the pixel features of the K most
  similar normal images. A pixel with no close match is anomalous.
- Mechanism. Fit: store, for each normal image, its pooled layer-4 feature (2048-d) and its pixel
  feature pyramid. Nothing is trained. Test: (1) find the K = 50 nearest normal images by global
  feature (K = 1 on STC) and score the image (Eq. 2); (2) the gallery = all pixel features of those K
  images; (3) score each pixel by its nearest gallery feature (Eq. 3, kappa = 1); (4) pixel features =
  ends of blocks 1, 2, 3 (56, 28, 14 px grids) with equal weight; how they are resized is not stated
  (our reading: upsample to 56 x 56 and concatenate, 1792-d); (5) Gaussian smoothing, sigma = 4.
  Input: resize 256, crop 224; metrics at 256. Channel widths are our reading (torchvision WRN-50-2).
- Eq. (1): f_i = F(x_i). F = pretrained ResNet with global pooling; f_i = 2048-d feature of image x_i.
- Eq. (2): d(y) = (1/K) sum_{f in N_K(f_y)} ||f - f_y||^2. y = test image; f_y = its global feature;
  N_K(f_y) = the K nearest training global features; score = mean SQUARED Euclidean distance; K = 50.
- Eq. (3): d(y, p) = (1/kappa) sum_{f in N_kappa(F(y,p))} ||f - F(y,p)||^2. F(y, p) = test feature at
  pixel p; N_kappa = its kappa nearest gallery features. kappa = 1 in every experiment, so the pixel
  score is the squared distance to the single nearest gallery feature (resume-notes correction).
- Results (one-class, full-shot): image AUROC 85.5 (Table 1; this comes from the global kNN stage, not
  from the map), pixel AUROC 96.0 (Table 2), PRO 91.7 (Table 3). Layers (Table 6, PRO): all three 91.7,
  56 x 56 alone 87.74. Use SPADE's own 96.0, not PaDiM's printed 96.5 (Sec. 6).
- Limitations (Sec. 5): kNN cost grows linearly with data and dimension; pixel kNN is slow; features are
  not adapted. Fix: PaDiM replaces test-time kNN with one Gaussian per position. PaDiM measured SPADE at
  7.10 s per image vs 0.95 s for PaDiM-WR50-Rd550 on a CPU (PaDiM Table VII), and SPADE memory at
  37.0 GB on STC (PaDiM Table VIII).

PaDiM (2011.08785v1; ICPR 2020 workshop) as a delta from SPADE
- What SPADE did and why that hurt: kNN at test time, so cost scales with the training set (Sec. I-II).
  Image-level Gaussians (MahalanobisAD) ignore correlations between CNN levels.
- Key idea: one multivariate Gaussian per patch position (i, j), fitted to the N normal embeddings at
  that position; the embedding concatenates three CNN levels. Test cost does not depend on N.
- Mechanism: resize 256, crop 224; take ResNet layers 1-3 (EfficientNet-B5: layers 7, 20, 26); bring
  them to the grid of the largest map (our reading: nearest upsampling); concatenate (448-d for R18,
  Table II caption; 1792-d for WR50 is our reading); keep d random dimensions chosen once before fitting
  (d = 100 for R18, 550 for WR50; "550 of 1792" is our reading); fit mean and covariance per position;
  score with Mahalanobis distance; bicubic resize, Gaussian sigma = 4. Image score = max of the map, which
  differs from SPADE (resume-notes correction). Nothing is trained.
- Eq. (1): Sigma_ij = (1/(N-1)) sum_k (x_ij^k - mu_ij)(x_ij^k - mu_ij)^T + eps I. x_ij^k = embedding at
  (i, j) of normal image k; mu_ij = their mean; N = number of normal images; eps = 0.01; I = identity.
  The paper says eps I makes Sigma "full rank and invertible". Our reading: with d = 550 and a few
  hundred images, the sample covariance has rank at most N - 1 < d, so the term is required.
- Eq. (2): M(x_ij) = sqrt((x_ij - mu_ij)^T Sigma_ij^-1 (x_ij - mu_ij)). Directions in which normal data
  barely varies are weighted strongly. The W x H matrix M is the anomaly map.
- Storage: Table VIII prints 3.8 GB for WR50-Rd550 on MVTec AD; the Gaussians alone come to 3.80 GB
  (our arithmetic: 303,050 floats per position, 56 x 56 positions, float32).
- Results (one-class, full-shot, 224): WR50-Rd550 pixel AUROC 97.5, PRO 92.1 (Table III); image AUROC
  95.3, EfficientNet-B5 97.9 (Table V). R18-Rd100 96.7 / 90.1 (Table III). Layer ablation (Table I,
  R18): layers 1+2+3 summed without cross-level correlation 96.0 / 89.0 vs PaDiM-R18 97.1 / 90.8. Random
  vs PCA reduction (Table II): Rd 100 96.7 / 90.5 vs PCA 100 93.5 / 85.7.
- Alignment test, Rd-MVTec AD (Table VI; random rotation -10 to +10 degrees and random crop): PaDiM
  92.2 / 73.1, SPADE 87.2 / 69.0. Pixel AUROC drops by 5.3 points for PaDiM and 8.8 for SPADE (Sec.
  V-C). So PaDiM does not need perfect alignment, but it pays for misalignment.
- Timing is CPU only (i7-4710HQ, serial; Table VII): PaDiM R18-Rd100 0.23 s, WR50-Rd550 0.95 s.
- Limitations: fit time linear in N; memory grows with resolution (on MVTec AD, SPADE 1.4 GB vs PaDiM
  3.8 GB, Table VIII); a fixed position assumes correspondence across images. Fix: PatchCore's
  position-free memory bank (PatchCore Sec. 2 critique).

PatchCore (2106.08265v2; CVPR 2022) as a delta from SPADE and PaDiM
- What they did and why that hurt (Sec. 1-3): SPADE uses deep, ImageNet-biased features and a weak
  global preselection of K images; PaDiM limits each patch to its own position, so it relies on
  alignment and equal image shapes; random subsampling of a large bank loses coverage.
- Key idea: one position-free memory bank of locally aggregated mid-level patch features, thinned by a
  greedy coreset that keeps the bank's coverage. Score = distance from the worst test patch to its
  nearest stored normal patch.
- Mechanism. Fit (no gradient anywhere): resize 256, crop 224; WRN50 layer2 and layer3; aggregate each
  3 x 3 neighbourhood per layer (adaptive average pooling, stride 1); rescale layer3 bilinearly to the
  layer2 grid (28 x 28) and aggregate it with layer2; bank = every patch vector of every normal image;
  greedy coreset (25%, 10% or 1%). Test: nearest bank vector per patch (faiss); image score = max patch
  distance (Eq. 6), then reweighting (Eq. 7, paper only); map = bilinear upsample plus Gaussian sigma 4.
- Corrections from the code (resume notes): the random projection psi (code: 128-d) is used only for
  distances during coreset selection, and the bank keeps the 1024-d vectors; the two layers are
  pooled to 1024 per layer, joined end to end and pooled again to 1024 (code-only; no learned projection); Eq. 7 is
  not in the official code (image score = plain max, k = 1); hardware is printed "Nvidia Tesla V4".
- Shapes (our reading): 784 x 1024 patch features per image (28 x 28 grid). Bank = n x 784 vectors:
  toothbrush (60 images) 47,040, hazelnut (391) 306,544, all 15 categories 2,845,136 (our arithmetic
  from MVTec AD Table 1 counts).
- Eq. (1)-(3): the p x p neighbourhood of cell (h, w), p = 3; the locally aware feature = f_agg (adaptive
  average pooling) over that neighbourhood; the patch collection takes every s-th cell, s = 1.
- Eq. (4): M = union over normal images of their patch collections.
- Eq. (5), coreset objective ("minimax facility location"): M_C* = argmin_{M_C subset M} max_{m in M}
  min_{n in M_C} ||m - n||_2. Choose the subset whose worst-case distance from any bank vector m to its
  nearest kept vector n is smallest. That distance is the coverage radius. Exact solution is NP-hard.
- Algorithm 1: repeat l times: add the bank vector m whose distance (in psi space) to the current
  coreset is largest. The paper does not say how it starts (code: 10 random anchors).
- Eq. (6): m_test*, m* = argmax over test patches of argmin over M of ||m_test - m||_2; s* = ||m_test* -
  m*||_2. For each test patch find its nearest bank vector; take the test patch whose distance is
  largest; s* = that distance. The per-patch distances also form the map.
- Eq. (7): s = (1 - exp||m_test* - m*|| / sum_{m in N_b(m*)} exp||m_test* - m||) * s*. N_b(m*) = the b
  nearest bank vectors of m*. The weight rises when m* itself is isolated (a rare normal pattern). Our
  reading: the weight lies in [1 - 1/b, 1). b is not stated in the paper, README or code.
- Results (one-class, full-shot, WRN50, 224): image AUROC / pixel AUROC / PRO: PatchCore-25% 99.1 /
  98.1 / 93.4; -10% 99.0 / 98.1 / 93.5; -1% 99.0 / 98.0 / 93.1 (Tables 1-3). Three-backbone ensemble at
  320 px, 99.6 / 98.2 / 94.9 (Table 4; the abstract's "up to 99.6"). Time per image (Table 5): 100% 0.6
  s, 10% 0.22 s, 1% 0.17 s; batch size not stated. Re-runs in Table 5: SPADE 85.3 / 96.6 / 91.5, PaDiM
  95.4 / 97.3 / 91.8. Low-shot image AUROC, PatchCore-10 (Table S5): 1-shot 83.4, 16-shot 95.4, 50-shot
  97.5. Stride instead of coreset: s = 2 gives 97.6 (Sec. 4.4.2). No VisA numbers in the paper.
- Limitations: "limited by the transferability of the pretrained features" (Conclusion); remaining
  errors from label ambiguity, high normal variance, fine defects, and one defect cut by the center crop
  (supp. C.4). Follow-ups below adapt the features or add position context.

Later feature-embedding papers (per the branch summary). Each uses a different backbone or resolution
from PatchCore's WRN50 at 224, so the gains are not controlled comparisons.
- CFA (2206.04325v1): adds a small trainable patch descriptor that pulls normal features toward their
  nearest memory-bank centres (coupled hyperspheres). CFA++ (WRN50-2, 224 crop; ensembles cropped and
  resized inputs): image 99.5, pixel 98.5 (Table 2). Delta: the features are adapted, not frozen.
- PNI (2211.12634v3): conditions the memory score on neighbourhood features (an MLP) and per-position
  histograms; a refine net trained on synthetic defects sharpens maps. WRN-101, 480 crop: image 99.56,
  pixel 98.98 (Table 2). Delta: adds position and neighbourhood context back.
- ReConPatch (2305.16713v3, WACV 2024): learns a linear modulation of patch features by contrastive
  learning with similarity pseudo-labels, then scores PatchCore-style. Three-backbone ensemble at 480:
  image 99.72, pixel 98.67 (Table 6).

### 4.2 Flows: DifferNet / CFLOW-AD / CS-Flow -> FastFlow -> MSFlow

A normalizing flow is an invertible network. Because it is invertible, the exact likelihood of any
feature under the learned density is computable; low likelihood = anomaly.

Predecessors (per the branch summary):
- DifferNet (2008.12577v1): flow on pooled multi-scale AlexNet features; image-level only, scored over
  several test-time transforms. MVTec AD image AUROC 94.9 (Table 1); no pixel metric.
- CFLOW-AD (2107.12571v1, WACV 2022): conditional flows on each pyramid level of a frozen encoder, with a
  positional encoding as the condition, so likelihoods are per pixel. WRN-50, per-class best input size
  (256 or 512), mean of 4 runs: image 98.26, pixel 98.62 (Table 4).
- CS-Flow (2110.02855v1): fully convolutional flow over three scales jointly, with cross-scale links.
  EfficientNet-B5 at 768/384/192: image AUROC 98.7 (Table 1); no pixel metric.
- FastFlow's critique (Sec. 1, 2.3): 1D flows flatten the feature map, which "destroys the inherent
  spatial positional relationship"; DifferNet gives no exact localization; CFLOW-AD needs hard-coded
  position embeddings and a sliding window, so inference is slow.

FastFlow (2111.07677v2; venue not stated; AAAI-22 template only)
- Key idea: make the flow 2D. Coupling subnets are fully convolutional (3 x 3 and 1 x 1), so a C x H x W
  feature map maps to a C x H x W latent trained to be standard normal; the per-position likelihood is
  the anomaly map, in one pass.
- Mechanism: frozen ImageNet backbone (Sec. 4.7). Table 6: CaiT-M48-distilled, 448 px, block 40, 28 x
  28; DeiT-base-distilled, 384 px, block 7, 24 x 24; ResNet18 and WRN50-2, 256 px, blocks 1-3 at 64 / 32 /
  16, one separate flow per scale. 20 flow steps for ViTs, 8 for ResNets; subnets alternate 3 x 3 and 1 x
  1 ("3-1"), except ResNet18 (all 3 x 3). Fig. 2(b) step (figure reading): ActNorm, channel permute,
  split, two affine couplings with exp(s). Adam lr 1e-3, weight decay 1e-5, 500 epochs, batch 32. Test:
  sum the per-channel probabilities, bilinear upsample; ResNets average the three scales. The image
  score rule is not stated. Channel widths are our reading.
- Gradient: backbone frozen; flow subnets (and ActNorm) trained.
- Eq. (1): p_X(x) = p_Z(z) |det(dz/dx)|. x = backbone feature; z = f(x); p_Z = standard normal; the
  determinant corrects for how much the flow stretches or squeezes volume around x.
- Eq. (2): log p_X(x) = log p_Z(f_theta(x)) + log |det(d f_theta(x) / dx)|, with z ~ N(o, I) (printed
  "o"). theta = flow parameters. First term: how close the warped feature is to the centre of the
  Gaussian. Second term: the log volume change of the warp.
- Eq. (3), affine coupling: y_a, y_b = split(y); y_a' = y_a; y_b' = s(y_a) (elementwise) y_b + b(y_a);
  y' = concat(y_a', y_b'). The split is along channels; s and b are conv subnets of y_a. Because y_a
  passes unchanged, the step is invertible and its Jacobian is triangular. Eq. 3 has no exp; Fig. 2(b)
  uses exp(s).
- Training loss: not printed. The only loss equation is commented out of the LaTeX (lines 355-361).
  Our reading: minimize -log p_X(x) from Eq. 2 at every position, averaged.
- Results (one-class, full-shot, MVTec AD; image and pixel AUROC only, no PRO, no VisA): CaiT at 448,
  99.4 / 98.5; WRN50 at 256, 99.3 / 98.1 (Table 1). Table 2 repeats 99.4 / 98.5 without naming the
  backbone. In FastFlow's own Table 2, CFlow leads pixel AUROC (98.6 vs 98.5). The 99.4 uses training
  augmentation (Table 8: CaiT without it, 99.3). Speed (Table 1; GTX 1080Ti, batch not stated): CaiT
  FastFlow 3.08 FPS with 9 ms additional time; WRN50 FastFlow 21.8 FPS, 34 ms additional; WRN50 +
  PatchCore 5.88 FPS, 159 ms additional. "A.d." time excludes the backbone; FPS includes it.
- Limitations (supp. Sec. 6-7): misses small, unobvious anomalies; augmentation must suit each category;
  small backbones prefer 3-3 subnets. Later critique: EfficientAD (Sec. 4) says the anomalib FastFlow
  uses early stopping on test images; its re-run without it gives MVTec AD image AU-ROC 96.9
  (EfficientAD Table 2).

MSFlow (2308.15300v1; branch): parallel flows per scale plus a fusion flow that exchanges information
across scales; different scale aggregation for detection and for localization. WRN-50 at 512 (256 for
transistor): image AUROC 99.7 (Table I). MVTec AD 2 ran it at 256 (not the official 512): AU-PRO_0.30 on
MVTec AD 91.2 (MVTec AD 2 Table 1); under mixed lighting its AU-PRO_0.05 falls from 24.3% to 11.9%
(MVTec AD 2 Sec. 4.3.1).

### 4.3 Synthetic anomalies: CutPaste, DRAEM, then NSA / MemSeg / DSR / SimpleNet / GLASS

CutPaste (2104.04015v1; CVPR 2021)
- Prior methods and why that hurt (Sec. 1, 2.1): reconstruction lacks high-level semantics; rotation
  prediction learns objectness, not regularity (MVTec AD image AUC 73.1, Table 1); Cutout is too easy a
  proxy task (71.3).
- Key idea: train a classifier to tell normal images from images in which a patch of the same image was
  cut and pasted elsewhere. Keep the representation, discard the head, and fit a Gaussian density
  estimator (GDE) to normal representations.
- Mechanism: (1) augment: area 2-15% of the image, aspect (0.3, 1) U (1, 3.3); CutPaste-Scar 2-16 px
  wide, 10-25 px long, rotated -45 to +45 degrees; colour jitter up to 0.1 (App. A.2). (2) Train
  ResNet-18 from scratch plus an MLP head, 256 x 256; the 3-way task (normal / CutPaste / Scar) is best.
  (3) Fit GDE on pooled features (whether before or after the MLP head is not stated). (4) Image score =
  GDE log-density (our reading: the negative is a Mahalanobis distance). (5) Localization: a separate
  patch model; at test 32 x 32 patches at stride 4 give 57 x 57 x 512 embeddings, each scored by GDE,
  then Gaussian upsampling. One GDE per location for aligned objects; one shared GDE for textures,
  hazelnut and screw (App. A.5), which is PaDiM's per-position idea.
- Gradient: stage 1 trains the whole network by cross-entropy; stage 2 (GDE) has no gradient.
- Eq. (1): L_CP = E_{x in X} { CE(g(x), 0) + CE(g(CP(x)), 1) }. X = normal images; CP = the cut-paste
  augmentation; g = binary classifier; CE = cross-entropy. Class 0 for x, class 1 for its pasted copy.
  The 3-way variant adds a Scar class; no separate equation is printed.
- Eq. (2): log p_gde(x) proportional to -(1/2)(f(x) - mu)^T Sigma^-1 (f(x) - mu). f(x) = learned
  representation; mu, Sigma = mean and covariance of normal representations.
- Eq. (3): Eq. 1 applied to random crops c(x), so the representation describes a patch.
- Results (one-class, full-shot, 256, mean of 5 seeds; no PRO, no VisA, no latency): ResNet-18 3-way
  image AUROC 95.2 (+-0.6) (Table 1); ensemble of five 96.1; EfficientNet-B4 fine-tuned, pool feature
  96.6 (Table 3; the abstract headline); level-7 feature 97.1. Pixel AUROC 96.0 from the patch model
  (Table 2); GradCAM on the image model 88.3 (Sec. 4.2).
- Evidence for the risk "the model learns your fakes": in the t-SNE (Fig. 3) CutPaste examples "are
  almost not overlapped" with real defects, yet the features still separate real defects from normal.
- Limitations: "a crude approximation of a real anomaly distribution" (Sec. 1); failures on missing,
  flipped or dislocated components (App. B.1); weak on semantic outliers (CIFAR-10 69.4 vs rotation 91.3).

DRAEM (2108.07610v2; ICCV 2021) as a delta from CutPaste
- What CutPaste-style training did and why that hurt (DRAEM's claim, Sec. 1-2): a purely discriminative
  network trained on synthetic anomalies over-fits to the synthetic look. Autoencoders reconstruct subtle
  anomalies too well and need hand-made post-processing.
- Key idea: make a fake anomaly; train one network to repair it and a second network to segment the
  fake from the input and the repair side by side. The segmenter learns "how far is this from its own
  repaired version", which transfers to real defects.
- Mechanism: (1) Perlin noise, binarized at a random threshold -> mask M_a; a DTD texture with 3 random
  augmentations; blend inside the mask (Eq. 4). (2) Reconstructive subnet maps I_a to I_r, trained
  toward the clean I. (3) Discriminative U-Net takes concat(I_r, I_a) and outputs M_o, trained with focal
  loss against M_a. (4) Joint training, end to end. (5) Image score = max of the mean-filtered M_o (Eq.
  5). Training images are rotated -45 to +45 degrees.
- Gradient: the reconstructive subnet gets L_rec and L_seg (code: I_r is not detached); the
  discriminative subnet gets L_seg only; nothing is pretrained or frozen.
- Shapes (code-only, 256 x 256): reconstructive encoder-decoder, base width 128, 1024 x 16 x 16
  bottleneck, no skips; discriminative U-Net, base 64, 6 -> 2 channels. Parameters 69.0M + 28.4M = 97.4M
  (our arithmetic over the code).
- Eq. (1): L_SSIM(I, I_r) = (1/N_p) sum_{i,j} 1 - SSIM(I, I_r)_(i,j). N_p = pixel count; SSIM at the
  patch centred on (i, j).
- Eq. (2): L_rec = lambda L_SSIM + l2. lambda is not given in the paper (code: 1).
- Eq. (3): L = L_rec + L_seg. L_seg is focal loss (Lin et al. 2017); its formula is not printed (code:
  gamma = 2).
- Eq. (4): I_a = (not M_a) (elementwise) I + (1 - beta)(M_a (elementwise) I) + beta (M_a (elementwise) A).
  Outside the mask keep I; inside it mix (1 - beta) of the image with beta of the texture A; beta drawn
  from [0.1, 1.0]. A small beta gives a faint, harder anomaly.
- Eq. (5): eta = max(M_o * f_{s_f x s_f}). f = mean filter; * = convolution; s_f not given (code: 21).
- Results (one-class, full-shot; no PRO, no VisA, no latency): image AUROC 98.0 (Table 1); pixel AUROC
  97.3, pixel AP 68.4 (Table 2). In DRAEM's Table 2, PaDiM has 97.4 pixel AUROC and 55.0 AP. Ablation
  (Table 3, image AUROC / pixel AUROC / AP): discriminative subnet only 93.9 / 92.7 / 62.5; random-colour
  anomalies 96.2 / 92.6 / 56.5; rectangles 96.9 / 96.8 / 65.1; full 98.0 / 97.3 / 68.4. So the
  reconstruct-then-compare design matters more than realistic fakes. Fewer than 10 texture images
  suffice (Fig. 9).
- Limitations: missing parts are hard (Sec. 4.1); label ambiguity. In the multi-class setting DRAEM
  falls (UniAD re-run 88.1; ADer 54.5 / 55.2; see Sec. 6).

Later synthetic papers (per the branch summary). The trend is from image-space fakes to feature-space
fakes.
- NSA (2109.15222v3, ECCV 2022): Poisson-blends resized patches from other normal images into seamless
  fakes and predicts their pixel labels. Image AUROC 97.2, mean of 5 seeds (Table 1). Delta vs CutPaste:
  seamless blending, patches from other images, pixel-level targets.
- MemSeg (2205.00908v1): U-Net segmenter on simulated defects, guided by a memory of normal features.
  ResNet18, 256: image 99.56, pixel 98.84 (Table 1).
- DSR (2208.01521v2, ECCV 2022): samples fakes in a learned quantized feature space, so
  near-in-distribution defects can be made. Image AUROC 98.2 (Table 3).
- SimpleNet (2303.15140v2, CVPR 2023): Gaussian noise on adapted pretrained patch features; a small
  discriminator separates them. WRN50, 224 crop: 99.6 / 98.1 (Table 1). EfficientAD (Sec. 4) says it
  tunes training duration on test images; EfficientAD's re-run gives 98.2 on MVTec AD (Table 2).
- GLASS (2407.09359v1): feature-level noise steered by gradient ascent, plus image-level fakes, to cover
  weak defects. GLASS-j at 288 px: 99.9 / 99.3 (Table 1); GLASS-j picks GLASS-m or GLASS-h per category
  by spectrogram analysis.

### 4.4 Student-teacher: Uninformed Students, STPM, RD4AD, AST -> EfficientAD

The idea: train a student to imitate a teacher's features on normal images only. On a defect the student
was never shown, the two disagree, and the gap is the anomaly map.

Predecessors (per the branch summary):
- Uninformed Students (1911.02357v2, CVPR 2020): an ensemble of students regresses a pretrained
  teacher's patch descriptors; regression error plus student variance flags anomalies. MVTec AD AU-PRO
  (FPR 0.3), multiscale, 0.914 (Table 3); no image AUROC. EfficientAD's critique: 1.6 to 2.7 M
  parameters per network but no downsampling, so it is slow (EfficientAD lines 225-229).
- STPM (2103.04257v3, BMVC 2021): one student with the teacher's architecture matches the teacher's
  feature pyramid at several scales (ResNet-18 conv2_x to conv4_x). Image AUC-ROC 0.955 (Table 2).
- RD4AD (2201.10703v2, CVPR 2022): reverse distillation; a student decoder rebuilds the teacher
  encoder's multi-scale features from a compact one-class bottleneck. WRN50, 256: 98.5 (Table 1).
- AST (2210.07829v2, WACV 2023): architectural asymmetry; a normalizing-flow teacher and a plain CNN
  student diverge on anomalies. EfficientNet-B5, 768 px, mean of 5 runs: 99.2 (Table 2). EfficientAD
  re-ran it at 98.9 ("not able to fully reproduce", EfficientAD App. B). EfficientAD replaces exactly
  this asymmetry (resume-notes: AST added as a branch).

EfficientAD (2303.14535v3; WACV 2024 oral) as a delta from Uninformed Students and AST
- What prior methods did and why that hurt: deep backbones are slow, and their long-range dependencies
  let one defect make distant features anomalous (Sec. 3.1); ensembles, pyramids and AST's asymmetry
  cost runtime; more training images can teach the student to imitate anomalies too; autoencoders
  reconstruct fine patterns poorly, so their maps give false positives (Sec. 3.3).
- Key idea: a 4-layer patch description network (PDN), distilled from WideResNet-101, serves as teacher
  and student. A "loss-induced asymmetry" (hard feature loss plus an ImageNet penalty) replaces
  architectural asymmetry at no test-time cost. An autoencoder that the student also imitates adds a
  global map for logical anomalies.
- Mechanism. Stage 1, once: teacher T learns to match WRN-101 features with PatchCore's postprocessing
  (384-d; Psi at 512 x 512, T at 256 x 256, both 384 x 64 x 64), on ImageNet, 60,000 steps, batch 16.
  Stage 2, per scenario: T frozen; student S (768 output channels) and autoencoder A trained jointly for
  70,000 iterations, batch 1. The first 384 student channels imitate the normalized teacher with the
  hard loss; the penalty pushes student outputs on a random ImageNet image toward zero; A reconstructs
  the teacher features of an augmented image; the last 384 student channels imitate A. Stage 3: map
  quantiles on validation images. Inference: one pass each of T, S, A; local map = channel mean of
  (T - S_first)^2; global map = channel mean of (A - S_second)^2; normalize both, average, image score =
  max of the combined map.
- Gradient: stage 1, T only; stage 2, S (from L_ST and L_STAE) and A (from L_AE and L_STAE); stage 3 and
  inference, nothing. Per-scenario training takes "twenty minutes" (line 625).
- Shapes (our reading, from Tables 6-8): PDN-S maps 3 x 256 x 256 to 384 x 64 x 64 with padding; each
  output vector sees a 33 x 33 patch. Teacher 2,694,144 and student 4,267,392 parameters; autoencoder
  bottleneck 64-d. Totals T + S + A: 8,057,856 (S) and 20,738,432 (M), matching Table 16's "8" and "21"
  million. Every reported number uses padding disabled (App. E); the output size is then 56 x 56 or 58 x
  58 (our reading).
- Equations (none is numbered in the paper; all inline):
  - E1: D_{c,w,h} = (T(I)_{c,w,h} - S(I)_{c,w,h})^2, the elementwise squared difference.
  - E2, hard feature loss: L_hard = mean of the elements of D at or above their p_hard-quantile.
    p_hard = 0.999, which "corresponds to using, on average, ten percent of the values in each of the
    three dimensions". Our reading: about 1,573 of 1,572,864 elements get a gradient per image.
  - E3: local map M_{w,h} = C^-1 sum_c D_{c,w,h}.
  - E4: L_ST = L_hard + (CWH)^-1 sum_c ||S(P)_c||_F^2. P = a random ImageNet image; the penalty stops
    the student from imitating the teacher outside the normal data.
  - E5: L_AE = (CWH)^-1 sum_c ||T(I)_c - A(I)_c||_F^2 (A reconstructs features, not pixels).
  - E6: L_STAE = (CWH)^-1 sum_c ||A(I)_c - S'(I)_c||_F^2. The student learns A's systematic errors on
    normal images, but not its errors on anomalies. Total = L_ST + L_AE + L_STAE.
  - E8: M_hat = 0.1 (M - q_a)(q_b - q_a)^-1 per map, q_a = 0.9 and q_b = 0.995 validation quantiles;
    M = 0.5 M_hat_ST + 0.5 M_hat_AE; image score = max M.
- Results. Table 1 is not MVTec AD: it is the mean of three collection means (MVTec AD, VisA, LOCO), so
  LOCO logical anomalies weigh about one sixth (resume-notes correction). EfficientAD-S 95.4 AU-ROC /
  92.5 AU-PRO, 2.2 ms, 614 img/s; EfficientAD-M 96.0 / 93.3, 4.5 ms, 269 img/s (RTX A6000). For one
  dataset use Table 2 (image AU-ROC): S 98.8 MVTec AD, 97.5 VisA, 90.0 LOCO; M 99.1, 98.1, 90.7. AU-PRO
  (Table 12): S 93.1 / 93.1, M 93.5 / 94.0 (MVTec AD / VisA). p_hard (Table 3): 0 -> 94.9, 0.999 -> 96.0.
  Pixel AUROC is printed only up to FPR 5% (Table 14), so leave that scoreboard cell empty.
- All baselines are EfficientAD's re-runs with changed configs (App. B): PatchCore with a 1% coreset,
  224 px, WRN-101, no center crop scores 98.7 on MVTec AD (Table 2), not PatchCore's own 99.1.
- Limitations (Sec. 5): fine logical anomalies ("a screw that is two millimeters too long"); per-scenario
  training. On a Jetson under a continual protocol it collapsed (Sec. 5 here).

### 4.5 Unified (multi-class): UniAD -> HVQ-Trans / DiAD / MambaAD -> Dinomaly

UniAD (2206.03687v3; NeurIPS 2022)
- Core problem: one model per product does not scale. UniAD trains one model on all 15 MVTec AD classes
  with no class label at train or test time.
- What prior methods did and why that hurt: reconstruction networks fall into the "identical shortcut"
  (Sec. 1 also says "identity shortcut"): they copy the input, so defects reconstruct too. Evidence: Fig.
  2a (training loss falls while test AUROC peaks and then drops), Fig. 2b (decoded reconstructions keep
  the defect), Fig. 4 (the unified case is worse than the separate case at a lower loss). One-class
  methods drop in the unified case (Table 1, image AUROC, separate copied vs unified re-run): PaDiM 95.5
  -> 84.2; CutPaste 96.1 -> 77.5; DRAEM 98.0 -> 88.1.
- Key idea: reconstruct frozen CNN features with a transformer that cannot copy: a learnable query in
  every decoder layer (LQD), attention that hides each token's own 7 x 7 neighbourhood (NMA), and
  Gaussian noise on input tokens so the task is denoising (feature jitter, FJ).
- Mechanism: 224 x 224 input; frozen EfficientNet-b4 stages 1-4 resized to 14 x 14 and concatenated to
  272 x 14 x 14 (per-stage 24 / 32 / 56 / 160 channels is our reading); 196 tokens projected to 256;
  jitter (training only); 4-layer neighbour-masked encoder; 4-layer layer-wise query decoder; project
  back to 272; MSE against the clean features. Map = per-token L2 error, bilinear upsample; image score
  = max of the average-pooled map (pool size not stated).
- Gradient: projections, encoder, decoder, query and position embeddings. The backbone is frozen;
  training it gives 65.9 / 65.5 vs 96.5 / 96.8 (Table A5).
- Eq. (1): y = x+ w + b. x+ = K normal tokens of C channels; an FC layer can learn w -> I, b -> 0 and so
  copy anomalies too.
- Eq. (2): y = softmax(q x+^T / sqrt(C)) x+. With a learned query q, the attention map can reach the
  identity only if q is tied to normal inputs, so anomalous inputs cannot be copied.
- Eq. (3): D ~ N(0, (alpha ||f_tok||_2 / C)^2). f_tok = one token; alpha = 20; added with probability
  p = 1. Our reading: with C = 256 the noise std is 1.25 times the token's per-channel RMS.
- Eq. (4): L = (1/(H x W)) ||f_org - f_rec||_2^2. Eq. (5): s = ||f_org - f_rec||_2, an H x W map.
- Results (multi-class, 224, MVTec AD only; no VisA, no PRO, no latency): image AUROC 96.5 (+-0.08),
  pixel 96.8 (+-0.02) (Tables 1-2). Ablation (Table 5a, image / pixel): no query 69.5 / 79.4; one query
  87.6 / 92.8; layer-wise query 95.0 / 96.5; one query + NMA 96.1 / 96.3; all three 96.5 / 96.8. The
  learned query is the largest single effect. Neighbour 1 x 1 gives 94.6 vs 7 x 7 96.5 (Table 5c). 7.48M
  learnable parameters, 6.46 GFLOPs (Table A7).
- Limitations: class labels unused (Sec. 5); 1000 epochs. ADer's re-run: 92.5 at 100 epochs, 96.8 at
  300 (ADer Table 3).

ADer as the yardstick for this family (multi-class, 256, final epoch, one L40S)
- Memory banks do not collapse: PatchCore 98.8 image / 98.3 pixel / 94.2 AU-PRO on MVTec AD (Table 3),
  the best row, but its test time is 9h22m on MVTec AD and it runs out of memory on VisA (Table 1).
- One-class designs that collapse in ADer's runs: DRAEM 54.5, CFA 57.6, PyramidalFlow 70.2 (Table 3,
  100 epochs). PaDiM, CutPaste, FastFlow, EfficientAD and Dinomaly are absent from ADer.
- Table 1 times are totals for the whole test set, not per-image latency.

Branches (per the branch summary; multi-class MVTec AD image AUROC from each paper's own Table 1):
- HVQ-Trans (2310.14228v1, NeurIPS 2023): a hierarchy of learned discrete normal prototypes (vector
  quantization), so defects cannot pass through. 98.0 (+-0.11).
- DiAD (2312.06607v1, AAAI 2024): a semantically guided Stable Diffusion denoiser reconstructs the
  image; pretrained features of input and reconstruction are compared. 97.2. ADer: 88.9 / 92.0, 1331M
  parameters (ADer Tables 1, 3).
- MambaAD (2404.06564v4, NeurIPS 2024): a Mamba decoder over ResNet34 features, scanning in eight
  directions (global context at linear cost). 98.6 (+-0.3), 256 px. ADer: 97.8 / 98.5.

Dinomaly (2405.14325v5; CVPR 2025) as a delta from UniAD and its branches
- What they did and why that hurt (Sec. 2-3): each fix is a bespoke module (neighbour mask plus jitter,
  vector quantization, diffusion, state space). Dinomaly reframes the shortcut as "over-generalization"
  and notes that softmax attention and convolution can both learn an identity.
- Key idea: four deliberately simple parts on a strong frozen backbone: (1) scalable foundation
  Transformers, (2) a noisy bottleneck, (3) linear attention, (4) loose reconstruction (= loose
  constraint + loose loss). Each makes copying harder.
- Mechanism: resize 448, center crop 392, patch 14 -> 28 x 28 = 784 tokens; frozen DINOv2-R ViT-B/14;
  the middle 8 of 12 layers (indices 2-9 are our reading from Fig. 2) are summed and passed through an
  MLP bottleneck with Dropout 0.2; an 8-layer decoder with linear attention; loose constraint = 2 groups
  of 4 layers, summed, reconstructed group to group; loose loss = global cosine with hard mining. Map =
  per-point cosine distance, mean of the 2 groups; image score = mean of the top 1% map pixels. 10,000
  iterations on MVTec AD and VisA; RTX 3090; batch size not stated.
- Gradient: bottleneck and decoder only; the encoder is frozen.
- Delta table vs UniAD: encoder EfficientNet-b4 -> DINOv2-R ViT-B/14; noise feature jitter -> Dropout in
  the bottleneck; anti-copy attention neighbour mask -> linear attention; target concatenated layers ->
  two summed groups; loss point-by-point MSE -> global cosine with hard mining.
- Eq. (3): LA(Q, K, V) = (phi(Q) phi(K^T)) V = phi(Q) (phi(K^T) V), phi(x) = elu(x) + 1. There is no
  softmax, so no sharp peak on one token; K^T V is a d x d summary of all tokens; cost O(N d^2) instead
  of O(N^2 d). The paper: linear attention "cannot concentrate", which "is exactly what the
  reconstruction decoder favors".
- Eq. (4)-(6): L = D_cos(F(f_E), F(f_D_hat)); D_cos(a, b) = 1 - a^T b / (||a|| ||b||). f_E, f_D =
  grouped encoder and decoder maps; F flattens the whole map, so the cosine is global; f_D_hat shrinks
  the gradient to 0.1x at the k% best-reconstructed points of the batch (a full stop-gradient was
  unstable); k rises from 0 to 90% over the first 1,000 iterations.
- Results (multi-class, 392 px, ViT-B, seed 1; Table 1, I-AUROC / P-AUROC / AUPRO): MVTec AD 99.6 / 98.4
  / 94.8; VisA 98.7 / 98.7 / 94.5; Real-IAD 89.3 / 98.8 / 93.9. Class-separated (Table 2): MVTec AD 99.7
  image; VisA 98.9; Real-IAD 92.0. So the gap closes on MVTec AD and VisA but not on Real-IAD.
- Ablation (Table 3, MVTec AD I-AUROC): no components (close to ViTAD) 98.41; noisy bottleneck only
  99.06; all four 99.60. Swaps: feature jitter 99.59 vs Dropout 99.60 (Table A8); softmax plus neighbour
  mask 99.56 vs linear attention 99.60 (Table A7). Our reading in the note: most of the gain over UniAD
  comes from the backbone and framework. Backbones (Table A1): DINOv2-R 99.60, DINO 98.97, supervised
  DeiT 98.19, MAE 96.27. Scale (Table 4): ViT-S 99.26, ViT-B 99.60, ViT-L 99.77.
- Limitations (App. F): ViTs are expensive; zero-shot, few-shot and noisy training sets not covered; the
  PatchCore-style center crop can hide anomalies at image edges (App. C).

### 4.6 Zero-shot and few-shot: WinCLIP -> branches; AnomalyCLIP -> branches; AnomalyDINO

WinCLIP (2303.14814v1; CVPR 2023)
- Prior methods and why that hurt: one-class methods need many images per product; naive CLIP with
  "normal [o]" vs "anomalous [o]" gives 74.0 image AUROC on MVTec AD (Table 1); CLIP patch tokens are not
  aligned with text (22.4 pixel AUROC, main Table 8 p. 8).
- Key idea: score each image window's class token, not each patch token, against an ensemble of normal
  and anomalous prompts; spread the window scores back to pixels.
- Mechanism (nothing is trained; CLIP is frozen): (1) prompts = 7 normal states x 22 templates and 4
  anomaly states x 22 templates (154 and 88 prompts, our arithmetic); [o] is the category name from the
  dataset, so zero-shot means "no images", not "no product knowledge"; the text embeddings are averaged
  per label into two prototypes. (2) OpenCLIP ViT-B/16+ (LAION-400M), shorter edge 240, 225 patch tokens
  (15 x 15, our reading). (3) Windows of 2 x 2 patches (32 px) and 3 x 3 patches (48 px), stride one
  patch; each window is a forward pass on its unmasked patches only; its class token is scored. (4) Per
  pixel, harmonic mean over the windows that cover it, then over scales. (5) Image score = zero-shot
  score of the whole image plus multi-crop. WinCLIP+ (K = 1, 2, 4 normal images): store small-window,
  mid-window and patch-token features as three memories; score by nearest cosine distance; average;
  fuse with the language map (the fusion formula is not given).
- Eq. (1): p(s_i | x) = exp(<f(x), g(s_i)>/tau) / sum_{s in S} exp(<f(x), g(s)>/tau). f = image
  encoder; g = text encoder; <.,.> = cosine similarity; S = {normal, anomalous} prototypes. ascore_0 =
  p(anomalous) is our reading; the paper defines it in words; tau is not given.
- Eq. (2): F^W_ij = f(x (elementwise) w_ij). w_ij = binary window mask; F^W_ij = its class token.
- Eq. (3): pixel score = harmonic mean of the scores of all windows covering the pixel. One confidently
  normal window (score near 0) pulls the pixel toward normal.
- Eq. (4): M_ij = min_{r in R} (1/2)(1 - <F_ij, r>), half the cosine distance to the nearest stored
  normal feature. Eq. (5): average of the three memory maps. Eq. (6): image score = (1/2)(ascore_0 of the
  image + max of the vision map).
- Results: zero-shot MVTec AD image 91.8 / pixel 85.1 / PRO 64.6; VisA 78.1 / 79.6 / 56.8 (Tables 1, 4).
  1-shot WinCLIP+: MVTec AD 93.1 / 95.2; VisA 83.8 / 96.4. Ablation (Table 3): two-class 74.0 -> state
  ensemble 89.8 -> prompt ensemble 90.8 -> multi-crop 91.8. 389.4 ms per image, EC2 G4dn (main Table 8).
- Limitations: logical anomalies need a normal reference; tiny defects; low-shot F1-max below 60% even
  with pixel AUROC above 95%.

Branches after WinCLIP (per the branch summary):
- APRIL-GAN (2305.17382v3; AnomalyCLIP calls it VAND): trained linear layers map CLIP patch features into
  text space; few-shot adds a patch memory. Zero-shot MVTec AD image AUROC 86.1 (Table 3), layers trained
  on the VisA test set.
- AnomalyGPT (2308.15366v4, AAAI 2024): feeds a text-matched anomaly map to a large vision-language model
  through learned prompts; it answers whether and where, with no threshold. 1-shot MVTec AD 94.1 (+-1.1)
  (Table 2), trained on VisA.
- MuSc (2401.16753v1, ICLR 2024): training-free; scores each test patch by how few similar patches it
  finds in the other unlabeled test images. Zero-shot MVTec AD 97.8 (Table 1). It uses the whole test
  set at once, not one image.
- InCTRL (2403.06495v3, CVPR 2024): one CLIP-based model trained on auxiliary data scores residuals
  between a query and a few normal prompts. 2-shot MVTec AD 0.940 (+-0.015) (Table 1), trained on VisA.
- PromptAD (2404.05231v2, CVPR 2024): learns prompts per class from the k normal images only; anomaly
  prompts by suffixes, with an explicit margin. 1-shot MVTec AD 94.6 (+-1.7) (Table 1).

AnomalyCLIP (2310.18961v12; ICLR 2024) as a delta from WinCLIP
- What WinCLIP did and why that hurt (Sec. 5): many hand-written prompts that name the object, and many
  forward passes per image. CoOp-style learned prompts align global object semantics and segment
  badly (33.3 pixel AUROC on MVTec AD, Table 1).
- Key idea: learn two prompts whose only fixed words are "object" and "damaged object", once, on one
  auxiliary dataset, and reuse them on unseen products.
- Mechanism: prompts [V_1..V_12][object] and [W_1..W_12][damaged][object]; 4 learnable tokens replace
  token positions in the first 9 text layers (layers 2-9 is our reading); CLIP ViT-L/14@336px frozen, 518
  px input (37 x 37 patches, our reading); DPAM = V-V attention from layer 6 in a parallel path that
  feeds only the patch tokens; image score P(g_a, f_i); local maps per patch. Adam lr 0.001, batch 8, 15
  epochs. Auxiliary protocol: train on the VisA test split to test MVTec AD, on the MVTec AD test split
  to test everything else.
- Gradient (our reading): both losses depend on the prompts only through the outputs of the frozen text
  encoder, so the gradient passes back through frozen text layers to the learnable words and tokens. The
  vision side has nothing to train.
- Eq. (1): P(g_c, f_i) = softmax over {g_n, g_a} of <g_c, f_i>/tau (a misplaced parenthesis as printed);
  P(g_a, f_i) is the anomaly score.
- Eq. (2): L_total = L_global + lambda sum_{layers} L_local, lambda = 4. L_global = cross-entropy on the
  image-level probability (in words only; no formula). L_local = Focal(Up([S_n, S_a]), S) + Dice(Up(S_n),
  I - S) + Dice(Up(S_a), S): S = ground-truth mask; one Dice term asks the normal map to cover the normal
  region, the other asks the abnormal map to cover the defect.
- Eq. (3): at each of layers 2-9, overwrite the first Q = 4 token positions with fresh learnable tokens
  and discard the outputs at those positions.
- DPAM: attention weights from v_i . v_j instead of q_i . k_j; no new weights; each patch mostly keeps
  its own content.
- Results (zero-shot, 518 px; Table 1; image (AUROC, AP), pixel (AUROC, PRO)): MVTec AD (91.5, 96.2),
  (91.1, 81.4); VisA (82.1, 85.4), (95.5, 87.0). WinCLIP still leads MVTec AD image AUROC (91.8 vs 91.5).
  Ablation (Table 4): DPAM moves pixel metrics only (MVTec AD pixel AUROC 46.8 -> 68.4); the learned
  object-agnostic prompts give the big jump (to 90.8 image), but that step mixes "learn prompts" with
  "drop the name". The agnostic-minus-aware gain is only 0.0 to 1.0 points on MVTec AD and VisA, 4.4 on
  MPDD image AUROC (Fig. 5). Replacing "damaged" by other words barely matters (Table 8). No latency.
- Limitations: still needs an auxiliary dataset with pixel masks; weaker on medical data (Sec. 4.2).

Branches after AnomalyCLIP (per the branch summary):
- AdaCLIP (2407.15795v1, ECCV 2024): static (shared) and dynamic (per-image) learnable prompts inside
  CLIP layers, trained on VisA and ColonDB. Zero-shot MVTec AD image AUROC 89.2 (Table 1).
- UniVAD (2412.03342v3, CVPR 2025): training-free few-shot; segments components with clustering and
  foundation models, then matches patches and component graphs. 1-normal-shot MVTec AD 97.8 (Table 1).

AnomalyDINO (2405.14529v3; WACV 2025 oral) as a delta from PatchCore and the CLIP methods
- What they did and why that hurt: CLIP methods need prompt engineering or meta-training on a related
  dataset; PatchCore's few-shot numbers are low (1-shot 83.4, as WinCLIP ran it) and its coreset
  shrinks a bank that is already tiny.
- Key idea: PatchCore-style patch nearest neighbour on DINOv2 ViT-S/14 patch tokens, tuned for few
  shots. Nothing is trained.
- Mechanism: smaller edge 448 (or 672); a square 448 image gives 32 x 32 patches (our reading); layer
  not stated (our reading: last). Bank = all patches of the k references, enlarged by rotations (angles
  not stated), no coreset. A PCA mask removes test background only for categories that pass a one-time
  masking test (5 of 15 on MVTec AD; all 12 on VisA; Table 8). Cosine distance to the nearest bank
  patch (FAISS); image score = mean of the top 1% patch distances; map = bilinear upsample plus
  Gaussian sigma 4.
- Delta table vs PatchCore: backbone CNN mid-levels with aggregation -> DINOv2 tokens, no aggregation;
  Euclidean -> cosine; coreset -> keep all, add rotations; no background handling -> PCA mask; max
  patch score -> top-1% mean.
- Eq. (1): M = union of the patch features of the reference images. Eq. (2): d_NN(p; M) = min over M of
  d(p, p_ref). Eq. (3): d(x, y) = 1 - <x, y> / (||x|| ||y||). Eq. (4): s = mean of the top 1% of the
  d_NN values ("empirical tail value at risk"); it follows the worst patches but resists one outlier.
- Results (one-class bank, k-shot, mean of 3 runs; baselines copied, mostly from WinCLIP): MVTec AD
  1-shot image AUROC 96.5 (448) / 96.6 (672) vs WinCLIP+ 93.1 (Table 2); 16-shot 98.3 / 98.4. VisA
  1-shot 85.6 / 87.4 vs APRIL-GAN 91.2 (Table 3); APRIL-GAN leads image AUROC at k = 1, 2, ties at k = 4
  (92.6), and AnomalyDINO leads at k = 8, 16 and on every pixel metric. Full-shot (Table 5) MVTec AD 448:
  99.3 / 97.9 / 93.9; 672: 99.5 / 98.2 / 95.0. Top-1% vs max aggregation (Table 10): 96.5 vs 95.0.
  Runtime 60 ms per image (S, 448, 1-shot, A40; Table 9).
- Limitations: logical anomalies (cable swap 50.2, chance level); uninformative references; rotation
  can hide rotated defects; the PCA mask fails on close-ups.

## 5. Speed: desktop vs edge

Desktop and server timings (each with its own protocol):
| Source, loc | What is timed | Device | Protocol | Value |
|---|---|---|---|---|
| EfficientAD Table 17 | EfficientAD-S / -M; re-runs: GCAD, SimpleNet, FastFlow, DSR, PatchCore, AST, S-T, PatchCore-Ens | RTX A6000 (also A5000, V100, 3080, 2080 Ti) | batch 1, mean of 1000 passes after 1000 warm-up, float16, from CPU->GPU transfer until the map is on the CPU | 2.2 / 4.5; 10.7, 12.0, 16.5, 17.2, 32.0, 53.1, 74.7, 147.6 ms |
| EfficientAD Table 1 | throughput, same methods | RTX A6000 | batch 16 | S 614, M 269 img/s |
| PatchCore Table 5 | PatchCore-100% / 10% / 1% | printed "Nvidia Tesla V4" | per image, includes backbone; batch not stated | 0.6 / 0.22 / 0.17 s |
| FastFlow Table 1 | FastFlow and baselines | GTX 1080Ti + Xeon E5-2680 V4 | FPS includes the backbone; "A.d." ms excludes it; batch not stated | WRN50 FastFlow 21.8 FPS, 34 ms A.d. |
| WinCLIP main Table 8 | zero-shot WinCLIP | Amazon EC2 G4dn (GPU not named) | batch not stated | 389.4 ms |
| AnomalyDINO Table 9 | AnomalyDINO-S, 448, 1-shot | NVIDIA A40 | per sample, warm-up, CUDA sync | 60 ms |
| Dinomaly Table 4 | ViT-S / ViT-B / ViT-L | RTX 3090 | throughput, batch 16 | 153.6 / 58.1 / 24.2 img/s |
| MVTec AD 2 Sec. 4.3.3 | PatchCore at high resolution, Rice | RTX 2080 Ti | batch 1, mean of 1000 after warm-up | about 2 s per image |
| PaDiM Table VII | SPADE, PaDiM R18-Rd100, WR50-Rd550 | CPU i7-4710HQ, serial | batch not stated | 7.10 / 0.23 / 0.95 s |

- EfficientAD Fig. 6: the ranking is the same on all five GPUs, except DSR is slightly faster than
  FastFlow on the V100 and the RTX 3080.
- The accuracy that pairs with EfficientAD's latency in Table 1 is the three-collection mean, not MVTec
  AD image AUROC; for an MVTec AD axis use Table 2. All its baselines are its own re-runs.
- ADer Table 1 times are totals for the whole test set on one L40S, not per-image latency. UniAD,
  CutPaste, DRAEM, AnomalyCLIP and Anomalib (compiled paper) report no latency.

Edge timings (`edge.md`):
| Source | Device | Protocol | Latency | Paired accuracy |
|---|---|---|---|---|
| 2605.24251v2 Tab. 4 | Jetson Orin Nano | 30 single-image runs; framework and precision not stated | EfficientAD* 34.7, DNE 60.7, DINOSaur 91.6, PatchCore* (ResNet-18, 224) 119.0, UCAD 956.5 ms; IUF OOM | continual MVTec AD image AUROC: 0.529 (a collapse), 0.500, 0.968, 0.925, 0.798 (Tab. 1) |
| 2606.16119v1 Tab. 4 | Jetson Orin Nano Super, MAXN 25W | TensorRT FP16, mean of 100 after 20 warm-up | EdgeZSAD 32.83 ms; CLIP ViT-L/14 probe 130.26; DINOv2 ViT-L/14 probe 301.33 | EdgeZSAD measured on device, MVTec AD image 91.6 (zero-shot on targets, supervised on a Real-IAD source); the probes are backbone-only and their AUROC is a copied host number of another method |
| 2505.24334v1 Sec. 4.4 | "Jetson NX" / "Jetson AGX" | not stated | KairosAD 211 / 218 ms | none on device; KairosAD is supervised |
| 2407.02968v1 Tab. 5 | Jetson Xavier NX, 30 W | "Avg. Time"; batch not stated | STFPM FP32 52.14 -> TRT FP16 24.98 ms; RD 166.60 -> 19.68 ms | multi-class "Mean AUROC", level not stated: STFPM 0.94 -> 0.94; RD 0.96 -> 0.82 |
| 2603.20288v1 Tab. II | desktop CPU i5-9600K | per test image; precision not stated | PatchCore 345 / -Edge 30 / -Lite 130; PaDiM 22553 / 35 / 24; STFPM 123 / 13; PaSTe 9 ms | MVTec AD image AUROC PatchCore 0.98 / 0.95 / 0.86; PaDiM 0.96 / 0.94 / 0.85 |
| 2410.11591v1 (PaSTe) | none | MACs and MB only | no ms | pixel F1 only |

What can be compared:
- Methods inside one table with one protocol: EfficientAD Table 17 (desktop GPUs); 2605.24251v2 Tab. 4
  (one Orin Nano); 2603.20288v1 Tab. II (one CPU); 2407.02968v1 Tab. 5 (one Xavier NX, FP32 vs TRT).
- Relative claims inside a paper: TensorRT FP16 halves STFPM latency at the same 0.94; RD keeps its
  speed-up but loses accuracy (0.96 -> 0.82).

What cannot be compared:
- No source times PatchCore, PaDiM, EfficientAD, FastFlow, DRAEM, a CLIP method and a DINO method on one
  device. FastFlow has no edge timing; DRAEM has only a copied 159 ms with no device (2505.24334v1).
- The Orin Nano AUROC axis is continual-protocol; EfficientAD* at 0.529 is a collapse, not its normal
  accuracy; its PatchCore is ResNet-18 at 224, not WRN50.
- Latencies across papers mix batch sizes, precisions (float16, FP32, TensorRT FP16), devices and what is
  inside the timer (with or without the backbone and the CPU<->GPU transfers).
- PatchCore-Lite and PaSTe are not Jetson results; KairosAD's Jetson points have no paired accuracy.

## 6. Open caveats

### 6.1 Paper vs code
- PatchCore Eq. 7: the reweighted image score is in the paper; the official code does not implement it
  (image score = plain max of patch scores, k = 1), and b is never stated. Our reading: the map uses
  unweighted patch scores either way.
- PatchCore aggregation: the paper "aggregates" layer3 with layer2; the code average-pools each layer to
  1024 and then the stack to 1024 (no concatenation, no learned projection). psi (128-d, code-only) is
  used only for coreset selection; the coreset starts from 10 random anchors (code-only). faiss
  IndexFlatL2 returns squared distances (our reading; AUROC unchanged).
- DRAEM: beta in [0.1, 1.0] weighting the texture (Eq. 4) vs code texture weight (0.2, 1.0]; random
  Perlin threshold vs fixed 0.5; LR x0.1 at epochs 400 and 600 vs MultiStepLR at 0.8 and 0.9 of the
  epochs with gamma 0.2; rotation (-45, 45) vs (-90, 90) with probability 0.3; 700 epochs vs released
  checkpoints named 800; lambda, s_f and the focal gamma appear only in code (1, 21, 2). README reports
  98.1 / 97.5 / 68.9 for released checkpoints vs the paper's 98.0 / 97.3 / 68.4.
- FastFlow: the loss equation is commented out of the LaTeX; Eq. 3 has no exp, Fig. 2(b) uses exp(s).
- PatchCore README "WR50-baseline" 99.2 / 98.1 / 94.4 vs paper Tables 1-3 (PatchCore-25%) 99.1 / 98.1 / 93.4.

### 6.2 Protocol mismatches to label on every cell
- One-class vs multi-class vs k-shot vs zero-shot vs continual vs supervised (KairosAD, EdgeZSAD source
  training).
- Resolution and backbone: PatchCore 224 (WRN50), FastFlow headline 448 (CaiT), Dinomaly 392 (ViT-B),
  WinCLIP 240 (ViT-B/16+), AnomalyCLIP 518 (ViT-L/14), PNI and ReConPatch 480, AST 768.
- Center crop: PatchCore crops 76.6% of the image; EfficientAD and MVTec AD 2 disable it.
- Test-set tuning: SimpleNet (training duration), anomalib FastFlow (early stopping); EfficientAD with
  early stopping reaches 99.8 on MVTec AD (line 406), which is not its reported protocol.
- Auxiliary data in "zero-shot": AnomalyCLIP, APRIL-GAN, AnomalyGPT and InCTRL train on another
  benchmark's test split; MuSc scores the whole unlabeled test set together.
- EfficientAD Table 1 averages three collections; VisA splits used by later papers are not stated; ADer
  does not state its AU-PRO FPR limit; Anomalib states no settings.

### 6.3 Numbers that conflict across or within papers (both sides; not reconciled)
- SPADE pixel AUROC: 96.0 (SPADE Table 2) vs 96.5 (PaDiM Table III); PaDiM's own per-class SPADE values
  average 96.03, and its "8.8 p.p." drop matches 96.0.
- PaDiM text: EfficientNet-B5 beats every model by "at least 2.6p.p"; Table V gives 2.1 over
  MahalanobisAD (2.6 is the margin over PaDiM-WR50-Rd550). R18-Rd100: 96.7 / 90.5 (Table II, 10 seeds)
  vs 96.7 / 90.1 (Table III).
- PaDiM image AUROC as printed elsewhere: 95.5 (DRAEM Table 1, UniAD separate case), 97.9 (FastFlow
  Table 2), 95.4 (PatchCore re-run, Table 5), 0.891 (Anomalib Table 1); own 95.3 (WR50), 97.9 (B5).
- CutPaste: 95.2 (ResNet-18, Table 1), 96.1 (ensemble; used as UniAD's separate case), 96.6 (B4 pool,
  abstract), 97.1 (B4 level-7; FastFlow Table 2). Training patch 64 x 64 (Sec. 4.2) vs 32 x 32 (App. A.5).
- PatchCore full-shot MVTec AD image AUROC elsewhere: 99.6 / 98.2 (WinCLIP Table 2), 99.0 / 98.1
  (AnomalyCLIP Table 7), 98.7 (EfficientAD re-run), 98.8 (ADer, multi-class), 99.1 (Dinomaly Table 2).
  Inside PatchCore: "a third of all classes are solved perfectly" (Sec. 4.2) vs "six of the 15" (supp. B);
  WRN50-10% 98.9 (Table S6) vs 99.0 (Table 1).
- DRAEM AP margin: "13.4" (Sec. 4.1) vs "13.5" (Conclusion). DRAEM multi-class image AUROC: 88.1 (UniAD
  re-run) vs 54.5 / 55.2 (ADer); pixel 87.2 vs 47.6 / 48.7.
- UniAD: 96.5 / 96.8 (own; 224, 1000 epochs) vs 92.5 / 96.8 image at 100 / 300 epochs (ADer). Size:
  7.48M learnable, 6.46 GFLOPs (UniAD Table A7) vs 24.5M, 3.4G (ADer Table 1; our reading: ADer counts the
  frozen backbone).
- Dinomaly says most multi-class baselines come from ADer, but its UniAD 96.5, DiAD 97.2 and MambaAD
  98.6 equal each paper's own Table 1, not ADer's cells (our reading from comparing the notes).
  Class-separated MVTec AD P-AUROC "99.9" (Table 2) looks like a typo. ViT-S MACs 26.2G (Table A11) vs
  26.3G (Table 4).
- AST 99.2 (own) vs 98.9 (EfficientAD re-run); DSR 98.2 (own) vs 98.1 (re-run).
- FastFlow does not lead pixel AUROC in its own Table 2 (CFlow 98.6 vs 98.5).
- WinCLIP: 91.8 zero-shot image AUROC (with multi-crop, Table 3) vs 90.8 (supp. Table 9); two different
  "Table 8"s (main p. 8, supp. p. 13); supp. Table 8 WinCLIP+ 92.7 / 94.0 / 94.8 vs Table 1 93.1 / 94.4 /
  95.2. As re-reported by VisualAD (EdgeZSAD Table 1): WinCLIP 90.4 / 82.3 / 62.0 and AnomalyCLIP 91.6 /
  91.0 / 81.7 vs their own 91.8 / 85.1 / 64.6 and 91.5 / 91.1 / 81.4.
- AnomalyCLIP v12: "top feature map only" (Sec. 4.1, App. A) vs four blocks in Fig. 2 and a layer axis in
  Fig. 7(d); Fig. 7(b) text says performance drops at M = 9, which is the chosen value.
- MVTec AD 2 Table 1: the PatchCore, EfficientAD, SimpleNet and DSR cells for MVTec AD and VisA equal
  EfficientAD Table 12 (our reading; provenance not stated); its EfficientAD row mixes -M and -S.
- EfficientAD latency: Table 1 rounded (e.g. 148 ms) vs Table 17 one decimal (147.6 ms) for the same runs.
- VisA test split: ADer "962 test anomaly, 1,200 test normal" vs 1,200 anomalous in the VisA paper and
  962 normal / 1200 anomalous in AnomalyCLIP Table 6.
- MVTec AD: "over 70" defect types (abstract) vs 73; "almost 1900" regions vs 1888. STC: 13 scenes
  (PaDiM) vs 12 (SPADE).
- Edge: KairosAD Table 2 latency columns look shifted vs Table 1; Hazelnut 99.17 vs 100; AGX (218 ms)
  slower than NX (211 ms). Xavier NX paper: RD 0.95 (Tab. 2) vs 0.96; 166.60 ms vs "66.60" (Tab. 7); US
  INT-8 size 11.8 vs 10.4 MB; "256-core" vs "384-core". PaSTe text 31.1 MB for PaDiM vs 49.4 MB in Tab. 3;
  "4-fold" vs about 40x. PatchCore-Lite peak memory 0.02 to 0.03 MB looks implausible (as printed).

### 6.4 Missing sources
- MVTec LOCO IJCV paper (`sources/mvtec-loco/` is empty): image counts, the fifth scenario name and the
  sPRO formula are unknown. Ask the user to save the PDF to `sources/mvtec-loco/paper.pdf`.
- MVTec AD IJCV 2021 extension: credited with AU-PRO and with the VisA quote (Student-Teacher 92.2% AU-ROC
  vs 59.9% AU-PR); not in `sources/`.
- EfficientAD `per_scenario_results.json` (full-range pixel AUROC): not in `sources/`.
- Branch papers (19 in `branches-a.json`, 15 in `branches-b.json`): one claim each; no equations, shapes
  or limitations were extracted. AnomalyCLIP and EfficientAD code were not read; AnomalyDINO rotation
  angles are code-only; Anomalib's throughput table is commented out of the LaTeX and not citable.
- Claims tagged `our_reading` need the check-page lint taught, or must stay marked uncertain (resume notes).
