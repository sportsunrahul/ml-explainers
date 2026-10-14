# PatchCore -- Towards Total Recall in Industrial Anomaly Detection

Chapters: ch-patchcore and the interlude ch-patchcore-pipeline. Lens: method. PatchCore trains
nothing. "Fit" = run a frozen ImageNet backbone over the normal images, store patch features, thin
them with a greedy coreset. No loss, no optimizer, no epochs, no gradient anywhere.

Sources read: `sources/patchcore/src/main.tex`, `supplementary_arxiv.tex`, `tables/*.tex`,
`figures/*.tex`; `paper.pdf` (18 pages) for equation numbers, page numbers and figures;
`abs.html`. Official repo (code-only values): `sources/patchcore/README.md` plus three files
fetched from `amazon-science/patchcore-inspection` main: `sources/patchcore/code/{patchcore,common,sampler}.py`
(read only, not run). Training image counts: `sources/mvtec-ad/paper.pdf`, Table 1.

## 1. Identity
- Title: Towards Total Recall in Industrial Anomaly Detection.
- arXiv 2106.08265, version read: v2 (5 May 2022). v1 submitted 15 Jun 2021 (abs.html). PDF built
  8 May 2022.
- Authors: Karsten Roth (Univ. Tuebingen; internship at Amazon AWS) ... Peter Gehler (Amazon AWS).
  Six authors: Roth, Pemula, Zepeda, Schoelkopf, Brox, Gehler.
- Venue: CVPR 2022 (abs.html comment "Accepted to CVPR 2022"; main.tex `\confYear{2022}`).
- Repo: the abstract prints `github.com/amazon-research/patchcore-inspection`; the live repo is
  `github.com/amazon-science/patchcore-inspection` (the README fetched from there).

## 2. Core problem
Cold-start anomaly detection: fit a model with nominal (defect-free) images only, then detect and
localize defects at test time, for many product categories without per-class hand tuning. The
paper wants high recall of nominal context at test time, low ImageNet bias, and fast inference.

## 3. What prior methods did and why that hurt (per the paper, Sec. 1, 2, 3.1, 3.2)
- DN2 / SPADE use the deepest feature level: loses localized nominal information, and deep
  ImageNet features are biased toward natural-image classification (Sec. 3.1).
- SPADE's full patch memory is too large to search, so it preselects K nearest training images
  with a weak global (last-layer averaged) image-level kNN; low-resolution, ImageNet-biased, can
  hurt detection and localization (Sec. 3.2).
- PaDiM: one Gaussian per patch position, Mahalanobis distance. Limits each test patch to the
  statistics of its own position, so it relies on image alignment and needs equal train/test image
  shapes (Sec. 2, last paragraph).
- Random subsampling of a big memory bank loses coverage of the nominal feature space (Sec. 3.2,
  Fig. 3, Fig. 5).

## 4. The one key idea
Store every locally aggregated mid-level patch feature of every normal image in one memory bank
that any test patch can search (position-free), then keep only a greedy coreset that preserves
the bank's coverage (minimax facility location), so nearest-neighbor search stays fast. Anomaly
score = distance from the worst test patch to its nearest stored normal patch.

## 5. Mechanism, step by step
Fit (no gradients; backbone frozen, `torch.no_grad()` in code):
1. Resize each normal image to 256, center crop to 224 (Sec. 4.1). No augmentation.
2. Forward through ImageNet-pretrained WideResNet50 (supp. A); keep the final outputs of blocks 2
   and 3 (`layer2`, `layer3`) (supp. A; Sec. 4.4.1 "2 + 3, which is chosen as the default").
3. Local neighbourhood aggregation per layer: for each position (h, w), aggregate the p x p
   neighbourhood (p = 3, Eq. 1-2) with adaptive average pooling; stride s = 1 keeps every
   position (Eq. 3).
4. Bilinearly rescale the layer-3 patch collection to the layer-2 grid so the two collections
   have the same size, and aggregate each layer-3 patch with its layer-2 partner (Sec. 3.1).
   Result: one d-dimensional vector per layer-2 position.
5. Memory bank M = union of all patch vectors of all training images (Eq. 4).
6. Greedy coreset (Algorithm 1): project with a random linear map psi to d* < d dimensions for
   the distance computations only; repeatedly add the bank element whose distance to the current
   coreset is largest; stop at l elements (25 %, 10 % or 1 % of |M|). Store the original
   d-dimensional vectors of the chosen elements (M <- M_C).
Test:
7. Same preprocessing and feature path for the test image -> one vector per patch.
8. Nearest neighbor in M (faiss) for every test patch -> patch score (Eq. 6 inner argmin).
9. Image score: maximum patch score s* (Eq. 6), then reweighting by the neighbourhood of m* in M
   (Eq. 7). Paper: "more robust than just the maximum patch distance".
10. Segmentation: put the patch scores back on their grid, upscale bilinearly to input
    resolution, smooth with a Gaussian, sigma = 4 ("did not optimize this parameter") (Sec. 3.3).

Code-only differences (official repo, `code/`), important for the stepper:
- Image score in code = plain max over patch scores (`PatchMaker.score`: repeated `torch.max`),
  where each patch score = mean distance to `anomaly_scorer_num_nn` nearest bank vectors
  (`NearestNeighbourScorer.predict`, README sets 1). Eq. 7 reweighting is NOT implemented in
  the three official files read; `b` appears nowhere. code_only.
- faiss `IndexFlatL2` is used; our reading: its `search` returns squared L2 distances, so code
  patch scores are squared distances (monotone; AUROC unchanged, map shape slightly changes).
- Aggregation in code: `torch.nn.Unfold(kernel 3, stride 1, padding 1)` per layer -> C x 3 x 3
  per position; layer-3 unfolded patches are bilinearly interpolated 14x14 -> 28x28
  (`align_corners=False`); each layer vector (C*9 values) is `adaptive_avg_pool1d`-ed to
  `pretrain_embed_dimension` = 1024 (`MeanMapper`); the 2 x 1024 stack is `adaptive_avg_pool1d`-ed
  to `target_embed_dimension` = 1024 (`Aggregator`). No learned projection.
- Our reading of that arithmetic (PyTorch adaptive pooling windows; not run): layer 3 has
  1024*9 = 9216 values -> 1024, windows of exactly 9 = the 3x3 mean of one channel; layer 2 has
  512*9 = 4608 -> 1024, windows of 5 values within one channel (two half-neighbourhoods,
  overlapping at the centre); the final 2048 -> 1024 averages adjacent pairs. So the final
  vector is 512 dims from layer 2 (3x3 mean per channel, centre double-weighted) followed by 512
  dims from layer 3 (mean of channel pairs 2c, 2c+1, each a 3x3 mean).
- Coreset in code (`ApproximateGreedyCoresetSampler`, README `approx_greedy_coreset`):
  psi = `torch.nn.Linear(D, 128, bias=False)` with default (random, untrained) init;
  10 random starting points; initial "distance to coreset" = mean distance to those 10 points
  (they are not added to the coreset); then loop argmax / min-update; coreset size
  `int(len(features) * percentage)` (floor). Avoids the N x N matrix. code_only.
- Segmentation in code: `F.interpolate(..., mode="bilinear", align_corners=False)` to 224x224,
  then `scipy.ndimage.gaussian_filter(sigma=4)` (`RescaleSegmentor.smoothing = 4`).

## 6. Equations (TeX verbatim from main.tex; numbers from the v2 PDF)
Custom macros: `\DeclareMathOperator*{\argmax}{arg\,max}`, `\DeclareMathOperator*{\argmin}{arg\,min}`.
Only Eq. 1, 3, 5, 7 carry `\label`s (`eq:neighbourhood`, `eq:patchcollection`, `eq:coreset`,
`eq:image_level_anomaly_score`); Eq. 2, 4, 6, 8 have none.

Eq. 1 (p. 3, Sec. 3.1), neighbourhood:
```
\mathcal{N}_p^{(h, w)} = \{(a, b) | &a \in [h - \lfloor p/2 \rfloor, ..., h + \lfloor p/2 \rfloor], \\&b \in  [w - \lfloor p/2 \rfloor, ..., w + \lfloor p/2 \rfloor]\},
```
Paper: "uneven patchsizes p (corresponding to the neighbourhood size considered)". Gloss: the p x p
grid cells centred on (h, w) of one feature map; p = 3 gives the 3x3 block.

Eq. 2 (p. 3), locally aware feature:
```
\phi_{i, j}\left(\mathcal{N}_p^{(h, w)}\right) = f_\text{agg}\left(\{\phi_{i, j}(a, b) | (a, b) \in \mathcal{N}_p^{(h, w)}\}\right),
```
Paper: "f_agg some aggregation function ... For PatchCore, we use adaptive average pooling. This is
similar to local smoothing over each individual feature map, and results in one single
representation at (h, w) of predefined dimensionality d". Gloss: phi_{i,j} = feature map of image
x_i at hierarchy j (c* x h* x w*); phi_{i,j}(a, b) = the c*-dim vector at cell (a, b); f_agg pools
the p^2 vectors of the neighbourhood into one d-dim vector. The paper never states d (code: 1024).

Eq. 3 (p. 3), patch collection:
```
\mathcal{P}&_{s, p}(\phi_{i, j}) = \{ \phi_{i, j}(\mathcal{N}_p^{(h, w)}) | \\&h,w \text{ mod } s = 0, h < h^*, w < w^*, h, w \in \mathbb{N} \},
```
Gloss: all locally aware features of one map, taken every s cells (s = 1 by default). As printed the
bounds are strict (`h < h^*`), while the text says h in {1, ..., h*}; do not reconcile.

Eq. 4 (p. 4), memory bank:
```
\mathcal{M} = \bigcup_{x_i\in\mathcal{X}_N} \mathcal{P}_{s, p}(\phi_j(x_i)).
```
Gloss: X_N = all nominal training images; M = every patch vector of every training image.

Eq. 5 (p. 4, Sec. 3.2), coreset objective ("minimax facility location"):
```
\mathcal{M}_C^* = \argmin_{\mathcal{M}_C\subset\mathcal{M}} \max_{m\in\mathcal{M}}\min_{n\in\mathcal{M}_C} \left\Vert m - n \right\Vert_2.
```
Paper: "to ensure approximately similar coverage of the M-coreset M_C in patch-level feature space
as compared to the original memory bank M"; exact solution "NP-Hard", so "the iterative greedy
approximation suggested in [Sener & Savarese]". Gloss: pick the subset M_C (of the target size)
that minimizes the worst-case distance from any bank element m to its nearest kept element n,
i.e. the coverage radius. This radius is what `viz-patchcore` draws.

Algorithm 1 (p. 5), "PatchCore memory bank", verbatim core lines:
```
\For{$i\in[0, ..., l-1]$}{
	$m_i \leftarrow \underset{m\in\mathcal{M}-\mathcal{M}_C}{\argmax}\underset{n\in\mathcal{M}_C}{\min} \left\Vert \psi(m) - \psi(n) \right\Vert_2$\\
	$\mathcal{M}_C \leftarrow \mathcal{M}_C \cup \{m_i\}$\\
}
$\mathcal{M}\leftarrow\mathcal{M}_C$\\
```
Input: "Pretrained phi, hierarchies j, nominal data X_N, stride s, patchsize p, coreset target l,
random linear projection psi". psi: `\psi: \mathbb{R}^d \rightarrow \mathbb{R}^{d^*}` with
`d^* < d`, justified by Johnson-Lindenstrauss (Sec. 3.2). d* is not given in the paper
(code: 128). Gloss: each step adds the bank element farthest from everything chosen so far.
psi is used only inside the distance; M_C holds the original vectors. The algorithm starts with
M_C = {} and does not say how the first element is picked (code: 10 random anchors, see Sec. 5).
Naming: "PatchCore-n%" = memory bank subsampled to n %; "PatchCore-1% a 100x times reduction".

Eq. 6 (p. 5, Sec. 3.3), worst patch and its raw score:
```
m^{\text{test}, *}, m^* &= \argmax_{m^\text{test} \in \mathcal{P}(x^\text{test})}\argmin_{m \in \mathcal{M}} \left\Vert m^\text{test} - m \right\Vert_2\\
s^* &= \left\Vert m^{\text{test},*} - m^* \right\Vert_2.
```
Paper: "the maximum distance score s* between test patch-features in its patch collection
P(x^test) ... to each respective nearest neighbour m* in M". Gloss: for every test patch find its
nearest bank vector; m^{test,*} = the test patch whose nearest-neighbour distance is largest; m* =
that nearest bank vector; s* = that distance. The per-patch nearest-neighbour distances also form
the segmentation map.

Eq. 7 (p. 5), reweighted image score:
```
s = \left(1-\frac{\exp{\left\Vert m^{\text{test}, *} - m^* \right\Vert_2}}{\sum_{m \in \mathcal{N}_b(m^*)} \exp{\left\Vert m^{\text{test},*} - m \right\Vert_2}}\right) \cdot s^*,
```
Paper: "we use scaling w on s* to account for the behaviour of neighbour patches: If memory bank
features closest to anomaly candidate m^{test,*}, m*, are themselves far from neighbouring
samples and thereby an already rare nominal occurrence, we increase the anomaly score", with
"N_b(m*) the b nearest patch-features in M for test patch-feature m*". "We found this
re-weighting to be more robust than just the maximum patch distance." Gloss: a softmax-style
weight over the b bank neighbours of m*. b is not given in the paper or README, and the official
code does not implement this step. Our reading: m* is the nearest bank element to m^{test,*}, so
each denominator term is >= the numerator; the ratio is <= 1/b and the weight lies in
[1 - 1/b, 1). The weight is near 1 - 1/b when m* sits in a dense cluster (neighbours as close as
m*) and near 1 when m* is isolated. Whether N_b(m*) contains m* itself is not stated. Note: the
paper text says "Eq. 7 (first line) requires ... the argmax-operation"; the argmax line is Eq. 6
(as printed). Our reading: Eq. 7 reweights only the image score and the map uses the unweighted
patch scores (the paper does not say; the code map is unweighted, but the code skips Eq. 7 entirely).

Eq. 8 (p. 7, Sec. 4.4.2), only for the "learned proxies" subsampling baseline (not PatchCore):
```
\mathcal{L}_\text{rec}(m_i) &= \left\Vert m_i - \sum_{p_k\in\mathcal{P}}\frac{e^{\left\Vert m_i - p_k\right\Vert_2}}{\sum_{p_j\in\mathcal{P}} e^{\left\Vert m_i - p_j\right\Vert}} p_k\right\Vert_2^2,
```
Gloss: |P| = p_target * |M| proxies learned to reconstruct each bank vector m_i as a soft
combination. It loses to greedy coreset (Fig. 5).

## 7. Config / setup (method lens)
| Item | Value | Source |
|---|---|---|
| Trainable weights | none; backbone frozen, fit = feature extraction + coreset | Sec. 1 "without requiring training on the dataset at hand"; code `torch.no_grad()` |
| Backbone | WideResNet50, ImageNet-pretrained (torchvision / timm) | supp. A |
| Layers | final outputs of blocks 2 and 3 (`layer2`, `layer3`) | supp. A; README `-le layer2 -le layer3` |
| Input | resize 256x256, center crop 224x224; no augmentation | Sec. 4.1 (README `--resize 256 --imagesize 224`) |
| Neighbourhood p | 3 | Sec. 4.4.1; README `--patchsize 3` |
| Stride s | 1 | Sec. 3.1 |
| Aggregation | adaptive average pooling | Sec. 3.1 |
| d (per-layer and final dim) | 1024 and 1024 | code_only: README `--pretrain_embed_dimension 1024 --target_embed_dimension 1024` |
| Coreset method | greedy (k-center style), minimax facility location | Sec. 3.2, Alg. 1 |
| psi output dim d* | 128 | code_only: `sampler.py` `dimension_to_project_features_to=128` |
| Coreset starting points | 10 | code_only: `sampler.py` |
| Coreset percentages | 25 %, 10 %, 1 % (tables); 100 % in Table 5; Fig. 5 x-axis is log subsampling percentage (ticks 10^0, 10^-2) | Tables 1-5, Fig. 5 |
| README default | `-p 0.1` (10 %) for WR50; `-p 0.01` for the ensemble example | code_only: README |
| kNN for patch score | 1 | code_only: README `--anomaly_scorer_num_nn 1`; paper Eq. 6 uses the single nearest |
| b (Eq. 7) | not stated | paper, README, code all silent |
| Upsampling | bilinear to input resolution | Sec. 3.3 |
| Gaussian smoothing | sigma = 4, not tuned | Sec. 3.3; code `smoothing = 4` |
| NN search | faiss (exact `IndexFlatL2`; IVFPQ variant in Table 5) | supp. A; code `common.py` (IVFPQ: 512 centroids, 64 sub-quantizers, 8 bits, code_only) |
| Hardware | "Nvidia Tesla V4 GPUs" (as printed; no such product name, do not correct) | supp. A |
| Software | Python 3.7, PyTorch (README: Python 3.8) | supp. A; README |
| Batch size at test | not stated | -- |

Tensor shapes for the stepper (our reading; WRN-50-2 at 224: layer2 = stride 8, layer3 = stride 16;
SPADE `sources/spade/src/main.tex` line 153 states the block grids 56x56, 28x28, 14x14 for the same
backbone and crop; channel counts 512 and 1024 are the standard torchvision `wide_resnet50_2`
outputs, not printed in any source here):
| Stage | Shape per image |
|---|---|
| input | 3 x 224 x 224 |
| layer2 | 512 x 28 x 28 |
| layer3 | 1024 x 14 x 14 |
| local 3x3 neighbourhoods (code: unfold, pad 1) | layer2: 784 x 512 x 3 x 3; layer3: 196 x 1024 x 3 x 3 |
| layer3 upsampled to layer2 grid | 784 x 1024 x 3 x 3 |
| per-layer pooled (code MeanMapper) | 784 x 2 x 1024 |
| final patch features (d = 1024) | 784 x 1024 (a 28 x 28 grid) |
| memory bank M (category with n training images) | (n * 784) x 1024 |
| psi(M) used only for coreset selection | (n * 784) x 128 |
| coreset M_C (stored) | floor(n * 784 * pct) x 1024 |
| test: patch features | 784 x 1024 |
| test: NN distances | 784 (28 x 28) |
| image score | 1 scalar (max of 784; paper: then Eq. 7) |
| anomaly map | 28 x 28 -> bilinear -> 224 x 224 -> Gaussian sigma 4 -> 224 x 224 |

Memory bank size per MVTec AD category (our reading). n = training images from MVTec AD
(`sources/mvtec-ad/paper.pdf`, Table 1, "# Train"; total 3629, matching PatchCore's "5354 images,
1725 of which are in the test set"). Bank = n * 784 vectors (Eq. 4 with s = 1, 28 x 28 grid).
Coreset counts use the code's floor rule `int(N * pct)`. MB assumes float32 x 1024 (4096 bytes per vector).
| Category | n | Full bank | 25 % | 10 % | 1 % | Full MB | 10 % MB |
|---|---|---|---|---|---|---|---|
| bottle | 209 | 163,856 | 40,964 | 16,385 | 1,638 | 671.2 | 67.1 |
| cable | 224 | 175,616 | 43,904 | 17,561 | 1,756 | 719.3 | 71.9 |
| capsule | 219 | 171,696 | 42,924 | 17,169 | 1,716 | 703.3 | 70.3 |
| carpet | 280 | 219,520 | 54,880 | 21,952 | 2,195 | 899.2 | 89.9 |
| grid | 264 | 206,976 | 51,744 | 20,697 | 2,069 | 847.8 | 84.8 |
| hazelnut | 391 | 306,544 | 76,636 | 30,654 | 3,065 | 1255.6 | 125.6 |
| leather | 245 | 192,080 | 48,020 | 19,208 | 1,920 | 786.8 | 78.7 |
| metal nut | 220 | 172,480 | 43,120 | 17,248 | 1,724 | 706.5 | 70.6 |
| pill | 267 | 209,328 | 52,332 | 20,932 | 2,093 | 857.4 | 85.7 |
| screw | 320 | 250,880 | 62,720 | 25,088 | 2,508 | 1027.6 | 102.8 |
| tile | 230 | 180,320 | 45,080 | 18,032 | 1,803 | 738.6 | 73.9 |
| toothbrush | 60 | 47,040 | 11,760 | 4,704 | 470 | 192.7 | 19.3 |
| transistor | 213 | 166,992 | 41,748 | 16,699 | 1,669 | 684.0 | 68.4 |
| wood | 247 | 193,648 | 48,412 | 19,364 | 1,936 | 793.2 | 79.3 |
| zipper | 240 | 188,160 | 47,040 | 18,816 | 1,881 | 770.7 | 77.1 |
| all 15 | 3629 | 2,845,136 | -- | -- | -- | -- | -- |

## 8. Results worth remembering
Protocol for all MVTec rows unless noted: one model per category (one-class), full-shot (all
training images), 224 px crop, WRN-50, class-averaged. VisA: not in this paper (VisA, 2207.14315,
came later); take PatchCore-on-VisA numbers from the VisA paper or later papers.

MVTec AD, PatchCore (p. 6-7):
| Variant | Image AUROC | Pixel AUROC | PRO | Loc |
|---|---|---|---|---|
| PatchCore-25% | 99.1 | 98.1 | 93.4 | Tables 1, 2, 3 |
| PatchCore-10% | 99.0 | 98.1 | 93.5 | Tables 1, 2, 3 |
| PatchCore-1% | 99.0 | 98.0 | 93.1 | Tables 1, 2, 3 |
| PatchCore-100% | 99.1 | 98.0 | 93.3 | Table 5 only |
| PatchCore-100% + IVFPQ | 98.0 | 97.9 | 93.0 | Table 5 |
- Table 1 errors 0.9 / 1.0 / 1.0; misclassifications at F1-optimal threshold 42 / 47 / 49 (of 1725
  test images). Supp. C.4: 19 false positives + 23 false negatives (variant not named; 19 + 23 = 42
  matches the PatchCore-25% count).
- "a reduction from an error of 2.1% (PaDiM) to 0.9% for PatchCore-25% means a reduction of the
  error by 57%" (Sec. 4.2; 2.1 is PaDiM*, 97.9).
- Sec. 4.2 says "a third of all classes are solved perfectly"; supp. B says PatchCore-25% "solves
  six of the 15". As printed.

Table 4 (p. 7), PatchCore-1% with bigger setups (image AUROC / pixel AUROC / PRO):
- DenseN-201 & RNext-101 & WRN-101 (2+3), image size 320: 99.6 / 98.2 / 94.9 (the abstract's
  "up to 99.6%").
- WRN-101 (2+3), image size 280: 99.4 / 98.2 / 94.4.
- WRN-101 (1+2+3), image size 280: 99.2 / 98.4 / 95.0.
README (code_only): "WR50-baseline" 99.2% / 98.1% / 94.4%; "Ensemble" 99.6% / 98.2% / 94.9%.

Inference time, Table 5 (p. 7): "Mean inference time per image on MVTec AD", includes backbone
forward (Sec. 4.3); hardware per supp. A "Nvidia Tesla V4 GPUs"; batch size not stated.
| Method | Time (s) | Scores (img AUROC, pixel AUROC, PRO) |
|---|---|---|
| PatchCore-100% | 0.6 | (99.1, 98.0, 93.3) |
| PatchCore-10% | 0.22 | (99.0, 98.1, 93.5) |
| PatchCore-1% | 0.17 | (99.0, 98.0, 93.1) |
| PatchCore-100% + IVFPQ | 0.2 | (98.0, 97.9, 93.0) |
| SPADE (re-implemented, WRN50) | 0.66 | (85.3, 96.6, 91.5) |
| PaDiM (re-implemented, WRN50) | 0.19 | (95.4, 97.3, 91.8) |

Baselines and provenance:
- Copied from the original papers (cited; Table 1 caption: "We did not have individual anomaly
  scores for competing methods"): Table 1 image AUROC SPADE 85.5, PatchSVDD 92.1, DifferNet 94.9,
  PaDiM 95.3, Mah.AD 95.8, PaDiM* 97.9 ("result from [PaDiM] with problem-specific backbone
  selection"; supp. Table S1 caption: "which we could not reproduce"). Table 2 pixel AUROC:
  AE_SSIM 87, gamma-VAE + grad. 88.8, CAVGA-R_w 89, PatchSVDD 95.7, SPADE 96.0, PaDiM 97.5.
  Table 3 PRO: AE_SSIM 69.4, Student 85.7, SPADE 91.7, PaDiM 92.1.
- Re-run by PatchCore authors with WRN50: Table 5 rows above (SPADE 85.3 / 96.6 / 91.5; PaDiM
  95.4 / 97.3 / 91.8), Fig. 6 and Table S5 low-shot rows.

Ablations (text values only; plots not read for numbers):
- Fig. 4 (p. 6): neighbourhood size p = 1..7, optimum motivates p = 3; hierarchies 1, 2, 3, 1+2,
  2+3, 1+2+3; "features from hierarchy level 2 can already achieve state-of-the-art performance,
  but benefit from" 2 + 3 (default).
- Stride instead of coreset: s = 2 -> image AUROC 97.6%, s = 3 -> 96.8% (Sec. 4.4.2).
- Bank usage at test time: "less than 30%" of the full bank is used; after coreset to 1% "nearly
  95%" (Sec. 4.4.2).
- Fig. 5: greedy coreset beats random and learned proxies; no subsampling "comparable to a
  coreset-reduced memory bank that is two orders of magnitudes smaller"; joint performance
  partly increases between around 50% and 10%.
- Fig. S4: image sizes 288, 360, 448 with P in 3, 5, 7, 9: detection "slightly increased" then
  saturates; segmentation consistently increases.
- Table S6 backbones (p. 17), image AUROC at 10% / 1%: ResNet50 99.0 / 98.7; WideResNet50 98.9 /
  99.0; ResNet101 98.6 / 98.7; WideResNet101 99.1 / 99.0; ResNeXt101 98.9 / 98.7. Note: WRN50-10%
  is 98.9 here but 99.0 in Table 1 (as printed).

Low-shot, Table S5 (p. 16), image AUROC, mean +- std, shots 1 / 2 / 5 / 10 / 16 / 20 / 50
(retained 0.4 / 0.8 / 2.1 / 4.1 / 6.6 / 8.3 / 21 %); protocol one-class, k-shot normals, 224 px:
- SPADE (re-run): 71.6, 73.4, 75.2, 77.5, 78.9, 79.6, 81.1
- PaDiM (re-run): 76.1, 78.9, 81.0, 83.2, 85.5, 86.5, 90.1
- DifferNet (copied, 16-shot only): 87.3
- PatchCore-10: 83.4, 86.4, 90.8, 93.6, 95.4, 95.8, 97.5
- PatchCore-25: 84.1, 87.2, 91.0, 93.8, 95.5, 95.9, 97.7
- PatchCore-10 pixel AUROC: 92.0, 93.1, 94.8, 96.2, 96.8, 96.9, 97.8; PRO: 82.4, 85.1, 88.7,
  90.9, 91.8, 92.0, 93.0.
Other benchmarks, Table 6 (p. 8): mSTC pixel AUROC PatchCore-10 91.8 (hierarchies 3 and 4) vs
PaDiM 91.2, SPADE 89.9; MTD image AUROC PatchCore-10 97.9 vs DifferNet 97.7.

## 9. Limitations the paper states
- "applicability is generally limited by the transferability of the pretrained features
  leveraged. This can be addressed by merging the effectiveness of PatchCore with adaptation of
  the utilized features" (Conclusion). Later work in this survey's branch list targets exactly
  this: CFA 2206.04325 (coupled-hypersphere feature adaptation), ReConPatch 2305.16713; PNI
  2211.12634 adds position and neighbourhood information. Not verified from those sources here.
- Remaining errors (supp. C.4): label ambiguity, high nominal variance, fine-grained defects that
  need higher resolution, one defect cropped out by the center crop.
- Full-bank search cost grows with |X_N| (Sec. 3.2), the reason for the coreset.

## 10. Outline feedback (ch-patchcore, ch-patchcore-pipeline)
1. Stepper stage "random projection -> greedy coreset": psi is used only to compute distances
   during selection; the stored coreset keeps the 1024-d vectors (Alg. 1, code). Draw psi as a
   side branch of the coreset step, not as a stage that shrinks the bank.
2. Stepper stage "reweighting": Eq. 7 is in the paper, but the official code does not implement
   it (image score = max patch distance, k = 1), and b is never given. Label it "paper Eq. 7; not
   in the official code; b not stated" or show it as optional. Our reading: the anomaly map does not
   use it (the paper is silent; the code map is unweighted).
3. "upsample and concatenate": the paper "aggregates" layer-3 patches with layer-2 patches after
   bilinear rescaling; the code averages them with adaptive average pooling to 1024 (no concat,
   no learned projection). Suggest "upsample layer 3, average-pool both layers to d = 1024".
   Also, "projection to the target dimension" in the ch-patchcore walkthrough is pooling, not a
   learned or random projection; d = 1024 is code_only.
4. Order in the code: 3x3 neighbourhood aggregation happens per layer at native resolution, then
   layer 3 is upsampled 14 -> 28. The outline order (aggregation, then upsample) matches.
5. Coreset percentages: tables use 25 %, 10 %, 1 % (100 % only in Table 5). README default 10 %.
   The viz slider can show the bank sizes from the table above.
6. The `viz-patchcore` coverage radius is exactly the Eq. 5 objective; the code seeds the greedy
   loop with mean distance to 10 random points, the paper does not say how it starts.
7. Scoreboard: no VisA number in this paper; PaDiM/SPADE have two different numbers each
   (copied vs re-run); label which. Inference time 0.17 s (PatchCore-1%) is per image on the
   paper's "Tesla V4" GPU, batch size unstated; ch-speed should say so.
8. Equation numbers for v2: Eq. 1 neighbourhood, 2 aggregation, 3 patch collection, 4 memory bank,
   5 coreset objective, 6 worst-patch score, 7 reweighted image score, Algorithm 1 greedy coreset.
