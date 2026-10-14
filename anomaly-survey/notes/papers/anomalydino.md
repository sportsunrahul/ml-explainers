# AnomalyDINO (2405.14529v3)

## 1. Identity
- Title: "AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2".
- arXiv 2405.14529. Version read: v3 (2025-03-13). v1: 2024-05-23. v2: 2024-09-12 (abs.html submission history).
- Authors: Simon Damm ... Asja Fischer (4 authors; Ruhr University Bochum, University of Hamburg).
- Venue: WACV 2025 (Oral) (abs.html comments; LaTeX footnote says WACV 2025).
- Repo: https://github.com/dammsi/AnomalyDINO (LaTeX line 151).
- Sources read: `src/arxiv_paper_v3.tex`, `src/tables/*.tex`, PDF page 7 (Fig. 3 layout).

## 2. Core problem
Few-shot industrial anomaly detection: given k in {1, 2, 4, 8, 16} normal reference images of one category, detect and
localize defects. Recent few-shot leaders are vision-language (CLIP) methods with prompt engineering or meta-training on
a related dataset. The paper asks whether strong visual features alone (DINOv2) are enough.

## 3. What prior methods did and why that hurt (per the paper)
- CLIP-based methods (WinCLIP, APRIL-GAN, ADP, AnomalyCLIP) need "significant prompt engineering" (APRIL-GAN uses
  35 x 7 normal prompts) or fine-tuning of prompt embeddings; APRIL-GAN and AnomalyCLIP need meta-training on a related
  dataset (train on VisA to test MVTec AD and vice versa) (Sec. 2, lines 190-196, 234-236).
- Patch kNN methods (SPADE, PatchCore) used ImageNet-supervised CNN features; their few-shot numbers are low
  (PatchCore 1-shot MVTec image AUROC 83.4, Table 2). PatchCore shrinks the memory bank for speed (coreset), which is the
  wrong goal when the bank is tiny (line 313).
- Feature engineering for CNN kNN (which layers, how to aggregate) is a design burden (line 251-252).

## 4. The one key idea
PatchCore-style patch nearest-neighbor scoring, but on DINOv2 ViT-S/14 patch tokens, tailored to few-shot: no coreset,
add rotation augmentation of the few references, mask the test background with DINOv2's own PCA, and score the image by
the mean of the top 1% patch distances. Nothing is trained.

## 5. Mechanism, step by step
1. Resize so the smaller edge is 448 (default) or 672 px; any multiple of 14 works (line 252, 360). Our reading: 448/14 =
   32, so a square 448 image gives 32 x 32 = 1,024 patches (a commented-out footnote in the LaTeX, line 361, says the
   same; not in the rendered paper). The paper says 200 to 1000 patches per image after masking (line 1235).
2. Feature extractor f = DINOv2 ViT-S (21M params), patch size 14 x 14. Which layer's tokens are used is not stated; the
   paper says DINOv2 makes the layer choice "unnecessary" (line 251). Our reading: final-layer patch tokens. Embedding
   dim not stated in the paper (our reading: 384 for ViT-S).
3. Build memory bank M from the k reference images: all patch features (Eq. 1). Default "agnostic" preprocessing always
   augments the references with rotations (line 349). The rotation angles are NOT stated in the paper (code-only if
   needed). No coreset subsampling; the paper notes it is advisable only if M grows large (footnote, line 483).
4. Zero-shot masking: threshold the first PCA component of the DINOv2 patch features to separate object from background
   (line 323), plus a center-crop rule (object should dominate the center crop, line 1006), dilation and morphological
   closing (line 329). WHEN: decided once per category by a "masking test" on the first reference sample (line 326-328);
   textures are never masked (line 331); masking is applied only to the test samples, not to M (line 1007). Outcome
   (Table 8): MVTec AD masks 5 of 15 categories (Capsule, Hazelnut, Pill, Screw, Toothbrush); 5 textures skipped (T),
   5 objects failed the test (MT: Bottle, Cable, Metal nut, Transistor, Zipper). VisA masks all 12.
5. For each kept test patch p: d_NN(p; M) = min cosine distance to M (Eq. 2, 3). GPU nearest-neighbor search (FAISS,
   cited as johnson2019billion).
6. Image score: q = mean of the 1% highest patch distances ("empirical tail value at risk", Eq. 4, line 297).
7. Pixel map: bilinear upsampling of patch distances + Gaussian smoothing sigma = 4.0, following PatchCore (line 306).
8. Trained: nothing. Gradient: none. "Fit" = run DINOv2 on k (augmented) images and store the patch tokens.

Delta vs PatchCore (for the chapter diagram). The PatchCore column is our reading from the PatchCore paper, not from this
paper; cross-check it against `notes/papers/patchcore.md`. This paper itself only calls max of the smoothed map the
"standard choice" (line 1225) and cites PatchCore for upsampling + sigma 4 and for coreset subsampling.
| | PatchCore | AnomalyDINO |
|---|---|---|
| Backbone | ImageNet-supervised CNN, mid-level layers + local aggregation | DINOv2 ViT-S/14 patch tokens, no aggregation |
| Distance | nearest neighbor (Euclidean in PatchCore) | nearest neighbor, cosine distance (Eq. 3) |
| Memory bank | greedy coreset to shrink it | keep all; enlarge it with rotations |
| Background | not handled | DINOv2 PCA mask on test patches (objects that pass the masking test) |
| Image score | max patch score with reweighting | mean of top 1% patch distances |
| Pixel map | upsample + Gaussian sigma 4 | same (copied from PatchCore) |
| Training | none | none |

## 6. Equations (LaTeX verbatim; custom macro `\method` = AnomalyDINO)
Eq. (1) `eq:memory_bank`:
```
\mathcal{M}:= \hspace{-2mm} \bigcup_{\mathbf{x}^{(i)} \in X_{\operatorname{ref}}}  \hspace{-2mm}  \bigl\{ \mathbf{p}_j^{(i)} \mid f(\mathbf{x}^{(i)}) = (\mathbf{p}_1^{(i)}, \dots, \mathbf{p}_n^{(i)}),\, j \in [n]\bigr\}  \enspace .
```
Paper: "we collect the nominal patch features and store them in a memory bank". Gloss: X_ref = the k reference images;
f maps an image to n patch features p_1..p_n; n depends on resolution and patch size 14; M = union of all their patches.

Eq. (2) `eq:NN-distance`:
```
d_{\operatorname{NN}}(\mathbf{p}; \mathcal{M}) := \min_{\mathbf{p}_{\operatorname{ref}} \in \mathcal{M}} d(\mathbf{p},\, \mathbf{p}_{\operatorname{ref}})
```
Paper: "the distance to the closest reference patch for a given test patch". Gloss: p = one test patch feature;
p_ref ranges over the memory bank; d = distance of Eq. 3.

Eq. (3) `eq:cosine_distance`:
```
d(\mathbf{x}, \mathbf{y}) := 1 - \frac{\langle \mathbf{x},\;  \mathbf{y}\rangle} {\Vert \mathbf{x} \Vert \, \Vert \mathbf{y} \Vert } \enspace.
```
Paper: "we set d as the cosine distance". Gloss: 0 for identical direction, up to 2 for opposite; length of vectors ignored.

Eq. (4) `eq:meantop1p`:
```
s(\mathbf{x}_\mathrm{test}) := q\big( \{ d_{\operatorname{NN}}(\mathbf{p}_1; \mathcal{M}), \dots, d_{\operatorname{NN}}(\mathbf{p}_n; \mathcal{M}) \} \big) \enspace .
```
with (inline, line 297) `q(\mathcal{D}):= \mathrm{mean}(H_{0.01}(\mathcal{D}))`. Paper: "the average distance of the 1% most
anomalous patches"; "an empirical estimate of the tail value at risk for the 99% quantile". Gloss: D = set of the n
patch distances; H_0.01(D) = its top 1% values; s = their mean. Trade-off: driven by the worst patches but robust to a
single outlier patch in a sparse bank.

Appendix D (batched zero-shot, not the main setting): M_j = all patches of the other test images (Eq. 5); patch score =
mean of the distances below the alpha = 0.1% quantile (Eqs. 6-7), because M_j may contain anomalous patches.

## 7. Config / setup table
| Item | Value | Source |
|---|---|---|
| Backbone | DINOv2 ViT-S (21M params); also B 86M, L 300M, G 1,100M in ablation | lines 359-360; Table 9d |
| Patch size | 14 x 14 px | line 259 |
| Input resolution | 448 and 672 px (smaller edge); ablations 224, 896 | line 360; Table 9c |
| Layer used | not stated (our reading: last layer) | line 251 |
| Distance | cosine | Eq. 3 |
| Image score | mean of top 1% patch distances | Eq. 4, line 297 |
| Pixel map | bilinear upsample + Gaussian sigma = 4.0 | line 306 |
| Augmentation | rotations of references, "agnostic" default (always) | line 349; Table 8 |
| Rotation angles | not stated | -- |
| Masking | PCA 1st component threshold + center-crop rule + dilation + closing; test images only; per-category decision from first reference | lines 323-331, 1006-1007 |
| Coreset | none | line 313, footnote 483 |
| Trainable params | none (training-free) | abstract, Table 1 |
| Shots | k in {1, 2, 4, 8, 16}; 3 runs (refs = first, second, third k train images) | line 375 |
| Protocol | one memory bank per category (one-class), k-shot | our reading of Sec. 3 |
| Hardware | single NVIDIA A40, all experiments | line 1098 |

## 8. Results worth remembering
All numbers in %, mean over 3 runs (std in the paper). AnomalyDINO-S, agnostic preprocessing.

MVTec AD (Table 2): image AUROC / pixel AUROC / PRO
| k | AnomalyDINO-S (448) | AnomalyDINO-S (672) | WinCLIP+ | APRIL-GAN | ADP_l (img only) | PatchCore (dagger) |
|---|---|---|---|---|---|---|
| 1 | 96.5 / 96.3 / 91.7 | 96.6 / 96.8 / 92.7 | 93.1 / 95.2 / 87.1 | 92.0 / 95.1 / 90.6 | -- | 83.4 / 92.0 / 79.7 |
| 2 | 96.7 / 96.5 / 92.0 | 96.9 / 97.0 / 93.1 | 94.4 / 96.0 / 88.4 | 92.4 / 95.5 / 91.3 | 95.4 | 86.3 / 93.3 / 82.3 |
| 4 | 97.6 / 96.7 / 92.4 | 97.7 / 97.2 / 93.4 | 95.2 / 96.2 / 89.0 | 92.8 / 95.9 / 91.8 | 96.2 | 88.8 / 94.3 / 84.3 |
| 8 | 98.0 / 97.0 / 92.7 | 98.2 / 97.4 / 93.8 | 94.6* / -- / -- | 93.1 / 96.2 / 92.4 | 97.0 | -- |
| 16 | 98.3 / 97.1 / 92.9 | 98.4 / 97.5 / 94.0 | 94.8* / -- / -- | 93.2 / 96.4 / 92.6 | 97.0 | -- |
GraphCore (image AUROC / pixel AUROC, no PRO, no std): 89.9/95.6, 91.9/96.9, 92.9/97.4, 95.9/97.8 at k = 1, 2, 4, 8.
SPADE (dagger) image AUROC 81.0, 82.9, 84.8 at k = 1, 2, 4.

VisA (Table 3): image AUROC / pixel AUROC / PRO
| k | AnomalyDINO-S (448) | AnomalyDINO-S (672) | WinCLIP+ | APRIL-GAN | ADP_l (img only) | PatchCore (dagger) |
|---|---|---|---|---|---|---|
| 1 | 85.6 / 97.5 / 90.7 | 87.4 / 97.8 / 92.5 | 83.8 / 96.4 / 85.1 | 91.2 / 96.0 / 90.0 | -- | 79.9 / 95.4 / 80.5 |
| 2 | 88.3 / 97.8 / 91.7 | 89.7 / 98.0 / 93.4 | 84.6 / 96.8 / 86.2 | 92.2 / 96.2 / 90.1 | 86.9 | 81.6 / 96.1 / 82.6 |
| 4 | 91.3 / 98.0 / 92.5 | 92.6 / 98.2 / 94.1 | 87.3 / 97.2 / 87.6 | 92.6 / 96.2 / 90.2 | 88.4 | 85.3 / 96.8 / 84.9 |
| 8 | 92.6 / 98.2 / 93.3 | 93.8 / 98.4 / 94.8 | 85.0* / -- / -- | 93.0 / 96.3 / 90.2 | 89.2 | -- |
| 16 | 93.8 / 98.3 / 93.8 | 94.8 / 98.5 / 95.3 | 85.0* / -- / -- | 93.2 / 96.3 / 90.2 | 90.1 | -- |
PaDiM (dagger) VisA image AUROC 62.8, 67.4, 72.8 at k = 1, 2, 4.

Baseline provenance (lines 382-384): dagger = numbers copied from the WinCLIP paper (jeong2023winclip); star = WinCLIP
re-implementation reported in ADP (kwak2024few, "Few-shot anomaly detection via personalization", IEEE Access 2024);
all other rows copied from each method's original publication. AnomalyDINO re-ran nothing. ADP is the class-label
variant ADP_l. APRIL-GAN = arXiv 2305.17382. PromptAD and InCTRL do not appear anywhere in this paper. AnomalyCLIP
appears only in the zero-shot block of Table 4.

Headline (abstract, Table 2): 1-shot MVTec AD image AUROC 93.1 (WinCLIP+) -> 96.6 (AnomalyDINO-S 672).
VisA image-level: APRIL-GAN (meta-trained) is best at k = 1, 2; tie at k = 4 (92.6); AnomalyDINO best at k = 8, 16.
VisA pixel-level: AnomalyDINO best on all three pixel metrics at every k.

Other settings (Table 4, image AUROC, MVTec AD / VisA): 0-shot WinCLIP 91.8 / 78.1; AnomalyCLIP 91.5 / 82.1; APRIL-GAN
86.1 / 78.0. Batched 0-shot: ACR 85.8 / not reported; MuSc 97.8 / 92.8; AnomalyDINO-S (448) 93.0 / 89.7;
AnomalyDINO-S (672) 94.2 / 90.7.

Full-shot (Table 5, AnomalyDINO-S, masking only, no augmentation, deterministic): image AUROC / pixel AUROC / PRO
- MVTec AD 448: 99.3 / 97.9 / 93.9; 672: 99.5 / 98.2 / 95.0.
- VisA 448: 97.2 / 98.7 / 95.0; 672: 97.6 / 98.8 / 96.1.

Runtime (Table 9; seconds per test image, mean over the 1,725 MVTec AD test images; single NVIDIA A40, GPU warmup and
CUDA sync; batch size not stated, per-sample timing):
- Default (S, 448, 1-shot, agnostic): 0.060 (std 0.012); memory bank build 0.52 s per category. Text: "roughly 60ms",
  "approximately 16.7 samples/s" (line 1165).
- Shots: 1: 0.060, 2: 0.059, 4: 0.059, 8: 0.060, 16: 0.063, full (masking only): 0.067, full (agnostic): 0.130.
- Preprocessing: none 0.055; rotate only 0.055; mask only 0.068; both 0.067.
- Resolution: 224: 0.043; 448: 0.060; 672: 0.086; 896: 0.141.
- Model size: S 0.060; B 0.084; L 0.141; G 0.306.
- Fig. 3 (log time axis, 1-shot MVTec AD): AnomalyDINO is "at least one order of magnitude faster" than SOTA few-shot
  competitors (line 1166). The plot labels say ADP (512) was "reported on GeForce RTX 3090, 2-shot" and WinCLIP+ (240)
  "reported on Amazon EC2 G4dn"; exact competitor times are only plotted, not tabulated.

Ablations:
- Aggregation (Table 10, image AUROC, S-448, k = 1): MVTec AD mean top 1% 96.5 vs max(D) 95.0 vs max(A) 94.9; VisA 85.6
  vs 82.4 vs 80.5.
- Preprocessing raises AUROC by "approximately 2%" (line 487). Rotation: Screw 1-shot 65.6 -> 89.2 (line 953). Rotation
  hurts PCB1/2/3, Macaroni1 (VisA) and slightly Transistor (line 967).
- Backbone (Fig. 17, 224 px, no preprocessing): DINOv2 beats ImageNet-supervised ViT-B/L by "at least +4%" AUROC.
- Model size (Fig. 16, 672 px): S best on MVTec AD; larger models slightly better on VisA.

## 9. Limitations stated
- Semantic/logical anomalies are missed: MVTec AD Cable "cable swap" detection AUROC 50.2 (std 4.9), chance level
  (App. B.1, line 788). Every test patch matches some reference patch.
- Uninformative references: a Capsule reference with the print hidden makes printed normal parts look anomalous; high
  variance at k = 1, 2 (App. B.2).
- Rotation augmentation can hide "misplaced"/rotated defects (Transistor, Fig. 13); preprocessing must suit the product.
- PCA masking fails on close-ups (object covers >= ~50% of patches), hence the masking test.
- Later fixes: not stated in these sources.

## 10. Outline feedback
- `viz-dinov2` shots slider: AnomalyCLIP has NO few-shot numbers in this paper (zero-shot only, Table 4). Usable
  same-setting comparisons: WinCLIP+ image AUROC at all 5 k (k = 8, 16 are the ADP re-implementation, starred), WinCLIP+
  pixel AUROC and PRO only at k = 1, 2, 4; APRIL-GAN all metrics at all k; ADP_l image AUROC at k = 2..16. PromptAD is
  absent. Suggest WinCLIP+ and APRIL-GAN as the CLIP lines, AnomalyCLIP as a zero-shot reference mark.
- "beats CLIP-based few-shot methods" holds on MVTec AD at every k and metric. On VisA image AUROC, APRIL-GAN wins at
  k = 1, 2 and ties at k = 4. Qualify the intuition sentence.
- Masking is applied to test patches only, and only for categories that pass the one-time masking test (5 of 15 on
  MVTec AD). The PatchCore delta diagram should show this.
- AnomalyDINO is evaluated one category at a time (one-class memory bank), so it does not sit on the multi-class axis.
- The full-shot rows (Table 5) are a fair scoreboard entry under "one-class, full-shot".
