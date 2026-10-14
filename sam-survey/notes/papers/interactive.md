# Interactive segmentation: DIOS -> RITM -> SimpleClick (+ FocalClick)

Scope: ch-clicks (delta cards, scene `clicks`). Architecture lens, light. All locations are in the versions read.

---------------------------------------------------------------------------------------------------
## DIOS -- Deep Interactive Object Selection

### 1. Identity
- Xu (Ning), Price, Cohen, Yang, Huang (last). UIUC + Adobe Research.
- arXiv 1603.04042v1, submitted 2016-03-13 (abs.html). Only v1 exists.
- Venue: not stated on abs.html (the "Computer Vision and Pattern Recognition" cell is the arXiv subject). SimpleClick Table 2 tags it "CVPR16".
- Source: latex/ only wraps a PDF (`\includepdf{deepselect.pdf}`); everything below is from paper.pdf.

### 2. Core problem
Earlier interactive tools (graph cut, geodesics, random walk) model fg/bg from low-level color and texture. They need many clicks when fg and bg look alike.

### 3. Prior methods and why that hurt
Low-level features only, no "objectness" (Sec. 2). Semantic-segmentation FCNs know objects but cannot be steered by a user and are tied to fixed classes (Sec. 2, Fig. 6).

### 4. Key idea
Encode clicks as images (two distance maps), stack them with RGB as a 5-channel input, and fine-tune an FCN on simulated clicks (Fig. 1).

### 5. Mechanism
1. Positive clicks S^1, negative clicks S^0 -> two Euclidean distance maps U^1, U^0, same H x W as the image (Sec. 3.1, Eq. 1).
2. Truncate distances to 255. If there are no negative clicks, U^0 is all 255 (Sec. 3.1).
3. Input = RGB ++ U^1 ++ U^0 (5 channels) -> FCN (Fig. 1). Extra filters for the two new channels in the first layer (Sec. 3.3).
4. Fine-tune FCN-32s -> FCN-16s -> FCN-8s from the semantic FCN of Long et al. (Sec. 3.3). About 20 epochs each; ~3 days FCN-32s, ~5 days FCN-16s and FCN-8s.
5. Output probability map Q is refined by graph cut, E(L) = lambda R(L) + B(L) (Sec. 3.4).
6. Training clicks are simulated at random (Sec. 3.2), not from the model's errors:
   - Positive: n in [1, N_pos] pixels inside the object O, at least d_step apart and d_margin from the boundary.
   - Negative, one of 3 strategies per pair, equal probability:
     S1: n in [0, N_neg1] random background pixels within distance d of the object (set G^c).
     S2: n_i in [0, N_neg2] on each other object O_i in the image.
     S3: N_neg3 clicks that cover the outside boundary evenly: first random in G^c, then farthest-point (Eq. 2).
   - Values (Sec. 4.1): d = 40, N_pos = 5, N_neg1 = 10, N_neg2 = 5, N_neg3 = 10, N_pairs = 15 per object, ~80k pairs. d_step and d_margin values are not printed.
   - Training data: PASCAL VOC 2012, 1464 train images + flips.
7. Evaluation user (Sec. 3.5): "places a seed at the mislabeled pixel that is farthest from the boundary of the current selection and the image boundaries, mimicing a user's behavior under the assumption that the user clicks in the middle of the region of greatest error." Max 20 clicks; if the IoU target is not reached in 20, NoC = 20 (Sec. 4.1).

### 6. Equations (transcribed from PDF p. 3; no TeX source)
- Eq. 1: `u^t_{ij} = f(p_{ij} | \mathcal{S}^t), \quad t \in \{0, 1\}`, with
  `f(p_{mn}|\mathcal{A}) = \min_{\forall p_{ij} \in \mathcal{A}} \sqrt{(m-i)^2 + (n-j)^2}`.
  Paper: "the operator f calculates the minimum Euclidean distance between a point and a set of points."
  Gloss: u^t_ij = pixel (i,j) of click map t; t = 1 positive, 0 negative; value = distance to the nearest click of that type (then capped at 255).
- Eq. 2: `p_{next} = \arg\max_{p_{ij} \in \mathcal{G}^c} f(p_{ij} | \mathcal{S}^0 \cup \mathcal{G})`.
  Gloss: the next Strategy-3 negative click is the background pixel (near the object) farthest from all existing negative clicks and from the object/far-region set G.

### 7. Config
| item | value | loc |
|---|---|---|
| input | RGB + 2 distance maps = 5 channels | Fig. 1, Sec. 3.1 |
| distance cap | 255 | Sec. 3.1 |
| network | FCN-8s (from FCN-32s, FCN-16s) + graph cut | Sec. 3.3-3.4 |
| train data | VOC 2012, 1464 images, ~80k pairs | Sec. 4.1 |
| click limit in eval | 20 | Sec. 4.1 |
| FCN time per map | < 100 ms on Titan X | Sec. 3.5 |

### 8. Results (Table 1, "Ours" row; mean clicks)
Pascal@85 6.88; GrabCut@90 6.04; Berkeley@90 8.65; COCO seen@85 8.31; COCO unseen@85 7.82. Graph cut on GrabCut@90: 11.10.
(SimpleClick Table 2 also lists DIOS DAVIS NoC90 = 12.58, not in DIOS itself.)

### 9. Limitations
FCN misses thin structures and fine details; graph cut helps only partly (Sec. 4.2 text near Fig. 5). Training clicks ignore the model's errors -> fixed by RITM's iterative sampling.

---------------------------------------------------------------------------------------------------
## RITM -- Reviving Iterative Training with Mask Guidance for Interactive Segmentation

### 1. Identity
- Sofiiuk (first), Petrov, Konushin (last).
- arXiv 2102.06583v1, 2021-02-12 (abs.html). Venue: none on abs.html; PDF header says "Preprint". Repo: github.com/saic-vul/ritm_interactive_segmentation.

### 2. Core problem
SOTA click models relied on inference-time optimization (BRS, f-BRS): backward passes at test time, slow, hard to deploy. RITM asks whether a plain feed-forward model can match them.

### 3. Prior methods and why that hurt
- BRS / f-BRS / IA+SA optimize at test time (Sec. 2).
- Random click sampling "without considering any relations between them" ignores that real clicks land on errors; this also blocks feeding back the previous mask (Sec. 3.2).
- Training sets (SBD / VOC, 20 classes, coarse masks) limit generality (Sec. 4.1).

### 4. Key idea
Iterative training: simulate clicks on the model's own errors and feed the previous mask back as a third input channel, trained on COCO+LVIS.

### 5. Mechanism
1. Backbone: HRNet+OCR (or DeepLabV3+). Clicks enter via "Conv1S": a conv block whose output is summed with the backbone's first conv output (64 ch) (Sec. 3.1).
2. Click encoding: disks of radius 5 (Disk5) beat distance transform (DT) and Disk3 (Sec. 5.2, Table 1). Paper's reason: disk changes are local; a DT map can change drastically when a click is added.
3. Input maps = positive disks, negative disks, previous mask ("the third channel together with two channels for positive and negative encoded clicks"). Empty mask for the first click (Sec. 3.3).
4. Training clicks (Sec. 3.2): random sampling as in DIOS first, then 0..N_iters iterative clicks. N_iters = 3 (Table 6). Each batch gets uniformly 0..N_iters iterations. No clicks are cached across epochs.
5. Where an iterative training click goes (Sec. 3.2): not the centre of the error; "from the region obtained by applying morphological erosion operation to the mislabelled region, so that the eroded region has 4 times less area." Reason: centre points overfit the NoC metric.
6. Loss: normalized focal loss (NFL), Eq. 2 (Sec. 3.4).
7. Evaluation user (Sec. 5, "Evaluation metric"), verbatim: "The next click is placed at the center of the region with any type of prediction error (false positive or false negative) with the largest area among other erroneous regions. The region center is defined as the point farthest from the boundaries of the corresponding region."

### 6. Equations (main.tex)
- Eq. 1 `\label{eq:FL}`: `FL(i,j)=-(1-p_{i,j})^{\gamma}\log p_{i,j}`. p_ij = confidence of the prediction at pixel (i,j); gamma = focusing exponent.
- Eq. 2 `\label{eq:NFL}`: `NFL(i,j,\hat{M})=-\frac{1}{P(\hat{M})}(1-p_{i,j})^{\gamma}\log p_{i,j}`, with `P(\hat{M})=\sum_{i,j}(1-p_{i,j})^\gamma`.
  Paper: "The gradient of NFL does not fade over time due to normalization and remains equal to the total gradient of BCE." M-hat = network output; P = total focal weight over the image.

### 7. Config (Sec. 5 "Implementation details")
| item | value |
|---|---|
| crops | 320x480, scale jitter 0.75-1.40 |
| optimizer | Adam, beta1 0.9, beta2 0.999 |
| schedule | 55 epochs on COCO+LVIS; lr 5e-4, /10 at epochs 50 and 53; backbone lr x0.1 |
| batch | 32 |
| hardware | HRNet-18s / 18 / 32 on 1 / 2 / 4 Tesla P40 |
| N_iters | 3 |
| click disk radius | 5 |
| COCO+LVIS | 104k images, 1.6M instance masks; COCO mask dropped if an LVIS mask has IoU > 80% with it (Sec. 4.2) |
| params | HRNet18+OCR 10.03M; HRNet32+OCR 30.95M (Table 4) |
| TTA | Zoom-In + flip averaging from f-BRS |

### 8. Results (Table 7, NoC@90, trained on COCO+LVIS)
| model | GrabCut | Berkeley | SBD | DAVIS |
|---|---|---|---|---|
| H18 IT-M | 1.54 | 2.26 | 6.06 | 5.74 |
| H32 IT-M | 1.56 | 2.10 | 5.71 | 5.34 |
SBD-trained H18 IT-M: 2.04 / 3.22 / 5.43 / 6.71. Previous best f-BRS-B: 2.98 / 4.34 / 8.08 / 7.81.
SAM's RITM baseline is "HRNet32 IT-M" trained on COCO+LVIS (SAM App. D.1, Baselines).
Prev-mask ablation (Table 6, N_iters 3): without mask 2.38 / 5.92 / 6.49, with mask 2.26 / 5.74 / 6.06 (Berkeley / DAVIS / SBD). The mask also removes IoU drops when clicks are added (Fig. 4).

### 9. Limitations
N_iters > 4 makes training unstable (runs collapsed after 10-20 epochs, Sec. 5.2). HRNet CNN backbone -> SimpleClick replaces it with a plain MAE ViT.

---------------------------------------------------------------------------------------------------
## SimpleClick -- Interactive Image Segmentation with Simple Vision Transformers

### 1. Identity
- Liu (Qin, first), Xu, Bertasius, Niethammer (last).
- arXiv 2210.11006v3 (v1 2022-10-20). abs.html comment: "Tech report. Update 03/11/2023: Add results on a tiny model and append supplementary materials".

### 2. Core problem
Interactive segmentation used hierarchical backbones. Can a plain ViT, pretrained as an MAE, work with minimal changes?

### 3. Prior methods
Hierarchical CNN backbones (HRNet, SegFormer); FocalClick adds a local refinement module (Sec. 4.1).

### 4. Key idea
Plain ViT backbone (MAE init) + a "symmetric patch embedding" for the click/mask maps, added element-wise to the image patch tokens.

### 5. Mechanism (Sec. 3.1)
1. Clicks = "two-channel disk map" (positive, negative). Concatenate previous segmentation -> 3-channel map.
2. Two patch embeddings, one on the image, one on the 3-channel map; outputs added element-wise before the first attention block.
3. ViT-B/L/H, MAE-pretrained on ImageNet-1K (MAE's released weights). Fine-tuned at 448x448 with non-shifting window attention plus a few global blocks (ViTDet style; "e.g. 2 for ViT-B").
4. Simple feature pyramid from the last feature map only (four convs with different strides), then an all-MLP head -> 1/4-scale probability map -> upsample -> threshold 0.5.
5. Training: random + iterative click simulation "inspired by RITM", previous mask as input, NFL loss (Sec. 3.2). Disk radius is not stated in the paper.
6. Evaluation user (Sec. 4), verbatim: "the next click will be put at the center of the region with the largest error." Targets 85% and 90%; max 20 clicks.

### 7. Config (Sec. 4 "Implementation Details", Table 1)
| item | value |
|---|---|
| input | 448x448 |
| epochs | 55; lr 5e-5 -> 5e-6 after epoch 50 |
| batch | 140 (B), 72 (L), 32 (H) |
| GPUs | 4 NVIDIA RTX A6000 |
| optimizer | Adam 0.9 / 0.999 |
| params (M) | ViT-H backbone 604.0, neck 25.8, head 1.3 (Table 1) |

### 8. Results (Table 2, NoC90)
| SimpleClick ViT-H | GrabCut | Berkeley | SBD | DAVIS |
|---|---|---|---|---|
| trained on SBD | 1.44 | 2.09 | 4.15 | 5.34 |
| trained on C+L | 1.50 | 1.75 | 4.70 | 4.78 |
Headline: "4.15 NoC@90 on SBD, improving 21.8% over the previous best result" (abstract; SBD-trained model).
SAM's SimpleClick baseline: "ViT-H448 trained on a combination of COCO and LVIS" (SAM App. D.1).

### 9. Limitations (Sec. 5)
ViT-H is much larger than prior models; fails on very thin, elongated shapes and cluttered occlusions.

---------------------------------------------------------------------------------------------------
## FocalClick (one line)
arXiv 2204.02574v2, CVPR 2022 (abs.html), Chen (Xi) ... Zhao (Hengshuang). Adds local, crop-based prediction: coarse segmentation on a resized "Target Crop", local refinement on a "Focus Crop" around the changed region, and "Progressive Merge" to edit preexisting masks without destroying correct parts (abstract).

---------------------------------------------------------------------------------------------------
## SAM's own point protocol (for contrast; 2304.02643v1)
- Eval (App. D.1 "Point sampling"): "The first point is chosen deterministically as the point farthest from the object boundary. Each subsequent point is the farthest from the boundary of the error region between ground truth and the previous prediction."
- Training (App. A "Training algorithm"): "subsequent points are selected uniformly from the error region between the previous mask prediction and the ground truth mask."

## 10. Outline feedback (ch-clicks)
1. "training samples the next point uniformly from the error region" is SAM's rule (App. A). RITM samples from the eroded error region (4x smaller area) to avoid overfitting NoC; DIOS training ignores errors (random sampling). The toy text should name SAM, or say "RITM: eroded error region".
2. The evaluation click ("centre of the largest error region" = point farthest from that region's boundary) is right. DIOS also excludes the image boundary from the distance.
3. Encoding lineage worth one line on the card: DIOS distance maps (cap 255) -> RITM disks radius 5 + prev-mask channel -> SimpleClick patch-embedded 3-channel map -> SAM mask prompt as logits.
4. NoC is capped at 20 clicks (NoC_20). Show "20" as the curve's x limit.
5. SimpleClick headline 4.15 is the SBD-trained model. A C+L comparison with RITM should use the C+L row.
