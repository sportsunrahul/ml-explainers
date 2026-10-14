# YOLO-World (spine, chapter 10)

## 1. Identity
- Title: "YOLO-World: Real-Time Open-Vocabulary Object Detection".
- arXiv 2401.17270, version read: v3. v1 2024-01-30, v2 2024-02-02, v3 2024-02-22 (abs.html submission history).
- Authors: Tianheng Cheng (first, equal contribution) ... Ying Shan (last); Lin Song and Xinggang Wang corresponding. Tencent AI Lab, ARC Lab Tencent PCG, HUST.
- Venue: the LaTeX uses the CVPR 2024 template (paper ID 3489). arXiv v3 comment: "Work still in progress"; no journal-ref. The README says "accepted by CVPR 2024" (update dated 2024-2-28) and its bibtex is CVPR 2024 (code_only).
- Repo: https://github.com/AILab-CVC/YOLO-World. The README describes the live repo (YOLO-World-v2 / V2.1 models), not the v3 paper models.
- Sources read: src/main.tex (main text + appendix A-C), paper.pdf (figures, numbering), README.md. Custom macros: `\name` = YOLO-World, `\tb` = bold, `\xmark`/`\cmark` from preamble (Table 4 uses `\checkmark`, not `\cmark`).

## 2. Core problem
Open-vocabulary detectors (GLIP, Grounding DINO, DetCLIP) are large (Swin backbones), encode the text prompt together with the image on every call, and are hard to deploy on edge devices. Nobody had shown that a small real-time detector can be pretrained for open-vocabulary detection (Sec. 1).

## 3. What prior methods did and why that hurt (per the paper)
- Distillation-based OVD (ViLD and others): limited by scarce training data with low vocabulary diversity, e.g. OV-COCO has 48 base classes (Sec. 1).
- Grounding-style pretraining (GLIP, GLIPv2, Grounding DINO, DetCLIP): good but "heavy computation burden" and "complicated deployment for edge devices"; heavy backbones such as Swin-L; they "simultaneously encode images and texts as input for prediction, which is time-consuming" (Sec. 1, Fig. 2b caption).
- ZSD-YOLO used YOLO with language-model alignment, but only in the standard OVD setting (Sec. 2.2).

## 4. The one key idea
Take YOLOv8, swap its class rows for frozen CLIP text embeddings, fuse text into the PAN neck cheaply (a per-pixel max-sigmoid gate and a 27-token image-pooling attention), and pretrain on region-text pairs at scale. At deployment the user's vocabulary is encoded once ("prompt-then-detect", offline vocabulary) and re-parameterized into ordinary 1x1 conv / linear weights, so the text encoder is removed from the inference path.

## 5. Mechanism, step by step
1. Data as region-text pairs: annotations `{B_i, t_i}` where `t_i` is a class name, noun phrase or description (Sec. 3.1). Captions and referring expressions are cut into noun phrases by "the simple n-gram algorithm" (Sec. 3.2).
2. Online vocabulary per mosaic sample (4 images): all positive nouns in the mosaic plus randomly sampled negative nouns from the same dataset, at most `M` = 80 nouns (Sec. 3.2).
3. Text encoder: CLIP Transformer text encoder, `W = TextEncoder(T)` in R^{C x D}, `C` = number of nouns. Frozen during pretraining (Sec. 4.2; snowflake on the encoder in Fig. 3). The ablation calls it "CLIP-base (ViT-base)" (Sec. 4.3). The patch-size variant (B/32 vs B/16) is NOT stated in the paper or README.
4. Image encoder: YOLOv8 Darknet backbone gives `{C3, C4, C5}` (Sec. 3.2, 3.3).
5. RepVL-PAN (Fig. 4): YOLOv8 PAN top-down then bottom-up paths produce `{P3, P4, P5}`. Every CSPLayer (C2f) after a fusion becomes a T-CSPLayer: after the last dark bottleneck, the image feature is gated by Eq. 2 and concatenated with the cross-stage (split) features. Fig. 4 shows 4 T-CSPLayers (2 top-down, 2 bottom-up) and 2 I-Pooling Attention blocks (one after the top-down path, one at the end). Our reading of Fig. 4: the bottom-up T-CSPLayers take the text embeddings already updated by the first I-Pooling block; the second I-Pooling block outputs the "Image-aware Embeddings" used by the head.
6. I-Pooling Attention (Eq. 3): max-pool each of the 3 levels to 3x3, giving 27 patch tokens; text embeddings attend to them (query = text, key = value = pooled image tokens), residual add.
7. Head: YOLOv8 decoupled head, "two 3x3 convs" per branch, regressing boxes `b_k` and object embeddings `e_k` for each of `K` predictions; text contrastive head gives `s_{k,j}` (Eq. 1).
8. Assignment and loss: task-aligned label assignment (TOOD), as in YOLOv8; each positive prediction gets a text index as its class label. `L_con` = cross entropy between region-text similarity and the assignment; box loss = IoU loss + "distributed focal loss" (sic, DFL). Box loss only for detection / grounding images (`lambda_I` = 1); 0 for pseudo-labelled image-text data (Sec. 3.4). Our reading: CC3M pseudo boxes still decide which predictions are positives for `L_con`; they only lose the regression term. The paper says "cross entropy" and does not say softmax vs per-class sigmoid. YOLOE (Sec. 3.5) uses binary cross entropy in a similar setup, but that is YOLOE's own statement, not evidence for YOLO-World's loss.
9. Inference, offline vocabulary: encode the user's prompts once; re-parameterize the text embeddings into weights of conv / linear layers (Sec. 3.3, App. A.1: Eq. 4-6). After COCO fine-tuning: "pre-compute the class text embeddings with given COCO categories and store the embeddings into the weights of the classification layers" (App. A.2).

What gets a gradient (pretraining): backbone, RepVL-PAN (T-CSPLayers, I-Pooling attention), both head branches, `alpha`, `beta`. Frozen: CLIP text encoder. Signals: `L_con` on all data; `L_iou + L_dfl` only on detection / grounding data. Initialization of the YOLO backbone is not stated. Fine-tuning (Sec. 4.4): all weights; CLIP text encoder now trained at 0.01x learning rate.

### What the paper says about the YOLOv8 base (everything it says; nothing more is in the paper)
- "a Darknet backbone [YOLOv1, YOLOv8] as the image encoder, a path aggregation network (PAN) for multi-scale feature pyramids, and a head for bounding box regression and object embeddings" (Sec. 3.2).
- PAN: "follows the top-down and bottom-up paths in [PAFPN, YOLOv8] to establish the feature pyramids {P3, P4, P5} with the multi-scale image features {C3, C4, C5}" (Sec. 3.3).
- CSPLayer: "the cross-stage partial layers (CSPLayer) are utilized after the top-down or bottom-up fusion ... the CSPLayer (also called C2f) of [YOLOv8]"; Fig. 4 inset draws C2f as Split -> Dark Bottleneck -> (Max-Sigmoid) -> Concat with the split branch.
- Head: "the decoupled head with two 3x3 convs" (Sec. 3.2).
- Assignment: "task-aligned label assignment [TOOD]" (Sec. 3.4). Loss: "IoU loss and distributed focal loss" (Sec. 3.4).
- Augmentation: color augmentation, random affine, random flip, 4-image mosaic (Sec. 4.2).
- Sizes S / M / L follow YOLOv8 (Sec. 4.1); built on MMYOLO + MMDetection.
- NOT in the paper: input resolution, strides, feature-map sizes, number of predictions, channel widths per level, backbone depth, C2f internals beyond Fig. 4, the TAL metric, the DFL bin formulation, "anchor-free". These need an outside source (YOLOv8 docs, TOOD, GFL) or an "our reading" label.

### Shapes (our reading unless marked)
- Input 640x640: our reading (YOLOv8 convention). README model card lists 640 for the pretraining (PT, 100e) V2.1 models and 800 / 1280 for continued-pretraining models (code_only; not the paper's models).
- `C3/P3`, `C4/P4`, `C5/P5` at strides 8, 16, 32 (our reading: the `C_l` index convention means stride 2^l) -> 80x80, 40x40, 20x20 -> K = 6400 + 1600 + 400 = 8400 predictions per image.
- `W`: C x D, C <= 80 in training. `D` not stated; 512 if it is the CLIP ViT-B text projection (our reading, unverified).
- `s`: K x C = 8400 x 80 logits in training (our reading).
- I-Pooling: 3 levels x 3x3 = 27 tokens (sourced, Sec. 3.3; App. A.1 says X~ is B x D x 27).
- Eq. 2 writes `X_l` in R^{H x W x D}, the same D as text. YOLOv8 neck widths differ per level and per model size; the paper does not say how channels are matched to D (our reading: a projection must exist).
- Number of attention heads in Eq. 3: not stated.

## 6. Equations
Numbering: 3 numbered equations in the main text (Eq. 1-3), 4 in the appendix (Eq. 4-7). The total loss is inline and unnumbered (a commented-out equation environment sits under it in the tex).

**Eq. (1), text contrastive head** (Sec. 3.2)
```tex
s_{k,j} = \alpha\cdot \texttt{L2-Norm}(e_k) \cdot \texttt{L2-Norm}(w_j)^{\top} + \beta,
```
Paper: "a text contrastive head to obtain the object-text similarity"; "Both the L2 norms and the affine transformations are important for stabilizing the region-text training."
- `s_{k,j}`: score of prediction k for noun j (the class logit).
- `e_k`: object embedding of prediction k, from the head's embedding branch, in R^D.
- `w_j`: j-th row of the text embedding matrix W (one noun).
- `L2-Norm`: scale to unit length, so the dot product is a cosine.
- `alpha`: learnable scale (acts as an inverse temperature). `beta`: learnable shift.

**Unnumbered total loss** (Sec. 3.4, inline)
```tex
\mathcal{L}(I) = \mathcal{L}_{\text{con}} + \lambda_I\cdot(\mathcal{L}_{\text{iou}} + \mathcal{L}_{\text{dfl}}),
```
- `L_con`: region-text contrastive loss, cross entropy between similarities and TAL assignments.
- `L_iou`, `L_dfl`: box losses (IoU, distribution focal).
- `lambda_I`: 1 for detection or grounding images, 0 for image-text (pseudo-labelled) images.

**Eq. (2), max-sigmoid attention in the T-CSPLayer** (Sec. 3.3)
```tex
X_l' = X_l\cdot\delta(\max_{j\in\{1..C\}}(X_lW_j^{\top}))^{\top},
```
Paper: "we adopt the max-sigmoid attention after the last dark bottleneck block to aggregate text features into image features ... the updated X_l' is concatenated with the cross-stage features as output. The delta indicates the sigmoid function."
- `X_l`: image feature map at level l in {3,4,5}, H x W x D.
- `W_j`: text embedding of noun j.
- `X_l W_j^T`: per-pixel dot product with noun j (an H x W map).
- `max_j`: keep, per pixel, the best-matching noun's score.
- `delta`: sigmoid, turning it into a gate in (0, 1).
- Effect (our reading, consistent with Eq. 4): each pixel's feature vector is multiplied by one scalar gate; pixels that match any prompt are kept, others damped.

**Eq. (3), Image-Pooling Attention** (Sec. 3.3)
```tex
W' = W + \texttt{MultiHead-Attention}(W, \tilde{X}, \tilde{X})
```
Paper: "Rather than directly using cross-attention on image features, we leverage max pooling on multi-scale features to obtain 3x3 regions, resulting in a total of 27 patch tokens X~ in R^{27 x D}."
- `W`: text embeddings (queries). `X~`: 27 pooled image tokens (keys and values). `W'`: image-aware text embeddings (residual update).

**Eq. (4), re-parameterized T-CSPLayer** (App. A.1), verbatim including the source typo `X\times`
```tex
X' = X \odot \texttt{Sigmoid}(\texttt{max}(\texttt{Conv}(X, W), \texttt{dim=1})),
```
followed by "where $X\times \in \mathbb{R}^{B\times D\times H\times W}$ and $X'\in \mathbb{R}^{B\times D\times H\times W}$ are the input and output image features. $\odot$ is the matrix multiplication with \texttt{reshape} or \texttt{transpose}."
Paper: "reshaping the text embeddings W in R^{C x D x 1 x 1} into the weights of a 1x1 convolution layer (or a linear layer)".
- `Conv(X, W)`: 1x1 conv whose C output filters are the C text embeddings -> B x C x H x W similarity maps.
- `max(..., dim=1)`: max over the C nouns -> B x 1 x H x W. Sigmoid -> gate. Our reading: the `odot` acts as a broadcast per-pixel multiply.

**Eq. (5), pooled tokens for re-parameterized I-Pooling** (App. A.1)
```tex
\tilde{X} = \texttt{cat}(\texttt{MP}(X_3, 3),\texttt{MP}(X_4, 3), \texttt{MP}(X_5, 3)),
```
- `MP(., 3)`: max pooling to a 3x3 output. `cat`: concatenation ("concentration" in the source). X~ flattened is B x D x 27.

**Eq. (6), re-parameterized I-Pooling update** (App. A.1), verbatim, parentheses unbalanced in the source
```tex
W' = W + \texttt{Softmax}(W\odot\tilde{X}), \texttt{dim=-1}) \odot W,
```
- Flag: as written the attention weights multiply `W`, not `X~`, and there are no projections and a single head, so it does not match Eq. 3. Likely a transcription error; do not repair it on the page. Our reading: X~ still comes from the current image, so this text update runs per image even after re-parameterization.

**Eq. (7), pseudo-labeller** (App. B)
```tex
\{B_i, t_i, c_i\}_{i=1}^N = \texttt{GLIP-Labeler}(I, T),
```
- GLIP-L proposes boxes `B_i` with noun `t_i` and confidence `c_i` for caption nouns T of image I.
- Inline rescoring (App. B, step 4): `$\tilde{c_i} = \sqrt{c_i * s^r_i}$`, `s^r_i` = CLIP region-text similarity of the crop. Image score (step 6): `$s=\sqrt{s^{img}*s^{region}}$`.

## 7. Config / setup
| Item | Value | Source |
|---|---|---|
| Base detector | YOLOv8 (Darknet backbone, PAN, decoupled head), sizes S / M / L | Sec. 3.2, 4.1 |
| Text encoder | CLIP Transformer text encoder, "CLIP-base (ViT-base)"; frozen in pretraining | Sec. 3.2, 4.2, 4.3 |
| Online vocabulary | <= M = 80 nouns per 4-image mosaic (positives + random negatives) | Sec. 3.2 |
| I-Pooling tokens | 27 (3 levels x 3x3) | Sec. 3.3 |
| Fusion blocks | 4 T-CSPLayers, 2 I-Pooling Attention blocks | Fig. 4 (our count) |
| Pretrain optimizer | AdamW, initial lr 0.002, weight decay 0.05 | Sec. 4.2 |
| Pretrain schedule | 100 epochs, 32 NVIDIA V100, total batch 512 | Sec. 4.2 (README: 4 nodes x 8 GPUs, code_only) |
| README config name | `yolo_world_l_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py` (name says 2e-4; paper says 0.002; not reconciled) | README, code_only |
| Augmentation | color, random affine, random flip, 4-image mosaic | Sec. 4.2 |
| Pretrain data | O365v1, GQA, Flickr30k (COCO images removed from GoldG), CC3M pseudo-labelled 246k | Sec. 4.2, Table 1 |
| Eval | LVIS minival zero-shot, Fixed AP, max 1000 predictions per image | Sec. 4.2 |
| Speed | 1 V100, no FP16 / TensorRT (Table 2); Table 6 FPS is WITH TensorRT | Sec. 4.1, Table 2, Table 6 captions |
| Fine-tune | 80 epochs, AdamW, lr 0.0002, wd 0.05, CLIP text encoder at 0.01x lr | Sec. 4.4, App. A.2 |
| COCO fine-tune | RepVL-PAN removed (all T-CSPLayers and I-Pooling) | Sec. 4.4, App. A.2 |
| LVIS fine-tune | train on LVIS-base (common + frequent), test rare as novel | Sec. 4.4 |
| Input size | not stated in paper; README: 640 for PT V2.1 models (code_only); our reading 640 | - |

Pretraining data, Table 1 (Dataset / Type / Vocab / Images / Anno.):
- Objects365V1 / Detection / 365 / 609k / 9,621k
- GQA / Grounding / - / 621k / 3,681k
- Flickr / Grounding / - / 149k / 641k
- CC3M-dagger / Image-Text / - / 246k / 821k

Pseudo-labelling pipeline (Sec. 3.4, App. B, Fig. 8): (1) n-gram noun extraction from the caption; (2) GLIP-L proposes boxes for those nouns; (3) CLIP image-text score `s^img`; (4) CLIP score of each crop with its noun `s^r_i`; optional re-labelling of a crop to its best noun; (5) rescoring `sqrt(c_i * s^r_i)`; per-noun NMS at 0.5, drop proposals below 0.3; (6) image score `sqrt(s^img * s^region)`, keep images above 0.3. "The whole pipeline is automatic without human verification." Result: 246k CC3M images, 821k pseudo annotations.

## 8. Results worth remembering
**Table 2, zero-shot LVIS minival, Fixed AP, FPS on 1 V100 without TensorRT.** YOLO-World cells: "re-parameterized (no bracket) / original (bracket)" for Params and FPS. Baseline rows are as reported in YOLO-World Table 2.

| Method | Backbone | Params | Data | FPS | AP | APr | APc | APf |
|---|---|---|---|---|---|---|---|---|
| MDETR | R-101 | 169M | GoldG | - | 24.2 | 20.9 | 24.3 | 24.2 |
| GLIP-T | Swin-T | 232M | O365,GoldG | 0.12 | 24.9 | 17.7 | 19.5 | 31.0 |
| GLIP-T | Swin-T | 232M | O365,GoldG,Cap4M | 0.12 | 26.0 | 20.8 | 21.4 | 31.0 |
| GLIPv2-T | Swin-T | 232M | O365,GoldG | 0.12 | 26.9 | - | - | - |
| GLIPv2-T | Swin-T | 232M | O365,GoldG,Cap4M | 0.12 | 29.0 | - | - | - |
| Grounding DINO-T | Swin-T | 172M | O365,GoldG | 1.5 | 25.6 | 14.4 | 19.6 | 32.2 |
| Grounding DINO-T | Swin-T | 172M | O365,GoldG,Cap4M | 1.5 | 27.4 | 18.1 | 23.3 | 32.7 |
| DetCLIP-T | Swin-T | 155M | O365,GoldG | 2.3 | 34.4 | 26.9 | 33.9 | 36.3 |
| YOLO-World-S | YOLOv8-S | 13M (77M) | O365,GoldG | 74.1 (19.9) | 26.2 | 19.1 | 23.6 | 29.8 |
| YOLO-World-M | YOLOv8-M | 29M (92M) | O365,GoldG | 58.1 (18.5) | 31.0 | 23.8 | 29.2 | 33.9 |
| YOLO-World-L | YOLOv8-L | 48M (110M) | O365,GoldG | 52.0 (17.6) | 35.0 | 27.1 | 32.8 | 38.3 |
| YOLO-World-L | YOLOv8-L | 48M (110M) | O365,GoldG,CC3M-dagger | 52.0 (17.6) | 35.4 | 27.6 | 34.1 | 38.0 |

- Text: "Compared to DetCLIP, YOLO-World achieves comparable performance (35.4 v.s. 34.4) while obtaining 20x increase in inference speed" (Sec. 4.2). 52.0 / 2.3 = 22.6 (arithmetic, our note).
- Our reading: the bracketed Params minus the plain Params is 62M-64M across S/M/L; that is roughly a CLIP ViT-B text tower, consistent with the text encoder being dropped after re-parameterization. The bracketed FPS includes running the text encoder.

**Table 3, pretraining-data ablation** (YOLO-World-L, zero-shot LVIS minival; table does not restate Fixed AP). AP / APr / APc / APf:
O365 23.5 / 16.2 / 21.1 / 27.0; O365,GQA 31.9 / 22.5 / 29.9 / 35.4; O365,GoldG 32.5 / 22.3 / 30.6 / 36.0; O365,GoldG,CC3M-dagger 33.0 / 23.6 / 32.0 / 35.5. Text: GQA adds 8.4 AP; CC3M (8% of the data) adds 0.5 AP and 1.3 APr.

**Table 4, RepVL-PAN ablation** (YOLO-World-L; GQA / T->I / I->T; AP / APr / APc / APf):
O365 only: none 22.4 / 14.5 / 20.1 / 26.0; T->I 23.2 / 15.2 / 20.6 / 27.0; both 23.5 / 16.2 / 21.1 / 27.0. With GQA: none 29.7 / 21.0 / 27.1 / 33.6; both 31.9 / 22.5 / 29.9 / 35.4. T->I = T-CSPLayers, I->T = I-Pooling Attention. Text: +1.1 AP on O365; larger gain with GQA (+2.2 AP, our arithmetic).

**Table 5, text encoder** (AP / APr / APc / APf): BERT-base frozen 14.6 / 3.4 / 10.7 / 20.0; BERT-base fine-tune 18.3 / 6.6 / 14.6 / 23.6; CLIP-base frozen 22.4 / 14.5 / 20.1 / 26.0; CLIP-base fine-tune 19.3 / 8.6 / 15.7 / 24.8. Fine-tuning used 0.01x lr. Paper's reason for the CLIP drop: fine-tuning on O365 (365 classes) "may degrade the generalization ability of the pre-trained CLIP". Our reading: Table 5's CLIP-frozen row equals Table 4's no-fusion row, so this ablation ran without RepVL-PAN, on O365 only.

**Table 6, COCO val2017** (AP / AP50 / AP75 / FPS; FPS on V100 WITH TensorRT):
- From scratch: YOLOv8-S 44.4 / 61.2 / 48.1 / 386; YOLOv8-M 50.5 / 67.3 / 55.0 / 238; YOLOv8-L 52.9 / 69.9 / 57.7 / 159 (YOLOv6, YOLOv7 rows also present).
- Zero-shot: S (O+G) 37.6; M (O+G) 42.8; L (O+G) 44.4; L (O+G+C) 45.1.
- Fine-tuned with RepVL-PAN: S 45.9; M 51.2; L (O+G+C) 53.3.
- Fine-tuned without RepVL-PAN: S 45.7 / 373 FPS; M 50.7 / 231; L (O+G+C) 53.3 / 156.

**Table 7, LVIS fine-tuning** (trained on LVIS-base; bbox AP; eval split not named in the caption; AP / APr / APc / APf):
ViLD 27.8 / 16.7 / 26.5 / 34.2; RegionCLIP 28.2 / 17.1; Detic 26.8 / 17.8; FVLM 24.2 / 18.6; DetPro 28.4 / 20.8 / 27.8 / 32.4; BARON 29.5 / 23.2 / 29.3 / 32.5. YOLOv8 trained on FULL LVIS (base + novel, class-balanced sampling, gray rows): S 19.4 / 7.4 / 17.4 / 27.0; M 23.1 / 8.4 / 21.3 / 31.5; L 26.9 / 10.2 / 25.4 / 35.8. YOLO-World: S 23.9 / 12.8 / 20.4 / 32.7; M 28.8 / 15.9 / 24.6 / 39.0; L 34.1 / 20.4 / 31.1 / 43.5. Text: L beats YOLOv8-L by 7.2 AP and 10.2 APr.

**Table 8, open-vocabulary instance segmentation** (mask AP on LVIS val2017 + box AP^b), L rows: COCO, seg head only 16.2 / APr 12.4, box APr 18.0; LVIS-base, seg head only 19.1 / 14.2, box APr 18.0; LVIS-base, all modules 28.7 / 15.0, box APr 17.4. Text: fine-tuning all gives +9.6 mask AP but -0.6 box APr.

**Table 9 (App. C), YOLO-World-S data scale** (Samples, AP / APr): O365 0.61M 16.3 / 9.2; O365+GoldG 1.38M 24.2 / 16.4; O365+CC3M-245k 0.85M 16.5 / 10.8; O365+CC3M-520k 1.13M 19.2 / 10.7; O365+CC3M-750k 1.36M 18.2 / 11.2.

**Paper-internal discrepancies (flag, do not reconcile):**
- Table 2 YOLO-World-L on O365,GoldG = 35.0 AP; Table 3 same data = 32.5 AP. Not explained.
- Sec. 4.3 text: CLIP gives "+10.1 AP for rare categories" over BERT; Table 5 gives APr 14.5 vs 3.4 (frozen) or 6.6 (fine-tuned). Neither difference is 10.1.
- "246k" (Sec. 3.4, Table 1) vs "CC3M-245k" (Table 9).
- "20x" vs 52.0 / 2.3 = 22.6.
- LaTeX comments hold extra rows (GLIP-L, 10k-detection fixed AP) and ms timings. They are not in the PDF; do not use.

## 9. Limitations and failure cases stated
- arXiv comment: "Work still in progress".
- Fine-tuning the CLIP text encoder on O365 hurts zero-shot LVIS badly (Table 5).
- Fine-tuning all modules for segmentation "may result in the degradation of the zero-shot capabilities" (Sec. 4.5; Table 8 box APr 18.0 -> 17.4).
- For small models, "using an excessive amount of pseudo-labeled data may have some negative effects" (App. C, Table 9: 520k -> 750k drops AP 19.2 -> 18.2).
- Pseudo labels are fully automatic, without human verification (App. B).
- No failure-case figure; Fig. 5-7 show only successes.

## 10. Outline feedback
1. Frame line "text fused into the neck, then folded into conv weights": true for the T-CSPLayer gate (Eq. 4, 1x1 conv whose filters are text embeddings) and the classifier (App. A.2). But I-Pooling Attention takes 27 tokens from the current image (Eq. 5-6), so the text side stays image-dependent after re-parameterization. Suggest: "text gates the neck and becomes 1x1 conv weights; a small image-to-text attention over 27 pooled tokens still runs per image".
2. Archflow shapes (640 -> 80x80 / 40x40 / 20x20 -> 8400) are not in the paper. Label them "our reading (YOLOv8 convention)"; the README's 640 is code_only and refers to V2.1 models. "Vocab of at most 80 nouns per mosaic" is sourced (Sec. 3.2).
3. The from-scratch YOLO part (Darknet, C2f internals, TAL formula, DFL, anchor points) is only named in this paper. Teach it from an outside source (Ultralytics YOLOv8 docs, TOOD, GFL) or label it as background. Fig. 4 is enough for the C2f / T-CSPLayer picture.
4. `viz-speed`: metric is Fixed AP on LVIS minival, max 1000 detections, 1 V100 without TensorRT. Plot both YOLO-World FPS values (re-parameterized vs with text encoder): 74.1 vs 19.9 for S. Use a log FPS axis (0.12 to 74.1). Baseline rows are "as reported in YOLO-World Table 2"; cross-check them with the GLIP and Grounding DINO agents. Never place Table 6 FPS (TensorRT) on the same axis.
5. Text encoder: write "CLIP ViT-base text encoder, frozen". B/32 is not verifiable from the paper or README.
6. Equations: add App. Eq. 4 for `btn-repvl-reparam` (it is the literal "text becomes a 1x1 conv" step). Quote Eq. 6 only with a typo note, or skip it.
7. Ablation numbers (Tables 3-5) come from a different setting than Table 2 (32.5 vs 35.0 for the same data). Do not mix them in one chart.
8. Cross-links: Table 5 (fine-tuning CLIP on 365 classes destroys rare-class AP: APr 14.5 -> 8.6) supports chapter 12/13 "fine-tuning forgets". The GLIP-L -> CLIP-filter pseudo-labelling is a teacher-student loop like OWLv2 (chapter 6).
9. Venue: say "CVPR 2024 (per the repo)", or cite arXiv v3; arXiv itself gives no venue.
10. YOLOE (branch): it REMOVES YOLO-World's cross-modal fusion (YOLOE Table 5) and puts text back at the classifier only, with a foldable FFN on cached text embeddings. Color it "classifier only" in the family tree. It also belongs in chapter 11 (visual prompts, SAVPE). See notes/papers/yoloe.md.
