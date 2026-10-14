# Prehistory delta cards (ch-before): FCN, U-Net, Mask R-CNN, MaskFormer, Mask2Former

Delta cards, not full extractions. Architecture lens, light. All numbers from LaTeX unless noted.
Shared theme for the chapter: every model below is trained on one labelled dataset with a fixed class list
(PASCAL VOC 21 incl. background; ISBI cell data with 2 output classes; COCO 80 things; COCO panoptic 133;
ADE20K 150).

## Task definitions as the papers state them
- Semantic: "each pixel is labeled with the class of its enclosing object or region" (FCN, Sec. 1).
  "classify each pixel into a fixed set of categories without differentiating object instances"
  (Mask R-CNN, Sec. 1, with footnote: "per-pixel classification without differentiating instances").
- Instance: "requires the correct detection of all objects in an image while also precisely segmenting each
  instance"; it "combines elements" of detection (box per object) and semantic segmentation (Mask R-CNN, Sec. 1).
- Panoptic: "has been proposed to unify both semantic and instance segmentation tasks" (Mask2Former, Sec. 2,
  citing Kirillov et al.). Operationally: "Panoptic and semantic segmentation tasks are evaluated on the union
  of 'things' and 'stuff' categories while instance segmentation is only evaluated on the 'things' categories"
  (Mask2Former, Sec. 4, Datasets). COCO = 80 things + 53 stuff = 133.
- Universal (Mask2Former abstract): "a new architecture capable of addressing any image segmentation task
  (panoptic, instance or semantic)". Intro caveat: "universal architectures are still trained separately for
  different tasks and datasets, albeit having the same architecture."

---

## 1. FCN (fcn)
- Identity: "Fully Convolutional Networks for Semantic Segmentation", arXiv 1411.4038, read v2. v1 2014-11-14,
  v2 2015-03-08. Jonathan Long ... Trevor Darrell (with Evan Shelhamer). Comments: "to appear in CVPR (2015)".
- Key idea (first in list; delta from patchwise/hybrid convnets): turn a classification net (VGG16) into a
  fully convolutional net trained "end-to-end, pixels-to-pixels", so one forward pass gives a class score map
  for every pixel. Then fuse coarse deep layers with finer shallow ones ("skip") to recover detail.
- Mechanism:
  1. Convolutionalize VGG16: fc layers become convolutions; output is a coarse score map at stride 32
     (Table 1: "max stride 32" for AlexNet, VGG16, GoogLeNet).
  2. Append a 1x1 conv with 21 channels = "scores for each of the PASCAL classes (including background)" at each
     coarse location (Sec. 4.1).
  3. FCN-32s: one in-network deconvolution "bilinearly upsample[s]" stride-32 scores to pixels (Sec. 4.1, Fig. 3).
  4. FCN-16s: 1x1 score layer on pool4 (stride 16) + 2x upsampling of conv7 scores, summed; then upsample to
     image. "We initialize the 2x upsampling to bilinear interpolation, but allow the parameters to be learned"
     (Sec. 4.2). Initialized from FCN-32s.
  5. FCN-8s: same again with pool3 (stride 8).
  6. Loss: "per-pixel multinomial logistic loss" (softmax cross-entropy over 21 classes), whole-image training.
     "Final layer deconvolutional filters are fixed to bilinear interpolation, while intermediate upsampling
     layers are initialized to bilinear upsampling, and then learned" (Sec. 4.3, Dense Prediction).
- Toy-figure shapes (our reading): HxW image -> H/32 x W/32 x 21 scores (32s); fuse at H/16 (16s), H/8 (8s);
  final upsample x32 / x16 / x8 to H x W x 21; argmax per pixel.
- Equations (Sec. 3): loss summed over spatial positions, TeX verbatim:
  `\ell(\mathbf x; \theta) = \sum_{ij} \ell'(\mathbf x_{ij}; \theta)` -- "its gradient will be a sum over the
  gradients of each of its spatial components". Mean IU (Sec. 5):
  `(1/n_{\text{cl}}) \sum_i n_{ii} /\left(t_i + \sum_j n_{ji} - n_{ii}\right)` (n_ij = pixels of class i
  predicted as j, t_i = pixels of class i, n_cl = number of classes).
- Results:
  - Table 2 (VOC2011 val subset, mean IU): FCN-32s 59.4, FCN-16s 62.4, FCN-8s 62.7 (FCN-32s-fixed 45.4).
  - Table 3 (test, mean IU): FCN-8s 62.7 VOC2011, 62.2 VOC2012, ~175 ms; SDS 52.6 / 51.6, ~50 s. Abstract:
    "20% relative improvement to 62.2% mean IU on 2012".
  - Training data: VOC2011 train 1112 images; SBD-extended 8498 images (Sec. 4.3).
  - Fine-tuning: "three days on a single GPU for the coarse FCN-32s version, and about one day each to upgrade
    to the FCN-16s and FCN-8s versions" (Sec. 4.3).
- Limitation stated: "The 32 pixel stride at the final prediction layer limits the scale of detail" (Sec. 4.2);
  gains past 8s diminish. No instances: one label per pixel.

## 2. U-Net (unet)
- Identity: "U-Net: Convolutional Networks for Biomedical Image Segmentation", arXiv 1505.04597v1, 2015-05-18.
  Olaf Ronneberger ... Thomas Brox (with Philipp Fischer). "conditionally accepted at MICCAI 2015".
- Key idea (delta from FCN): make the upsampling path as wide as the downsampling path ("a large number of
  feature channels" in the upsampling part) so it is "more or less symmetric" -> a U; skips concatenate
  feature maps (not sum scores); trains from very few images with heavy elastic-deformation augmentation.
- Mechanism (Sec. 2, Fig. 1):
  1. Contracting path: repeated [two 3x3 unpadded convs + ReLU] then 2x2 max pool stride 2. "At each
     downsampling step we double the number of feature channels": 64 -> 128 -> 256 -> 512 -> 1024.
  2. Five resolution levels in Fig. 1 (four poolings) (our reading of the figure): 572^2 -> 570 -> 568 | 284 ->
     282 -> 280 | 140 -> 138 -> 136 | 68 -> 66 -> 64 | 32 -> 30 -> 28 ("32x32 pixels in the lowest resolution").
  3. Expansive path: upsample + 2x2 "up-convolution" that halves channels, "concatenation with the
     correspondingly cropped feature map from the contracting path", two 3x3 convs + ReLU. "The cropping is
     necessary due to the loss of border pixels in every convolution."
  4. Final 1x1 conv maps 64 channels to classes (2 in Fig. 1). "In total the network has 23 convolutional
     layers." Input tile 572x572 -> output map 388x388 (Fig. 1).
  5. Overlap-tile: valid convs only, so output covers pixels with full context; tile large images, mirror at
     borders (Fig. 2). Input tile must give even sizes at every 2x2 pool.
  6. Weighted loss for touching cells: precomputed weight map w(x) puts large weight on thin background
     borders between touching cells (Fig. 3d). Batch of 1 image, momentum 0.99.
- Equations (Sec. 3), TeX verbatim:
  - softmax: `{p}_k(\vec{x}) = \exp({a_k(\vec{x})}) / \left(\sum_{k' = 1}^K \exp(a_{k'}(\vec{x}))\right)`
  - energy: `E = \sum_{\vec{x} \in \Omega} w(\vec{x}) \log({p}_{\ell(\vec{x})}(\vec{x}))` (sic: no minus sign
    in source; text says it is the cross entropy that "penalizes ... the deviation of p_l(x)(x) from 1").
  - weight map: `w(\vec{x}) = w_c(\vec{x}) + w_0 \cdot \exp\left( - \frac{(d_1(\vec{x}) + d_2(\vec{x}))^2}{2\sigma^2}\right)`
    w_c = class-frequency balance; d1, d2 = distance to border of nearest and second-nearest cell;
    `w_0 = 10`, `\sigma \approx 5` pixels. (`\mathds` needs dsfont.)
- Results:
  - Table 1 (ISBI EM, warping error): u-net 0.000353, rank 1; IDSIA (Ciresan sliding window) 0.000420.
    Text: rand error 0.0382.
  - Table 2 (ISBI cell tracking 2015, IOU): PhC-U373 0.9203 vs second-best 0.83; DIC-HeLa 0.7756 vs 0.46.
    Text: "92%" and "77.5%".
  - Data: EM 30 training images (512x512); PhC-U373 35 partially annotated; DIC-HeLa 20 partially annotated.
  - "10 hours on a NVidia Titan GPU (6 GB)" (Sec. 5); 512x512 "less than a second" (abstract).
- Classes: binary per-dataset (cell vs membrane/background); one model per dataset.

## 3. Mask R-CNN (maskrcnn)
- Identity: "Mask R-CNN", arXiv 1703.06870, read v3. v1 2017-03-20, v3 2018-01-24. Kaiming He ... Ross
  Girshick (with Georgia Gkioxari, Piotr Dollar). Code: github.com/facebookresearch/Detectron.
- Key idea (delta from FCN/U-Net): instance-first. Detect each object (Faster R-CNN), then run a small FCN on
  each region to predict its binary mask, "in parallel" with class and box. Two fixes make it work:
  RoIAlign (no quantization) and decoupled per-class sigmoid masks.
- Mechanism (Sec. 3):
  1. Backbone (ResNet/ResNeXt-50/101, C4 or FPN) on the whole image; RPN proposes boxes.
  2. RoIAlign: "we use x/16 instead of [x/16]"; bilinear interpolation "at four regularly sampled locations in
     each RoI bin"; no rounding of RoI or bins. RoIPool quantizes twice (RoI, then bins) -> misalignment.
  3. Box head: class + box offset. Mask head (FPN, Fig. 4): RoI 14x14x256 -> four 3x3 convs (x4) -> deconv
     2x2 stride 2 -> 28x28x256 -> 1x1 conv -> 28x28x80. Output is `Km^2`: K binary masks of m x m, one per class.
  4. Per-pixel sigmoid + average binary cross-entropy; for an RoI of GT class k "L_mask is only defined on
     the k-th mask". The class branch picks which mask to use. No competition between classes.
  5. Inference: box branch on proposals, NMS, mask branch on top 100 boxes; take mask k of predicted class;
     resize m x m float mask to the RoI, threshold at 0.5.
- Equation (Sec. 3, inline, verbatim): `L = L_{cls} + L_{box} + L_{mask}` ("multi-task loss on each sampled RoI";
  L_cls, L_box as in Fast R-CNN). Mask target = intersection of RoI and its GT mask; positive RoIs only (IoU>=0.5).
- Results:
  - Table 1 (COCO test-dev mask AP): ResNet-101-FPN 35.7; ResNeXt-101-FPN 37.1; FCIS+++ 33.6; MNC 24.6.
  - Table 2b (R50-C4, minival): softmax 24.8 vs sigmoid 30.3 AP (+5.5).
  - Table 2c (R50-C4): RoIPool 26.9, RoIAlign 30.3 AP (avg) / 30.2 (max). Table 2d (C5, stride 32): RoIPool 23.6
    vs RoIAlign 30.9 (+7.3).
  - Table 2e: FCN mask head 33.6 vs MLP 31.5.
  - Speed: 195ms per image on Tesla M40 (+15ms CPU); "5 fps". Training R50-FPN 32 hours on 8 GPUs; R101-FPN 44 h.
- Classes: COCO, 80 (seen as the 80-channel mask output in Fig. 4 and "80*28^2" in Table 2e; the paper does not
  state "80 categories" in prose. Our reading: COCO detection 80).
- Limitation implied by the design: bounding boxes as the segment representation; Mask2Former Sec. 3.1:
  "uses bounding boxes as the representation which limits its application to semantic segmentation."

## 4. MaskFormer (maskformer)
- Identity: "Per-Pixel Classification is Not All You Need for Semantic Segmentation", arXiv 2107.06278, read v2.
  v1 2021-07-13, v2 2021-10-31. Bowen Cheng ... Alexander Kirillov (with Alexander G. Schwing). NeurIPS 2021,
  Spotlight.
- Key idea (delta from Mask R-CNN and FCN): no boxes and no per-pixel softmax. Predict a set of N
  (class, binary mask) pairs with DETR-style queries; the same model, loss and training solve semantic and
  panoptic segmentation. "mask classification is sufficiently general to solve both semantic- and
  instance-level segmentation tasks in a unified manner".
- Mechanism (Sec. 3.3, Fig. 2; config Sec. 4.1):
  1. Backbone -> feature map F, `C_F x H/S x W/S`, S = 32.
  2. Pixel decoder (light FPN, 256 channels, stride 32 -> 4) -> per-pixel embeddings E_pixel, `C_E x H x W` in
     the formula; implemented at stride 4 (Sec. 4.1) (our reading: H/4 x W/4), C_E = 256.
  3. Transformer decoder (DETR design, 6 layers, 100 queries by default) attends to F -> N per-segment
     embeddings Q, `C_Q x N`.
  4. Linear + softmax on Q -> p_i over K+1 classes (K + "no object"). MLP (2 hidden layers, 256) -> mask
     embeddings E_mask, `C_E x N`.
  5. Mask i = sigmoid(dot of mask embedding i with every pixel embedding). Masks may overlap (no softmax across
     masks, "beneficial").
  6. Train: bipartite matching with cost `-p_{i}(c_{j}^{\text{gt}}) + \mathcal{L}_{\text{mask}}(m_{i},  m_{j}^\text{gt})`,
     then L_mask-cls. Inference semantic: argmax_c sum_i p_i(c) m_i[h,w].
- Equations (TeX verbatim, `\x` = `\times`):
  - per-pixel CE (Sec. 3.1): `\mathcal{L}_{\text{pixel-cls}}(y,  y^\text{gt}) = \sum\nolimits_{i=1}^{H \cdot W}-\log p_{i}( y_{i}^\text{gt})`
  - output set (Sec. 3.2): `z = \{(p_{i}, m_{i})\}_{i=1}^{N}`
  - Eq. 1 (`eq:mask_cls`): `\mathcal{L}_{\text{mask-cls}}(z, z^\text{gt}) = \sum\nolimits_{j=1}^N \left[-\log p_{\sigma(j)}( c_{j}^\text{gt}) + \mathds{1}_{c_{j}^\text{gt}\neq\varnothing} \mathcal{L}_{\text{mask}}(m_{\sigma(j)},  m_{j}^\text{gt})\right].`
    sigma = the matching; mask loss only for matched non-empty GT. (`\mathds` needs dsfont.)
  - mask (Sec. 3.3): `m_{i}[h, w] = \text{sigmoid}(\mathcal{E}_\text{mask}[:, i]^{\text{T}} \cdot \mathcal{E}_\text{pixel}[:, h, w])`
  - L_mask = lambda_focal * focal + lambda_dice * dice, `\lambda_{\text{focal}} = 20.0`, `\lambda_{\text{dice}} = 1.0`;
    no-object class weight 0.1 (Sec. 4.1).
- Results:
  - Table 1 (ADE20K val, 150 categories): Swin-L (IN-22K) 55.6 mIoU m.s. (54.1 s.s.); prior SOTA Swin-UperNet 53.5.
  - Table 2 (R50, mIoU): gain over PerPixelBaseline+ grows with classes: Cityscapes 19 classes +0.0 (78.5);
    ADE20K 150 +2.6 (44.5); COCO-Stuff 171 +2.9 (37.1); ADE20K-Full 847 +3.5 (17.4).
  - Table 3 (COCO panoptic val, 133 categories): Swin-L 52.7 PQ; R50 46.5 PQ vs DETR 43.4.
  - Number of queries (wrap table, Sec. 4.4): 100 best on ADE20K/COCO-Stuff/ADE20K-Full; 1000 drops to 35.4 mIoU on ADE20K.
- Datasets/classes (Sec. 4): ADE20K 150, COCO-Stuff-10K 171, Cityscapes 19, Mapillary Vistas 65; COCO panoptic
  80 things + 53 stuff; ADE20K-Panoptic 100 things + 50 stuff.
- Limitation (Mask2Former Sec. 2): "MaskFormer cannot segment instances well"; memory: "can only fit a single
  image in a GPU with 32G memory" (Mask2Former Sec. 3.3). Fixed by Mask2Former.

## 5. Mask2Former (m2f)
- Identity: "Masked-attention Mask Transformer for Universal Image Segmentation", arXiv 2112.01527, read v3.
  v1 2021-12-02, v3 2022-06-15. Bowen Cheng ... Rohit Girdhar (with Ishan Misra, Alexander G. Schwing,
  Alexander Kirillov). CVPR 2022.
- Key idea (delta from MaskFormer): same meta-architecture, new Transformer decoder. Masked attention: each
  query's cross-attention only looks inside the foreground of its own mask predicted by the previous layer.
  Plus multi-scale features round robin, self/cross order swapped, learnable query features, no dropout, and a
  point-sampled mask loss. First universal model to beat specialized ones on all three tasks.
- Mechanism (Sec. 3, Fig. 2; config Sec. 4.1):
  1. Backbone; pixel decoder = 6 MSDeformAttn layers on 1/8, 1/16, 1/32 maps, then upsample to 1/4 = per-pixel
     embeddings.
  2. N learnable query features X_0 (100 queries default; 200 for the best Swin-L models) predict M_0 before the
     decoder.
  3. Each decoder layer: masked cross-attention (with mask M_{l-1} binarized at 0.5 and resized to K_l), then
     self-attention, then FFN ("we switch the order of self- and cross-attention").
  4. Round robin scales: layers get 1/32, 1/16, 1/8 in turn, each with sinusoidal pos. embedding + learnable
     level embedding; the 3-layer block repeats L = 3 times -> 9 layers.
  5. After each layer, masks = query . per-pixel embedding (MaskFormer head) -> next layer's attention mask;
     auxiliary loss at every layer and on X_0.
  6. Loss on K = 12544 = 112x112 sampled points, uniform for matching cost, importance sampling for the final
     loss: memory 18GB -> 6GB per image. L_mask = 5.0 BCE + 5.0 dice; lambda_cls 2.0 (0.1 for no object).
- Equations (Sec. 3.2.1, TeX verbatim; `\mathbfcal` = `\DeclareMathAlphabet\mathbfcal{OMS}{cmsy}{b}{n}`, `\x` = `\times`):
  - standard: `\mathbf{X}_{l} = \text{softmax}(\mathbf{Q}_{l}\mathbf{K}^{\text{T}}_{l})\mathbf{V}_{l} + \mathbf{X}_{l-1}.`
  - masked: `\mathbf{X}_{l} = \text{softmax}(\mathbfcal{M}_{l-1} + \mathbf{Q}_{l}\mathbf{K}^{\text{T}}_{l})\mathbf{V}_{l} + \mathbf{X}_{l-1}.`
  - mask: `\mathbfcal{M}_{l-1}(x, y) = \left\{\begin{array}{ll} 0  & \text{if~} \mathbf{M}_{l-1}(x,y)=1 \\ -\infty & \text{otherwise} \end{array}\right..`
  - Gloss: X_l in R^{N x C} query features; Q_l = f_Q(X_{l-1}); K_l, V_l in R^{H_l W_l x C} image features at
    the layer's scale; M_{l-1} in {0,1}^{N x H_l W_l}; adding -inf zeroes attention outside the predicted mask.
- Results ("universal" table rows; each model trained per task/dataset):
  - Table 1 (COCO panoptic val2017, 133 categories): Swin-L, 200 queries, 100 epochs: 57.8 PQ (MaskFormer
    Swin-L 52.7; K-Net 54.6). R50: 51.9 PQ in 50 epochs vs MaskFormer 46.5 in 300.
  - Table 2 (COCO instance val2017, 80 categories): Swin-L 50.1 AP (Swin-HTC++ 49.5); R50 43.7 vs Mask R-CNN
    (400 epochs) 42.5.
  - Table 3 (ADE20K val, 150 categories): Swin-L-FaPN 57.7 mIoU m.s. (Swin-L 57.3); BEiT-UperNet 57.0.
  - Table 4a: removing masked attention costs -5.9 AP, -4.8 PQ, -1.7 mIoU (R50). Table 4c: cross-attention 37.8
    AP vs masked attention 43.7.
- Limitations (Sec. 4.6, Table 7): still "needs to be trained for those specific tasks": panoptic-trained COCO
  model 41.7 AP / 61.7 mIoU vs task-trained 43.7 AP / 61.5 mIoU; "struggles with segmenting small objects".
  Goal stated: "a model that can be trained only once for multiple tasks and even for multiple datasets."

---

## Outline feedback (ch-before)
- The outline says "Four ways"; the scene tabs are FCN / U-Net / Mask R-CNN / Mask2Former. MaskFormer and
  Mask2Former share the same output head (query . pixel embedding -> sigmoid), so one tab is fine; label it
  "MaskFormer / Mask2Former" and put masked attention in the delta card.
- FCN and U-Net are both "label every pixel", but U-Net's actual data is 2-class biomedical, not a class list;
  the "fixed list from one labelled dataset" point is strongest with FCN (VOC 21), Mask R-CNN (COCO 80),
  MaskFormer (Table 2: 19 -> 847 classes) and Mask2Former (133 / 80 / 150). For U-Net say "one dataset, one
  model, binary classes".
- Good hook for later chapters: Mask2Former's own stated goal (train once for multiple tasks and datasets) and its
  "trained separately" caveat, which SAM's promptable task answers differently.
- MaskFormer source inconsistency: Sec. 4 text says ADE20K-Full keeps "874 classes"; Table 2 header says
  "847 classes"; the source figure file is ade20k_full_847.pdf. Cite 847.
- MaskFormer's E_pixel is written as `C_E x H x W` but implemented at stride 4; the toy can show H/4.
