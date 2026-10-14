# ViLD -- Open-vocabulary Object Detection via Vision and Language Knowledge Distillation

## 1. Identity
- Title: "Open-vocabulary Object Detection via Vision and Language Knowledge Distillation"
- arXiv 2104.13921, read v3. v1 2021-04-28 (Wed, 28 Apr 2021 17:58:57 UTC). v2 2021-10-13. v3 2022-05-12 (abs.html submission history).
- Authors: Xiuye Gu (first), Tsung-Yi Lin, Weicheng Kuo, Yin Cui (last). Google Research; Lin at Nvidia (work done at Google).
- Venue: ICLR 2022 (abs.html Comments "ICLR Camera Ready"; src uses iclr2022_conference.sty).
- Repo: https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild
- Source read: `src/arxiv_after_camera_ready.tex` (equations, tables), `paper.pdf` (page and table numbers).

## 2. Core problem
Detect objects named by arbitrary text, while detection boxes exist only for base classes. Detection datasets cannot scale to long-tail vocabularies: LVIS has 1,203 categories and by Zipf's law the rare ones need much more data (Sec. 1). Image-text models (CLIP, ALIGN) know many concepts, but at image level, not object level.

## 3. What prior methods did and why that hurt (per the paper)
- Zero-shot detection (ZSD) aligns region features to pretrained text embeddings (GloVe-style) on base classes; "there is a large performance gap to supervised counterparts" (Sec. 2). In this paper GloVe as classifier gives 3.0 APr vs 10.1 with CLIP text (Table 3).
- OVR-CNN (Zareian et al. 2021) pretrains the backbone on image captions, then fine-tunes on detection. ViLD instead uses an off-the-shelf classifier as teacher, no detection-tailored pretraining (Sec. 2, 4.5).
- All prior methods were evaluated on tens of categories; ViLD claims to be the first on more than 1,000 (Sec. 2).
- The naive R-CNN-style fix (crop every proposal, classify with CLIP) works on novel classes but is "very slow for inference, because it feeds object proposals one-by-one into the classification model" (Sec. 1). Its scores also ignore box quality (App. B).

## 4. The one key idea
Make the detector's region embedding live in CLIP's space. Two signals: (a) ViLD-text classifies region embeddings with frozen CLIP text embeddings of base classes plus a learned background vector; (b) ViLD-image regresses the region embedding onto CLIP's image embedding of the cropped proposal (L1). (b) reaches proposals that contain novel objects, which no label touches. At test time, add novel class names as more text rows.

## 5. Mechanism, step by step
1. Localize class-agnostically (Sec. 3.1): take Mask R-CNN, replace class-specific box regression and mask layers with class-agnostic ones (one box and one mask per RoI for all classes).
2. Offline teacher pass (Sec. 3.2): train an RPN on base classes C_B, extract M proposals per training image offline; crop each at 1x and 1.5x (more context), resize, run the frozen CLIP image encoder V, sum the two embeddings and L2-normalize (Eq. 1). Store the M embeddings.
3. Offline text pass (Sec. 3.2, App. D): for each class name, fill 63 prompt templates (e.g. "a photo of {category} in the scene"), also synonyms when provided, run the frozen CLIP text encoder T, ensemble. This gives T(C_B) for training and T(C_B u C_N) at test.
4. Student forward (Sec. 3.3): image -> backbone phi -> RPN (online proposals P, N = |P| per image) -> RoIAlign -> lightweight head R -> projection layer (resizes region embedding to the text-embedding dim; Fig. 3 caption) -> region embedding e_r. "we take outputs before the classification layer as region embeddings."
5. ViLD-text loss (Eq. 2): cosine of e_r with e_bg and every t_i in T(C_B), divide by tau = 0.01, softmax, cross-entropy against the matched label (unmatched proposals -> background).
6. ViLD-image loss (Eq. 3): for the M offline proposals, run the same head R on their boxes and take L1 distance to the ensembled CLIP crop embedding. Labels are not used, so novel objects in proposals get a training signal.
7. Total (Eq. 4): L_text + w * L_image; w = 0.5 for the ResNet-50 ViLD row.
8. Inference: same architecture for ViLD-text, ViLD-image and ViLD; classifier = text embeddings of C_B u C_N (sometimes C_N only), "ViLD-image distillation only happens in training time" (Sec. 3.3). Max 300 detections per image (Sec. 4).
9. Variants (Sec. 3.4): ViLD-text+CLIP (R-CNN style: ensemble a ViLD-text detector's scores with CLIP-on-crop scores via Eq. 5; slow). ViLD-ensemble: two heads of identical architecture, one trained with Eq. 2 and one with Eq. 3, scores combined with Eq. 5 (p_ViLD-image replaces p_cls). Separate heads reduce the "competition" between the two losses.

Trainable vs frozen (Sec. 3, 4, Fig. 3; trainable list is our reading of "train the detector with ViLD-text from scratch"):
- Trainable: backbone (from scratch, no ImageNet init stated for ViLD; "We train the model from scratch for 180,000 iterations", Sec. 4), FPN, RPN, RoI head R, projection layer, class-agnostic box and mask heads, background embedding e_bg.
- Frozen: CLIP image encoder V and text encoder T. Their outputs are precomputed offline (text embeddings "offline", Sec. 3.2; "precompute the M image embeddings", Sec. 3.3). The gradient stops at the stored embeddings.
- Note: the PASCAL VOC supervised baseline in Table 5 is ImageNet-initialized; ViLD itself is from scratch.

## 6. Equations
Custom macros: none needed beyond standard `\mathcal`, `\mathbf`, `\tilde`, `\widetilde`. The source writes the offline proposal set both as `\tilde{P}` (prose) and `\widetilde{P}` (equations); kept as written. Eq. numbers verified against the PDF.

### Eq. (1) -- crop ensemble (Sec. 3.2, `\begin{equation}`, unlabelled, number 1)
```tex
\mathcal{V}(\text{crop}(I, \tilde{r}_{\{1\times, 1.5\times\}})) = \frac{\mathbf{v}}{\|\mathbf{v}\|}, ~\text{where}~\mathbf{v} = \mathcal{V}(\text{crop}(I, \tilde{r}_{1\times})) + \mathcal{V}(\text{crop}(I, \tilde{r}_{1.5\times})).
```
Paper: "We ensemble the image embeddings from 1x and 1.5x crops, as the 1.5x crop provides more context cues. The ensembled embedding is then renormalized to unit norm."
- `I`: the image. `\tilde{r}`: one offline proposal box.
- `crop(I, r_{1x})`: the box itself; `r_{1.5x}`: the box enlarged 1.5x for context. Crops are resized to 224x224 (App. B).
- `\mathcal{V}`: frozen CLIP image encoder. `v`: sum of the two embeddings; divided by its norm -> unit vector, the distillation target.

### Eq. (2) -- ViLD-text (Sec. 3.3, `align` block, `\label{eq:vild-text}`; first two lines are `\nonumber`, only the loss line carries (2))
```tex
\begin{align}
    &\mathbf{e}_r = \mathcal{R}(\phi(I), r) \nonumber \\
    &\mathbf{z}(r) = 
    \begingroup % keep the change local
    \setlength\arraycolsep{2pt}
    \begin{bmatrix}
    sim(\mathbf{e}_r, \mathbf{e}_{bg}), & 
    sim(\mathbf{e}_r, \mathbf{t}_1), &
    \cdots, & 
    sim(\mathbf{e}_r, \mathbf{t}_{|C_B|})
    \end{bmatrix}
    \endgroup
    \nonumber \\
    &\mathcal{L}_{\text{ViLD-text}} = \frac{1}{N}\sum_{r \in P}\mathcal{L}_\text{CE}\Big(softmax \big(\mathbf{z}(r) / \tau \big), y_r \Big), \label{eq:vild-text}
\end{align}
```
followed by: "where $sim(\mathbf{a}, \mathbf{b}) = \mathbf{a}^\top \mathbf{b} / (\|\mathbf{a}\|\|\mathbf{b}\|)$, $\mathbf{t}_i$ denotes elements in $\mathcal{T}(C_B)$, $y_r$ denotes the class label of region $r$, $N$ is the number of proposals per image ($|P|$), and $\mathcal{L}_{CE}$ is the cross entropy loss."
Paper: "Since the text 'background' does not well represent these unmatched proposals, we allow the background category to learn its own embedding e_bg ... Then we apply softmax activation with a temperature tau to compute the cross entropy loss."
- `\phi(I)`: backbone features. `r`: an online RPN proposal in `P`. `\mathcal{R}`: lightweight head (incl. projection) -> region embedding `e_r`.
- `e_bg`: learnable background vector (the only learned "row" of the classifier). `t_i`: frozen CLIP text embedding of base class i; |C_B| = 866 on LVIS.
- `z(r)`: 1 + |C_B| cosine similarities. `tau` = 0.01 (Sec. 4), so logits span [-100, 100].
- `y_r`: matched base label or background. `N`: proposals per image (value not stated).

### Eq. (3) -- ViLD-image (Sec. 3.3, `\label{eq:vild-image}`)
```tex
\mathcal{L}_{\text{ViLD-image}} &= \frac{1}{M} \sum_{\tilde{r} \in \widetilde{P}} \| \mathcal{V}(\text{crop}(I, \tilde{r}_{\{1\times, 1.5\times\}})) - \mathcal{R}(\phi(I), \tilde{r}) \|_1. \label{eq:vild-image}
```
Paper: "We apply an L1 loss between the region and image embeddings to minimize their distance. The ensembled image embeddings ... are used for distillation." "These proposals can contain objects in both C_B and C_N, as the network can generalize. In contrast, ViLD-text can only learn from C_B."
- `\widetilde{P}`: the M offline proposals of this image (from an RPN trained on base classes). `M`: value not stated.
- First term: Eq. 1 teacher target (unit norm). Second term: student region embedding for the same box.
- `\|.\|_1`: L1 norm. Table 7 shows L1 beats L2 on APr.

### Eq. (4) -- total (Sec. 3.3, `\label{eq:vild}`)
```tex
\mathcal{L}_{\text{ViLD}} = \mathcal{L}_{\text{ViLD-text}} + w \cdot \mathcal{L}_{\text{ViLD-image}}, \label{eq:vild}
```
Paper: "where w is a hyperparameter weight for distilling the image embeddings."
- `w`: 0.5 for R50 ViLD in Table 3; 1.0 for R152 ViLD and the EfficientNet-b7 ensembles; 2.0 for R152 ViLD-ensemble (Table 8). ViLD-image alone is trained with weight 1.0 (Sec. 4.4).

### Eq. (5) -- ensemble (Sec. 3.4, `\label{eq:ensemble}`)
```tex
&p_{i, \text{ensemble}} = \begin{cases}
    p_{i, \text{ViLD-text}}^\lambda \cdot p_{i, \text{cls}}^{(1-\lambda)}, & \text{if } i \in C_B\\
    p_{i, \text{ViLD-text}}^{(1-\lambda)} \cdot p_{i, \text{cls}}^\lambda. & \text{if } i \in C_N
\end{cases} \label{eq:ensemble}
```
Paper: "we introduce a weighted geometric average for the ensemble ... lambda is set to 2/3, which weighs the prediction of ViLD-text more on base categories and vice versa."
- `p_{i,ViLD-text}`: detector score for class i on proposal r~. `p_{i,cls}`: teacher (CLIP on crop) score; in ViLD-ensemble it is replaced by `p_{i,ViLD-image}` from the second head.
- `lambda` = 2/3: base classes trust the text-trained head (2/3), novel classes trust the image-distilled branch (2/3).

Eqs. 6-8 (Sec. 4.7, attribute expansion Pr(v_i, a_j | e_r) = Pr(v_i | e_r) Pr(a_j | e_r)) are qualitative-only; not needed for the page.

## 7. Config / setup
| Item | Value | Location |
|---|---|---|
| Detector | Mask R-CNN, ResNet FPN backbone, class-agnostic box and mask heads | Sec. 3.1, Sec. 4 |
| Backbone (main) | ResNet-50; also ResNet-152 (Table 8), EfficientNet-b7 (stronger teachers) | Table 3, 8 |
| Input size | 1024x1024 | Sec. 4 |
| Augmentation | large-scale jittering [0.1, 2.0] ([0.5, 2.0] for EfficientNet-b7 runs) | Sec. 4, App. D |
| Batch | 256 (sync BN batch size) | Sec. 4 |
| Optimizer details | weight decay 4e-5, initial LR 0.32; LR /10 at 0.9x, 0.95x, 0.975x of iterations (optimizer type not stated) | Sec. 4 |
| Schedule | 180,000 iterations, from scratch (LVIS); 45,000 on COCO; 84,375 for the Table 7 sweep | Sec. 4, 4.5, Table 7 |
| Teacher | CLIP ViT-B/32, 224x224 input; stronger: CLIP ViT-L/14 (768-d), ALIGN EfficientNet-l2 + BERT-large (1,376-d) | Sec. 4 footnote, App. D |
| CLIP ViT-B/32 embedding dim | not stated; 1,376 is "2.7x CLIP embedding dimension" -> about 512 (our reading) | App. D |
| Region embedding / projection | projection layer maps region embedding to text-embedding dim (Fig. 3); size not stated | Fig. 3 caption |
| Head widths for bigger teachers | ViT-L/14: FC 1,024, FPN 512; ALIGN: ViLD-text head FC 2,048, FPN 1,024, ViLD-image head uses MBConvBlocks + global avg pool; RoI features from P3 only | App. D |
| tau | 0.01 | Sec. 4 |
| w | 0.5 (R50 ViLD) | Table 3 |
| lambda | 2/3 | Sec. 3.4 |
| N (online proposals/img) | symbolic only, N = |P|; value not stated | Sec. 3.3 |
| M (offline proposals/img) | symbolic only, M = |P~|; value not stated | Sec. 3.3 |
| Max detections | 300 per image | Sec. 4 |
| Prompts | 63 templates + synonyms; single template costs 0.4 APr | App. D, Table 9 |
| R-CNN-style pipeline only | Mask R-CNN R50-FPN second-stage boxes, class-agnostic NMS 0.9, 1000 proposals; class-specific NMS 0.6; top 300 | App. D (NOT the ViLD N or M) |
| Hardware / compute | not stated; repo is under tensorflow/tpu (TPU is our reading) | -- |
| Data | LVIS v1 train, base annotations only (866 classes); COCO 48/17 split | Sec. 4.1 |

LVIS split (Sec. 4.1): 1,203 categories; "We take its 866 frequent and common categories as the base categories C_B, and hold out the 337 rare categories as the novel categories C_N. APr, the AP of rare categories, is the main metric." ViLD gives no definition of rare/common/frequent and no separate frequent vs common counts. Detic (experiments.tex, LVIS paragraph) says only: "The classes are divided into three groups - frequent, common, rare based on the number of training images." The thresholds are in the LVIS paper, which is not in `sources/`.

Shapes (our reading, for viz-vildflow): 1024x1024x3 -> R50-FPN levels -> RPN proposals -> RoIAlign (7x7 standard, not stated) -> head -> projection -> e_r in R^d (d = 512 for CLIP B/32, our reading) -> cosine with [e_bg; T(C_B)] (1 + 866 rows at train; 1 + 1,203 at LVIS test) -> /0.01 -> softmax.

## 8. Results worth remembering
Table 1 (p.6), RPN bounding-box AR on LVIS novel (rare) classes, R50:
| Supervision | ARr@100 | ARr@300 | ARr@1000 |
|---|---|---|---|
| base | 39.3 | 48.3 | 55.6 |
| base + novel | 41.1 | 50.9 | 57.0 |
"slightly worse by ~2 AR" (Sec. 4.2).

Table 2 (p.7), mask AP, R-CNN-style CLIP on crops: Supervised (base only) APr 0.0 / APc 22.6 / APf 32.4 / AP 22.5; CLIP on crops w/o objectness 13.0 / 10.6 / 6.0 / 9.2; CLIP on crops (with objectness, geometric mean) 18.9 / 18.8 / 16.0 / 17.7; Supervised (base+novel) 4.1 / 23.5 / 33.2 / 23.9; Supervised-RFS 12.3 / 24.3 / 32.4 / 25.4.

Table 3 (p.7), LVIS v1 mask AP, R50 rows averaged over 3 runs. dagger: R-CNN style, "runtime is 630x of Mask R-CNN style".
| Backbone | Method | APr | APc | APf | AP |
|---|---|---|---|---|---|
| ResNet-50+ViT-B/32 | CLIP on cropped regions (dagger) | 18.9 | 18.8 | 16.0 | 17.7 |
| ResNet-50+ViT-B/32 | ViLD-text+CLIP (dagger) | 22.6 | 24.8 | 29.2 | 26.1 |
| ResNet-50 | Supervised-RFS (base+novel) | 12.3 | 24.3 | 32.4 | 25.4 |
| ResNet-50 | GloVe baseline | 3.0 | 20.1 | 30.4 | 21.2 |
| ResNet-50 | ViLD-text | 10.1 | 23.9 | 32.5 | 24.9 |
| ResNet-50 | ViLD-image | 11.2 | 11.3 | 11.1 | 11.2 |
| ResNet-50 | ViLD (w=0.5) | 16.1 | 20.0 | 28.3 | 22.5 |
| ResNet-50 | ViLD-ensemble (w=0.5) | 16.6 | 24.6 | 30.3 | 25.5 |
| EfficientNet-b7 | ViLD-ensemble w/ ViT-L/14 (w=1.0) | 21.7 | 29.1 | 33.6 | 29.6 |
| EfficientNet-b7 | ViLD-ensemble w/ ALIGN (w=1.0) | 26.3 | 27.2 | 32.9 | 29.3 |
| ResNeSt269+HTC | 2020 Challenge winner (fully supervised, ddagger) | 30.0 | 41.9 | 46.0 | 41.5 |
Text (Sec. 4.4): ViLD APr is 6.0 above ViLD-text, 4.9 above ViLD-image, 3.8 above Supervised-RFS (not plain Supervised, which is 4.1). ViLD-text+CLIP beats ViLD by 6.5 APr. ALIGN ensemble trails the Challenge winner by 3.7 APr. ViLD-image is the only variant that uses objectness ensembling.

Table 8 (p.18), box AP mirror of Table 3 + ResNet-152: e.g. R50 ViLD (w=0.5) box APr 16.3 / AP 24.4; R152 ViLD (w=1.0) box APr 19.1, mask APr 18.7; R152 ViLD-ensemble (w=2.0) box APr 19.8, mask APr 18.7; ALIGN ensemble box APr 27.0.

Table 6 (p.16), box AP: CLIP on crops APr 19.5 / AP 18.6; ALIGN on crops APr 39.6 / APc 32.6 / APf 26.3 / AP 31.4.

Table 7 (p.18), mask AP, R50, 84,375 iters (shorter): no distill APr 10.4; L1 w=0.05 12.9, 0.1 14.0, 0.5 16.3, 1.0 17.3 (APf falls 31.7 -> 25.1); L2 w=0.5 13.7. Trade-off APr vs APc/APf.

Table 4 (p.8), COCO 48 base / 17 novel, box AP50, R50, 45,000 iters:
| Method | Novel AP | Base AP | Overall AP |
|---|---|---|---|
| Zareian et al. 2021 (OVR-CNN) | 22.8 | 46.0 | 39.9 |
| CLIP on cropped regions | 26.3 | 28.3 | 27.8 |
| ViLD-text | 5.9 | 61.8 | 47.2 |
| ViLD-image | 24.1 | 34.2 | 31.6 |
| ViLD (w=0.5) | 27.6 | 59.5 | 51.3 |
Text: +4.8 novel, +13.5 base vs OVR-CNN (Sec. 4.5); abstract says +11.4 overall.

Table 5 (p.9), transfer without finetuning, LVIS-trained R50, box AP, background embedding kept from LVIS:
| Method | VOC AP50 | VOC AP75 | COCO AP | COCO AP50 | COCO AP75 | O365 AP | O365 AP50 | O365 AP75 |
|---|---|---|---|---|---|---|---|---|
| ViLD-text | 40.5 | 31.6 | 28.8 | 43.4 | 31.4 | 10.4 | 15.8 | 11.1 |
| ViLD | 72.2 | 56.7 | 36.6 | 55.6 | 39.8 | 11.8 | 18.2 | 12.6 |
| Finetuning | 78.9 | 60.3 | 39.1 | 59.8 | 42.4 | 15.2 | 23.9 | 16.2 |
| Supervised | 78.5 | 49.0 | 46.5 | 67.6 | 50.9 | 25.6 | 38.6 | 28.0 |
VOC 2007 test, COCO val, Objects365 v1 val. VOC supervised is ImageNet-initialized.

Inference cost: the only number is Table 3 dagger: R-CNN-style methods (CLIP on crops, ViLD-text+CLIP) have runtime "630x of Mask R-CNN style". No absolute latency anywhere. Conclusion: ViLD "surpasses its supervised counterpart at the same inference speed."

## 9. Limitations and failure cases
- Missed detections are the most common failure on LVIS novel classes: "the elevator car is not detected" (App. A, Fig. 9a, p.15).
- Misclassification by visual similarity, less common: "half of the waffle iron is detected as a calculator" (Fig. 9b). Fine-grained birds: "Western Gull" assigned to "Horned Puffin" (Fig. 7b); "fails at recognizing visually non-distinctive species" (Sec. 4.7).
- Mask head follows low-level appearance, not semantics, on a VOC example (Fig. 10, App. A).
- Interactive detection "fails to recognize poses of animals" (Sec. 4.7).
- CLIP on crops (App. B, Figs. 11-12, p.17): (a) visual similarity; (b) aspect-ratio distortion from resizing proposals to 224x224 (thin structures, oven vs fridge); (c) multiple objects in a box, "a corner of an aquarium dominates the prediction", because CLIP pairs a whole image with its caption.
- "Confidence scores predicted by CLIP do not reflect the localization quality": CLIP gives the highest score to partial boxes; geometric mean with the RPN objectness score fixes the ranking (Fig. 12). Mask R-CNN has good boxes but wrong classes. Paper: need both an open-vocab classifier and detection supervision.
- Competition between ViLD-text and ViLD-image: APr rises with w while APc/APf fall (Table 7, Sec. 4.4).
- Transfer gaps: 3-6 AP below finetuning, larger below supervised (Sec. 4.6).
- Base-only RPN loses about 2 AR on novel classes (Table 1).

## 10. Outline feedback
- Ch. 1 LVIS bar: ViLD Sec. 4.1 gives only 1,203 total, 866 base (= frequent + common together) and 337 rare/novel. It gives no separate frequent and common counts and no definition. No source in `sources/` gives the thresholds or the separate counts (Detic only says "based on the number of training images"). Either show 866 / 337 / 1,203, or add the LVIS paper (Gupta et al. 2019) to sources.
- Ch. 4 `viz-vildflow`: the detector is trained from scratch (no ImageNet init) for 180,000 iterations; show the backbone as trainable from random init. Teacher embeddings (text and image) are precomputed offline, so the frozen encoders are not even in the training graph. The ViLD-image branch uses M offline proposals, not the online proposals; that they come from a separately trained base-class RPN is our reading (Sec. 3.2 says "We train a proposal network on base categories C_B and extract the region proposals offline"; App. D says both M and N come "from the first-stage RPN").
- Ch. 4 shapes: N, M, region-embedding dim and projection size are not stated. Only "d is about 512" can be inferred (our reading). Do not print values for N or M as sourced. The "1000 proposals" in App. D belongs to the R-CNN-style pipeline only.
- Ch. 4 sourced table: GloVe and supervised RFS are in Table 3 (mask AP). The "CLIP on crops" row is in Tables 2 and 3. Keep it mask AP (18.9); the box AP version (19.5, Table 6) must not be mixed in.
- Ch. 4 add ViLD-ensemble (16.6) and the ALIGN result (26.3) as a "scales with the teacher" note; and the 630x runtime footnote as the cost of CLIP-on-crops.
- Ch. 13 item 4: the ViLD source is App. B, Fig. 12 (p.17). Ch. 13 item 2 ("ViLD failure cases") fits weakly: ViLD's failures are missed objects and visual-similarity confusion (Fig. 9), not texture/region targets.
- Ch. 3: Table 1 compares RPN recall at 100/300/1000 proposals; good input for the `viz-rpn` proposals slider.
