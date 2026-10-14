# DETR: End-to-End Object Detection with Transformers (arXiv 2005.12872v3)

Carion, Massa, Synnaeve, Usunier, Kirillov, Zagoruyko (Facebook AI). v3 submitted 28 May 2020.
Sources: tex/main.tex + related.tex, method.tex, experiments.tex, table1_alt.tex, supplementary.tex, preliminaries.tex; paper.pdf for equation/section/table numbering; official repo README (github.com/facebookresearch/detr, fetched 2026-09-29) where marked "README".

Source note: the tex uses `\oldnew{old}{new}`, defined as `#2`. Only the second argument appears in v3. All numbers below come from the second argument; first-argument text (e.g. 900 epochs, 128 GPUs, 23 FPS, 37.3/40.3 AP) is stale v1 text and must not be used (these are deliberately NOT claims).

Claim ids in [brackets] point to ../claims/detr.json.

---

## Core problem

Object detection is naturally a **set prediction** problem: output an unordered set of (class, box) pairs, one per object. Modern detectors do not predict that set directly. They define surrogate regression and classification problems on a large set of proposals, anchors, or window centers, then collapse near-duplicates with postprocessing (Section 1). DETR asks: can a network emit the final set directly, with no duplicates and no postprocessing, and be competitive on COCO?

## Prior methods and their pain

- **Surrogate tasks on initial guesses.** Two-stage detectors (Faster R-CNN, Cascade R-CNN) predict boxes relative to proposals; single-stage detectors predict relative to anchors (RetinaNet) or a grid of object centers (CenterNet, FCOS). The paper cites Zhang et al. ("Bridging the gap between anchor-based and anchor-free detection") to argue that "the final performance of these systems heavily depends on the exact way these initial guesses are set" (Section 2.3).
- **Hand-designed components that encode prior knowledge**: anchor sets, heuristic target-to-anchor assignment rules, and non-maximum suppression (NMS) to remove duplicates (Section 1, abstract). Performance is "significantly influenced" by these steps.
- **Non-unique assignment.** Modern detectors assign many predictions to one ground-truth box (non-unique assignment rules) and then need NMS (Section 2.3, "Set-based loss").
- **Earlier set-prediction attempts.** Learnable NMS and relation networks still use hand-crafted context features such as proposal box coordinates. Recurrent (RNN) set detectors used bipartite matching but decoded autoregressively, one box at a time, and were evaluated only on small datasets, not against modern baselines (Section 2.3, "Recurrent detectors").
- **Customized layers.** DETR's pitch is that it needs no specialized library or custom layers; any framework with a standard CNN and transformer suffices (Section 1, Section 3.2).

## The one key idea

Predict a **fixed-size set of N slots in parallel** with a transformer decoder, and train with a **one-to-one (bipartite, Hungarian) matching** between slots and ground-truth objects, padding the ground truth with "no object". Because each ground-truth object can be claimed by exactly one slot, the loss itself punishes duplicates; the decoder's self-attention between slots lets them coordinate so only one slot claims each object. Result: no anchors, no NMS.

Two ingredients, as the paper states them (Section 3): (1) a set prediction loss forcing unique matching between predicted and ground-truth boxes; (2) an architecture that predicts, in a single pass, a set of objects and models their relations.

## Mechanism step by step

1. **Backbone.** Image `x_img` in R^{3 x H0 x W0} (batched with zero-padding to the largest image in the batch) goes through an ImageNet-pretrained ResNet (frozen batchnorm) giving `f` in R^{C x H x W}, with C = 2048 and H, W = H0/32, W0/32 [detr-backbone-dims] (Section 3.2).
2. **Project and flatten.** A 1x1 conv reduces C to d, giving `z0` in R^{d x H x W}; flatten spatial dims to a d x HW sequence (Section 3.2). Main model uses d = 256 [detr-d-model].
3. **Encoder.** Standard transformer encoder layers (multi-head self-attention + FFN). Because attention is permutation-invariant, fixed 2D sine positional encodings are **added to the input of every attention layer** (added to queries and keys, Appendix A.1/A.3), not just once at the input. The 2D encoding uses d/2 sine/cosine functions per spatial coordinate, concatenated to d channels (Appendix A.4, "Spatial positional encoding"). 6 encoder layers, 8 heads [detr-layers] [detr-heads].
4. **Object queries.** The decoder input is N learned positional embeddings called **object queries**, N = 100 [detr-n-queries]. Per Appendix A.3, the decoder's content input starts at zero and the object queries are added (as positional encodings) at every decoder attention layer; the first self-attention of the first decoder layer can be skipped. The queries must differ because the decoder is permutation-invariant; different queries are what make slots produce different outputs (Section 3.2).
5. **Decoder.** 6 decoder layers [detr-layers] of self-attention (slots talk to each other) plus encoder-decoder cross-attention (slots read the image). Unlike the original transformer, all N objects are decoded **in parallel** at each layer, not autoregressively (Section 3.2).
6. **Prediction heads.** Each of the N output embeddings is decoded **independently** by shared heads: a 3-layer MLP with ReLU and hidden dim d predicts the normalized box (center x, center y, height, width relative to image), and a linear layer with softmax predicts the class, including an extra "no object" class [detr-pred-ffn] [detr-box-format] (Section 3.2).
7. **Matching (training).** Pad ground truth to N with no-object. Find the permutation of the N slots with lowest total matching cost (Eq. 1) using the Hungarian algorithm. Matching cost combines class probability and box similarity (L1 + GIoU).
8. **Loss (training).** For the matched pairs compute the Hungarian loss (Eq. 2): negative log-likelihood of the target class for every slot (no-object term down-weighted by a factor 10 [detr-noobj-downweight]) plus the box loss for slots matched to a real object. Box loss = lambda_iou * GIoU + lambda_L1 * L1 with lambda_L1 = 5, lambda_iou = 2 [detr-lambda-l1] [detr-lambda-iou]. Losses are normalized by the number of objects in the batch, across all GPUs (Appendix A.2).
9. **Auxiliary decoding losses.** Prediction FFNs (shared parameters, plus a shared layer-norm) and a Hungarian loss are attached after **every** decoder layer; found helpful especially for predicting the correct number of objects of each class (Section 3.2).
10. **Inference.** Take the N predictions directly, no NMS. To optimize AP, slots predicting the empty class are overridden with their second highest scoring class and its confidence, which improves AP by 2 points vs filtering them out [detr-empty-slot-override] (Section 4, "Technical details").

Why duplicates disappear: in Fig. 4 the paper runs standard NMS on each decoder layer's outputs. NMS helps the first layer's predictions (a single layer cannot compute cross-correlations between outputs, so it makes duplicates) but the benefit shrinks with depth and NMS slightly hurts AP at the last layers by removing true positives (Section 4.2) [detr-nms-thresh].

What the attention learns (qualitative, for visuals): encoder self-attention at the last layer already separates individual instances (Fig. 3). Decoder cross-attention is fairly local and focuses on object extremities such as heads and legs (Fig. 6); the paper hypothesizes the encoder separates instances globally so the decoder only needs extremities for class and boundaries (Section 4.2). Each query slot specializes on certain areas and box sizes, with several modes, and almost all slots have an image-wide-box mode (Fig. 7) [detr-query-viz].

## Equations

Macro key (from main.tex and math_commands.tex), needed to render the verbatim LaTeX below:
`\hy = \hat{y}`, `\hb = \hat{b}`, `\hp = \hat{p}`, `\noobject = \varnothing`, `\indic{#1} = \mathds{1}_{\{#1\}}`, `\lmatch{#1} = {\cal L}_{\rm match}(#1)`, `\hloss{#1} = {\cal L}_{\rm Hungarian}(#1)`, `\bloss{#1} = {\cal L}_{\rm box}(#1)`, `\iouloss{#1} = {\cal L}_{\rm iou}(#1)`, `\dmodel = d`, `\dk = d'`, `\ques = Q`, `\keys = K`, `\vals = V`, `\Nkv = N_{\rm kv}`, `\argmin` = `\DeclareMathOperator*{\argmin}{arg\,min}`. Attention-weight macros (`\xq`, `\xkv`, `\posq`, `\poskv`, `\weit`) are also in math_commands.tex.

### Eq. (1): optimal bipartite matching (Section 3.1)

```latex
\hat{\sigma} = \argmin_{\sigma\in\Sigma_N} \sum_{i}^{N} \lmatch{y_i, \hy_{\sigma(i)}},
```

Plain words: over all ways of assigning the N ground-truth entries (real objects plus no-object padding) one-to-one to the N prediction slots, pick the assignment with the smallest total matching cost.
- `\hat{\sigma}`: the chosen (optimal) assignment; `\hat{\sigma}(i)` is the slot assigned to ground-truth entry i.
- `\Sigma_N`: the set of all permutations of N elements (every possible one-to-one assignment).
- `\sigma`: one candidate permutation.
- `y_i`: the i-th ground-truth entry, `y_i = (c_i, b_i)`; padded with `\varnothing` so there are N of them.
- `\hy_{\sigma(i)}`: the prediction in slot `\sigma(i)`.
- `\lmatch`: pairwise matching cost (below).
- Solved with the Hungarian algorithm (Section 3.1).

### Matching cost (inline, unnumbered, Section 3.1)

```latex
\lmatch{y_i, \hy_{\sigma(i)}} = -\indic{c_i\neq\noobject}\hp_{\sigma(i)}(c_i) + \indic{c_i\neq\noobject} \bloss{b_{i}, \hb_{\sigma(i)}}
```

(The tex writes this as prose: "we define `\lmatch{...}` as `-\indic{...}\hp... + \indic{...}\bloss{...}`"; the `=` is added here for readability.)

Plain words: if ground-truth entry i is a real object, the cost of giving it to slot `\sigma(i)` is "minus how much probability that slot puts on the right class" plus "how far that slot's box is from the true box". If entry i is no-object, the cost is zero regardless of the slot (the paper notes this cost is a constant, independent of the prediction).
- `c_i`: target class label (may be `\varnothing`).
- `b_i` in [0,1]^4: ground-truth box, center coordinates plus height and width relative to image size.
- `\hp_{\sigma(i)}(c_i)`: predicted probability of class `c_i` in slot `\sigma(i)`.
- `\hb_{\sigma(i)}`: predicted box in slot `\sigma(i)`.
- `\indic{c_i\neq\noobject}`: 1 if entry i is a real object, else 0.
- Design note (Section 3.1): uses probabilities, not log-probabilities, so the class term is commensurable with the box term; this gave better empirical performance.

### Eq. (2): Hungarian loss (Section 3.1)

```latex
\hloss{y, \hy} = \sum_{i=1}^N \left[-\log  \hp_{\hat{\sigma}(i)}(c_{i}) + \indic{c_i\neq\noobject} \bloss{b_{i}, \hb_{\hat{\sigma}}(i)}\right]\,,
```

Plain words: after fixing the best assignment, every slot is trained to predict its assigned class (a real class or no-object) with cross-entropy, and slots assigned to real objects are additionally trained to match the box.
- `\hat{\sigma}`: the optimal assignment from Eq. (1).
- `-\log \hp_{\hat{\sigma}(i)}(c_i)`: negative log-likelihood of the target class for the slot matched to entry i. Applies to all N entries, including no-object ones.
- In practice this log-probability term is down-weighted by a factor 10 when `c_i = \varnothing` (class imbalance; analogous to Faster R-CNN subsampling negatives) [detr-noobj-downweight].
- `\indic{c_i\neq\noobject} \bloss{...}`: box loss only for real objects.
- Typo in the source: `\hb_{\hat{\sigma}}(i)` should read `\hb_{\hat{\sigma}(i)}` (the prediction in slot `\hat{\sigma}(i)`); kept verbatim above.
- Contrast with the matching cost: log-probabilities here, probabilities there; indicator on the class term there, not here.

### Box loss (inline in Section 3.1 "Bounding box loss"; numbered as Eq. (9) in Appendix A.2)

Main-text inline form:

```latex
\bloss{b_{i}, \hb_{\sigma(i)}} = \lambda_{\rm iou}\iouloss{b_{i}, \hb_{\sigma(i)}} + \lambda_{\rm L1}||b_{i}- \hb_{\sigma(i)}||_1
```

(prose "defined as" rendered with `=`.)

Eq. (9), Appendix A.2, verbatim:

```latex
\bloss{b_{\sigma(i)}, \hb_i} = \lambda_{\rm iou}\iouloss{b_{\sigma(i)}, \hb_i} + \lambda_{\rm L1}||b_{\sigma(i)}- \hb_i||_1\,,
```

Plain words: box error is a weighted sum of a scale-invariant overlap loss (GIoU) and plain coordinate L1 distance.
- `\lambda_{\rm iou}`, `\lambda_{\rm L1}`: weights, 2 and 5 respectively [detr-lambda-iou] [detr-lambda-l1].
- `||\cdot||_1`: L1 norm over the 4 normalized box coordinates.
- Why both (Section 3.1): boxes are predicted directly (not as deltas from anchors), and L1 alone has different scales for small and large boxes even at similar relative error; GIoU is scale-invariant.
- Index convention swap: the appendix (Eq. 9, 10) writes `b_{\sigma(i)}, \hb_i` (ground truth indexed by the permutation), the reverse of the main text's `b_i, \hb_{\sigma(i)}`. Same meaning; pick one for the explainer.
- Both losses are normalized by the number of objects in the batch.

### Eq. (10): generalized IoU loss (Appendix A.2)

```latex
\iouloss{b_{\sigma(i)}, \hb_i} = 1 - \bigg(\frac{ | b_{\sigma(i)}\cap \hb_i|}{|b_{\sigma(i)}\cup \hb_i|} - \frac{|B(b_{\sigma(i)},\hb_i)  \setminus b_{\sigma(i)} \cup \hb_i|}{|B(b_{\sigma(i)},\hb_i)|}\bigg)\,.
```

Plain words: 1 minus (IoU minus a penalty for the empty space in the smallest box enclosing both). Unlike plain IoU it still gives a useful signal when boxes do not overlap.
- `|.|`: area; `\cap`, `\cup` of box coordinates stand for intersection and union regions.
- `B(b_{\sigma(i)}, \hb_i)`: the smallest box containing both boxes (the paper says "largest box containing"; the GIoU reference defines it as the smallest enclosing box).
- Computed via min/max of linear functions of box coordinates, so well-behaved for stochastic gradients.

### Eq. (8): attention weights (Appendix A.1), for the architecture chapter

```latex
\alpha_{i,j} = \frac{e^{\frac{1}{\sqrt{\dk}}\ques_i^T\keys_j}}{Z_i} \text{~~where~} Z_i = \sum_{j=1}^{\Nkv}  e^{\frac{1}{\sqrt{\dk}}\ques_i^T\keys_{j}}\,.\label{eq:posenc}
```

Plain words: standard scaled dot-product softmax attention. Query i's weight on key j.
- `\ques_i`, `\keys_j`: query i and key j embeddings (Q, K).
- `\dk = d'` = d/M, per-head dimension with M heads.
- `\Nkv`: key-value sequence length.

### Eq. (7): where positional encodings enter (Appendix A.1)

```latex
[\ques; \keys; \vals] = [\weit'_1 (\xq+\posq);\weit'_2 (\xkv+\poskv); \weit'_3 \xkv]
```

Plain words: positional encodings are added to the query input and key input before projection, but **not** to the values. This is how both spatial encodings (encoder) and object queries (decoder) are "added at every attention layer".
- `\xq`, `\xkv`: query and key-value sequences; `\posq`, `\poskv`: their positional encodings; `\weit'_{1,2,3}`: per-head projection weights.

Other numbered equations in v3, not needed for the explainer: (3) multi-head attention signature, (4) self-attention as special case, (5)-(6) head concatenation and residual/dropout/layernorm, (11) DICE loss for panoptic masks.

Complexity (Appendix A.3, "Computational complexity"): each encoder self-attention is `O(d^2 HW + d (HW)^2)`; decoder self-attention `O(d^2 N + d N^2)`; cross-attention `O(d^2 (N+HW) + d N HW)`, much lower than the encoder since N << HW. The `(HW)^2` term is why higher-resolution features (DC5) are expensive.

## Results worth remembering

- **Headline: parity with a tuned Faster R-CNN.** DETR R50: 42.0 AP [detr-headline-ap] (text: "42 AP" [detr-headline-ap-text]), AP50 62.4 [detr-ap50], AP75 44.2 [detr-ap75] vs Faster RCNN-FPN+ 42.0 AP [detr-frcnn-fpn-plus-ap], with 41M params [detr-params-table] vs 42M [detr-frcnn-plus-params], and 86/28 GFLOPS/FPS [detr-flops-fps] (text: 28 FPS [detr-fps-text]) vs 180/26 [detr-frcnn-plus-flops]. Detectron2 zoo Faster RCNN-FPN (3x): 40.2 AP [detr-frcnn-fpn-ap]. The Faster R-CNN+ rows were strengthened with GIoU, random crops, and a 9x (109 epoch) schedule, adding 1-2 AP [detr-frcnn-enhancements] [detr-frcnn-epochs].
- **Large vs small objects (the signature trade-off).** DETR R50: AP_L 61.1 [detr-apl] vs 53.4 for FPN+ [detr-frcnn-plus-apl]; AP_S 20.5 [detr-aps] vs 26.6 for FPN+ [detr-frcnn-plus-aps]. AP_M 45.8 [detr-apm] vs 45.4 [detr-frcnn-plus-apm]. The text states "+7.8" AP_L and "-5.5" AP_S [detr-apl-delta-text], but these deltas do not reproduce from Table 1 (see claim; use the raw table values in visuals).
- **Bigger variants.** DETR-DC5: 43.3 AP [detr-dc5-ap], AP_S 22.5 [detr-dc5-aps], at 187/12 GFLOPS/FPS [detr-dc5-flops]. DETR-R101: 43.5 AP [detr-r101-ap]. DETR-DC5-R101: 44.9 AP [detr-dc5-r101-ap] vs Faster RCNN-R101-FPN+ 44.0 AP [detr-frcnn-r101-plus-ap].
- **Encoder matters (global reasoning).** 0 / 3 / 6 / 12 encoder layers: 36.7 [detr-enc0-ap] / 40.1 [detr-enc3-ap] / 40.6 [detr-enc6-ap] / 41.6 AP [detr-enc12-ap]. Removing the encoder costs 3.9 AP overall and 6.0 AP on large objects [detr-no-encoder-drop].
- **Decoder depth removes duplicates.** +8.2 AP / +9.5 AP50 from the first to the last decoder layer [detr-decoder-gain]; NMS helps only early layers (Fig. 4).
- **FFNs matter.** Removing transformer FFNs: 41.3M -> 28.7M params, -2.3 AP [detr-ffn-removal].
- **Positional encodings.** No spatial encodings at all: 32.8 AP, -7.8 [detr-posenc-none]; encodings only at input once: 39.2 AP, -1.4 [detr-posenc-input-once]; learned instead of sine: 39.6, -1.0 [detr-posenc-learned]; no spatial encodings in encoder only: 39.3, -1.3 [detr-posenc-no-encoder]; baseline 40.6 [detr-posenc-baseline].
- **Loss components.** Without GIoU (class + L1): 35.8 AP, -4.8 [detr-loss-no-giou]; without L1 (class + GIoU): 39.9 AP, -0.7 [detr-loss-no-l1]; all three: 40.6 AP [detr-loss-all]. GIoU carries most of the performance; L1 helps AP_M and AP_L when combined.
- **Training tricks that move AP.** Random crop augmentation about +1 AP [detr-crop-aug]; empty-slot override at inference +2 AP [detr-empty-slot-override]; long vs short schedule +1.5 AP per text [detr-long-schedule-gain] (40.6 vs 42.0 in Section 4.2 [detr-short-long-ap]).
- **Generalization to counts.** No training image has more than 13 giraffes, yet DETR finds all 24 in a synthetic image [detr-giraffes].
- **Panoptic extension.** DETR R50 43.4 PQ [detr-pano-pq-r50], DETR-R101 45.1 PQ [detr-pano-pq-r101] vs PanopticFPN++ R101 44.1 PQ [detr-pano-pfpn-r101]; trained on 53 stuff + 80 things categories [detr-pano-categories], mask head 25 epochs on frozen DETR [detr-pano-head-epochs]; strongest on stuff (37.0 vs 33.6 PQ^st) [detr-pano-stuff]; 46 PQ on COCO test [detr-pano-test].
- **Simplicity.** Inference in under 50 lines of PyTorch [detr-inference-50-lines].

## Limitations (as stated by the paper)

- **Long training schedule.** "The new model requires extra-long training schedule and benefits from auxiliary decoding losses in the transformer" (Section 1). Concretely: 300 epochs for ablations [detr-epochs-short] and 500 epochs for the headline [detr-epochs-long], vs about 109 epochs for the Faster R-CNN+ baselines [detr-frcnn-epochs]; 300 epochs take 3 days on 16 V100s [detr-train-cost]. The README gives about 6 days for 300 epochs on one 8-V100 machine [detr-readme-epoch-time]. Note: the paper states the long schedule as a requirement and a "new challenge"; it does not use the phrase "slow convergence" or present a convergence curve.
- **Small objects.** "It obtains, however, lower performances on small objects" (Section 1); still lagging in AP_S, and DETR-DC5 "is still significantly behind in AP_S too" (Section 4.1) [detr-aps] [detr-frcnn-plus-aps]. The authors expect a fix analogous to what FPN did for Faster R-CNN (Section 1). DETR uses only a single stride-32 feature map [detr-backbone-dims].
- **Cost of higher resolution.** Getting better small-object AP via DC5 costs 16x in encoder self-attention and about 2x overall compute [detr-dc5-cost]; DC5 drops from 28 to 12 FPS [detr-flops-fps] [detr-dc5-flops]. Root cause is the quadratic `(HW)^2` encoder self-attention term (Appendix A.3).
- **Conclusion's summary.** "This new design for detectors also comes with new challenges, in particular regarding training, optimization and performances on small objects" (Section 5).
- **Hard cap on object count.** "By design, DETR cannot predict more objects than it has query slots, i.e. 100 in our experiments" (Appendix A.5). In the repeated-grid test it detects all instances up to 50 but only 30 on average when all 100 are present [detr-saturation]; the authors attribute this to being far from the training distribution.
- **Optimization sensitivity.** A backbone LR about an order of magnitude smaller than the rest is "important to stabilize training, especially in the first few epochs" (Appendix A.4) [detr-optimizer].
- **Panoptic mask AP.** "a severe deficit of up to 8 mAP compared to the baselines on the mask AP computation" for things, though PQ^th stays competitive (Section 4.4) [detr-pano-mask-ap].
- **Loss weighting not explored.** "We only studied simple ablations of different losses (using the same weighting every time), but other means of combining them may achieve different results" (Section 4.2).

## Facts useful for visuals

- Number of object queries / prediction slots: N = 100 [detr-n-queries].
- Backbone: ImageNet-pretrained ResNet-50 (main) or ResNet-101, torchvision, frozen batchnorm; output C = 2048 channels at stride 32 [detr-backbone-dims]. DC5 variant: stride 16 (2x resolution) [detr-dc5-cost].
- Transformer: d = 256 [detr-d-model], 8 heads [detr-heads], 6 encoder + 6 decoder layers [detr-layers], dropout 0.1 [detr-dropout].
- Parameters: 41.3M total = 23.5M ResNet-50 + 17.8M transformer [detr-params-text] [detr-params-backbone] [detr-params-transformer].
- Heads: 3-layer MLP for box, linear layer for class incl. no-object [detr-pred-ffn]; box = (cx, cy, h, w) normalized to [0,1] [detr-box-format].
- Loss weights: lambda_L1 = 5 [detr-lambda-l1], lambda_iou = 2 [detr-lambda-iou], no-object class term down-weighted by factor 10 [detr-noobj-downweight].
- Optimizer: AdamW, transformer LR 10^-4, backbone LR 10^-5, weight decay 10^-4 [detr-optimizer]; grad clip max norm 0.1 [detr-grad-clip].
- Schedules: 300 epochs with LR /10 after 200 (ablations) [detr-epochs-short]; 500 epochs with LR drop after 400 (headline) [detr-epochs-long]; Faster R-CNN+ 9x = 109 epochs [detr-frcnn-epochs].
- Hardware: 16 V100, 4 images/GPU, batch 64, 3 days for 300 epochs [detr-train-cost]. README: 28 min/epoch on 8 V100 [detr-readme-epoch-time]; a 150-epoch run reaches 39.5/60.3 AP/AP50 [detr-readme-150ep]; R50 inference 0.036 s/img [detr-readme-inftime].
- Input sizes: training scale augmentation, short side 480-800, long side at most 1333 [detr-scale-aug]; random crop with p = 0.5 then resize to 800-1333 [detr-crop-aug]. The Appendix A.6 listing uses a dummy 800x1200 input, 91 classes (+1 no-object), 100 queries [detr-listing-input].
- Dataset: COCO 2017, 118k train / 5k val [detr-coco-size]; 7 instances per image on average, up to 63 [detr-coco-instances]. (Useful contrast: N = 100 slots vs 7 average objects, so most slots predict no-object.)
- Speed/compute: DETR R50 86 GFLOPS / 28 FPS [detr-flops-fps] vs Faster RCNN-FPN+ 180 / 26 [detr-frcnn-plus-flops]. Table 2 lists the 6-layer model at 86/23 [detr-fps-table2]; treat as inconsistent, prefer Table 1.
- NMS experiment threshold: 0.7 [detr-nms-thresh].
- Figure hooks: Fig. 1 (pipeline with bipartite matching), Fig. 2 (architecture), Fig. 3 (encoder self-attention separates cows), Fig. 4 (AP per decoder layer with/without NMS), Fig. 5 (24 giraffes) [detr-giraffes], Fig. 6 (decoder attends to extremities), Fig. 7 (20 of 100 slot box-center distributions) [detr-query-viz], Fig. 10 (detailed transformer), Fig. 12 (missed instances vs count) [detr-saturation].

## Inconsistencies found in the source (do not smooth over in the explainer)

1. Section 4.1 text "+7.8" AP_L / "-5.5" AP_S does not match Table 1 arithmetic against either Faster R-CNN R50 row [detr-apl-delta-text].
2. "adds 1.5 AP" for the long schedule vs 40.6 -> 42.0 (1.4) in Section 4.2 [detr-long-schedule-gain].
3. FPS of the baseline: 28 in Table 1 and Section 4.2 vs 23 in Table 2 row '6' [detr-fps-table2].
4. Eq. (2) subscript typo `\hb_{\hat{\sigma}}(i)`; Eq. (9)/(10) index swap relative to main text.
