# Deformable DETR (arXiv 2010.04159v4, ICLR 2021)

Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection".
Source: LaTeX in `sources/deformable-detr/tex/` (main `iclr2021_conference.tex`, body in `src/*.tex`).
Code: github.com/fundamentalvision/Deformable-DETR (stated in the abstract).
Section numbering used below (as compiled): 1 Intro, 2 Related Work, 3 Revisiting Transformers and DETR, 4 Method (4.1 Deformable Transformers, 4.2 Improvements and Variants), 5 Experiment (5.1 vs DETR, 5.2 Ablation, 5.3 SOTA), 6 Conclusion, Appendix A.1-A.7.
Claim ids refer to `notes/claims/deformable-detr.json`.

## Delta from DETR (problem it fixes)

DETR's two stated problems (Sec 1, Sec 3), both blamed on Transformer attention over image feature maps:

1. Slow convergence. DETR needs 500 epochs on COCO, "around 10 to 20 times slower than Faster R-CNN" [ddetr-detr-500ep, ddetr-detr-10-20x-slower]. Why: at initialization, with N_k keys, attention weights are about 1/N_k (nearly uniform over all pixels), giving "ambiguous gradients"; the final attention maps are very sparse (object extremities), so training has to travel a long way from uniform to sparse [ddetr-uniform-init-attn].
2. Weak on small objects. Small objects want high-resolution (multi-scale) feature maps, but encoder self-attention costs O(H^2 W^2 C), quadratic in spatial size, so high-res maps are "unacceptable" [ddetr-detr-enc-complexity]. In Table 1, DETR AP_S is 20.5 vs 26.6 for Faster R-CNN + FPN [ddetr-t1-detr-aps, ddetr-t1-frcnn-aps].

Deformable DETR keeps DETR's set prediction / Hungarian-loss design and replaces every attention module that reads image feature maps (encoder self-attention, decoder cross-attention) with (multi-scale) deformable attention. Decoder self-attention among object queries is left unchanged (Sec 4.1).

## The one key idea

Each query attends to only a small fixed number K of sampling points placed around a reference point, with the point locations (offsets) and their weights both predicted from the query feature by a linear layer, instead of comparing against every pixel. It is deformable convolution's sparse, learned sampling plus the Transformer's multi-head weighted aggregation. Because K does not depend on H x W, cost stops growing quadratically, and because the query only has K candidates (not ~HW), attention is sparse from the start. The same trick extends across L feature levels, which lets the module do FPN-style cross-scale fusion by itself.

The paper calls the sampled points "a pre-filter for prominent key elements out of all the feature map pixels" (Sec 1); if the sampling points traversed all locations it would be equivalent to Transformer attention (Sec 4.1).

## Mechanism step by step

Deformable attention for one query q (Eq. 2, Fig. 2):
1. Query q has content feature z_q and a 2-d reference point p_q.
2. Feed z_q into one linear projection with 3MK output channels: the first 2MK channels are the offsets Delta p_mqk (an (x,y) pair per head per point), the remaining MK channels go through a softmax to give attention weights A_mqk [ddetr-3mk-proj].
3. For each head m and point k, sample the value map at p_q + Delta p_mqk. The location is fractional, so use bilinear interpolation.
4. Project the sampled features with W'_m, take the weighted sum with A_mqk over the K points, then apply W_m and sum over heads.
There is no query-key dot product: the weights come straight from the query, not from comparing to keys.

Multi-scale version (Eq. 3): same thing, but for each head the query samples K points on each of L feature levels (LK points total), using the normalized reference point rescaled to each level by phi_l. The softmax is normalized jointly over all L x K points of a head, so one head can weigh evidence across scales.

Encoder (Sec 4.1):
- Multi-scale inputs, no FPN: x^1..x^3 from ResNet stages C3-C5 (each through a 1x1 conv); x^4 from a 3x3 stride-2 conv on C5 (called C6). L = 4, all with C = 256 channels [ddetr-L4, ddetr-C256]. C_l has resolution 2^l lower than the input image (so strides 8, 16, 32, 64 for C3-C6; derived from "2^l", not listed in the paper) [ddetr-derived-strides].
- Queries and keys are both pixels of all levels; each query pixel's reference point is itself.
- A learned scale-level embedding e_l is added (on top of the fixed positional embedding) so each pixel knows which level it is on.
- Output is multi-scale feature maps with the same resolutions as the input. Encoder parameters are shared across feature levels (Sec 5, Implementation Details).

Decoder (Sec 4.1):
- Cross-attention replaced by multi-scale deformable attention; self-attention among object queries unchanged.
- Each object query's reference point p_hat_q is predicted from its query embedding by a linear layer + sigmoid.
- The detection head predicts the box as offsets relative to the reference point: the reference point is the initial guess of the box center (Appendix A.3). This ties the decoder's attention to the predicted box and the paper says it "also accelerates the training convergence".

Iterative bounding box refinement (Sec 4.2, Appendix A.4):
- Each decoder layer d refines the box from layer d-1: new box = sigmoid(Delta b^d + sigmoid^{-1}(previous box)), for x, y, w, h. D = 6 decoder layers given as an example [ddetr-D6].
- Initial box: center = reference point, w = h = 0.1 [ddetr-init-wh]; 0.05, 0.1, 0.2, 0.5 all worked about equally [ddetr-init-wh-robust].
- Gradient is stopped at sigmoid^{-1}(previous box) (only flows through Delta b^d), following RAFT, to stabilize training.
- Prediction heads are not shared across decoder layers.
- In layer d's cross-attention, the previous box center is the new reference point, and the sampling offsets are scaled by the previous box's width and height: (Delta p_x * w^{d-1}, Delta p_y * h^{d-1}). So the sampling points follow the box's center and size. This "boxes as reference points, refined layer by layer" idea is what DAB-DETR and DINO build on.

Two-stage Deformable DETR (Sec 4.2, Appendix A.4):
- Motivation: in DETR, object queries are "irrelevant to the current image".
- Stage 1: an encoder-only Deformable DETR. Every pixel of every level in the encoder output is a query that directly predicts a box (3-layer FFN regression + linear foreground/background classifier), trained with the Hungarian loss. Each pixel's box is predicted relative to its location, with a level-dependent default size 2^(l_i - 1) * s, base scale s = 0.05 [ddetr-two-stage-s].
- The decoder is dropped for stage 1 because making every pixel a decoder query would make decoder self-attention (quadratic in number of queries) too expensive.
- Top-scoring boxes become region proposals. No NMS is applied before stage 2.
- Stage 2: proposals are fed to the decoder as initial boxes for iterative refinement; the object queries' positional embeddings are set to positional embeddings of the proposal coordinates. (This "queries initialized from encoder top-k" is the ancestor of DINO's query selection.)

Initialization (Appendix A.4), useful for animating the start of training:
- Weights of the linear projection that predicts A and Delta p are initialized to zero; biases are set so A_mlqk = 1/(LK) (uniform over the head's points) and the M = 8 heads' offsets point in 8 compass directions: head 1 (-k,-k), head 2 (-k,0), head 3 (-k,k), head 4 (0,-k), head 5 (0,k), head 6 (k,-k), head 7 (k,0), head 8 (k,k), with k in {1..K} so each head's K points march outward along its direction [ddetr-init-offsets].
- With iterative refinement, decoder offset biases are further multiplied by 1/(2K) so all initial sampling points lie inside the previous layer's box.

## Equations

Equation numbers below are from the compiled v4 source: the three numbered `equation` environments are in `src/3-method.tex` in order (Eq. 1, 2, 3). The appendix box equations are unnumbered (`equation*`). Verified: `math_commands.tex` and the main `.tex` contain no equation environments.

Macros (from `math_commands.tex`) needed to render these in KaTeX/MathJax: `\vz`, `\vx`, `\vp`, `\vb` = bold vectors (z, x, p, b); `\mW`, `\mU`, `\mV` = bold matrices W, U, V; `\emA` = italic A (element of A); `\sR` = blackboard R; `\sigmoid` = `\sigma`; `\softmax` = `\mathrm{softmax}`.

### Eq. 1: standard multi-head attention (Sec 3, the baseline)

```latex
\text{MultiHeadAttn}(\vz_q, \vx) = \sum_{m=1}^{M} \mW_m \big[\sum_{k\in\Omega_k} \emA_{mqk} \cdot \mW'_m \vx_k \big],
```
with `\emA_{mqk} \propto \exp\{\frac{\vz_q^T \mU_m^T~ \mV_m \vx_k}{\sqrt{C_v}}\}`, normalized so `\sum_{k\in\Omega_k} \emA_{mqk} = 1`.

- z_q: feature of query q (dimension C). x_k: feature of key k. Omega_k: the set of all keys (every pixel, for image features).
- m: head index, M heads. W'_m (C_v x C): value projection for head m. W_m (C x C_v): output projection. C_v = C/M by default.
- A_mqk: attention weight of query q on key k in head m, from a softmax over scaled dot products between U_m z_q and V_m x_k.
- Meaning: every query looks at every key, weights them by similarity, sums values.

### Eq. 2: deformable attention (Sec 4.1)

```latex
\text{DeformAttn}(\vz_q, \vp_q, \vx) = \sum_{m=1}^{M} \mW_m \big[\sum_{k=1}^{K} \emA_{mqk} \cdot \mW'_m \vx(\vp_q + \Delta\vp_{mqk})\big],
```

- x: a single input feature map, C x H x W.
- z_q: query content feature. p_q: the query's 2-d reference point (pixel coordinates).
- k: now indexes sampled points, not all keys; K points per head, K << HW.
- Delta p_mqk: 2-d offset of point k in head m, real-valued and unconstrained, predicted from z_q.
- x(p_q + Delta p_mqk): feature at a fractional location, read by bilinear interpolation.
- A_mqk: scalar weight in [0, 1], sum over k of A_mqk = 1 (softmax over the K points), predicted from z_q (no dot product with keys).
- W'_m, W_m: same value and output projections as Eq. 1.
- Meaning: "look at K learned spots near the reference point, weigh them, sum", per head.

### Eq. 3: multi-scale deformable attention (Sec 4.1)

```latex
\text{MSDeformAttn}(\vz_q, \hat{\vp}_q, \{\vx^l\}_{l=1}^{L}) = \sum_{m=1}^{M} \mW_m \big[\sum_{l=1}^{L} \sum_{k=1}^{K} \emA_{mlqk} \cdot \mW'_m \vx^l(\phi_{l}(\hat{\vp}_q) + \Delta\vp_{mlqk})\big],
```

- {x^l}, l = 1..L: the multi-scale feature maps, x^l is C x H_l x W_l.
- p_hat_q in [0,1]^2: normalized reference point; (0,0) is the top-left image corner, (1,1) bottom-right.
- phi_l(p_hat_q): rescales the normalized reference point into level l's pixel coordinates, so the same image location is used on every level.
- Delta p_mlqk, A_mlqk: offset and weight for point k on level l in head m.
- Normalization: sum over l and k of A_mlqk = 1, i.e. one softmax across all LK points of a head (levels compete).
- Meaning: identical to Eq. 2, except each head samples LK points spread over L levels instead of K points on one map.
- Special cases stated in the paper: with L = 1, K = 1 and W'_m fixed to identity, it degenerates to deformable convolution. If sampling points traversed all locations, it equals Transformer attention.

### Box parameterization relative to reference point (Appendix A.3, unnumbered)

```latex
\hat{\vb}_q = \{\sigmoid\big({b}_{qx} + \sigmoid^{-1}(\hat{p}_{qx})\big), \sigmoid\big({b}_{qy} + \sigmoid^{-1}(\hat{p}_{qy})\big), \sigmoid({b}_{qw}), \sigmoid({b}_{qh})\}
```
- b_q{x,y,w,h}: raw real outputs of the detection head. sigma / sigma^{-1}: sigmoid and inverse sigmoid, which keep b_hat_q in [0,1]^4.
- Meaning: the head predicts the center as a nudge in logit space away from the reference point.

### Iterative refinement (Appendix A.4, unnumbered)

```latex
\hat{\vb}^d_q = \{\sigmoid(\Delta b^d_{qx} + \sigmoid^{-1}(\hat{b}^{d-1}_{qx})), \sigmoid(\Delta b^d_{qy} + \sigmoid^{-1}(\hat{b}^{d-1}_{qy})), \sigmoid(\Delta b^d_{qw} + \sigmoid^{-1}(\hat{b}^{d-1}_{qw})), \sigmoid(\Delta b^d_{qh} + \sigmoid^{-1}(\hat{b}^{d-1}_{qh}))\},
```
- d in {1..D}: decoder layer. b_hat^{d-1}_q: previous layer's normalized box. Delta b^d: layer d's predicted correction. Gradients blocked at sigma^{-1}(b_hat^{d-1}).
- Meaning: each layer edits the previous layer's box (all four coordinates) in logit space.

### Two-stage proposal box (Appendix A.4, unnumbered)

```latex
\hat{\vb}_i = \{\sigmoid(\Delta b_{ix} + \sigmoid^{-1}(\hat{p}_{ix})), \sigmoid(\Delta b_{iy} + \sigmoid^{-1}(\hat{p}_{iy})), \sigmoid(\Delta b_{iw} + \sigmoid^{-1}(2^{l_i - 1}s)), \sigmoid(\Delta b_{ih} + \sigmoid^{-1}(2^{l_i - 1}s))\},
```
- i: an encoder pixel on level l_i with normalized coordinate p_hat_i. s = 0.05: base object scale; the default size doubles per level (2^(l_i - 1) s).
- Meaning: each pixel proposes a box centered on itself, with a prior size set by its pyramid level, plus learned corrections.

## Complexity argument (as the paper states it)

Standard attention (Sec 3): Eq. 1 costs O(N_q C^2 + N_k C^2 + N_q N_k C). For images N_q = N_k >> C, so the third term dominates: O(N_q N_k C), quadratic in feature map size [ddetr-mha-complexity].
- DETR encoder self-attention: O(H^2 W^2 C) [ddetr-detr-enc-complexity].
- DETR decoder cross-attention: N_q = N, N_k = HW: O(HWC^2 + NHWC), linear in spatial size.
- DETR decoder self-attention: O(2NC^2 + N^2 C), acceptable for moderate N.

Deformable attention (Sec 4.1, derivation in Appendix A.1):
- Predicting offsets and weights: O(3 N_q C M K).
- Evaluating Eq. 2 given them: O(N_q C^2 + N_q K C^2 + 5 N_q K C) if W'_m is applied after sampling; the 5 comes from bilinear interpolation plus the weighted sum. Or O(N_q C^2 + HW C^2 + 5 N_q K C) if W'_m x is precomputed on the whole map (it does not depend on the query).
- Overall: O(N_q C^2 + min(HW C^2, N_q K C^2) + 5 N_q K C + 3 N_q C M K). With M = 8, K <= 4, C = 256, 5K + 3MK < C, so it simplifies to O(2 N_q C^2 + min(HW C^2, N_q K C^2)) [ddetr-deform-complexity].
- In the encoder (N_q = HW): O(HW C^2), linear in spatial size (vs quadratic for DETR).
- In decoder cross-attention (N_q = N): O(N K C^2), independent of HW.

Measured cost (Table 1, Sec 5.1): FLOPs on par with Faster R-CNN + FPN and DETR-DC5 (173G vs 180G vs 187G) [ddetr-t1-ddetr-flops, ddetr-t1-frcnn-flops, ddetr-t1-dc5-500-flops]. Runtime 1.6x faster than DETR-DC5 but 25% slower than Faster R-CNN + FPN [ddetr-speed-vs-dc5, ddetr-speed-vs-frcnn], because deformable attention trades DETR-DC5's heavy memory access for unordered memory access, which is still slower than plain convolution.

## Results worth remembering

All COCO 2017. Table 1 and 2: val set; Table 2 is ResNet-50 ("ResNet-50 ... for ablations", Sec 5). Table 1 does not name its backbone; R50 is inferred because its Deformable DETR row equals Table 2's R50 full-model row. Table 3: test-dev.

Headline, 10x fewer epochs (Table 1):
- DETR, 500 epochs: 42.0 AP, AP_S 20.5 [ddetr-t1-detr-ap, ddetr-t1-detr-aps, ddetr-detr-500ep].
- DETR-DC5, 500 epochs: 43.3 AP, AP_S 22.5 [ddetr-t1-dc5-500-ap, ddetr-t1-dc5-500-aps].
- Deformable DETR, 50 epochs: 43.8 AP, AP_S 26.4 [ddetr-t1-ddetr-ap, ddetr-t1-ddetr-aps, ddetr-t1-ddetr-epochs].
- So: +1.8 AP over DETR and +5.9 AP_S [ddetr-derived-gain-vs-detr], with 50 vs 500 epochs (derived from the rows above; the paper's own words are "better performance (especially on small objects) with 10x less training epochs" [ddetr-10x-fewer-epochs]).
- Note AP_L: Deformable DETR 58.0 vs DETR 61.1, i.e. worse on large objects in this row [ddetr-t1-ddetr-apl, ddetr-t1-detr-apl]. Iterative refinement recovers it to 61.7 [ddetr-t1-refine-apl].
- Training cost: 325 GPU hours vs 2000 (DETR) and 7000 (DETR-DC5) [ddetr-t1-ddetr-gpuh, ddetr-t1-detr-gpuh, ddetr-t1-dc5-500-gpuh].
- Inference FPS: 19 vs 28 (DETR) vs 12 (DETR-DC5) vs 26 (Faster R-CNN + FPN) [ddetr-t1-ddetr-fps, ddetr-t1-detr-fps, ddetr-t1-dc5-fps, ddetr-t1-frcnn-fps].
- Fair same-schedule baseline: DETR-DC5 at 50 epochs 35.3 AP, and DETR-DC5+ (with Focal Loss and 300 queries, the same tweaks Deformable DETR uses) at 50 epochs 36.2 AP, AP_S 16.3 [ddetr-t1-dc5-50-ap, ddetr-t1-dc5plus-ap, ddetr-t1-dc5plus-aps].
- Faster R-CNN + FPN, 109 epochs: 42.0 AP, AP_S 26.6 [ddetr-t1-frcnn-ap, ddetr-t1-frcnn-aps, ddetr-t1-frcnn-epochs]. This row is identical to DETR's enhanced 'Faster RCNN-FPN+' (GIoU, random crop, 9x schedule), not the Detectron2 3x baseline (40.2 AP).

Variants (Table 1, 50 epochs):
- + iterative bounding box refinement: 45.4 AP, AP_S 26.8 [ddetr-t1-refine-ap, ddetr-t1-refine-aps].
- ++ two-stage: 46.2 AP, AP_S 28.8, 340 GPU hours [ddetr-t1-twostage-ap, ddetr-t1-twostage-aps, ddetr-t1-twostage-gpuh].

Ablation (Table 2, 50 epochs, R50):
- Single-scale, K = 1, no MS attention (roughly deformable conv): 39.7 AP, AP_S 21.2 [ddetr-t2-ss-k1].
- + multi-scale inputs: 41.4 AP (+1.7), AP_S 24.1 (+2.9) [ddetr-t2-ms-k1, ddetr-ablate-ms-inputs-gain].
- + K = 4: 42.3 AP (+0.9) [ddetr-t2-ms-k4, ddetr-ablate-k-gain].
- + multi-scale deformable attention (cross-level exchange): 43.8 AP (+1.5) [ddetr-t2-full, ddetr-ablate-msattn-gain].
- Adding FPN: 43.8 AP; BiFPN: 43.9 AP, i.e. no gain [ddetr-t2-fpn, ddetr-t2-bifpn].

SOTA (Table 3, test-dev, refinement + two-stage):
- R50 46.9 AP; R101 48.7; ResNeXt-101 49.0; ResNeXt-101 + DCN 50.1; + TTA 52.3 AP [ddetr-t3-r50, ddetr-t3-r101, ddetr-t3-x101, ddetr-t3-x101dcn, ddetr-t3-tta].

Training setup (Sec 5): M = 8, K = 4 by default; 50 epochs, LR decayed by 0.1 at epoch 40; Adam, base LR 2e-4, beta1 0.9, beta2 0.999, weight decay 1e-4; LR of the reference-point and offset projections scaled by 0.1; Focal Loss with weight 2; object queries raised from 100 to 300; V100 for timing [ddetr-M8-K4, ddetr-sched, ddetr-lr, ddetr-queries-300, ddetr-focal].

## Limitations / what follow-ups fixed

Stated in the paper (first bullet), or directly visible in its tables (last two; these are my reading, the paper does not call them limitations):
- Speed: still 25% slower than Faster R-CNN + FPN at similar FLOPs, due to unordered memory access of sampling (Sec 5.1) [ddetr-speed-vs-frcnn].
- The main comparison changes more than attention: Focal Loss and 300 queries are added (Sec 5). The paper addresses this with the DETR-DC5+ baseline, but only at 50 epochs [ddetr-t1-dc5plus-ap].
- Base Deformable DETR has lower AP_L than DETR (58.0 vs 61.1) (Table 1) [ddetr-t1-ddetr-apl].

Observations for the explainer (my reading, not claims from this paper):
- Decoder queries in the one-stage model are still image-independent learned embeddings, and the reference point is only a 2-d point predicted from the query; box size enters only via iterative refinement's offset scaling. DAB-DETR later makes queries explicit 4-d anchor boxes (x, y, w, h) refined layer by layer, generalizing this paper's iterative refinement and reference points.
- Two-stage's "top-scoring encoder proposals as decoder queries" is the ancestor of DINO's mixed query selection; DINO also modifies how gradients flow through the layer-by-layer box refinement ("look forward twice") relative to this paper's stop-gradient at sigma^{-1}(b^{d-1}).
- These follow-up descriptions are not in this paper and have no claim ids here; verify against the DAB-DETR and DINO notes.

## Facts useful for visuals

- M = 8 heads, K = 4 points per head per level (defaults) [ddetr-M8-K4].
- L = 4 levels: C3, C4, C5 via 1x1 conv, plus C6 via 3x3 stride-2 conv on C5 [ddetr-L4].
- C = 256 channels on every level [ddetr-C256]; C_v = C/M = 32 per head (derived from "C_v = C/M by default") [ddetr-derived-cv].
- Per query per head: L x K = 16 sampling points; per query overall: M x L x K = 128 (derived from the defaults, not stated in the paper) [ddetr-derived-points].
- Offsets + weights come from a single linear layer with 3MK channels (single-scale description) [ddetr-3mk-proj]; for the multi-scale case this is 3MLK by the same logic (derived, not stated).
- Softmax over all L x K points of a head; weights start uniform at 1/(LK) [ddetr-init-offsets].
- Initial offsets: head m points in one of 8 directions (a 3x3 neighborhood minus center), with point k at distance k; draw as 8 rays of 4 dots each around the reference point [ddetr-init-offsets].
- Reference point: encoder = the query pixel itself; decoder = sigmoid(linear(query embedding)); with refinement = previous box center, and offsets scaled by previous box w, h.
- Decoder: D = 6 layers (example value given), initial box w = h = 0.1 [ddetr-D6, ddetr-init-wh].
- Two-stage proposal prior size: 2^(l-1) x 0.05, i.e. 0.05, 0.1, 0.2, 0.4 for levels 1-4 (last three derived) [ddetr-two-stage-s, ddetr-derived-prior-sizes].
- Object queries: 300 (vs 100 in DETR) [ddetr-queries-300].
- Convergence plot (Fig. 3): the prose gives no values. The figure itself annotates a few point labels (41.1, 43.8, 44.9, 45.3, 45.5 for Deformable DETR schedules, 43.6 for DETR-DC5), but the schedules behind each label are not specified; do not build claims on them.
- Appendix A.6 visualization: in the decoder, sampling points cover the whole foreground instance, not only extreme points as in DETR; gradient maps show it looks at left/right boundaries for x and w, top/bottom for y and h, and interior pixels for the category (A.5, A.6).
