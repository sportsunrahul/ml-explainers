# Understanding: the DETR family (DETR -> RF-DETR)

Synthesis across seven papers. Per-paper detail, verbatim equations and every number live in
`papers/*.md`; claim ids in [brackets] resolve in `claims.json` (678 claims, 40 marked uncertain;
all fact-checked against the LaTeX sources by independent agents on 2026-09-29).

| Paper | arXiv | First posted | Version read |
|---|---|---|---|
| DETR | 2005.12872 | 2020-05-26 | v3 |
| Deformable DETR | 2010.04159 | 2020-10-08 | v4 |
| DAB-DETR | 2201.12329 | 2022-01-28 | v4 |
| DN-DETR | 2203.01305 | 2022-03-02 | v3 (journal extension) |
| DINO (detection) | 2203.03605 | 2022-03-07 | v4 |
| RT-DETR | 2304.08069 | 2023-04-17 | v3 (CVPR 2024) |
| RF-DETR | 2511.09554 | 2025-11-12 | v2 |

## The core problem

Object detection outputs an unordered set of (class, box) pairs. Pre-2020 detectors never predict
that set directly: they score a large grid of anchors/proposals and then remove near-duplicates
with non-maximum suppression (NMS), a hand-tuned post-processing step whose thresholds trade
missed overlapping objects against leftover duplicates. Anchor design and target assignment are
likewise hand-crafted priors.

## DETR: the one key idea

Predict a fixed set of N = 100 slots in parallel with a transformer decoder [detr-n-queries], and
train with a one-to-one Hungarian matching between slots and ground truth padded with "no object".
A duplicate slot cannot also claim a matched object, so the loss pushes it to "no object"; decoder
self-attention lets slots coordinate. No anchors, no NMS.

Mechanism: ResNet features (stride 32, C = 2048) -> 1x1 conv to d = 256 -> 6-layer encoder with
sine positional encodings added at every attention layer -> 6-layer decoder over 100 learned
object queries -> shared FFN heads per slot (box as normalized cx, cy, h, w; class incl.
no-object) [detr-backbone-dims, detr-d-model, detr-layers].

Equations in words:
- Eq. 1: over all one-to-one assignments of padded ground truth to slots, pick the cheapest.
- Matching cost: for a real object, minus the slot's probability of the right class, plus the
  box distance; no-object entries cost nothing.
- Eq. 2 (Hungarian loss): after fixing the assignment, cross-entropy on every slot's class
  (no-object term down-weighted 10x) plus box loss for slots matched to real objects.
- Box loss: 2 x GIoU loss + 5 x L1 [detr-lambda-iou, detr-lambda-l1]; GIoU because it is
  scale-invariant and gives signal even without overlap.

Results: 42.0 AP, matching an enhanced Faster RCNN-FPN+ (42.0) [detr-headline-ap,
detr-frcnn-fpn-plus-ap]; much better on large objects (AP_L 61.1 vs 53.4) and much worse on small
(AP_S 20.5 vs 26.6) [detr-apl, detr-frcnn-plus-apl, detr-aps, detr-frcnn-plus-aps]. First to last
decoder layer adds 8.2 AP, and NMS stops helping in the last layers [detr-decoder-gain].

Limitations as stated: extra-long training (500 epochs for the headline vs ~109 for the baseline)
[detr-epochs-long, detr-frcnn-epochs]; weak on small objects; the encoder's (HW)^2 self-attention
makes higher-resolution features expensive (DC5: 28 -> 12 FPS) [detr-dc5-flops]. Paper-internal
inconsistencies (text vs table deltas, 28 vs 23 FPS) are marked uncertain; use table values.

## The follow-ups as deltas

### Deformable DETR: sparse, learned sampling (fixes cost and small objects)
Each query attends to K = 4 sampled points per head around a reference point, per each of L = 4
feature levels, with offsets and weights predicted directly from the query (no query-key dot
product) [ddetr-M8-K4, ddetr-L4]. Cost becomes linear in HW in the encoder and independent of HW
in decoder cross-attention [ddetr-deform-complexity], so multi-scale maps become affordable.
Eq. 2 = "look at K learned spots near the reference point, weigh them, sum"; Eq. 3 = same across
levels with one softmax per head. Result: 43.8 AP at 50 epochs vs DETR 42.0 at 500, AP_S 26.4 vs
20.5 [ddetr-t1-ddetr-ap, ddetr-t1-detr-ap, ddetr-t1-ddetr-aps]; iterative box refinement and a
two-stage variant (encoder proposals as queries) push to 45.4 and 46.2 [ddetr-t1-refine-ap,
ddetr-t1-twostage-ap]. Still 25% slower than Faster R-CNN at similar FLOPs (unordered memory
access) [ddetr-speed-vs-frcnn].

### DAB-DETR: the query is a box
A query = content part + positional part. DAB-DETR makes the positional part an explicit 4D anchor
(x, y, w, h): sinusoidal encoding + shared MLP gives the positional query (Eq. 1-2); w, h scale the
positional attention into an oval (Eq. 6); the anchor is refined after each of 6 layers. The
update formula is not in the paper; the official code uses sigmoid(delta + inverse_sigmoid(old))
with a detach [dab-repo-update-rule, uncertain: code not paper]. Temperature T = 20 instead of
10000 [dab-T20]. Anchor update alone is worth +1.7 AP in the ablation [dab-abl-no-update];
headline 45.7 AP at 50 epochs for DC5-R50 with 3 pattern embeddings [dab-dc5-r50-star-ap].
The paper's diagnosis of slow training (multi-modal learned positional queries) is stated as a
conjecture.

### DN-DETR: denoising stabilizes matching
Hungarian matching flips a query's assigned object between epochs early in training (measured by
IS, Eq. 1-2; max 14.72 on COCO val [dn-is-max]). Fix: feed noised ground-truth boxes and labels as
extra queries whose target is simply the original (center shift |dx| < lambda_1 w/2, scaling in
[(1 - lambda_2) w, (1 + lambda_2) w], label flip rate 0.2, lambda_1 = lambda_2 = 0.4)
[dn-lambda1, dn-lambda2, dn-gamma], in groups, behind an attention mask (Eq. 7) so matching
queries cannot see the answers. Without the mask denoising collapses (24.0 vs 43.4 AP)
[dn-abl-nomask]. Zero test-time cost; about the same accuracy in half the epochs [dn-t1b-dc5r50].
Several 12-epoch rows use 3 pattern embeddings though unstarred (labelled in claims).

### DINO: contrastive denoising + better queries
DINO = DAB anchors + DN denoising + Deformable attention, plus (1) contrastive denoising:
positive noised queries reconstruct the box, negative (further) ones must predict "no object";
the paper's thresholds are lambda_1 = 1.0, lambda_2 = 2.0 (different meaning from DN-DETR's
lambdas) and the code implements them as corner shifts under 0.2w vs 0.2w-0.4w
[dino-cdn-lambdas, dino-repo-cdn-noise, uncertain: code reading]; (2) mixed query selection:
encoder top-K boxes initialize the anchors while content queries stay learned; (3) look forward
twice: layer i also gets gradient from layer i+1's loss (Eq. 2). Results, R50, no extra data:
49.0 AP (4-scale) at 12 epochs, 51.3 (5-scale) at 24 [dino-t1-4s, dino-t2-24]. The 63.2 / 63.3
headline needs Swin-L, Objects365 pre-training and TTA [dino-t3-val, dino-t3-testdev].

### RT-DETR: real time, NMS counted
YOLO speeds usually exclude NMS, whose time grows with the number of boxes: YOLOv8 NMS kernel time
is 2.36 / 1.73 / 1.06 ms at confidence 0.001 / 0.01 / 0.05 (T4, TRT FP16) [rtdetr-t1-conf0001,
rtdetr-t1-conf001, rtdetr-t1-conf005]; box counts per threshold exist only as a figure (read off
as approximate) [rtdetr-fig2-counts, uncertain]. RT-DETR: self-attention only on the coarsest map
(AIFI) plus a CNN cross-scale fusion (CCFF) (Eq. 1); uncertainty-minimal query selection (Eq. 2-3:
the class loss also sees box quality; +0.8 AP) [rtdetr-t4-um]; decoder layers can be dropped at
inference. R50: 53.1 AP at 108 FPS vs YOLOv8-L 52.9 at 71, end to end, 640 px, T4 TRT FP16
[rtdetr-r50-ap, rtdetr-r50-fps, rtdetr-yolov8l].

### RF-DETR: weight-sharing NAS for any latency
LW-DETR lineage with a pretrained DINOv2 backbone; one weight-sharing network trained with a
random patch size, resolution and window count per step; decoder depth (0-6) and query count
(50-300) cut at inference. A grid search over 6,468 configs picks a Pareto curve [rfdetr-search-count]
(about 10,000 T4 GPU hours, per backbone family). COCO: nano 48.0 AP at 2.3 ms (+5.3 over D-FINE N),
2XL 60.1 at 17.2 ms (first real-time > 60) [rfdetr-nano-beats-dfine, rfdetr-first-60ap]; D-FINE
still leads at M, L, XL on COCO AP. Latency protocol: T4, TensorRT 10.4, 200 ms pause between runs,
accuracy measured on the timed engine; FP16 inferred [rfdetr-fp16-main, uncertain]. RT-DETR
baselines in RF-DETR's tables appear mislabelled / O365-pretrained (flagged); use D-FINE and YOLO
as comparison points.

## Through-line (what each paper inherits)

- Queries: learned embeddings (DETR) -> reference points (Deformable) -> explicit 4D anchors (DAB)
  -> anchors initialized from encoder top-K (DINO mixed selection, RT-DETR uncertainty-minimal).
- Training speed: 500 epochs (DETR) -> 50 (Deformable, DAB) -> 12 for 49 AP (DINO), via sparse
  attention, box priors, and denoising.
- Speed: DETR 28 FPS -> real time on T4 (RT-DETR) -> Pareto family from one training (RF-DETR).
- Constant: set prediction + one-to-one matching, no NMS.

## Uncertain items worth remembering
- DETR's text deltas vs its own tables; 28 vs 23 FPS.
- DAB anchor-update rule and temperature placement come from code.
- DINO noise-scale mechanics come from code; ablation epoch count inferred.
- RT-DETR box counts per threshold are read from a figure.
- RF-DETR FP16 and batch size for the main tables are inferred; Figure 1 curve values are visual reads.
