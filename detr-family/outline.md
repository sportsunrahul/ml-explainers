# Outline: The DETR family, from set prediction to real time

Reader: comfortable with CNNs and transformers. Skip attention basics; slow down on matching,
query design, and why training was slow. 9 narrated chapters, ~100 s each (~15 min total), then an
unnarrated family tree, summary and "where to go deeper".

Visual honesty key: [toy] = illustrative, labelled "Toy model" on the page; [sourced] = drawn
from numbers in claims.json; [computed] = real formula from the paper evaluated on toy inputs.

## Act I: DETR (foundational, 4 chapters)

### 1. `ch-set-problem` Detection is a set problem
- Story: a detector's real answer is an unordered set of (class, box). Classic detectors instead
  score thousands of anchors and then delete duplicates with NMS, a hand-tuned cleanup step.
- ONE intuition: NMS is a heuristic bolted on after the network; it cannot tell "duplicate" from
  "two overlapping objects".
- Interactive `viz-nms-overlap` [toy]: two people standing close, each with a cloud of candidate
  boxes. Drag them together; an NMS IoU-threshold slider shows the trade-off: low threshold
  deletes the second person, high threshold leaves duplicates. Buttons: apart / overlapping /
  run NMS.
- Math: none (IoU in words).

### 2. `ch-matching` One prediction per object: Hungarian matching
- Story: predict a fixed set of N = 100 slots in parallel, pad the ground truth with "no object",
  and train with the one-to-one assignment that costs least. A duplicate slot cannot also claim
  the object, so it is pushed to "no object": the loss itself removes duplicates.
- ONE intuition: one-to-one matching makes duplicates expensive, which is why DETR needs no NMS.
- Interactive `viz-match-matrix` [computed]: a "solve it yourself" cost matrix, 5 predicted boxes x
  3 objects (+ no-object padding) on a toy image. Click cells to assign; the page shows your total
  cost vs the Hungarian optimum. Cost = -class prob + 5 x L1 + 2 x GIoU loss, DETR's actual
  weights [sourced lambdas]. Buttons: show optimal / reset / add a duplicate.
- Math: Eq. 1 (argmin over permutations), the matching cost, Eq. 2 (Hungarian loss), GIoU in words.

### 3. `ch-queries` Object queries and a decoder that talks to itself
- Story: the decoder input is 100 learned "object queries". Slots read the image through
  cross-attention and talk to each other through self-attention, so they can agree who takes
  which object. Early layers still produce duplicates; later layers do not (the paper's NMS-per-
  layer experiment; +8.2 AP from first to last decoder layer).
- ONE intuition: duplicate removal is learned, layer by layer, by slots coordinating.
- Interactive `viz-decoder-layers` [toy, with one sourced annotation]: a layer scrubber 1..6 over
  a toy scene; duplicate boxes fade as slots coordinate. A caption cites the sourced per-layer
  claim (first -> last layer gain) and says the animation itself is illustrative.
- Math: none new; one line on "positional encodings added at every attention layer".

### Interlude `ch-training` One training step, end to end (not narrated; added after the run)
- Stepper `viz-train-step` [computed shapes + toy scenes]: Batch (scale aug, zero-pad, mask, per-image
  target lists) -> Backbone (2048 x H/32 x W/32) -> Tokens (1x1 conv to 256, flatten, sine pos) ->
  Encoder (6 layers, padding masked) -> Decoder (100 zero-init slots, 6 layers, all outputs kept) ->
  Heads (shared across layers; 92 logits, sigmoid box) -> Match (per image, per layer, no grad) ->
  Loss and update (aux losses summed, box losses / objects across GPUs, clip 0.1, AdamW).
- Links back to ch-matching's equations rather than re-deriving them.

### 4. `ch-detr-limits` What it cost: 500 epochs and small objects
- Story: DETR matches a tuned Faster R-CNN (42.0 AP) with no hand-designed parts, but trains
  for 500 epochs vs ~109, and is much worse on small objects (AP_S 20.5 vs 26.6) while better on
  large ones (AP_L 61.1 vs 53.4). Root cause of the small-object gap: dense self-attention cost
  grows with (HW)^2, so DETR can only afford a coarse stride-32 map.
- ONE intuition: dense attention over every pixel is what makes DETR slow to train and blind to
  small things.
- Interactive `viz-attn-cost` [computed + sourced]: a feature-map-resolution slider (stride 32 ->
  16 -> 8) showing number of pixel pairs for dense self-attention vs linear cost, alongside
  sourced bars for AP_S / AP_L (DETR vs Faster R-CNN+) and epochs.
- Math: O(d^2 HW + d (HW)^2) from DETR's appendix, glossed.

## Act II: making DETR train fast (3 chapters)

### 5. `ch-deformable` Deformable DETR: look at a few points, not every pixel
- Story: each query samples only K = 4 points per head around a reference point, with offsets and
  weights predicted from the query; do it across 4 feature scales. Cost becomes linear, attention
  is sparse from the start, and multi-scale maps become affordable.
- ONE intuition: attention does not need to compare against every pixel; a query can learn where
  to look.
- Interactive `viz-deform-points` [sourced init + toy image]: drag a reference point; see the
  8 heads x 4 points initialization pattern (the paper's actual initialization), toggle "after
  training" (illustrative learned offsets clustering on the object) and "multi-scale" (points on
  4 levels). A cost readout compares dense vs deformable for the current resolution.
- Result: 43.8 AP at 50 epochs vs DETR 42.0 at 500; AP_S 26.4 vs 20.5.
- Math: Eq. 2 (DeformAttn), Eq. 3 (multi-scale), in plain words; complexity line.

### 6. `ch-anchor-queries` DAB-DETR: a query is a box
- Story: a DETR query has a content part and a positional part. DAB-DETR makes the positional
  part an explicit 4D anchor (x, y, w, h), refined after every decoder layer, and uses w and h to
  stretch the positional attention into an oval matched to the object.
- ONE intuition: when the query IS a box, the decoder knows where and how big to look, and can
  refine that box layer by layer.
- Interactive `viz-anchor-attn` [computed]: drag and resize an anchor box on a toy image; the
  positional-attention heatmap (Eq. 5 / 6, real sinusoidal encodings with T = 20) updates live.
  Buttons: modulation on / off, temperature 20 vs 10000, and a 6-step "refine" animation
  (update rule sigma(delta + sigma^-1(old)) from the official code, toy deltas).
- Result: +1.7 AP from the anchor update alone (ablation); 45.7 AP at 50 epochs (DC5-R50, 3
  patterns).
- Math: Eq. 1-2 (anchor to positional query), Eq. 6 (modulated attention).

### 7. `ch-denoising` DN-DETR and DINO: teach with noisy answers
- Story: early in training, Hungarian matching keeps changing which slot owns which object, so
  the targets flip. DN-DETR adds noised copies of the ground-truth boxes as extra queries with a
  fixed target: "fix this box", behind an attention mask so answers do not leak. DINO adds hard
  negatives: slightly-further boxes must say "no object" (contrastive denoising), plus mixed
  query selection and look-forward-twice.
- ONE intuition: give the decoder an easy, stable homework problem (denoise a nearby box), and
  training stops thrashing.
- Interactive `viz-noise-box` [computed, noise scheme from the papers and DINO's code]: a ground
  truth box; "sample noise" draws noised queries. Toggle DN (all positives, center shift and
  scale with lambda = 0.4) vs CDN (positive inner shell, negative outer shell), with each query
  labelled by its training target (reconstruct / no object). A small matching-flip counter shows
  the instability idea [toy].
- Results: mask matters (24.0 vs 43.4 AP); DINO R50 49.0 AP in 12 epochs (4-scale); ladder
  43.4 -> 47.9 in the ablation; 63.2 / 63.3 only with Swin-L + Objects365 + TTA (labelled).
- Math: the noise ranges in words; Eq. 7 attention mask as a picture, not a formula.

## Act III: making DETR run fast (2 chapters)

### 8. `ch-rt-detr` RT-DETR: DETRs beat YOLOs once you count NMS
- Story: YOLO speed numbers usually omit NMS. RT-DETR measures end to end, puts self-attention
  only on the coarsest map (AIFI) and fuses scales with CNNs (CCFF), and selects queries whose
  class score also implies a good box. No NMS means the threshold does not change speed.
- ONE intuition: NMS time is hidden latency that grows with the number of boxes; a set predictor
  does not pay it.
- Interactive `viz-nms-threshold` [sourced + approximate]: confidence-threshold stepper over the
  paper's 6 thresholds; bars show YOLOv5 / YOLOv8 box counts (read from Figure 2, labelled
  approximate) and, at the 3 measured points, YOLOv8 NMS time and AP (Table 1). RT-DETR line
  stays flat. Buttons per threshold.
- Results: R50 53.1 AP at 108 FPS vs YOLOv8-L 52.9 at 71 FPS (T4, TensorRT FP16, 640 px, NMS
  included); decoder layers can be dropped at inference (-0.1 AP, -0.5 ms for one layer).
- Math: Eq. 2 uncertainty in words.

### 9. `ch-rf-detr` RF-DETR: train once, pick your latency
- Story: RF-DETR swaps in a pretrained DINOv2 ViT backbone and trains one weight-sharing network
  where every step samples a random patch size, resolution and window count; decoder depth and
  query count are cut at inference. Afterwards a grid search over thousands of configs reads off
  an accuracy-latency curve for your hardware and dataset.
- ONE intuition: instead of designing N/S/M/L models by hand, train one network that contains
  them all and choose a point on the curve.
- Interactive `viz-latency-budget` [sourced]: a latency-budget slider (T4 ms); the page picks the
  best table-backed RF-DETR config under the budget (N..Max: resolution, patch, windows, decoder
  layers, AP) and plots it against D-FINE / YOLOv11 / RT-DETR baselines from the same table.
  Buttons: COCO / show baselines.
- Results: nano 48.0 AP at 2.3 ms; 2XL 60.1 AP at 17.2 ms (first real-time > 60 AP); D-FINE still
  wins at M, L, XL on COCO (said plainly).
- Math: tokens per side = resolution / patch (the knob that matters most).

## Unnarrated closing sections
- `ch-timeline` family tree: DETR (2020-05) -> Deformable DETR (2020-10) -> DAB-DETR (2022-01)
  -> DN-DETR (2022-03) -> DINO (2022-03) -> RT-DETR (2023-04) -> RF-DETR (2025-11), each edge
  labelled with what it inherited.
- `ch-summary` one paragraph.
- `ch-deeper` where to go deeper: specific sections / equations / tables per paper.
