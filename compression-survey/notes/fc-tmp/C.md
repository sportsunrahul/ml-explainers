# Fork C: ch-ptq, kits 43/50/51/53, atlas calib

Scope checked: parts/ch-ptq.html (prose, captions, 3 tables, eq. labels, pipeline step texts, aside),
parts/kit/43-calib.js, 50-cle.js, 51-adaround.js, 53-ptqpipe.js (readout text functions),
atlas/parts/calib.article.html, calib.dialogs.html.
All 69 data-claim spans in scope match their claim values. Equation numbers (Nagel 2021 Eq. 16, 21, 22,
26, 27, 28, 29, 30, 33, 34, 35; DFQ Eq. 11; AdaRound Eq. 22, 25) match the PDFs. Atlas Wu Table 5 numbers
match wu20-t5-* claims; the TensorRT 11 calibrator-removal statement matches the 10.x-to-11.x migration guide.

## Must-fix

1. parts/ch-ptq.html, intro: "using at most a few hundred unlabeled images".
   Problem: contradicts the same chapter. Nagel 2021 Sec. 3.5 footnote says 500 to 1000 images usually
   suffice; AdaRound uses 1024 (2048 for its Table 7). Data-free variants use none.
   FIXED: "using no data or a small set of unlabeled images (a few hundred to a few thousand)".

2. parts/ch-ptq.html, AdaRound prose: "its output error is almost half that of the nearest corner".
   Problem: with w = (0.42, 0.35) and H = [[1, .5], [.5, 1]] the costs are 0.446 (nearest, (0,0)) and
   0.256 (best, (1,0)): 57 percent of nearest, 43 percent lower (the kit readout prints "43% lower").
   FIXED: "about 40 percent lower than that of the nearest corner".

3. parts/ch-ptq.html, viz-adaround caption: "uses the 2 by 2 Hessian of Example 1 in the AdaRound paper
   (Sec. 2): inputs with unit variance and correlation 0.5".
   Problem: in AdaRound Example 1 (Eq. 7) the matrix is the task-loss Hessian H(w); the paper does not
   describe it as an input correlation. Reading it as E[x x^T] is ours.
   FIXED: says it is a task-loss Hessian in the paper and that we read it as an input correlation.

4. parts/ch-ptq.html, BRECQ Table 5 caption: "reconstruction in the backbone only".
   Problem: BRECQ App. B.4.4 says block reconstruction is applied only in the backbone, and "a naive layer
   reconstruction is applied to the rest of the network" (e.g. the FPN).
   FIXED: "block reconstruction in the backbone and layer-wise reconstruction in the rest, such as the FPN".

## Nits

- parts/ch-ptq.html, AdaRound setup sentence follows the DeeplabV3+ (Table 9) numbers, but that run used
  512 images, not the default 1024 (AdaRound Sec. 5.x "we only use 512 images"). FIXED by adding "and the
  DeeplabV3+ run used fewer images because they are larger" (no new number, so no new claim id needed;
  there is no claim id for the 512).
- "the total error falls by the same part" (bias correction): in the toy at 8 bits, t = 0, the RMS error
  falls 12.80% -> 9.81% while the mean shift falls 8.25% -> 0.58%; it falls in quadrature, not by the
  same amount. Not edited (wording is vague, not false).
- "Nagel et al. report that this choice is optimal for the combined precision": Nagel 2021 says DFQ
  (Nagel 2019) proves it; DFQ maximizes the total per-channel precision. Acceptable.
- "It is the only source here with 4-bit and 2-bit post-training rows for COCO detectors": OBC
  (frantar2022obc) has post-training 4w4a YOLOv5 COCO results, but only as mixed-precision curves (Fig. 3),
  not uniform rows, and no 2-bit. Statement holds as worded; fragile if other chapters cite more.
- Atlas calib article: "Here it clips fewer [than the percentile]" verified by running the kit code:
  max T=3.731 (0 clipped), p99.99 T=2.514 (9), entropy T=2.762 (7), MSE T=3.068 (3). The "because it
  looks at the tail's shape" reason is our explanation, not a source claim.
- CLE toy numbers in prose (3 of 255 levels, ~13% -> ~1%, 120-fold spread, identical FP32 output) match
  notes/sim-cle.txt and the kit code.

## Open doubts

- On a detector aside: "For a ResNet-FPN detector at 8 bits, per-channel weights and a good activation
  range are usually enough" is advice; it is consistent with Wu Table 5 (Faster/Mask R-CNN R50-FPN lose
  0.1 to 0.6 points) and BRECQ 8/8 rows but has no span or citation.
- Badges: viz-cle and viz-adaround "Computed" are honest (exact arithmetic on stated toy inputs);
  viz-ptqpipe "Sourced" is a redraw of Nagel Fig. 8 with our summary texts, labelled as such.
