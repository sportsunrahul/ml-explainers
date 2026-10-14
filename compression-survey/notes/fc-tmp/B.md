# Fact check, fork B: ch-grid, ch-granularity, ch-qat; kits 40, 41, 42, 52; atlas grid, granularity, qat

Checked against notes/claims.json and the primary sources (Jacob main.tex and appendix.tex, Wu main.tex and PDF,
Nagel 2021 PDF and Fig. 5 render, Krishnamoorthi PDF, LSQ tex, Bengio PDF, PACT PDF, TensorRT 11.4 txt pages).
All data-claim values in scope match their claims. Toy numbers were checked against notes/sim-grid.txt and
notes/sim-ste.txt.

## Must-fix

1. parts/ch-qat.html, eqno under the LSQ quantizer: "With a zero-point the same operation is Nagel et al. Eq. 7
   ... and Wu et al. Eq. 12". Problem: Wu Eq. 12 is `x_hat = dequantize(quantize(x, b, s), b, s)`, which is
   scale (symmetric) quantization with no zero-point. Only Nagel Eq. 7 has a zero-point. FIXED: the sentence
   now says Nagel Eq. 7 writes it with a zero-point and Wu Eq. 12 writes it as dequantize(quantize(x)).
2. parts/ch-qat.html, after the Wu Table 7 card: "Faster R-CNN and Mask R-CNN end slightly below their PTQ
   scores, and Wu et al. call such differences noise." Problem: Wu Sec. 5.1 names only ResNeXt-101, Mask R-CNN
   and GNMT as the PTQ-beats-QAT cases and calls those noise level. The text does not mention Faster R-CNN,
   although its Table 7 row also drops (36.82 to 36.76). FIXED: "Wu et al. put the Mask R-CNN gap at the noise
   level (their text does not discuss Faster R-CNN)."
3. parts/ch-qat.html, figure viz-qatflow badge "Computed". Problem: the ArchFlow diagram computes nothing and
   plots no paper numbers; its own caption says "a generic FPN detector, our example; shapes are symbolic".
   Under the page-authoring rule (computed = runs the paper's algorithm; sourced = plots the paper's numbers;
   else toy) it is a toy. FIXED: `data-toy` and a "Toy model" badge.

## Nits (not edited)

- ch-grid: "N_{levels} is 256 for 8 bits" sits in the gloss of Krishnamoorthi's Eq. 1-2 but its span cites
  jacob-levels-8bit (Jacob Sec. 3). Same value; the attribution is to the other paper.
- ch-grid: Krishnamoorthi Eq. 1-2 is quoted with clamp(0, N_levels-1, x_int); the source prints clamp(a, b, x)
  with the argument last in its definition but first-arguments-as-bounds in Eq. 2. The page matches Eq. 2.
- ch-granularity: "the toy uses Krishnamoorthi's asymmetric, per-channel variant, which changes the counts by
  about one bit" is our estimate (a symmetric grid can lose at most half its codes on a channel). Not sourced.
- ch-granularity: "every calibrator except 99.9% stays close to FP32": RetinaNet at 99.99% loses 0.75 mAP.
  "Close" is a judgement.
- ch-qat On a detector: "in Wu's measurements QAT added nothing" for ResNet-FPN detectors. RetinaNet QAT is
  +0.06 mAP over PTQ (39.25 vs 39.19), Faster and Mask R-CNN slightly lower. "Nothing measurable" would be more
  exact.
- ch-qat: Wu Table 7 rows are "COCO 2017 val" for DeepLabV3 too (segmentation on COCO), which the card states;
  fine, but readers may expect Pascal VOC as in the Nagel card above it.
- ch-granularity calibration figure uses kit 43-calib.js (fork C's scope); its caption and prose agree with
  notes/sim-grid.txt (p99.99 clips 9, entropy 7, MSE 3). Kit 43 readouts not checked here.

## Open doubts

- ch-granularity toy shape "two all-zero channels" and "rest within about +/-16" are read off Nagel Fig. 5
  (render of p. 11). Six wide channels (5, 15, 16, 28, 29, 30) and a max near 95 are confirmed by eye.
  Channels 11 and 23 look flat but the figure cannot prove they are exactly zero.
- Nagel's Sec. 3.6 text says W8A8 stays "within 0.7%" for all models, yet its Table 6 shows EfficientDet-D1
  losing 1.41 mAP (40.08 to 38.67). The page uses the table, which is the right choice; the paper is
  self-inconsistent.
- Badges "Computed" on viz-ruler, viz-intmm and viz-ste run the papers' quantizer, integer pipeline and LSQ
  gradient on toy inputs. That matches the rule as written, but a reader might read "Computed" as "paper
  numbers". The captions say "toy" clearly, so left as is.
