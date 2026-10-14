# Fact check, fork A: cost, map, distill, closing, hero; kits 10/20/25/26/30; atlas cost, distill

## Must-fix

1. parts/ch-cost.html, "On a detector": "The 1x1, depthwise and small-map head convolutions are bound by memory, so they speed up only when their activations get shorter too, not only their weights."
   Problem: the page's own roofline toy contradicts this for the head conv. Its bytes are 98% weights, and "4-bit weights" cuts its time from 2.41 to 0.64 us (computed with KIT.roofCost). The chapter body also says the head conv speeds up with 4-bit weights.
   FIXED: the aside now names only the 1x1 and depthwise convs, and states that the 5x5 head conv is the exception.

2. parts/ch-map.html, claim card: "titles naming LLMs make up a growing share ... (in 2025, from about a sixth for distillation ...)".
   Problem: the share does not grow every year. Quantization falls from 29.4% (2024) to 26.7% (2025), and distillation from 15.0% to 14.9%. 187/1257 = 14.9%, which is about a seventh, not a sixth.
   FIXED: the card now says "a much larger share ... than before", "about a seventh", and "the share does not rise every year".

3. atlas/parts/distill.article.html: "Hinton's own example is a BMW that might be taken for a garbage truck but never for a carrot."
   Problem: Hinton says the carrot mistake is much less probable, not impossible ("still many times more probable than mistaking it for a carrot"; introduction11.tex l.25). The whole point of soft targets is that the probability is small but not zero.
   FIXED: the text is reworded to match the paper.

4. atlas/parts/distill.article.html: "At T = 1 the teacher's wrong-class probabilities are almost zero, so they teach nothing."
   Problem: this overstates the paper. Hinton says these probabilities have very little influence on the loss, not none.
   FIXED: changed to "so they barely affect the loss".

## Verified, no change
- Roofline toy statements match KIT.roofCost: FP32 to FP16 on the 3x3 conv is 8x (math-bound); depthwise is 2x; 4-bit weights leave the 3x3 conv at 18.87 us. The roofText readout uses only toy values. The Toy model badge is honest.
- Temperature toy numbers match KIT.tempCore: bus 0.047 and cat <0.001 at T=1; at T=3, 0.21 and 0.014 versus the student's 0.096; about 55x at T=20; about 7x with T^2. The Computed badge is honest.
- Hinton: Eq. 1-4, the quotes, MNIST 67/146/74 and 1200/800 units, T 20/8/2.5-4, speech Table 1, weight 0.5, T=2, Table 5 (44.5/57.0/58.9), "development set". All checked against the tex and the PDF.
- Wang: the three reasons, psi=0.5, the per-level FPN mask, Table 5 = COCO FPN (31.2/34.8/36.9) and Table 1 = KITTI toy (45.96/52.63/62.63, +0.9 for -D, -8.9 for -F). The code-only facts (0.01 loss weight; teacher .eval() and requires_grad=False) are confirmed in code/trainval_net_sup.py.txt.
- FitNets: 11th student layer guided by teacher layer 2, about 2.5M vs 9M parameters. DeiT: Eq. 2 is printed student first, tau=3.0 and lambda=0.1, Table 3 (81.8/81.8 soft/83.4), 86M and 84M parameters.
- Scan counts in 25-map-data.js match all 117 scan-* claims. The tree arXiv ids match every abs.html. TensorRT docs version is 11.4.0.
- Closing "Where to go deeper": all section, table, equation and version references are checked against the PDFs and abs.html. The TRT 11.0.0 note on the removal of the implicit-quantization calibrators is confirmed. "Tenths of a point" matches Wu Table 5 (detector best-calibration drops of 0.09-0.13 mAP).

## Nits (not edited)
- ch-distill: "it is the only lever that cuts operations and bytes in every layer at once" is debatable, because uniform channel pruning does too. "the only one that needs a full training run" is also debatable: pruning with retraining (Han) and lottery tickets need full runs as well.
- ch-distill: "From about T = 6 on it stays near the Eq. 4 value". At T=6, gT^2 is 84% of the limit; it reaches 91% only at T=10. "Approaches" would be more exact.
- ch-distill: "The MNIST student has two hidden layers of 800 units and no regularization". The paper says "no regularization" of the hard-label baseline. The distilled net is "regularized solely" by the soft targets. The page says that later, so the meaning is correct.
- ch-cost: "Its bytes are mostly activations" (3x3 conv). In the toy, activations are 58% of bytes at FP16. That is a majority, but only just.

## Open doubts
- None of the tree descriptions in 26-map.js are paper results; they are one-line summaries. The "builds on" edges are labelled as our reading. The dfq -> SmoothQuant edge is the most interpretive.
