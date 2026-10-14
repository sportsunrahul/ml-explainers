# Fact check: explainer page and atlas text

Scope: parts/ch-*.html, parts/kit/*.js (including the KIT.*Text readouts and KIT.guideSteps), the figure badges, and the atlas/parts/*.html text. These were checked against notes/claims/*.json and the primary sources in sources/.
The work was split into six slices, A to F. Each section below is one slice's report.
Final state: `bash build.sh` ends PASS, and `node atlas/assemble.mjs` builds atlas.html.

Slices:
- A: cost, map, distill, closing, hero; kits 10/20/25/26/30; atlas cost, distill
- B: grid, granularity, qat; kits 40/41/42/52; atlas grid, granularity, qat
- C: ptq; kits 43/50/51/53; atlas calib
- D: prune, lottery; kits 60/61/62; atlas sparsity
- E: outliers, gptq-awq; kits 70/71/72; atlas outliers
- F: fp8, tensorrt, guide; kits 80/81/90; atlas fp8, guide


---

## Slice A

### Fact check, fork A: cost, map, distill, closing, hero; kits 10/20/25/26/30; atlas cost, distill

### Must-fix

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

### Verified, no change
- Roofline toy statements match KIT.roofCost: FP32 to FP16 on the 3x3 conv is 8x (math-bound); depthwise is 2x; 4-bit weights leave the 3x3 conv at 18.87 us. The roofText readout uses only toy values. The Toy model badge is honest.
- Temperature toy numbers match KIT.tempCore: bus 0.047 and cat <0.001 at T=1; at T=3, 0.21 and 0.014 versus the student's 0.096; about 55x at T=20; about 7x with T^2. The Computed badge is honest.
- Hinton: Eq. 1-4, the quotes, MNIST 67/146/74 and 1200/800 units, T 20/8/2.5-4, speech Table 1, weight 0.5, T=2, Table 5 (44.5/57.0/58.9), "development set". All checked against the tex and the PDF.
- Wang: the three reasons, psi=0.5, the per-level FPN mask, Table 5 = COCO FPN (31.2/34.8/36.9) and Table 1 = KITTI toy (45.96/52.63/62.63, +0.9 for -D, -8.9 for -F). The code-only facts (0.01 loss weight; teacher .eval() and requires_grad=False) are confirmed in code/trainval_net_sup.py.txt.
- FitNets: 11th student layer guided by teacher layer 2, about 2.5M vs 9M parameters. DeiT: Eq. 2 is printed student first, tau=3.0 and lambda=0.1, Table 3 (81.8/81.8 soft/83.4), 86M and 84M parameters.
- Scan counts in 25-map-data.js match all 117 scan-* claims. The tree arXiv ids match every abs.html. TensorRT docs version is 11.4.0.
- Closing "Where to go deeper": all section, table, equation and version references are checked against the PDFs and abs.html. The TRT 11.0.0 note on the removal of the implicit-quantization calibrators is confirmed. "Tenths of a point" matches Wu Table 5 (detector best-calibration drops of 0.09-0.13 mAP).

### Nits (not edited)
- ch-distill: "it is the only lever that cuts operations and bytes in every layer at once" is debatable, because uniform channel pruning does too. "the only one that needs a full training run" is also debatable: pruning with retraining (Han) and lottery tickets need full runs as well.
- ch-distill: "From about T = 6 on it stays near the Eq. 4 value". At T=6, gT^2 is 84% of the limit; it reaches 91% only at T=10. "Approaches" would be more exact.
- ch-distill: "The MNIST student has two hidden layers of 800 units and no regularization". The paper says "no regularization" of the hard-label baseline. The distilled net is "regularized solely" by the soft targets. The page says that later, so the meaning is correct.
- ch-cost: "Its bytes are mostly activations" (3x3 conv). In the toy, activations are 58% of bytes at FP16. That is a majority, but only just.

### Open doubts
- None of the tree descriptions in 26-map.js are paper results; they are one-line summaries. The "builds on" edges are labelled as our reading. The dfq -> SmoothQuant edge is the most interpretive.

---

## Slice B

### Fact check, fork B: ch-grid, ch-granularity, ch-qat; kits 40, 41, 42, 52; atlas grid, granularity, qat

Checked against notes/claims.json and the primary sources (Jacob main.tex and appendix.tex, Wu main.tex and PDF,
Nagel 2021 PDF and Fig. 5 render, Krishnamoorthi PDF, LSQ tex, Bengio PDF, PACT PDF, TensorRT 11.4 txt pages).
All data-claim values in scope match their claims. Toy numbers were checked against notes/sim-grid.txt and
notes/sim-ste.txt.

### Must-fix

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

### Nits (not edited)

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

### Open doubts

- ch-granularity toy shape "two all-zero channels" and "rest within about +/-16" are read off Nagel Fig. 5
  (render of p. 11). Six wide channels (5, 15, 16, 28, 29, 30) and a max near 95 are confirmed by eye.
  Channels 11 and 23 look flat but the figure cannot prove they are exactly zero.
- Nagel's Sec. 3.6 text says W8A8 stays "within 0.7%" for all models, yet its Table 6 shows EfficientDet-D1
  losing 1.41 mAP (40.08 to 38.67). The page uses the table, which is the right choice; the paper is
  self-inconsistent.
- Badges "Computed" on viz-ruler, viz-intmm and viz-ste run the papers' quantizer, integer pipeline and LSQ
  gradient on toy inputs. That matches the rule as written, but a reader might read "Computed" as "paper
  numbers". The captions say "toy" clearly, so left as is.

---

## Slice C

### Fork C: ch-ptq, kits 43/50/51/53, atlas calib

Scope checked: parts/ch-ptq.html (prose, captions, 3 tables, eq. labels, pipeline step texts, aside),
parts/kit/43-calib.js, 50-cle.js, 51-adaround.js, 53-ptqpipe.js (readout text functions),
atlas/parts/calib.article.html, calib.dialogs.html.
All 69 data-claim spans in scope match their claim values. Equation numbers (Nagel 2021 Eq. 16, 21, 22,
26, 27, 28, 29, 30, 33, 34, 35; DFQ Eq. 11; AdaRound Eq. 22, 25) match the PDFs. Atlas Wu Table 5 numbers
match wu20-t5-* claims; the TensorRT 11 calibrator-removal statement matches the 10.x-to-11.x migration guide.

### Must-fix

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

### Nits

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

### Open doubts

- On a detector aside: "For a ResNet-FPN detector at 8 bits, per-channel weights and a good activation
  range are usually enough" is advice; it is consistent with Wu Table 5 (Faster/Mask R-CNN R50-FPN lose
  0.1 to 0.6 points) and BRECQ 8/8 rows but has no span or citation.
- Badges: viz-cle and viz-adaround "Computed" are honest (exact arithmetic on stated toy inputs);
  viz-ptqpipe "Sourced" is a redraw of Nagel Fig. 8 with our summary texts, labelled as such.

---

## Slice D

### Fork D: ch-prune, ch-lottery, kits 60/61/62, atlas sparsity

All data-claim spans in ch-prune.html and ch-lottery.html (and the spans in 62-imp.js) match their claim values;
spot checks against han2015, han2016dc, frankle2019, frankle2020lmc, li2017filters, liu2017slim, liu2019rethink,
blalock2020 and mishra2021nm sources agree. Badges are honest: viz-sparse and viz-dc are Computed (toy weights,
exact algorithms; sourced numbers only in captions/tables), viz-imp is Toy model (the mask is computed, the
"training" is a seeded drift; only text-stated LeNet findings are printed). Kit readouts: 60-sparse prints only
computed counts plus the spanned 2x; the 2:4 storage figure (18 bits per kept FP16 value, 43.8% less) matches
Mishra's 36-vs-64-bit example. 61-dc prints only computed values. 62-imp round-to-P_m mapping (0.8^r) matches the
paper's stated P_m points (51.3, 21, 13.5, 3.6, 2.9).

### Must-fix

1. parts/ch-prune.html, eq. note: "Deep Compression (arXiv 1510.00149v5, Sec. 2) uses the same rule."
   Problem: DC Sec. 2 only says "all connections with weights below a threshold are removed" and builds on Han 2015;
   it never states the q x std rule (grep: no "standard deviation" in han2016dc). FIXED: now says DC builds on the
   method, quotes its sentence, and notes it does not restate the rule.
2. parts/ch-lottery.html, claim card "At scale": "73.7% ... no better than random reinitialization at 73.4%".
   Problem: 73.7 is 0.3 points above 73.4 (LMC Table 2), so "no better" overstates. FIXED: "only 0.3 points above".
3. parts/ch-lottery.html: "Matrix sizes must be multiples of 16 for FP16 and 32 for INT8".
   Problem: Mishra (software.tex l.71, results.tex l.48) says layers whose inner (GEMM-K) dimension is not a multiple
   are left unpruned; it is the summed dimension, not "matrix sizes", and it is a pruning rule, not a hard requirement.
   FIXED: "The paper prunes only layers whose summed dimension is a multiple of 16 ... 32 ...".

### Nits

- atlas/parts/sparsity.dialogs.html: "measured no whole network" was ambiguous (the paper measures whole-network
  accuracy, not speed). FIXED: "timed no whole network".
- ch-lottery "Their conclusion is that operation and parameter counts are a loose proxy for latency": Blalock et al.
  state this in Sec. 2 while citing others; their main conclusion is that results are hard to compare. Acceptable.
- ch-lottery Li et al. timing: "inference over the test set" is right for CIFAR-10 (10,000 images, Li appendix).

### Open doubts

- ch-lottery "Blalock et al. measure no hardware": ShrinkBench reports parameter and FLOP counts only, as far as the
  source shows; not verified exhaustively.
- ch-prune "Most weights in a trained network are close to zero" is a general statement with no source span.
- The new "0.3 points" in the LMC card is computed from two spanned values and is not itself spanned.

Test build: bash build.sh parts/ORDER build/fc-D -> PASS.

---

## Slice E

### Fork E: outliers, GPTQ/AWQ, outliers atlas

Scope: parts/ch-outliers.html, parts/ch-gptq-awq.html, parts/kit/70-actsurf.js, 71-gptq.js, 72-awq.js,
atlas/parts/outliers.article.html, atlas/parts/outliers.dialogs.html. All spanned claims were looked up in
notes/claims.json, and the prose around them was checked against sources/{dettmers2022,xiao2023,
yuan2022ptq4vit,frantar2023,lin2024,frantar2022obc,sun2024wanda,frantar2023sparsegpt}. Kit readouts were
checked against notes/sim-actsurf.txt, sim-gptq.txt and sim-awq.txt. Test build into build/fc-E: PASS.

### Must-fix

1. parts/ch-gptq-awq.html, GPTQ figure caption: "the paper shows that block size does not change the result".
   Problem: GPTQ runs no block-size experiment. It argues (Sec. 4, Step 2) that "the final rounding decisions
   for column i are only affected by updates performed on this very column". Changed to "the paper explains
   why blocking does not change the rounding decisions". FIXED.
2. parts/ch-gptq-awq.html, speed paragraph: "the paper notes that its kernels do not apply directly to large
   batches". Problem: GPTQ Sec. 6 says only that the study "does not consider ... speedups in batched
   execution". Changed to "the paper lists speedups in batched execution as outside its study". FIXED.
3. parts/ch-gptq-awq.html, pruning section: "Wanda ... is AWQ's idea applied to pruning". Problem: this is
   our framing, stated as fact. Wanda's own motivation is Dettmers et al.'s outlier features (main.tex l.129).
   It cites AWQ only in related work. Changed to "brings the same signal to pruning (our framing; the paper
   builds on LLM.int8()'s outlier features, not on AWQ)". FIXED.
4. atlas/parts/outliers.article.html: "the extra 16-bit multiply makes it slower than plain FP16 for smaller
   models". Problem: SmoothQuant Table 11 shows LLM.int8() slower than FP16 at OPT-13B (237.1 against
   152.6 ms) and at OPT-30B/256 (387.9 against 343.0 ms). The caption says "mostly slower". Dettmers Table 5
   shows a slowdown at 6.7B-sized layers and below. "For smaller models" understates this. Changed to
   "often makes it slower than plain FP16". FIXED.
5. atlas/parts/outliers.dialogs.html: "Dettmers et al. report up to 20x in models from 6.7B parameters up;
   Xiao et al. report about 100x in OPT-13B." Problem: Dettmers's "up to 20x" describes features that appear
   before the 6.7B phase shift too (intro l.128; Table 4 covers 125M-13B). Xiao's "~100x" is a general
   Sec. 3 statement and is not tied to OPT-13B; their OPT-13B Fig. 4 shows channels "greater than 70".
   Rewritten to say this. FIXED.

### Nits

- ch-outliers, PTQ4ViT paragraph: "With it, DeiT-S ... keeps 79.47%" credits the whole result to twin
  uniform quantization. The 79.47 is full PTQ4ViT, which also uses a Hessian-guided scaling metric. Not edited.
- ch-outliers: "That last step is memory-bound" (about decoding) is not in SmoothQuant. It is general
  background and is consistent with GPTQ Sec. 5. Not edited.
- outliers dialog: "so they cannot simply be clipped" is our inference from Dettmers's zeroing ablation.
  It is reasonable, but it is not stated in the paper. Not edited.
- atlas article: "SmoothQuant measured 2-3 levels". The paper computes the number from a formula; it does
  not measure it. Not edited.
- ch-gptq-awq: "Unlike GPTQ, AWQ does not fit the weights to the calibration data; it only measures
  activation sizes" follows the paper's own claim. AWQ's alpha search does evaluate L(s) on calibration
  inputs, and it also uses MSE-based weight clipping. The text says "The paper argues", which is fair.

### Verified, no change

- All 60+ spanned values in both chapters match claims.json and the primary sources. This includes Dettmers
  Tables 1/4/5, Sec. 4.1-4.2, SmoothQuant Tables 1/3/8/11 and Eq. 2-4, GPTQ Tables 1/2/5/6 and Eq. 1-2,
  Alg. 1, AWQ Tables 1/2/4/6/7/10 and Figs. 8/9, OBC Table 1, SparseGPT Table 1, and Wanda Table 3.
- Kit readouts (actsurfText, gptqText, awqText) print toy values only. They match the sim notes, and the
  chapter prose matches them: about 3 levels, 45% -> 1.6%, 4.7%/2.3% at alpha 0.5, output error 1.41 -> 0.40,
  GPTQ 5.4x below RTN, AWQ optimum at 0.42 with about 2x lower error. The only sourced number is 6.0, which
  is spanned.
- The badges are honest. actsurf is "Toy model" (invented data, papers' quantizers). GPTQ and AWQ are
  "Computed" (paper algorithms on toy data, and the captions list our conventions).
- Chapter cross-references (chapters 1, 5, 6, 9) match the ORDER numbering.

### Open doubts

- AWQ caption: "20 grid points spaced evenly from 0 to 1, including both ends" is labelled as our reading.
  The paper gives only "grid size of 20" over [0, 1].
- Dettmers caveat in the claim card ("the step to 6.7B also switches model family, GPT-J to fairseq") is
  labelled as our reading. Table 4 supports it (GPT-J 6.0B, FSEQ 6.7B), but the paper does not raise it.

---

## Slice F

### Fork F: ch-fp8, ch-tensorrt, ch-guide, kits 80/81/90, atlas fp8 + guide

All 151 data-claim spans in the three chapters match their claim values. Quotes from the TensorRT docs,
Micikevicius, van Baalen and TVM were found verbatim in sources/ (dash and "us" ASCII-fied). Section,
table and figure numbers checked: Micikevicius Sec. 2, 3.1, 4.3, Tables 1/2/5; van Baalen Sec. 2, 4.1,
Fig. 3, Fig. 5, Tables 1/2/6; Wu Tables 5/7/9; Nagel Tables 6/10; BRECQ Table 5; Mishra Table 4 (15 rows);
TVM Sec. 3. Toy byte counts in 81-fuse.js recomputed: 8.59 -> 7.03 -> 6.25 -> 3.91 MiB; BN fold saves
1.56 MiB, matching the prose. FP8 readouts match notes/sim-fp8.txt.

### Must-fix

1. parts/kit/90-guide.js (KIT.guideSteps, FP8 bullet): "TensorRT falls back to higher precision for FP8
   convolutions with kernels larger than 32." The step shows for every CC >= 8.9, including CC 10.0 (B200).
   Source (release notes 11.0.0, Limitations) limits it to SM89/90/120/121; SM100 is not listed.
   FIXED: now "On SM89, SM90, SM120 and SM121, TensorRT falls back to FP16 or FP32 kernels for FP8
   convolutions with kernel sizes larger than 32 (for example 7x7)."
2. parts/ch-fp8.html: "The answer is E4M3, by a small margin." Our own computation (notes/sim-fp8.txt)
   gives MSE INT8 1.96e-3 vs E4M3 6.56e-4, a 3x gap (4.50 vs 5.29 bits). FIXED: "with about a third of
   INT8's mean squared error, which is less than one bit of accuracy more" (data-toy span).

### Nits

- ch-fp8.html: "Van Baalen et al. state the same ordering for their three test shapes". Their Fig. 5 also
  has E2/E3 (E2 is best on the normal). FIXED wording: "find the same winners among these formats".
- ch-fp8.html, ch-tensorrt.html: the scale is "stored in FP32, FP16 or BF16". The schemes page says "a
  16-bit or 32-bit floating point scalar"; BF16 is not named there. Not edited.
- ch-fp8.html: "E4M3 puts ... eight marks into every doubling". True for normal binades only; the subnormal
  binades have fewer. Not edited.
- 90-guide.js ptqLoss.cnn = 0.2 is a round-up of 0.13 (Wu Table 7 Faster R-CNN); the flowchart prints
  "about 0.2 points ... (our reading)". Acceptable.
- 90-guide.js: 2:4 bullet appears for every CC >= 8.0. The saved support matrix does not list sparsity
  support; the chapter says so. Our reading, not sourced.

### Open doubts

- ch-fp8.html "On a detector": "That is simulated quantization on their own tooling". Van Baalen do not say
  "simulated" for Table 1, nor name the tool (AIMET is named only for Table 6). Likely true; unsourced.
  Same for "Qualcomm tooling" in ch-guide Table E and the guide YOLO step.
- 90-guide.js: "FP8 pays off for outlier-heavy transformer layers". Van Baalen say FP8-E4 is more accurate
  there in PTQ, but recommend W8A16 or QAT as more efficient fixes (Sec. 8). Wording may overstate.
- ch-guide.html badge "Sourced": the figure is our ordered recipe with sourced numbers in it; the caption
  says the order and stops are our reading. "Sourced" is defensible but generous.

Test build: `bash build.sh parts/ORDER build/fc-F` ends PASS.
