# Fact check: notes/claims/ptq-guide.json and notes/claims/grid-qat.json

Independent check against the primary sources under `sources/` (LaTeX, `pdftotext -layout`, abs.html,
and rendered figure PDFs). Scripts and extracted text live in the job scratch dir, not in `sources/`.
Backups of the unedited claim files: `$CLAUDE_JOB_DIR/tmp/fc/backup/`.

## 1. Summary

| item | count |
|---|---|
| claims checked mechanically | 1096 (911 ptq-guide + 185 grid-qat) |
| claims checked semantically | 1096 (see Sec. 3) |
| claims whose `value` was wrong | 0 |
| claims whose `claim` text was corrected | 79 (77 ptq-guide + 2 grid-qat) |
| claims newly marked uncertain | 0 |
| claims still marked uncertain | 9 (all figure readings; see Sec. 5) |

No `value`, `loc`, `arxiv`, `id` or `paper` field was changed. Both files are valid JSON, ids are unique,
the files are ASCII.

## 2. Mechanical pass

- `arxiv` field vs abs.html version: 1096/1096 match (Wu v1, Nagel 2021 v1, BRECQ v2, AdaRound v2,
  DFQ v3, Jacob v1, Krishnamoorthi v1, LSQ v3, PACT v2, Bengio v1). All date claims match abs.html.
- `value` string found in LaTeX or PDF text (boundary-aware match), and for table claims on the page of
  that table: 1090/1096. The 6 misses:

| id | value | why it is not a text match | status |
|---|---|---|---|
| wu20-fig2-max | 2.09 | read from figure vectors | already uncertain; consistent with rendered Fig. 2 |
| wu20-fig2-entropy | 1.39 | read from figure vectors | already uncertain; consistent with rendered Fig. 2 |
| wu20-fig2-p9999 | 1.12 | read from figure vectors | already uncertain; consistent with rendered Fig. 2 |
| wu20-fig3-most-sensitive | block0 depthwise | text is inside figure PDF only | verified from rendered Fig. 3 labels |
| brecq21-faster | 240x | source writes `240$\times$` (PDF "240x" with a times sign) | correct ASCII form |
| brecq21-iterations | 2x10^4 | source writes `$2\times 10^4$` | correct ASCII form |

If the page renders a Unicode times sign, the lint will not find "240x" or "2x10^4"; write them in ASCII.

## 3. Semantic pass (what was checked)

- Table cells, row AND column: every LaTeX table was parsed into a grid. For each of the 934 table claims
  with LaTeX source, the row named in `loc` was located by label, and the value was checked at the column
  index that all claims with the same column label map to; header text at that index was printed and read
  for every column label. 818 claims passed automatically (0 mismatches; the checker was tested by
  injecting two wrong values, and it caught both). The other 116 (BRECQ Tables 2-5 Brecq/Faster R-CNN rows,
  AdaRound Tables 1/7/9, Jacob Tables 4.4-4.8, LSQ Tables 1/3, Wu Table 9 config strings) failed only on
  parser label handling and were checked by eye against the printed grids: all correct.
- PACT Table 1 (PDF only, 10 claims): checked against the PDF text table: all correct.
- Table numbering: BRECQ PDF numbering is LaTeX order + 1 (a wraptable is Table 1); claims use PDF numbers
  correctly. Jacob tables are 4.1-4.8 as printed.
- Claim-text metadata: every distinct claim-text prefix per table (setting, metric, dataset, backbone,
  PTQ vs QAT, bit-widths, granularity, run averaging, relative vs absolute) was read against the caption
  and the method text. Corrections are in Sec. 4.
- All 152 non-table claims were read against the source sentence (section, footnote, abs.html, figure).
- Rendered and viewed: Wu Fig. 2 (calibration ranges), Wu Fig. 3 (sensitivity labels), Nagel 2021 Fig. 5
  (per-channel ranges).
- Equations in `notes/papers/*.md` for these 10 papers: every TeX snippet was compared, whitespace-
  normalized, to the LaTeX. Jacob, Krishnamoorthi, LSQ, AdaRound, DFQ and BRECQ snippets match verbatim.
  Wu Eq. 11 and Eq. 13, and Nagel Eq. 3, 21, 28 and 41 are multi-line `align`/`split`/`cases` bodies that
  the notes join into one line (the notes say so); the content matches. Bengio Eq. 3 and Eq. 13 match
  (the note adds trailing comments). PACT Eq. 1-3 (no TeX in the source) match the PDF. Equation numbers
  cited (Jacob 1-14, B.1; Wu 1-16; Nagel 1-41; Krishnamoorthi 1-15; LSQ 1-5; Bengio 3, 13) match the PDFs.
  The source typos are kept as printed: Jacob Eq. 12 `\max(x, a)` where r is meant; Wu Eq. 13 conditions
  on `y` where x is meant.

## 4. Every change (all are `claim`-text edits)

### 4.1 AdaRound Table 7 setup (19 claims)
Reason: AdaRound Sec. 5 "ImageNet" paragraph: "We use the same experimental setup as described earlier,
with the exception of optimizing AdaRound with 2048 images for 20k iterations." Loc: Table 7 (p8).
- old prefix: `PTQ methods compared, ImageNet val top-1 %, per-layer (per-tensor) weight scale; MobilenetV2 AdaRound cells use CLE as preprocessing (dagger); `
- new prefix: `... (dagger); AdaRound rows optimized with 2048 images for 20k iterations (Sec. 5 ImageNet paragraph), not the default 1024 / 10k; `
- ids: adaround20-t7-full-precision-32-32-r18, adaround20-t7-full-precision-32-32-r50,
  adaround20-t7-full-precision-32-32-incv3, adaround20-t7-full-precision-32-32-mnv2,
  adaround20-t7-nearest-w4-a32-r18, adaround20-t7-nearest-w4-a32-r50, adaround20-t7-nearest-w4-a32-incv3,
  adaround20-t7-nearest-w4-a32-mnv2, adaround20-t7-adaround-w4-a32-r18, adaround20-t7-adaround-w4-a32-r50,
  adaround20-t7-adaround-w4-a32-incv3, adaround20-t7-adaround-w4-a32-mnv2,
  adaround20-t7-dfq-our-implp-w4-a8-r18, adaround20-t7-dfq-our-implp-w4-a8-r50,
  adaround20-t7-dfq-our-implp-w4-a8-mnv2, adaround20-t7-adaround-w-act-quant-w4-a8-r18,
  adaround20-t7-adaround-w-act-quant-w4-a8-r50, adaround20-t7-adaround-w-act-quant-w4-a8-incv3,
  adaround20-t7-adaround-w-act-quant-w4-a8-mnv2

### 4.2 AdaRound setup claims (3)
- adaround20-calib-images (Sec. 5 Experimental setup, p6):
  "AdaRound optimization uses this many unlabeled ImageNet training images" ->
  "AdaRound default setup uses this many unlabeled ImageNet training images (Table 7 literature comparison
  uses 2048 images; Table 9 DeeplabV3+ uses 512)". Reason: 1024 is the default "unless otherwise stated".
- adaround20-iterations (Sec. 5, p6):
  "AdaRound: Adam, default hyper-parameters, 10k iterations per layer, batch size 32" ->
  "AdaRound default setup: Adam, default hyper-parameters, 10k iterations, batch size 32 (Table 7 literature
  comparison uses 20k iterations)". Reason: as 4.1. "per layer" removed; the source says "for 10k
  iterations" without "per layer".
- adaround20-256-images (Sec. 5 p7, Fig. 4):
  "... within 2% of FP32 accuracy (ResNet18, absolute)" ->
  "... within 2% of FP32 accuracy (ResNet18, Fig. 4; paper writes "within 2%", absolute vs relative not
  stated)". Reason: the paper does not say absolute.

### 4.3 Nagel 2021 prose claims that its own tables contradict (4)
- nagel21-w8a8-within (Sec. 3.6, p16): added "and per-channel (40.08 -> 38.67)" to the EfficientDet-D1
  exception. Reason: Table 6 per-channel W8A8 is also 1.41 points below FP32.
- nagel21-conclusion-w8a8 (Sec. 5, p25):
  "Conclusion: W8A8 PTQ within 1% of floating point for all networks (absolute)" ->
  "... (absolute points). Table 6 contradicts it for EfficientDet-D1: per-tensor 40.08 -> 38.29 (1.79),
  per-channel 40.08 -> 38.67 (1.41); quote rows, not this sentence".
- nagel21-qat-w4a4-dw-drop (Sec. 4.5, p24):
  "Prose: W4A4 QAT on depthwise-separable networks, even per-channel, can drop up to 5% (absolute points)" ->
  "... Excludes EfficientDet-D1: Table 10 per-channel W4A4 is 40.08 -> 28.68 (11.40 points); the next
  sentence says EfficientDet-D1 "remains more difficult to quantize"".
- nagel21-qat-pc-gap (Sec. 4.5, p24):
  "Prose: per-channel QAT brings MobileNetV2 and EfficientNet lite to less than 1.5% gap (absolute points)" ->
  "Prose: per-channel W4A8 QAT ... Table 10: MobileNetV2 71.72 -> 70.48 (1.24); EfficientNet lite
  75.42 -> 73.92 (1.50, not below 1.5)".

### 4.4 Wu Table 3 calibrator (25 claims)
Reason: the Table 3 caption and Sec. 4.1 do not name the calibrator; only Table 4 says "max calibration".
Loc: Table 3 (p7).
- old prefix: `INT8 weights only (activations FP32), max calibration, granularity and BN folding, ImageNet top-1 %`
- new prefix: `INT8 weights only (activations FP32), granularity and BN folding, ImageNet top-1 %; calibrator not stated for Table 3 (our reading: max, since its per-channel cells equal Table 4 max-calibration cells)`
- ids: wu20-t3-{mobilenet-v1, mobilenet-v2, resnet50-v1p5, resnext50, efficientnet-b0}-{fp32, pc, pc-foldbn,
  pt, pt-foldbn} (25 ids).

### 4.5 Wu Mask R-CNN metric (24 claims)
Reason: Wu Table 2 gives only "mAP" for Mask R-CNN; box AP vs mask AP is not stated.
- old: `Mask R-CNN (torchvision resnet50-fpn, COCO 2017 val mAP)`
- new: `Mask R-CNN (torchvision resnet50-fpn, COCO 2017 val mAP; box or mask AP not stated)`
- ids: wu20-t2-mask-r-cnn-fp32; wu20-t4-mask-r-cnn-{fp32, acc, rel}; wu20-t5-mask-r-cnn-{fp32, max, entropy,
  p99p9, p99p99, p99p999, p99p9999}; wu20-t7-mask-r-cnn-{fp32, ptq-calib, ptq-acc, ptq-rel, qat-acc,
  qat-rel}; wu20-t11-mask-r-cnn-{fp32, max, entropy, p99p9, p99p99, p99p999, p99p9999}.

### 4.6 Wu Fig. 3 (2)
Reason: values are only in the figure PDF; verified from the rendered x-axis labels.
- wu20-fig3-most-sensitive: appended "; read from the Fig. 3 x-axis labels (src/figures/sensitivity.pdf)".
- wu20-fig3-depthwise-count: "(the other 3 are project layers)" -> "(block0, 1, 3, 15, 4, 2, 14 depthwise;
  block15, 12, 14 project); read from the Fig. 3 x-axis labels".

### 4.7 grid-qat.json (2)
- jacob-resnet-within (Sec. 4.1.1): "... (paper wording 'within 2%')" -> "... (paper wording "within 2%").
  Table 4.1 depth 150 does not meet it: 78.8 -> 76.7 is 2.1 points".
- krishnamoorthi-ptq-within (Abstract item 1): appended "(prose). Table 3 Mobilenet-v2_1_224 misses it:
  asymmetric per-channel 0.697 and symmetric per-channel 0.698 vs float 0.719 (about 2.2 points)".

## 5. Remaining doubts

- Figure readings stay `uncertain`: wu20-fig2-max / -entropy / -p9999 / -logscale and nagel21-fig5-max /
  -min / -wide-count / -narrow-bound / -dead. The rendered figures agree: Wu Fig. 2 dashed lines near 1.12,
  1.4 and 2.1; Nagel Fig. 5 wide channels 5, 15, 16, 28, 29, 30, top whisker near 95 (ch. 28; ch. 30 near
  94), bottom near -60 (ch. 16), channels 11 and 23 flat, the rest within about +-16.
- Wu Table 5 FCN row: bold sits on 99.999% (63.90) while Entropy and 99.99% are 64.00. Table 7 picks
  Entropy 64.00. Do not use the bold as "best" for FCN.
- Wu Table 5 / Table 7 prefix "activations per-tensor ..., scale (symmetric)" is not in the captions. It
  rests on Sec. 3.2 ("For activations, only per-tensor quantization is practical") and Sec. 3.3 ("scale
  quantization is sufficient for int8 quantization of all the networks we studied"). Kept.
- BRECQ does not state the weight granularity (per-tensor or per-channel) for any table, or whether the
  first and last layers stay at 8 bits for COCO (stated only for weight-only ImageNet, Sec. 4.2). BRECQ's
  "decreases 0.21% mAP" (Sec. 4.4) is 34.55 -> 34.34, i.e. absolute points written as "%"; no claim
  records it.
- Jacob Tables 4.7/4.8: relative vs absolute not explicit (the claim text already says so).
- Nagel Table 9 MobileNetV2 "W4A8 baseline" is 0.10 for both PTQ and QAT; the claims map correctly. QAT does
  not recover this case without CLE (70.13 with CLE).
- Nagel 2021 Table 3 LaTeX comments out a "Replace ReLU6" row; claims follow the PDF (no such row). DFQ
  Table 1 does have that row; claims match.
- Wu App. A text calls RetinaNet "an additional segmentation network"; Wu Table 2 lists it under Detection
  with mAP. Claims use Table 2 (detection). Correct.

## 6. Statements in understanding.md / outline.md that the sources contradict

1. outline.md, intro: "the vision rows that the primary sources measure (Wu 2004.09602 Tables 4-7; Nagel
   2106.08295 Tables 6 and 9)". Source: Nagel Table 9 caption: "Ablation study with various PTQ
   initialization. ... ImageNet validation accuracy (%) averaged over 3 runs" (rows ResNet18 and
   MobileNetV2 only). The detector/segmenter QAT rows are Table 10 ("our standard QAT pipeline ...
   DeeplabV3 ... EfficientDet-D1"). Also, Wu Table 6 ("Partial post training quantization") has rows
   MobileNet v1, EfficientNet b0, EfficientNet b3, Transformer, BERT large: no detection or segmentation
   row. Fix: "Wu Tables 4, 5, 7 (and 8, 11); Nagel Tables 6 and 10".
2. outline.md, ch-ptq: "Nagel's "within 0.7%" prose fails for EfficientDet-D1 per-tensor". Source, Nagel
   Table 6: EfficientDet-D1 per-channel W8A8 is 38.67 vs FP32 40.08 (1.41 points), and MobileNetV2
   per-tensor W8A8 is 70.99 vs 71.72 (0.73). The prose also fails for those. The conclusion's "within only
   1% of the floating-point accuracy for all networks" fails for EfficientDet-D1 too (1.79 / 1.41).
3. outline.md, ch-qat: "LSQ keeps first/last layers at 8 bits; PACT leaves them unquantized; Jacob quantizes
   everything." Source: Jacob says nothing about first or last layers (no hit in main.tex or appendix.tex);
   it only says "a single set of quantization parameters for all values within each activations array and
   within each weights array". "Jacob quantizes everything" is our reading; label it. PACT: Sec. 4.2 "do not
   quantize the first and last layers" holds for accuracy runs, but Sec. 6 (hardware model) says "we still
   used 8 bit precision to execute the first and last layers". LSQ Sec. 2.3 ("except the first and last,
   which always use 8-bit") is correct.
4. understanding.md, Jacob: "Prior: binary/ternary nets that lost accuracy on efficient architectures and
   were not efficient on real hardware." Source, Jacob Sec. 1: "prior approaches have not been evaluated on
   a reasonable baseline architecture. The most common baseline architectures, AlexNet, VGG and GoogleNet,
   are all over-parameterized by design"; bit-shift (binary/ternary) nets "provide little benefit on
   existing hardware"; "1 bit quantization often leads to substantial performance degradation". The
   critique is "not tested on efficient architectures", not "lost accuracy on them".
5. understanding.md, guide summary: "within about a point ... except depthwise-heavy nets per-tensor
   (EfficientDet-D1 W8A8 40.08 -> 38.29) and 4-bit weights per-tensor (collapse). QAT recovers most of the
   rest." Sources:
   - Nagel Table 6 per-channel W8A8 EfficientDet-D1 is 38.67 (1.41 points): the exception is not
     per-tensor only.
   - Nagel Table 6 W4A8 per-tensor collapses only for EfficientDet-D1 (0.31); DeeplabV3 holds 70.80
     (2.14 below 72.94). BRECQ Table 5 Faster R-CNN R50 W4A8: Bias Correction 0.25, AdaRound 37.58, BRECQ
     38.29 (FP 38.55). The collapse depends on the rounding method, not only on granularity.
   - "QAT recovers most": Nagel Table 10 EfficientDet-D1 W4A4 per-channel is 28.68 (11.40 below FP32);
     Nagel Table 9 MobileNetV2 W4A8 QAT stays at 0.10 without CLE.
6. Prose vs own table, for the page to avoid quoting alone: Jacob Sec. 4.1.1 "within 2%" vs Table 4.1
   depth 150 (78.8 -> 76.7, 2.1 points); Krishnamoorthi abstract "within 2%" vs Table 3 Mobilenet-v2
   (0.719 -> 0.697 / 0.698); Nagel Sec. 4.5 "up to 5%" vs Table 10 EfficientDet-D1 W4A4 (11.40 points).

Everything else that outline.md and understanding.md state about these 10 papers checks out, including:
Jacob Eq. 1, 5-8, 11, 12 and Sec. 3.2 BN folding; the COCO MobileNet SSD row (320x320, per-tensor uint8,
QAT with a 500,000-step activation delay, single thread on Snapdragon 835, AP@[.5:.95]); latency curves are
Figs. 1.1c, 4.1, 4.2; Krishnamoorthi Eq. 1-2 and 14-15; Wu Eq. 13 and Sec. 3.4 (entropy in words, credited
to TensorRT; no MSE calibrator; percentiles 99.9 to 99.9999); Esser Eq. 5; Bengio Eq. 13 (plain identity
STE, Hinton lecture 15b); Wu Table 3 EfficientNet b0 12.93 vs 76.72; Wu Table 7 Faster R-CNN 36.82 vs
36.76 and DeepLabV3 67.50 both; Nagel Table 6 EfficientDet-D1 0.31 / 35.08; BRECQ Table 5 Faster R-CNN R50
38.55 / 38.29 / 34.23; Wu Fig. 3 (10 of 82 layers skipped, 7 of the 10 depthwise); the Nagel PTQ pipeline
order (CLE -> add quantizers -> weight range -> AdaRound or bias correction -> activation range).
