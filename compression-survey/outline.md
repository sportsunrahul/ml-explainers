# Outline: Making vision models cheap to run

Working title: "Smaller, fewer bits, fewer weights: compressing vision models for GPU inference".
Reader: basic deep learning only. Running example: a detector (ResNet-50-FPN Faster R-CNN / RetinaNet
class, plus EfficientDet-D1) and a segmenter (DeepLabV3), because those are the vision rows that the
primary sources measure (Wu 2004.09602 Tables 4-7; Nagel 2106.08295 Tables 6 and 9).
No narration (checkpoint 0). Cue ids and reset buttons are still built so audio can be added later.
Long page `explainer.html` plus `atlas.html`; each figure's logic is written once as a scene.

Color key for the whole page (one meaning each):
- blue = full-precision (FP32/FP16) values and the teacher
- yellow = the quantization grid / integer values / the student
- green = what the method keeps or protects (kept weights, salient channels, the fused kernel)
- red = error and loss (rounding error, clipping error, the loss term, the gradient)
- brown = removed or frozen (pruned weights, frozen teacher, dropped kernels)

## Story

Problem: a trained detector is too slow and too large for the appliance budget. Where does the time go?
Key insight: inference cost is bytes moved plus math done; there are four levers (a smaller model, fewer
bits per number, fewer numbers, better kernels), and a compiler turns the last three into real speed.
Mechanism: distillation -> quantization from first principles -> PTQ tools -> QAT -> pruning -> the LLM
era's outlier problem and its fixes -> FP8 -> the compiler. Results: sourced accuracy deltas on
detection and segmentation. What came next / what to do: the decision guide.

## Hero
`viz-hero`: a 16x16 block of FP32 weights (blue cells). Act 1: values snap onto 256 yellow levels
(quantize, 32 -> 8 bits). Act 2: small cells fade to brown (prune; 2:4 pattern). Act 3: the block shrinks
into a smaller student block (distill). Act 4: three kernel boxes merge into one green box (compile). Ends
on a static diagram of the four levers with "bytes per weight" labels. Replay button `btn-hero-replay`.
Toy, drawn.

## Act I: what costs time

### 1. `ch-cost` Where the time goes
- Intuition: a GPU layer is limited either by math or by moving bytes; for many detector layers, bytes
  dominate, so fewer bits per number speed things up twice (less traffic, faster integer/FP8 math units).
- Visual `viz-roofline`: pick a layer (3x3 conv in a ResNet stage, 1x1 conv, depthwise conv, an FC/
  attention projection at batch 1 vs 64) and a precision (FP32 / FP16 / INT8). FLOPs and bytes are computed
  exactly from the layer shape; the point moves on a roofline. The GPU roof is a toy with a labelled
  FP16:INT8 throughput ratio of 1:2 and a stated bandwidth. Buttons `btn-roof-fp32|fp16|int8`, layer chips.
- Prediction prompt: "does INT8 help a depthwise conv as much as a big 3x3 conv?"
- Then the four levers, one paragraph each, with links to their chapters.
- Toy (roof) + computed (layer FLOPs/bytes). A real counterpart to cite: AWQ Fig. 3, an RTX 4090 roofline.
  Precision support by GPU generation from the TensorRT 11.4 support matrix.

### 2. `ch-map` The field at a glance
- Intuition: compression research moved from CNNs on phones (2015-2020) to LLM weights (2022 on); the
  vision methods are older and more settled, which is good news for a detector.
- Visual `viz-map`: stacked bars of arXiv title counts per year for distillation / quantization / pruning,
  with a toggle for "titles naming LLMs" vs "titles naming detection or segmentation". Computed from the
  scan; every count is a `data-claim` whose loc is the query; labelled heuristic.
- Family tree `viz-tree`: the 11 spine papers + 23 branches + TensorRT on a time axis, colored by lever,
  with "builds on" edges (Hinton -> FitNets -> detector feature imitation -> DeiT; OBS/OBC -> GPTQ;
  DFQ CLE -> SmoothQuant scaling idea -> AWQ; Han 2015 -> Deep Compression -> LTH -> LMC; ...).
- Claim card for the scan (what it measured, what it cannot see: papers that use the idea without the word).

## Act II: a smaller student

### 3. `ch-distill` Knowledge distillation
- Intuition: a teacher's wrong answers carry information ("this truck looks a bit like a bus, not at all
  like a cat"); raising the softmax temperature exposes it, and a small student trained on those soft
  targets learns more per example than from hard labels.
- Visual `viz-temp`: teacher logits for one image over 6 classes; temperature slider T = 1..20 softens
  the distribution; overlay the student's distribution; readout of KL and gradient magnitude; a "x T^2"
  toggle shows why Hinton scales the soft loss by T^2. Toy logits; formulas from Hinton (Eq. 1-2, Sec. 2.1).
- Equation build: q_i = exp(z_i/T) / sum_j exp(z_j/T); loss = (1-a) CE(y, q^S) + a T^2 KL(q^T || q^S) (DeiT Eq. 2 prints KL(student, teacher); our direction is
  "our reading");
  gradient dC/dz_i = (1/T)(q_i - p_i) (Hinton Eq. 2) and the high-T approximation (Eq. 3-4).
- Architecture walkthrough `viz-distflow` (archflow): image -> frozen teacher -> logits/T (brown, gradient
  stops); image -> student -> logits/T and logits; two losses; backward reaches the student only.
- Delta for detection: logits are not enough for a detector; FitNets hints (2014) and feature imitation
  near objects (Wang 2019) distill intermediate features; DeiT distillation token for ViTs. One line each
  + one sourced number each.
- Sourced: Hinton MNIST (prose, Sec. 3: 67 / 146 / 74 errors at T = 20), speech Table 1 (ensemble of same-size
  models distilled into one, not a smaller student), 3% data Table 5; Wang Table 5 (R50-FPN Faster R-CNN:
  31.2 -> 34.8 AP, teacher 36.9) and Table 1 (full-map imitation hurts); DeiT-B 81.8 -> 83.4.
- Wording: the loss (1-a) CE + a T^2 KL is DeiT Eq. 2's form, labelled "our transcription" of Hinton's prose;
  quote Hinton's "condiderably [sic] lower weight"; truck/bus/cat is our example (Hinton: BMW / garbage truck /
  carrot); "frozen teacher" is our reading of Hinton (Wang's code freezes it; lambda 0.01 is code-only).

## Act III: fewer bits

### 4. `ch-grid` Numbers on a grid: scale and zero-point
- Intuition: quantization is a ruler with 256 marks laid over the range of a tensor; the scale is the mark
  spacing and the zero-point says which mark is 0.0. Too wide a ruler wastes marks (rounding error),
  too narrow clips (clipping error).
- Visual `viz-ruler`: a number line with a toy activation histogram; drag the range ends; bit-width
  buttons 8/4/2; symmetric vs asymmetric; readouts S, Z, rounding MSE, clipping MSE (computed exactly).
- Equation build: r = S (q - Z) (Jacob Eq. 1); quantize q = clamp(round(r/S) + Z) in Krishnamoorthi's form
  (1806.08342 Eq. 1-2); Jacob's own quantizer is Eq. 12 in (a, b, n) form. Quote source typos as printed.
- Second visual `viz-intmm`: an integer-only dot product stepper for one 3x3 conv output pixel: uint8
  inputs and weights -> int32 accumulator -> the zero-point cross terms (Jacob Eq. 7-8) -> requantize with
  M = S1 S2 / S3 written as M0 2^-n (Jacob Eq. 5-6) -> uint8 output. Numbers computed live.
- Bias in int32 with S_bias = S1 S2 (Jacob Sec. 2.4). Note: TensorRT uses symmetric (Z = 0) INT8 (docs).
- Sourced: Jacob results (ImageNet; COCO MobileNet SSD 320x320 per-tensor uint8 QAT, timed single-thread on a
  Snapdragon 835 CPU: a phone number, not a GPU number; latency curves are Figures 1.1c, 4.1, 4.2).

### 5. `ch-granularity` One scale or many? Calibration
- Intuition: one ruler per tensor fails when channels have very different ranges (depthwise convs after BN
  folding); one ruler per output channel fixes weights. Activations need a ruler chosen from data
  (calibration), and the best ruler clips the rare tail instead of covering it.
- Visual `viz-chanland` (3D, pe3): per-channel weight ranges of a toy depthwise layer (32 channels shaped
  like Nagel Fig. 5's spread) as a landscape; toggle per-tensor vs per-channel grids; readout of effective
  levels per channel and total MSE. Toy, labelled.
- Visual `viz-calib`: a long-tailed ReLU activation histogram (toy); calibrator buttons max / percentile
  99.99 / entropy (KL) / MSE pick a clip value computed by the actual algorithm; readout of MSE. Then the
  sourced table: Wu Table 5 (Faster R-CNN, Mask R-CNN, RetinaNet, DeepLabV3 under each calibrator).
- Equations: per-channel S_c; MSE range setting (Nagel). Wu has no MSE calibrator and no entropy equation:
  Sec. 3.4 describes entropy calibration in words and attributes it to TensorRT, so our KL implementation is
  labelled "our reading". Wu's calibrators: max, entropy, percentile 99.9 / 99.99 / 99.999 / 99.9999.
- Sourced: Nagel Table 6 W8A8/W4A8 per-tensor vs per-channel; Wu Table 3 (EfficientNet b0 per-tensor with folded
  BN 12.93 vs per-channel 76.72), Wu Table 5 (calibrators on detectors). Toy shaped by Nagel Fig. 5 (read off
  the figure, uncertain: 6 of 32 channels beyond +-40, max about 95, the rest within about +-16).

### 6. `ch-ptq` The post-training toolkit
- Intuition: before any retraining, you can rearrange the network so it quantizes better: move scale
  between consecutive layers (cross-layer equalization), correct the mean shift that rounding causes (bias
  correction), and choose up or down for each weight to minimise the layer's output error (AdaRound).
- Visual `viz-cle`: two consecutive layers with 6 channels; slider for per-channel scale s_i; ranges
  equalize while the function stays identical (shown by an output readout). Computed on toy weights.
- Visual `viz-adaround`: two weights on a 2D plane with the 4 grid corners; error contours from a toy
  Hessian; nearest rounding lands on a worse corner than the best one. Computed exactly. This figure
  returns in ch-gptq.
- Nagel standard PTQ pipeline as a stepper (CLE -> add quantizers -> weight range -> AdaRound or bias
  correction -> activation range), with the debugging flow chart as the bridge to the guide.
- Sourced: Nagel Tables 3-5 (CLE, bias correction, AdaRound), Table 6 standard PTQ incl. EfficientDet-D1
  W4A8 per-tensor collapse (0.31) and per-channel recovery (35.08); Nagel's "within 0.7%" prose fails for
  EfficientDet-D1 per-tensor, so quote table rows. DFQ Tables 3-4, AdaRound Table 9, BRECQ Table 5 (Faster
  R-CNN R50: FP 38.55, W4A8 38.29, W2A8 34.23). Wu Table 6 / Fig. 3: partial quantization (skip 10 of 82
  layers in EfficientNet b0, 7 of the 10 most sensitive are depthwise; classification only, Wu Table 6 has no
  detection rows). AdaRound Table 7 used 2048 images and 20k iterations. Nagel prose ("within 0.7%", "within
  1%", "up to 5%", "less than 1.5%") disagrees with its own tables for EfficientDet-D1: quote table rows only.

### 7. `ch-qat` Training through the rounding: QAT and the straight-through estimator
- Intuition: simulate the rounding in the forward pass so the network learns weights that survive it;
  the rounding has zero gradient almost everywhere, so pretend it is the identity inside the range.
- Visual `viz-ste`: the fake-quant staircase and its two derivatives (true: zero with spikes; STE: a box);
  a one-weight toy trained with true gradient (stuck) vs STE (moves); learned step size toggle (LSQ).
  Computed toy (simulated once in node before shipping).
- Equation build: fake quant x_hat = S (clamp(round(x/S) + Z) - Z); STE dx_hat/dx = 1 inside the range,
  0 outside: the clipped form is Krishnamoorthi Eq. 14-15, Wu Eq. 13, Esser Eq. 5; Bengio 2013 Eq. 13 is the
  plain identity STE (credited to Hinton's lecture). LSQ step-size gradient (Esser), one line.
- Contrast: Jacob does not learn ranges (min/max for weights, EMA for activations, "backprop as usual");
  PACT learns a clipping value, LSQ learns the step size. LSQ keeps first/last layers at 8 bits; PACT leaves
  them unquantized; Jacob does not discuss first/last layers.
- Architecture walkthrough `viz-qatflow` (archflow): one QAT fine-tuning step for a detector: images ->
  fake-quant conv blocks (BN folded per Jacob Sec. 3.2) -> FPN -> head -> detection loss -> backward via
  STE -> FP32 shadow weights and scales update. This replaces a separate training-step interlude.
- Sourced (Nagel Table 9 is an ImageNet ablation, not vision rows): Nagel Table 10 QAT (EfficientDet-D1, DeepLabV3), Wu Table 7 QAT vs PTQ (detectors: Faster R-CNN PTQ
  36.82 vs QAT 36.76; DeepLabV3 67.50 both), Jacob delay of activation quantization.

## Act IV: fewer weights

### 8. `ch-prune` Pruning and Deep Compression
- Intuition: most weights are near zero and can go; retrain and the rest compensate. But a GPU only goes
  faster if the zeros come in shapes it can skip.
- Visual `viz-sparse`: a weight matrix heatmap; sparsity slider; pattern buttons unstructured / 2:4 /
  whole channel; readout of what a dense GPU kernel skips (nothing / half the math with sparse tensor
  cores / a smaller dense layer). Toy weights.
- Deep Compression stepper `viz-dc`: prune -> k-means weight sharing (trained quantization) -> Huffman,
  with bits per weight and the sourced 35x (AlexNet) / 49x (VGG-16) totals.
- Equations: magnitude criterion; shared-weight gradient (sum of gradients in a cluster, Han Eq. 3).
- Sourced: Han Tables 1-6 (compression per network), Table 8 + Sec. 6.3 (timing: at batch 64 the sparse layer is
  slower in 16 of 18 cells; speedups appear at batch 1 FC layers); Blalock's critique.

### 9. `ch-lottery` Lottery tickets and structure a GPU can use
- Intuition: a big net contains a small subnetwork that would have trained just as well from the same
  starting weights; finding it needs the full training first, so it explains pruning more than it saves
  training. For speed, prune whole filters or use the 2:4 pattern.
- Visual `viz-imp`: iterative magnitude pruning loop stepper (train -> prune p% -> reset to the initial weights ->
  repeat; conv nets prune 10% or 15% per round; "rewind" is the LMC term for resetting to an early iteration)
  over a toy mask, with only the accuracy/sparsity points stated in Frankle's text (the curves have no
  tabulated values) and the random-reinit control. Frankle's "ResNet-18" is ResNet-20 (LMC Table 1).
- Claim card (LTH): claim, measurement (Conv-2/4/6, LeNet on MNIST/CIFAR10, random reinit control),
  falsifier (random reinit matches), and the scale caveat (needs rewinding to early training, LMC 1912.05671).
- Structured: filter pruning by L1 norm (Li 2017), network slimming by BN gamma (Liu 2017), "Rethinking"
  (training the pruned architecture from scratch is as good; Table 12 Faster R-CNN VOC row), and 2:4
  (Mishra 2021: train dense, prune 2:4 by magnitude, retrain with the same schedule and mask; for detectors,
  prune after the COCO fine-tune and repeat only that phase, Sec. 4.3). Table 4: 15 COCO/Cityscapes rows,
  dense FP16 / sparse FP16 / sparse INT8 (no dense INT8 column). No end-to-end speedup is measured: say
  "up to 2x math" (A100 INT8 GEMM microbenchmark, Fig. 3). Some efficient nets need channel permutation (Table 3).
- Sourced: Frankle, Li, Liu, Mishra tables; Blalock's critique.

## Act V: the LLM era

### 10. `ch-outliers` Outliers: LLM.int8() and SmoothQuant
- Intuition: at scale, transformers grow a few activation channels far larger than the rest (Dettmers: up
  to 20x; SmoothQuant: ~100x; cite each, do not merge);
  one ruler per tensor then wastes almost every level on them. Either keep those channels in FP16
  (LLM.int8) or move their difficulty into the weights with a per-channel rescale (SmoothQuant).
- Visual `viz-actsurf` (3D, pe3): activation magnitude over (token x channel) with a few outlier ridges
  (toy shaped like SmoothQuant Fig. 4); slider alpha (migration strength) smooths activations and roughens
  weights; readout of per-tensor INT8 error for X and for W. Computed on the toy.
- Equations: vector-wise quantization (Dettmers Eq. 7) and the mixed-precision decomposition (Eq. 8);
  SmoothQuant Eq. 2 (why per-channel activation scales cannot be factored out of a GEMM), then Eq. 3-4:
  Y = (X diag(s)^-1)(diag(s) W), s_j = max|X_j|^alpha / max|W_j|^(1-alpha).
- Toy shape: a few (3) ridges that persist across every token (SmoothQuant Fig. 4, read off a plot, so
  uncertain); show both papers' magnitude statements with their sources and do not reconcile them
  (Dettmers: 3-20x, in 6-7 dimensions, one-sided; SmoothQuant: ~100x).
- Speed: LLM.int8() is slower than FP16 for layers below 13B scale (Dettmers); SmoothQuant fixes that with
  pure INT8 GEMMs. The 175B no-degradation claim rests on Dettmers Fig. 1 (a plot) and SmoothQuant Table 3;
  Dettmers Table 1 stops at 13B.
- The claim card says Dettmers argues emergence tracks perplexity, not parameter count alone.
- Vision tie-in: ViTs have post-softmax and post-GELU distributions that a uniform grid fits badly
  (PTQ4ViT: a distribution-shape problem, not per-channel outliers; it has no detection results).
- Claim card for the emergent-outlier analysis (Dettmers: when outliers appear, by model size).
- Sourced: LLM.int8 degradation-free up to 175B; SmoothQuant W8A8 results and speedups.

### 11. `ch-gptq-awq` Four-bit weights: GPTQ and AWQ
- Intuition: token-by-token LLM decoding is memory-bound (callback to ch-cost), so shrinking weights to
  4 bits matters more than integer math. GPTQ rounds one column at a time and pushes each column's error
  onto the columns not yet rounded, guided by second-order statistics of real inputs. AWQ notices 1% of
  weight channels matter most (those that see large activations) and protects them by scaling.
- Visual `viz-gptq`: runs GPTQ on a toy 8x16 weight block with toy calibration inputs; column-by-column
  stepper; output-error readout vs round-to-nearest. Computed with the paper's algorithm. Recalls
  `viz-adaround` (same Hessian-shaped error).
- Visual `viz-awq`: per-channel activation magnitudes; slider alpha for s = s_X^alpha; readout of output
  error; computed on toy data.
- Equations: argmin ||WX - W_hat X||^2 (GPTQ Eq. 1); OBS update (Eq. 2-3); Cholesky reformulation;
  AWQ scaling (Eq. 4-5).
- Forward-only walkthrough: what is computed per layer from calibration data; no gradient anywhere.
- Sourced: GPTQ 175B in ~4 GPU hours, 3-4 bit perplexity; the single 80GB A100 result is 3-bit OPT-175B (the
  paper says 4-bit would not fit); speedups are batch-1 memory-bound decoding, not a detector claim. AWQ 1%
  salient (Table 1), Eq. 1-3 error ratio, Eq. 4-5 search; vision tie-in: OpenFlamingo COCO CIDEr (Table 6) and
  VILA (Tables 7, 10, incl. Jetson Orin). OBC has YOLOv5 pruning rows (Table 1) and runtimes (Table 7); quantization + 2:4 for YOLO only as figure curves. SparseGPT/Wanda
  one line each. AWQ v6 has internal inconsistencies (Fig. 2 vs Tables 1/3; 30 vs 33 tok/s): quote one, cite it.
- Toy algorithms: GPTQ Algorithm 1 with H = 2XX^T + lambda I, lambda = 0.01 mean(diag H), upper Cholesky factor
  U of H^-1 (not the plain inverse), err = (w - q(w)) / U_jj, W[:, j:] -= err U[j, j:]; per-row asymmetric
  min-max grid fixed before the loop ("our reading"); compare with RTN on the same grid; block size shrunk
  for the toy (label it; results identical). AWQ: Delta = max|w| / 2^(N-1) symmetric (state the convention for
  the top level), s = s_X^alpha with s_X = mean |X| per input channel, alpha on a 20-point grid in [0, 1]
  (spacing "our reading"), loss || Q(W diag s)(diag s^-1 X) - WX ||.

### 12. `ch-fp8` FP8: floating point in eight bits
- Intuition: a float spends bits on an exponent, so its marks are dense near zero and sparse far away;
  that suits bell-shaped tensors with outliers. E4M3 keeps precision (max 448), E5M2 keeps range (max
  57344). A per-tensor scale is still needed.
- Visual `viz-fp8`: all 256 codes of INT8 (scaled), E4M3 and E5M2 on a log-scaled number line; pick a
  toy distribution (Gaussian / Laplace / Gaussian + outliers) and see the MSE per format, computed by
  enumeration.
- Claim card (van Baalen 2303.17951): FP8 vs INT8 for inference, what was measured, hardware-cost argument.
  Their "FP8-E4" is a 255-value variant, not E4M3; their Fig. 5 uses uniform, normal and Student-t inputs, so
  viz-fp8 uses those three shapes and labels its MSE "our computation". Micikevicius has no detection rows.
- E4M3: bias 7, max 448, no inf, NaN only S.1111.111; E5M2: bias 15, max 57,344 (write "57,344" as printed),
  inf + NaN at e = 31. TensorRT rounds to nearest even and clips to +-448.
- Sourced: Micikevicius format table and results; van Baalen conclusions; TensorRT support matrix (which
  compute capabilities run FP8).

## Act VI: deployment

### 13. `ch-tensorrt` What an inference compiler does
- Intuition: the compiler rewrites the graph so fewer kernels touch memory (fusion), tries several
  kernels for each layer on the actual GPU and keeps the fastest (tactic selection), and places precision
  where the network says (Q/DQ nodes). The engine is tied to that GPU and that TensorRT version.
- Visual `viz-fuse`: a detector neck graph (conv -> BN -> ReLU -> conv -> add -> ReLU) with Q/DQ nodes;
  steps: fold BN, fuse conv+bias+ReLU, absorb Q/DQ into an INT8 conv, fuse the residual add; readout of
  kernels launched and bytes written. Then a tactic race: candidate kernels timed, fastest kept (toy times).
- TensorRT 11 change: implicit quantization (calibrators) removed in 11.0; explicit Q/DQ via Model
  Optimizer (release notes 11.0.0). TVM as the open-source analogue (fusion rules, learned cost model).
- Sourced: TensorRT 11.4 docs (how-trt-works, fusion catalog, quantized types, release notes 11.0.0).

### 14. `ch-guide` A decision guide for detection and segmentation
- Intuition: try the cheap, safe steps first, measure on your own validation set, and only pay for
  training when the measurement says so.
- Visual `viz-guide`: a decision widget: GPU generation (Turing / Ampere / Ada / Hopper / Blackwell),
  model family (CNN detector / depthwise-heavy detector / transformer detector or segmenter), accuracy
  budget, training data available -> an ordered recipe with the expected loss band and its sources.
- Table of sourced deltas, each row with backbone + dataset + toolchain, relative-percent (Wu) and absolute
  points (Nagel, BRECQ, van Baalen) in separate columns: Wu (Faster R-CNN R50-FPN, Mask R-CNN R50-FPN, RetinaNet
  ResNeXt101-FPN, FCN and DeepLabV3 R101, COCO), Nagel (EfficientDet-D1 COCO, DeepLabV3-MNv2 VOC), BRECQ
  (Faster R-CNN R50), van Baalen (YOLO), Jacob (MobileNet SSD, phone CPU).
- YOLO rows: van Baalen 2303.17951 (Qualcomm tooling, not TensorRT): Table 1 YoloV5 COCO mAP 56.3, INT8 PTQ -1.8;
  Table 6 PTQ rows (mostly INT8, some W8A16 layers; no dataset or mAP definition given) Yolov3 0.653 -> 0.649,
  Yolo-v5 0.557 -> 0.559. OBC 2208.11580 Table 1 has YOLOv5l pruning rows (baseline 66.97) and Table 7 YOLOv5s runtimes; its YOLO
  quantization + 2:4 results are only curves in figures. Label the toolchain; still say
  "measure on your own validation set". Do not reconcile van Baalen Tables 2-4 (different FP8-QAT numbers).
- TensorRT 11 notes for the guide (generic wording): implicit quantization/calibrators removed in 11.0 (use
  explicit Q/DQ, e.g. Model Optimizer); engines built by 8.x do not run on an 11.x runtime; FP8 convolutions
  with kernel size > 32 fall back to higher precision on SM 8.9/9.0/12.x; EfficientNMS_ONNX_TRT plugin removed in 11.0 (the removal list names only that variant).
- Precision by compute capability (TensorRT 11.4 support matrix): 7.5 FP16+INT8; 8.0/8.6 add TF32, BF16;
  8.9/9.0 add FP8 (FP4 emulated); 10.0+ native FP4. 2:4 sparsity support by GPU not in the saved pages.
- Order: FP16 engine -> INT8 PTQ (per-channel weights, percentile/entropy/MSE activations, calibration set
  from deployment-like images) -> per-layer sensitivity, keep sensitive layers in FP16 -> short QAT ->
  structural options (distill a smaller backbone; 2:4 on Ampere+; FP8 on Ada/Hopper+).

## Closing sections
- `ch-summary`: one paragraph.
- `ch-deeper`: per item: paper, section, equation/table number.

## Visual inventory (toy vs sourced)
| viz | kind |
|---|---|
| viz-hero | toy, drawn |
| viz-roofline | computed layer math + toy GPU roof |
| viz-map, viz-tree | computed from arXiv scan (heuristic) / sourced dates |
| viz-temp | toy logits, Hinton formulas |
| viz-distflow | archflow, sourced setup |
| viz-ruler, viz-intmm | computed (Jacob equations on toy numbers) |
| viz-chanland (3D), viz-calib | toy distributions, real calibrator algorithms; sourced table |
| viz-cle, viz-adaround | computed on toy weights |
| viz-ste, viz-qatflow | computed toy (node-simulated); archflow |
| viz-sparse, viz-dc | toy weights; sourced compression ratios |
| viz-imp | toy mask; sourced accuracy points |
| viz-actsurf (3D) | toy shaped like the paper's figure; computed smoothing |
| viz-gptq, viz-awq | computed with the papers' algorithms on toy data |
| viz-fp8 | computed by enumerating the formats |
| viz-fuse | toy graph, sourced fusion rules; toy timings |
| viz-guide | sourced deltas; our recipe, labelled "our reading" |

## Atlas (atlas.html), 10 stages, scenes shared with the long page
cost (roofline) -> distill (temperature) -> grid (ruler) -> granularity (3D channel landscape) ->
calib (calibrators) -> qat (STE) -> sparsity (patterns) -> outliers (3D surface + smoothing) ->
fp8 (formats) -> guide.
