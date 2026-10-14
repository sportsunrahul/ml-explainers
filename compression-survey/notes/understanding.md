# Understanding: model compression for GPU inference

Details, equations (verbatim TeX) and locations live in `notes/papers/<name>.md`; every number is in
`notes/claims/*.json`. This file is the frame: problem, key idea, lens, and each paper as a delta.

## The problem
A trained detector or segmenter runs too slowly or uses too much memory on the target GPU. Time per layer is
max(operations / math rate, bytes / bandwidth). Four levers act on that: a smaller network (distillation),
fewer bits per number (quantization, FP8), fewer numbers (pruning), and better kernels (an inference compiler).

## Lever 1: a smaller student
- **Hinton 2015 (1503.02531v1)**, training-method lens. Prior: Caruana's model compression matched logits.
  Key idea: train the student on the teacher's softmax at a raised temperature T; the relative probabilities of
  wrong classes carry information. Soft loss scaled by T^2 so its gradient magnitude stays comparable (Eq. 2-4;
  matching logits is the high-T limit). MNIST 67 / 146 / 74 errors (prose). The teacher is not updated (our
  reading; the paper does not say "frozen").
- Deltas: FitNets 2014 adds intermediate "hints" via a regressor; Wang 2019 imitates features only near
  objects on one anchor-level feature map (Sec. 3.1; full-map imitation hurts, Table 1); DeiT 2020 adds a distillation token with a convnet teacher.

## Lever 2: fewer bits
- **Jacob 2017 (1712.05877v1)**, training-method lens. Prior: binary/ternary nets evaluated on overparameterized
  architectures (not tested on efficient ones) and not efficient on real hardware. Key idea: an affine map r = S(q - Z) with
  integer q and zero-point Z so real zero is exact; integer-only GEMM with int32 accumulation, zero-point cross
  terms, and a fixed-point requantization multiplier M = 2^-n M0. Training simulates quantization in the forward
  pass with ranges from min/max (weights) and EMA (activations), BN folded; activation quantization delayed.
  Results timed on phone CPUs.
- **Krishnamoorthi 2018 (branch)**: the clamp(round(r/S) + Z) form, per-channel weights, the clipped STE,
  recommendations.
- **Wu 2020 (2004.09602v1)**, analysis/benchmark lens. Prior: inconsistent reports of INT8 accuracy. Delta:
  argues scale (symmetric) quantization for weights on cost grounds, per-channel weights, compares max /
  entropy / percentile calibration on classification, detection, segmentation and language; partial
  quantization by per-layer sensitivity; QAT with STE as a short fine-tune. Deltas are relative percent.
- **Nagel 2021 (2106.08295v1)**, method + analysis lens. Consolidates PTQ (range setting incl. MSE, CLE, bias
  correction, AdaRound, a standard pipeline and a debugging flow) and QAT (STE graph, BN folding options,
  initialization, standard pipeline) with EfficientDet-D1 and DeepLabV3-MobileNetV2 rows. Absolute points.
  Branch ancestors: DFQ 2019 (CLE + bias correction), AdaRound 2020 (learned up/down rounding from a Hessian
  argument), BRECQ 2021 (block reconstruction; Faster R-CNN rows).
- **STE and learned ranges (branches)**: Bengio 2013 defines the plain identity STE; PACT learns an activation
  clip; LSQ learns the step size with a scaled gradient.

## Lever 3: fewer numbers
- **Han 2016 Deep Compression (1510.00149v5)**, method lens. Delta from Han 2015 (prune + retrain): adds
  trained quantization by k-means weight sharing (gradients summed per cluster) and Huffman coding; 35x / 49x.
  Its own timing (Table 8) shows sparse layers are slower at batch 64 on GPU in most cells.
- **Frankle 2019 LTH (1803.03635v5)**, analysis lens (claim card). Iterative magnitude pruning with reset to the
  original initialization finds subnetworks that train as well; random reinit control fails. LMC 2019 (branch):
  at scale, rewind to an early iteration instead of 0.
- **Structured branches**: Li 2017 (L1 filter pruning), Liu 2017 (BN-gamma slimming), Liu 2019 (pruned
  architecture from scratch is as good), Blalock 2020 (no standard benchmarks), Mishra 2021 (2:4 pattern for
  sparse tensor cores; train dense, prune, retrain; detection rows within about a point; up to 2x math).

## Lever 2, the LLM era
- **LLM.int8() (2208.07339v2)**, method + analysis lens. Delta: vector-wise INT8 plus a mixed-precision
  decomposition that keeps outlier feature dimensions (threshold 6.0) in FP16; emergent outliers analysis
  (claim card). Slower than FP16 below 13B-scale layers.
- **SmoothQuant (2211.10438v7)**. Delta: migrate activation difficulty into weights with a per-channel scale
  s_j = max|X_j|^alpha / max|W_j|^(1-alpha), so both quantize to INT8 and the GEMM stays pure INT8.
- **GPTQ (2210.17323v2)**. Delta from OBS/OBC: quantize weights column by column in a fixed order, push each
  column's error onto the remaining columns with the inverse Hessian of the layer inputs (Cholesky form, lazy
  blocks), one-shot, 3-4 bit. Speedups are batch-1 decoding.
- **AWQ (2306.00978v6)**. Delta: protect ~1% salient weight channels (chosen by activation magnitude) by a
  per-channel scale s = s_X^alpha searched on a grid, no mixed precision, no backprop or reconstruction.
- **FP8 (2209.05433v2)**: E4M3 (max 448, no inf) and E5M2 (max 57,344) with per-tensor scaling; van Baalen
  2023 (branch, claim card): for inference INT8 is often as accurate and cheaper in hardware; FP8 wins with
  outliers. Its YOLO rows (PTQ, Qualcomm tooling) and OBC's YOLOv5l pruning table are the only YOLO numbers in the sources.

## Lever 4: the compiler
- **TensorRT 11.4 docs**: parse -> graph optimizations and fusions (conv+bias+activation, Q/DQ folding) ->
  tactic selection by timing kernels on the target GPU (engines are GPU- and version-specific) -> precision
  from the strongly typed network. 11.0 removed implicit quantization and calibrators; explicit Q/DQ (e.g.
  Model Optimizer) is the path. Precision support by compute capability from the support matrix.
- **TVM 2018 (branch)**: fusion categories and a learned cost model for tuning.

## What to remember for the guide
Vision INT8 PTQ with a good calibrator is within about a point on the sourced detectors and segmenters (Wu
Tables 5/7; Nagel Table 6), except depthwise-heavy nets (EfficientDet-D1 W8A8 per-tensor 40.08 -> 38.29, and
still 1.41 points per-channel) and 4-bit weights, where the outcome depends on the method (per-tensor plain PTQ
collapses; BRECQ holds Faster R-CNN). QAT recovers most but not all (EfficientDet-D1 W4A4 stays far below FP32). 2:4 sparsity with retraining holds
detection accuracy (Mishra Table 4). Distillation of detectors needs feature imitation, not logits alone.
