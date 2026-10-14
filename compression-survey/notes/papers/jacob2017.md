# jacob2017 (SPINE) -- Quantization and Training of NNs for Efficient Integer-Arithmetic-Only Inference

Lens: training-method paper. Sources read: `sources/jacob2017/src/main.tex`, `src/appendix.tex`, `paper.pdf`
(numbers of equations, tables, figures and sections taken from the PDF), `abs.html`.

## 1. Identity
- Title: "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference".
- arXiv 1712.05877v1 (only version), submitted 2017-12-15 (abs.html).
- Authors: Benoit Jacob ... Dmitry Kalenichenko (8 authors), Google Inc.
- Venue: not stated on arXiv (comment "14 pages, 12 figures"). The source uses `cvpr.sty` with `\cvprPaperID{777}`
  (our reading: a CVPR submission; do not cite a venue from this).
- Code: no repo README in sources. The paper points to TensorFlow `tf.contrib.quantize` (Sec. 3, ref [19]),
  TensorFlow Lite and gemmlowp (Sec. 2.1 footnotes 1-2).

## 2. Core problem
Phones need low-latency CNNs, and integer units on common mobile CPUs/DSPs are faster than float units.
Prior quantization work was tested on over-parameterized nets (AlexNet, VGG, GoogleNet) and rarely
measured real on-device speed. The paper wants integer-only inference for already-efficient nets
(MobileNets) with little accuracy loss.

## 3. What prior methods did and why that hurt (per the paper, Sec. 1)
- Baselines were over-parameterized, so large compression was "proof-of-concepts at best".
- Weight-only quantization (refs [4,8,9,35]) helps storage, not compute.
- Binary/ternary/bit-shift nets: bit shifts give "little benefit on existing hardware with multiply-add
  instructions"; rarely measured on device.
- 1-bit weights and activations: "substantial performance degradation".
- Plain post-training quantization (Sec. 3): works for large models but "leads to significant accuracy
  drops for small models". Failure modes: (1) more than 100x differences in weight ranges across output
  channels, under one per-layer scale; (2) outlier weight values.

## 4. The one key idea
Define quantization as an affine map r = S(q - Z) with an exact integer zero, so a whole layer
(matmul + bias + ReLU6 clamp) runs on uint8 inputs, int32 accumulators and one fixed-point multiply,
and train with the same rounding simulated in float ("fake quantization") so the weights learn to
survive it.

## 5. Mechanism, step by step
Inference (Sec. 2):
1. Each weights array and each activations array gets ONE (S, Z) pair (per-tensor; Sec. 2.1:
   "a single set of quantization parameters for all values within each activations array and within each
   weights array"). q is uint8; S is a positive real (float in software); Z has the type of q and is the
   q that represents real 0.0 exactly (needed for zero-padding).
2. For r3 = r1 r2 (N x N), substitute Eq. 2 into the matmul (Eq. 3) to get Eq. 4: the only non-integer
   is the multiplier M = S1 S2 / S3 (Eq. 5).
3. M is found empirically to be in (0, 1). Write M = 2^-n M0 with M0 in [0.5, 1) (Eq. 6). Store M0 as the
   int32 nearest to 2^31 M0 (at least 2^30, so at least 30 bits of relative accuracy). Multiply by M0 as a
   fixed-point multiply (ARM NEON SQRDMULH, App. B) and by 2^-n as a rounding right shift.
4. Expand Eq. 4 (Eq. 7): N Z1 Z2 - Z1 a2^(k) - Z2 abar1^(i) + sum_j q1 q2. The row/column sums (Eq. 8) cost
   2N^2 additions; the core sum (Eq. 9) costs 2N^3. So zero-points cost O(N^2), and the core is the
   same integer GEMM as a zero-point-free scheme.
5. Fused layer (Sec. 2.4): q1 = weights, q2 = activations, both uint8. Core accumulation is
   `int32 += uint8 * uint8` (Eq. 10); signed int32 accumulator.
6. Bias: int32, Z_bias = 0, S_bias = S1 S2 (Eq. 11), i.e. the accumulator's scale, so the bias is a plain
   int32 add. Reason given: bias error is added to many outputs and acts as an overall (nonzero-mean)
   bias.
7. Then: scale down by M (fixed-point multiply by M0 + rounding shift), saturating cast to uint8 [0, 255],
   then clamp to a sub-interval for ReLU/ReLU6. In practice training learns to use the whole [0, 255], so
   the activation clamp is subsumed by the saturating cast (Sec. 2.4, last paragraph).
8. ARM trick (App. B): subtract 128 to turn uint8 into int8 (`int32 += int8 * int8`, Eq. B.1). Weights are
   kept in [-127, 127] (never -128), so each product is below 2^14 in absolute value and two products can
   sum in an int16 before going into int32 (SMULL, SMLAL, SADALP).
9. Rounding: RSHL breaks ties upward; the paper needs round-to-nearest with ties away from zero
   (example: -12/2^3 should be -2, RSHL gives -1). The upward bias "has been observed to cause significant
   loss of end-to-end accuracy".
10. Add layers need rescaling (multiplier S1/S2); Concat layers force all inputs and the output to share
    the same (S, Z), so concat is lossless (App. A.2, A.3).

Training (Sec. 3):
1. Train from a float graph; insert fake-quant ops where tensors will be downcast at inference
   (Algorithm 1). All weights and biases are stored in float and updated in float.
2. Weights are fake-quantized before the conv. If BN is used, BN is folded into the weights first
   (Sec. 3.2, Eq. 14), then the folded weights are quantized.
3. Activations are fake-quantized where inference would quantize them: after the activation function, and
   after a bypass add or concat (Fig. C.4 adds a "conv quant" before the residual add).
4. Biases are not fake-quantized (they become int32 at inference; their (S, Z) is derived from weights and
   activations).
5. Ranges: weights a = min w, b = max w (tweaked so int8 weights stay in [-127, 127]). Activations: [a; b]
   observed in training, aggregated by EMA with smoothing close to 1. Activation quantization is disabled at
   the start of training ("say, for 50 thousand to 2 million steps"); all protocols in App. D use
   500,000 steps.
6. [a; b] is nudged so 0.0 is exactly an integer z(a, b, n); then S = s(a, b, n), Z = z(a, b, n) (Eq. 13).
7. Backward pass: "Backpropagation still happens as usual" (Sec. 3). The paper does NOT name or define the
   straight-through estimator, and does not learn ranges by gradient. Ranges are min/max (weights) and EMA
   (activations). This is the contrast with PACT and LSQ.
8. After training, build the integer inference graph (fold, fuse) and run it in the low-bit engine.
   Fused-op granularity must match fake-quant placement in training (Sec. 2.4).

Which weights get a gradient and from what: float weights, BN gamma/beta, and biases get the ordinary task-loss
gradient through the fake-quantized forward pass. Quantization ranges get no gradient (min/max, EMA).

## 6. Equations
Custom macros: `\defas` = `\coloneqq`; `\clamp` = `\DeclareMathOperator{\clamp}{clamp}`.

Eq. 1 (`eq:quant-scheme`, Sec. 2.1):
```
r = S(q - Z)
```
Paper: "our quantization scheme"; S and Z are "our quantization parameters". r = real value; q = quantized
integer (uint8); S = scale, positive real; Z = zero-point, same type as q, equal to the q that represents 0.

Eq. 2 (`eq:scalar-operands`): `r_\alpha^{(i,j)} = S_\alpha(q_\alpha^{(i,j)} - Z_\alpha).`
alpha = 1, 2, 3 for the two inputs and the output of r3 = r1 r2.

Eq. 3 (`eq:quant-product-1`):
```
S_3(q_3^{(i,k)}-Z_3) = \sum_{j=1}^N S_1(q_1^{(i,j)} - Z_1) S_2(q_2^{(j,k)} - Z_2),
```
Eq. 4 (`eq:quant-product-2`):
```
q_3^{(i, k)} = Z_3 + M \sum_{j=1}^N (q_1^{(i, j)} - Z_1) (q_2^{(j, k)} - Z_2),
```
Eq. 5 (`eq:multiplier`): `M \defas \frac{S_1 S_2}{S_3}.` Paper: "the only non-integer is the multiplier M ...
it can be computed offline."

Eq. 6 (no label in TeX): `M = 2^{-n} M_0` with "M_0 ... in the interval [0.5, 1) and n is a non-negative
integer." M0 as int32 = nearest int to 2^31 M0.

Eq. 7 (`eq:quant-product-matrix-expanded`):
```
q_3^{(i, k)} = Z_3 + M \left( \vphantom{\sum_{j=1}^N} NZ_1 Z_2 - Z_1 a_2^{(k)} \right. \\
\left. - Z_2 \bar a_1^{(i)} + \sum_{j=1}^N q_1^{(i, j)}q_2^{(j, k)}\right)
```
Eq. 8 (`eq:sums-slices`):
```
a_2^{(k)} \defas \sum_{j=1}^N q_2^{(j, k)},\;\;
\bar a_1^{(i)} \defas \sum_{j=1}^N q_1^{(i, j)}.
```
Gloss: a_2^(k) = column sum of the activations (q2) for output column k; abar_1^(i) = row sum of the weights
(q1) for output row i. N Z1 Z2 is a constant.

Eq. 9 (`eq:core-accumulation`): `\sum_{j=1}^N q_1^{(i, j)}q_2^{(j, k)}` -- "takes 2N^3 arithmetic operations".

Eq. 10 (`eq:accum-types`): `\texttt{int32 += uint8 * uint8}.`

Eq. 11 (no label): `S_{\mathrm{bias}} = S_1 S_2,\;\;Z_{\mathrm{bias}}=0.` Bias type int32.

Eq. 12 (`eq:fakequant`, Sec. 3), verbatim, including the source's `x` where `r` is meant (as printed in
TeX and PDF; do not "fix" when quoting):
```
\clamp(r;a,b) &\defas \min\left(\max(x, a), b\right) \nonumber \\
s(a,b,n) &\defas \frac{b - a}{n - 1} \nonumber \\
q(r;a,b,n) &\defas \left\lfloor \frac{\clamp(r;a,b) - a}{s(a,b,n)} \right\rceil s(a,b,n) + a,
```
Paper: "r is a real-valued number to be quantized, [a;b] is the quantization range, n is the number of
quantization levels, and \lfloor\cdot\rceil denotes rounding to the nearest integer. n is fixed for all
layers in our experiments, e.g. n = 2^8 = 256 for 8 bit quantization." Output is a float on the grid
(quantize then dequantize). Note the form is in (a, b, n), not (S, Z).

Eq. 13 (`eq:quant-param-mapping`): `S= s(a,b,n), \,\,\, Z=z(a,b,n)`. z(a, b, n) is never written out.

Eq. 14 (no label, Sec. 3.2):
```
w_{\mathrm{fold}} \defas \frac{\gamma w}{\sqrt{EMA(\sigma^2_B) + \varepsilon}}.
```
gamma = BN scale; EMA(sigma_B^2) = moving-average estimate of the batch variance of conv outputs;
epsilon = small constant. Folded bias (from Fig. C.6-C.8 node labels): `\beta - \gamma \mu / \sigma`;
folded weight label `w \gamma / \sigma`.

Eq. B.1 (`eq:accum-types-signed`, App. B): `\texttt{int32 += int8 * int8}.`

## 7. Config / setup
| item | value | loc |
|---|---|---|
| bit-widths | uint8 weights and activations; int32 bias and accumulator; 7-bit variant (n = 2^7) for Inception | Sec. 2.1, 2.4, 4.1.2 |
| granularity | per-tensor (one S, Z per array), weights and activations | Sec. 2.1 |
| layers kept in higher precision | none stated (every conv/FC is uint8); only biases (int32) and accumulators (int32) are wider | Sec. 2.1, 2.4 |
| symmetric? | affine (asymmetric) with zero-point; int8 weights restricted to [-127, 127] | Sec. 2.1, 3.1 |
| rounding | round to nearest; fixed-point shift must round to nearest (ties away from zero) | Eq. 12, App. B |
| weight range | min/max of (folded) weights | Sec. 3.1 |
| activation range | EMA of observed [a; b], smoothing close to 1 | Sec. 3.1 |
| activation-quant delay | "50 thousand to 2 million steps" (Sec. 3.1); 500,000 in all protocols | App. D.1, D.3, D.4; Sec. 4.2.2 |
| ResNet training | momentum 0.9, batch 32, lr 1e-5, x0.1 every 30 epochs, 50 async workers, ~100 epochs | App. D.1 |
| ResNet input | resize short side 256, center crop 224x224 | App. D.1 |
| Inception v3 | ~10 million steps, batch 32, 50 async workers, 299x299, RMSProp lr 0.045 x0.94 every 2 epochs, momentum 0.9, decay 0.9, eps 1.0, weight EMA 0.9999 | App. D.2 |
| COCO (MobileNet SSD) | 320x320, inputs in [-1, 1], RMSprop batch 32, lr 4e-3 x0.1 every 100 epochs, 20 async workers, ~6 million steps | App. D.3 |
| COCO model change | all regular convs in SSD prediction layers replaced by separable convs | Sec. 4.2.2 |
| COCO metric | AP at IoU=.50:.05:.95 | App. D.3 |
| face detection | 320x320, same optimizer as COCO, ~3 million steps | App. D.4 |
| face attributes | 128x128, Adagrad batch 32 lr 0.1, 12 async workers, 20 million steps | App. D.4 |
| start point | not stated whether QAT starts from a float checkpoint (our reading: ResNet lr 1e-5 suggests fine-tuning, not stated) | App. D |
| GPU type / hours | not stated | -- |
| float GEMM | Eigen | Sec. 4 |
| int8 GEMM | gemmlowp | Sec. 4 |
| hardware | Snapdragon 835 LITTLE (Pixel 2), 835 big (Pixel 2), 821 big (Pixel 1); all ARM CPU, no GPU | Sec. 4.2.1 |
| latency method | binary runs model on random inputs for 100 s via adb, taskset 1/2/4 cores | App. D.4 |
| COCO latency | single thread, S835 big and LITTLE | Sec. 4.2.2 |

## 8. Results worth remembering
Table 4.1 (ResNet, ImageNet top-1; "Accuracies ... within 2%"; depth labels as printed):
| depth | float | integer-quantized |
|---|---|---|
| 50 | 76.4% | 74.9% |
| 100 | 78.0% | 76.6% |
| 150 | 78.8% | 76.7% |

Table 4.2 (ResNet50, ImageNet top-1, other schemes):
| scheme | weight bits | act bits | top-1 |
|---|---|---|---|
| BWN | 1 | float32 | 68.7% |
| TWN | 2 | float32 | 72.5% |
| INQ | 5 | float32 | 74.8% |
| FGQ | 2 | 8 | 70.8% |
| Ours | 8 | 8 | 74.9% |

Table 4.3 (Inception v3, ImageNet; mean and std over runs):
| act | type | top-1 mean | std | recall-5 mean | std |
|---|---|---|---|---|---|
| ReLU6 | floats | 78.4% | 0.1% | 94.1% | 0.1% |
| ReLU6 | 8 bits | 75.4% | 0.1% | 92.5% | 0.1% |
| ReLU6 | 7 bits | 75.0% | 0.3% | 92.4% | 0.2% |
| ReLU | floats | 78.3% | 0.1% | 94.2% | 0.1% |
| ReLU | 8 bits | 74.2% | 0.2% | 92.2% | 0.1% |
| ReLU | 7 bits | 73.7% | 0.3% | 92.0% | 0.1% |
Paper: ReLU6 degrades less because [0, 6] is "a natural range for activations".

MobileNet ImageNet latency vs top-1 (Fig. 1.1c S835 LITTLE, Fig. 4.1 S835 big, Fig. 4.2 S821 big).
Values are exact pgfplots coordinates in main.tex (latency ms, top-1 %). The source does not label which
depth multiplier / resolution each point is. Float has 12 points; 8-bit has 13.
- S835 LITTLE float: (355,70.4)(266,69.3)(198,67.2)(142,64.1)(110,63.6)(82.3,62.1)(62.6,59.9)(42.8,56.2)(39.4,50.4)(30.2,49)(21,46)(14.6,41.3)
- S835 LITTLE 8-bit: (171,69.2)(144,67.9)(90.1,65.8)(59.3,61.8)(60.1,61.3)(49.8,60.2)(31.5,58.4)(21,54.9)(26.1,48.9)(21.3,46.7)(13.6,44.4)(9,40.3)(13.1,50)
- S835 big float: (167,70.4)(123,69.3)(87,67.2)(58,64.1)(50,63.6)(38.4,62.1)(28.3,59.9)(19.4,56.2)(17.2,50.4)(13,49)(9.2,46)(6.3,41.3)
- S835 big 8-bit: (75,69.2)(64.2,67.9)(40.6,65.8)(26.7,61.8)(27.4,61.3)(22.7,60.2)(14.4,58.4)(9.4,54.9)(12.1,48.9)(9.93,46.7)(6.3,44.4)(4.2,40.3)(5.7,50)
- S821 float: (120,70.4)(90.5,69.3)(64.3,67.2)(45.1,64.1)(36,63.6)(26.9,62.1)(18.8,59.9)(12.9,56.2)(12.4,50.4)(9.17,49)(6.7,46)(5.8,41.3)
- S821 8-bit: (84.7,69.2)(71.9,67.9)(44.8,65.8)(29.5,61.8)(30,61.3)(24.2,60.2)(15.3,58.4)(10.2,54.9)(11.5,48.9)(9.3,46.7)(4.9,44.4)(4.2,40.3)(6.42,50)
- Sec. 4.2.1: gap "quite substantial (~10%) for Snapdragon 835 LITTLE cores at the 33ms latency needed for
  real-time (30 fps)". Our reading of the coordinates: best float at <= 33 ms is 49 (30.2 ms), best 8-bit
  is 58.4 (31.5 ms), so ~10 is absolute top-1 points. Sec. 4.2.1 also: S821 is better optimized for float,
  so the quantized latency gain is smaller there.

Table 4.4 (COCO, MobileNet SSD, mAP = AP@[.5:.95], 320x320, single thread, Snapdragon 835):
| DM | type | mAP | LITTLE (ms) | big (ms) |
|---|---|---|---|---|
| 100% | floats | 22.1 | 778 | 370 |
| 100% | 8 bits | 21.7 | 687 | 272 |
| 50% | floats | 16.7 | 270 | 121 |
| 50% | 8 bits | 16.6 | 146 | 61 |
Text (Sec. 4.2.2): "up to a 50% reduction in running time, with a minimal loss in accuracy (-1.8% relative)".
Our reading: -1.8% relative = (21.7 - 22.1)/22.1; 50% = 61/121 (DM 50%, big core). Per-tensor, uint8, QAT
with 500k-step activation delay, ARM CPU; not a GPU result.

Table 4.5 (face detection, precision/recall averaged over IoU 0.5:0.05:0.95):
| DM | type | precision | recall |
|---|---|---|---|
| 100% | floats | 68% | 76% |
| 100% | 8 bits | 66% | 75% |
| 50% | floats | 65% | 70% |
| 50% | 8 bits | 62% | 70% |
| 25% | floats | 56% | 64% |
| 25% | 8 bits | 54% | 63% |

Table 4.6 (face detection latency, ms, Snapdragon 835, 320x320; columns = number of cores):
| DM | type | LITTLE 1 | 2 | 4 | big 1 | 2 | 4 |
|---|---|---|---|---|---|---|---|
| 100% | floats | 711 | -- | -- | 337 | -- | -- |
| 100% | 8 bits | 372 | 238 | 167 | 154 | 100 | 69 |
| 50% | floats | 233 | -- | -- | 106 | -- | -- |
| 50% | 8 bits | 134 | 96 | 74 | 56 | 40 | 30 |
| 25% | floats | 100 | -- | -- | 44 | -- | -- |
| 25% | 8 bits | 67 | 52 | 43 | 28 | 22 | 18 |
Text: "close to a 2x latency reduction ... at the cost of a ~2% drop in the average precision"; 25% model
real-time "1K/28 ~ 36 fps" on one big core vs float "1K/44 ~ 23 fps"; 4 cores give "1.5 to 2.2x" speedup.

Table 4.7 (face attributes, "relative average category precision ... compared with floating point";
rows = weight bits, columns = activation bits). Full table:
| wt\act | 8 | 7 | 6 | 5 | 4 |
|---|---|---|---|---|---|
| 8 | -0.9% | -0.3% | -0.4% | -1.3% | -3.5% |
| 7 | -1.3% | -0.5% | -1.2% | -1.0% | -2.6% |
| 6 | -1.1% | -1.2% | -1.6% | -1.6% | -3.1% |
| 5 | -3.1% | -3.7% | -3.4% | -3.4% | -4.8% |
| 4 | -11.4% | -13.6% | -10.8% | -13.1% | -14.0% |
Table 4.8 (age precision at 5 years, same layout): row 8: -1.3 -1.6 -3.2 -6.0 -9.8; row 7: -1.8 -1.2 -4.6 -7.0
-9.9; row 6: -2.1 -4.9 -2.6 -7.3 -9.6; row 5: -3.1 -6.1 -7.8 -4.4 -10.0; row 4: -10.6 -20.8 -17.9 -19.0 -19.5 (all %).
Whether these are relative percent or absolute points is not explicit beyond the caption word "relative"
(Table 4.7) and "compared with floating point". Paper's conclusions: weights more sensitive than activations;
8 and 7 bit close to float; at equal total bits keep weight and activation bits equal.

Fig. 4.3 (face attributes, S821, latency ms vs average precision) coordinates: float
(6,0.871)(4,0.861)(2,0.836)(17,0.889)(10,0.885)(6,0.869); 8-bit (4,0.872)(2,0.860)(1,0.834)(10,0.888)(6,0.881)(3,0.867).

Sec. 5: "In addition to the 4x reduction of model size".

Shape facts for toys:
- Sec. 3: PTQ failure mode "large differences (more than 100x) in ranges of weights for different output
  channels". Use for viz-chanland spread (per-tensor grid wastes levels on small channels).
- Sec. 4.1.2: ReLU6 gives activations the fixed range [0, 6].
- Fig. 1.1a labels the bias input and conv output "uint32"; the text says int32 (Sec. 2.4). Use int32.

## 9. Limitations the paper states
- Per-tensor scale for weights; the paper itself names the >100x per-channel range spread as a PTQ failure
  mode, and fixes it by training, not by per-channel scales. Per-channel weights came in Krishnamoorthi
  1806.08342 (Sec. 2.6, Tables 2-5).
- Math functions (tanh, logistic, softmax) are fixed-point but "we do not currently fuse them" (Sec. 2.4,
  App. A.1).
- Add layers cost more than in float because of rescaling (App. A.2).
- Fusing/optimizing the inference graph is "outside the scope of this paper" (Sec. 3).
- Speedup depends on hardware: S821 is "better optimized" for float, so less gain (Sec. 4.2.1).
- Ranges are not learned by gradient; PACT (1805.06085) and LSQ (1902.08153) learn them.
- BN folding with batch statistics causes jitter in quantized weights; Krishnamoorthi Sec. 3.2.2 adds a
  correction factor sigma_B/sigma and freezes BN statistics later in training.

## 10. Outline feedback
- ch-grid line 84 writes "quantize q = clamp(round(r/S) + Z)" as Jacob's. Jacob never writes that form.
  His quantizer is Eq. 12 in (a, b, n) form, mapped to (S, Z) by Eq. 13. The r -> q form is
  Krishnamoorthi Eq. 1-2 (`x_int = round(x/Delta) + z`, `x_Q = clamp(0, N_levels - 1, x_int)`). Cite that,
  or label the line "our reading".
- ch-grid Eq. refs are correct: Eq. 5-6 for M and M0 2^-n; Eq. 7-8 for the zero-point cross terms. Add
  Eq. 10 (`int32 += uint8 * uint8`) and Eq. 11 (S_bias) for viz-intmm. Note Jacob's matmul is q1 = weights
  (rows) and q2 = activations; the cross-term sums are a row sum of weights and a column sum of activations.
- ch-grid "Bias in int32 with S_bias = S1 S2 (Jacob Sec. 2.4)": correct (Eq. 11).
- ch-grid "TensorRT uses symmetric (Z = 0) INT8": Jacob is asymmetric uint8 with zero-points; keep the
  contrast explicit.
- ch-qat: Jacob does not define or name the STE ("Backpropagation still happens as usual") and does not learn
  ranges by gradient; ranges are min/max (weights) and EMA (activations). Say this on the page.
- ch-qat "BN folded per Jacob Sec. 3.2": correct (Eq. 14, Figs. C.5-C.8).
- ch-qat "Jacob delay of activation quantization": Sec. 3.1 says "50 thousand to 2 million steps"; every
  protocol (App. D.1, D.3, D.4) uses 500,000.
- ch-guide: Jacob's COCO row is MobileNet SSD on ARM CPU (Snapdragon 835, single thread), per-tensor uint8,
  QAT, 320x320. Say this when quoting it next to GPU/TensorRT rows.
- ch-guide "keep sensitive layers in FP16": Jacob keeps no layer in higher precision (none stated); LSQ keeps
  first/last layers 8-bit (Esser Sec. 2.3) and PACT leaves first/last unquantized (Choi Sec. 4.2).
- Tension to flag: Jacob Table 4.3 favors ReLU6 for quantization; Krishnamoorthi Sec. 5 / Fig. 16
  recommends ReLU over ReLU6.
- The MobileNet ImageNet "latency tables" in the outline are figures (Fig. 1.1c, 4.1, 4.2), not tables;
  the points are exact in the TeX but the DM/resolution of each point is not given.
