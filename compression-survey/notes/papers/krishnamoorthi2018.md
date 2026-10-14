# krishnamoorthi2018 (BRANCH) -- Quantizing deep convolutional networks for efficient inference: A whitepaper

Sources read: `sources/krishnamoorthi2018/src/main.tex`, `paper.pdf` (table and equation numbers from the PDF), `abs.html`.

## Identity
- arXiv 1806.08342v1 (only version), submitted 2018-06-21. Author: Raghuraman Krishnamoorthi (single author,
  Google). Venue: none stated (comment "37 pages"). Tools in TensorFlow / TFLite (Sec. 3).

## Key idea (two sentences)
Use per-channel (per output kernel) scales for weights and per-layer scales for activations: 8-bit PTQ then
lands within 2% of float on many CNNs, while per-layer weight scales collapse MobileNets because BN folding
spreads kernel ranges. QAT (fake quant with STE, BN correction and freezing) closes the remaining gap and
makes 4-bit weights usable.

## Defining equations (verbatim TeX)
Sec. 2.1, Eq. 1-2 (PDF-verified numbers; the r -> q form the outline wants):
```
x_{int}&= round\big(\frac{x}{\Delta}\big)+z \\
x_Q&=clamp(0,N_{levels} - 1, x_{int})
```
Eq. 3 (`eq:dequant`): `x_{float}=(x_Q  - z)\Delta`.
Sec. 2.4, Eq. 12-13 (SimQuant; note the sign of z differs from Eq. 1, as printed; do not reconcile):
```
x_{out}&=SimQuant(x) \\
       &=\Delta\: clamp\big(0,N_{levels}-1,round(\frac{x}{\Delta})-z\big)
```
Eq. 14 (`eq:ste`, the function used only to define the derivative): `x_{out}=clamp(x_{min},x_{max},x)`.
Eq. 15 (STE, "see [5]" = Courbariaux et al. 2015): `\delta_{out}=\delta_{in}I_{x \in S}  S:{x: x_{min} \leq x \leq x_{max}}`
-- gradient passes unchanged inside [x_min, x_max], zero outside.
Sec. 2.6: per-channel = "a different scale and offset for each convolutional kernel"; per-channel is not
used for activations "as this would complicate the inner product computations".

## Key table rows (ImageNet top-1, as fractions exactly as printed; do not convert)
Table 3 (PTQ, 8-bit weights and activations; activations per-layer):
| network | asym per-layer | sym per-channel | asym per-channel | activation only | float |
|---|---|---|---|---|---|
| Mobilenet-v1_1_224 | 0.001 | 0.591 | 0.703 | 0.708 | 0.709 |
| Mobilenet-v2_1_224 | 0.001 | 0.698 | 0.697 | 0.7 | 0.719 |
| Resnet-v1_50 | 0.75 | 0.751 | 0.751 | 0.751 | 0.752 |
| Inception-v3 | 0.78 | 0.78 | 0.78 | 0.78 | 0.78 |
Table 4 (8-bit QAT, same networks): Mobilenet-v1_1_224 asym per-layer QAT 0.70, sym per-channel QAT 0.707
(float 0.709); Mobilenet-v2_1_224 0.709 / 0.711 (float 0.719).
Table 5 (4-bit weights, 8-bit activations): Mobilenet_v1_1_224 asym per-layer PTQ 0.02, sym per-channel PTQ
0.001, sym per-channel QAT 0.65, float 0.709; Resnet_v1_50 0.002 / 0.54 / 0.732 / 0.752.
Table 7 (Pixel 2, single large core, ms): Mobilenet_v1_1_224 float 155, fixed 68, HVX NN-API 16;
Mobilenet_v1_1_224_SSD float 312, fixed 152.
Sec. 3.1.3: "Almost all the accuracy loss due to quantization is due to weight quantization"; per-layer
weight loss is "primarily due to batch normalization, which causes extreme variation in dynamic range across
convolution kernels in a single layer". App. A, Fig. 20: folded depthwise weights have "much larger outliers"
(no number). Fig. 18: Conv2d_1_depthwise of Mobilenet_v1_0.25_128 has 8 kernels.

QAT setup (Sec. 3.2.3): fine-tune from float checkpoint, SGD step size 1e-5. BN fix (Sec. 3.2.2): scale weights
by c = sigma_B / sigma before quantization; freeze BN statistics after "about 300000 steps in figure 15 and 200000 in figure 14" (Sec. 3.2.2; Fig. 14 = Mobilenet v1, Fig. 15 = Mobilenet v2; `freeze_bn_delay`).

## Recommendations (Sec. 8 conclusions; Sec. 7 accelerator list)
- Start with symmetric per-channel weight PTQ; fine-tune only if accuracy drops.
- QAT keeps 4-bit-weight nets within 5% of 8-bit.
- 8-bit gives 2x-3x CPU speedup, close to 10x on Qualcomm DSP with HVX; 4x model size cut.
- Match training forward pass to quantized inference; special BN handling; stochastic quantization
  underperforms deterministic; weight EMA may underperform during QAT.
- Larger models tolerate quantization better; trade width for bits.
- Use ReLU, not ReLU6 (Sec. 5, Fig. 16). This conflicts with Jacob Table 4.3.
- Accelerators: aggressive operator fusion, compressed memory access, 4/8/16-bit arithmetic, per-layer bit
  selection, per-channel weight quantization.

## Outline feedback
- Use this paper, not Jacob, as the source for the r -> q formula in ch-grid (Eq. 1-2) and for the
  clipped STE in ch-qat (Eq. 14-15).
- ch-granularity could quote Table 3 (Mobilenet-v1 per-layer 0.001 vs per-channel 0.703) as the cleanest
  per-tensor vs per-channel PTQ contrast; it is ImageNet, not detection.
- Flag the ReLU vs ReLU6 disagreement with Jacob.
