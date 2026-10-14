# nagel2019dfq -- Data-Free Quantization Through Weight Equalization and Bias Correction (BRANCH)

## Identity
- arXiv 1906.04721v3 (read). v1 submitted 2019-06-11; v3 last revised 2019-11-25 (abs.html). Comments: ICCV 2019.
- Authors: Markus Nagel (first) ... Max Welling (last). Qualcomm AI Research.

## Key idea (two sentences)
ReLU-like activations satisfy f(sx) = s f(x), so a per-channel scale can be moved from one layer's output channel to
the next layer's input channel without changing the function; choosing that scale to equalize the two layers' channel
ranges makes per-tensor INT8 work for depthwise-separable nets. The remaining mean shift caused by quantization is
then removed analytically from BN statistics (bias correction), so the whole method needs no data and no training.

## Defining equation
Eq. 11 (Sec. 4.1.2, p4), TeX from src/sections/method.tex:
`\vec{s}_i = \frac{1}{\vec{r}_i^{(2)}}\sqrt{\vec{r}_i^{(1)} \vec{r}_i^{(2)}}`
r_i^(1) = range of output channel i of W(1), r_i^(2) = range of input channel i of W(2). Result: for all i,
r_i^(1) = r_i^(2) after rescaling. Derived by maximizing total per-channel precision sum_i p_i^(1) p_i^(2) (Eq. 9),
p_i = r_i / R (Eq. 8). Reparameterization (Eq. 5-7): W^(2) S, S^-1 W^(1), S^-1 b^(1). Pairs are iterated until convergence.
Flow (Fig. 4): cross-layer equalization -> bias absorption -> quantization -> bias correction.

## Key results (all from the PDF tables; INT8 = W8A8)
- Table 1 (p6), MobileNetV2 ImageNet top-1, FP32 / INT8: original 71.72 / 0.12; replace ReLU6 71.70 / 0.11;
  + equalization 71.70 / 69.91; + absorbing bias 71.57 / 70.92; per-channel quantization 71.72 / 70.65 (own
  implementation of Jacob 2018 applied post-training). Identical to Nagel 2021 Table 3.
- Table 3 (p7), DeeplabV3+ (MobileNetV2 backend), Pascal VOC mIoU, FP32 / INT8: original 72.94 / 41.40; DFQ 72.45 / 72.33;
  per-channel 72.94 / 71.44.
- Table 4 (p7), MobileNetV2 SSD-lite, Pascal VOC mAP, FP32 / INT8: original 68.47 / 10.63; DFQ 68.56 / 67.91; per-channel
  68.47 / 67.52.
- Fig. 2 (p3) is the same MobileNetV2 first-depthwise-layer channel-range box plot as Nagel 2021 Fig. 5.

## Outline feedback
- Tables 3 and 4 are segmentation and detection rows the guide can use: per-tensor INT8 PTQ of a MobileNetV2 detector
  (10.63 mAP) and segmenter (41.40 mIoU) fails without equalization; DFQ or per-channel recovers within about 1 point.
- Nagel 2021 calls the segmenter "DeepLabV3 (MobileNetV2 backbone)"; DFQ and AdaRound call the same-FP32 (72.94) model
  "DeeplabV3+". Cite each paper's name as written.
