# vanbaalen2023fp8 -- FP8 versus INT8 for efficient deep learning inference (BRANCH, analysis lens)

## 1. Identity
- arXiv 2303.17951, read v2 (LaTeX `src/` + `paper.pdf`). Dates (abs.html): v1 2023-03-31, v2 2023-06-15.
- Authors: Mart van Baalen ... Tijmen Blankevoort (11 authors, Qualcomm AI Research). Whitepaper; no venue stated.
- Code: https://github.com/Qualcomm-AI-research/FP8-quantization (Sec 4, intro).

## 2. One key idea (two sentences)
With a flexible per-tensor/per-channel scale, INT8 and FP8 differ only in where the grid points sit: more exponent bits put more points near zero and fewer far away, so FP8 wins only when a tensor has significant outliers ("It's all about the outliers", Sec 4.1). Since FP8-E4 MACs cost 53% (FP16 accumulator) to 183% (FP32 accumulator) more gates than INT8, and QAT trains outliers away, the paper concludes FP8 does not warrant dedicated inference hardware over INT8.

## 3. Defining equations (Sec 2, verbatim TeX)
```
z &= s \cdot \hat{z} + b \\
\hat{z} &= z_1 \ldots z_8
```
```
z &= \left(1+\frac{\hat{m}}{2^M}\right) \cdot 2^{\hat{e}-B}\\
\hat{m} &= m_1 \ldots m_M \\
\hat{e} &= e_1 \ldots e_E
```
s = scale, b = optional bias (offset), z_i = bits; m-hat = mantissa, e-hat = exponent, B = exponent bias, M/E = mantissa/exponent bit counts. Subnormals (footnote): for e-hat = 0, z = (m-hat / 2^M) * 2^(1-B).
Their format (Appendix A) is NOT the NVIDIA/Arm/Intel E4M3: only `10000000` is reserved (one NaN-like code), no INF, clip to max on overflow, "255 unique FP8 values". The FP8-E0 and FP8-E1 variants equal INT8 (sign-magnitude). Notation: FP8-EX = X exponent bits; FP8-E4 ~ E4M3, FP8-E5 ~ E5M2.

## 4. Claim card
- Claim (abstract): "We also provide a hardware analysis showing that the FP formats are somewhere between 50-180% less efficient in terms of compute in dedicated hardware than the INT format. Based on our research and a read of the research field, we conclude that although the proposed FP8 format could be good for training, the results for inference do not warrant a dedicated implementation of FP8 in favor of INT8 for efficient inference."
- Claim (Sec 4.1): "If a distribution has very significant outliers, the FP8-E4/FP8-E5 format is more accurate, but if the distributions are well-behaved and more Gaussian-shaped, then INT8 or FP8-E2/FP8-E3 is expected to perform better."
- Claim (Sec 8): "Depending on the accumulator size, the FP8 MAC units are 50% to 180% less efficient than their INT8 counterparts."
- Measurement, hardware: count of 2-input gate equivalents (NAND2/NOR2) per multiply-accumulate unit, assumed proportional to area and power (Sec 3.3, Appendix B). INT8 with a 15+12 = 27-bit fixed-point (Kulisch) accumulator vs FP8-EX with fixed-point (FX+12; INT37 for E4), FP16, or FP32 accumulators. Floating-point accumulator modelled with the smallest gate count ("most optimistic comparison in favor of floating-point"). Not a synthesis; corroborated by the cited microexponents synthesis ("40% decrease in performance of FP8 versus INT8"). Result, Fig 3: E4 -> 16-bit acc "53% more gates", E4 -> 32-bit acc "183% more gates".
- Measurement, accuracy: (a) Fig 5 "bits of accuracy" (inverted normalized RMSE) on uniform, normal, student-t; (b) naive PTQ (Table 1): best over a grid of per-channel vs per-tensor, range setting (MSE / minmax for weights; MSE / allminmax / running minmax for activations), with/without BN re-estimation; no CLE, bias correction or AdaRound for either format; all layers quantized; (c) QAT (Table 2): fine-tune pretrained FP32 models, weights and activations quantized, LSQ range learning, identical setup for INT8 and FP8; ImageNet: 20 epochs, batch 32, Adam (model) + SGD without momentum (ranges), LR searched 1e-2..1e-6, cosine decay; DeepLabV3: 200 epochs, batch 8, SGD.
- Held fixed: same scale freedom for both formats (FP32-like scale, per-channel for weights), whole network in 8 bits (unlike Micikevicius, where only GEMM inputs are FP8 and other tensors are FP16).
- Falsifier: (1) a well-behaved (near-Gaussian) CNN where FP8-E4 beats INT8 after QAT with equal range learning; (2) a synthesized FP8-E4 MAC at or below INT8 area/energy with the same accumulator; (3) an outlier-heavy model where INT8 with W8A16 on the outlier layers or QAT cannot match FP8-E4.
- Scope caveat for our page: hardware argument is for dedicated edge accelerators, not NVIDIA GPUs whose FP8 and INT8 tensor-core throughput is fixed by the vendor. Our reading: on a GPU the choice is driven by which formats the GPU and TensorRT support (see tensorrt.md 7a).

## 5. Key rows (vision first; deltas are absolute points vs FP32, as printed)
Table 1, naive PTQ (p.8), FP32 then INT8 / E2 / E3 / E4 / E5:
| Model (metric) | FP32 | INT8 | FP8-E2 | FP8-E3 | FP8-E4 | FP8-E5 |
|---|---|---|---|---|---|---|
| ResNet18 (top-1) | 69.72% | -0.08% | -0.06% | -0.27% | -1.15% | -4.8% |
| ResNet50 | 76.06% | -0.07% | -0.05% | -0.8% | -0.99% | -3.82% |
| EfficientNet B0 | 77.35% | -14.65% | -4.78% | -2.8% | -3.7% | -9.18% |
| EfficientFormer | 80.21% | -50.35% | -4.95% | -0.41% | -1.43% | -7.41 % |
| DLA102 | 77.94% | -0.19% | -0.31% | -0.54% | -2.13% | -7.95% |
| MobileNetV2 | 71.70% | -0.76% | -0.64% | -1.08% | -5.65% | -22.19% |
| MobileNetV3 | 73.84% | -3.71% | -1.48% | -1.62% | -3.33% | -12.06% |
| ViT | 77.75% | -1.33% | -0.45% | -0.04% | -0.19% | -76.69% |
| YoloV5 (COCO mAP) | 56.3 | -1.8 | -0.5 | -0.3 | -2 | -9.9 |
| HRNet (Cityscapes mIoU) | 81.05 | -0.12 | -0.02 | -0.01 | -0.28 | -1.06 |
| CP-Pointpillar (nuScenes mAP) | 40.94 | -21.41 | -14.81 | -2.86 | -2.93 | -7.06 |
| RangeNet++ | 0.305 | -1.67 | -0.9 | -0.4 | -1.3 | -3.8 |
| FFNet (Cityscapes) | 79.16 | -1.15 | -0.26 | -0.31 | -0.41 | -1.07 |
| DeeplabV3 (MNv2, Pascal VOC) | 72.91 | -1.67 | -0.33 | -1.63 | -34.98 | -66.38 |
| SalsaNext | 55.80 | -1.58 | -0.28 | -0.13 | -0.68 | -2.72 |
| BERT (GLUE) | 83.06 | -12.03 | -2.75 | -0.45 | -0.26 | -0.25 |
| QuickSRNet | 32.79 | -0.8 | -0.44 | -1.78 | -3.24 | -6.8 |
Relative vs absolute: the "%" on classification deltas is part of how they are printed; our reading is absolute points (e.g. ViT E5 77.75 - 76.69 = 1.06 is a collapse). The RangeNet++ row (FP32 0.305, deltas -1.67 ...) has inconsistent units as printed; do not use it.

Table 2, QAT (p.9): FP32 / INT8 / E2 / E3 / E4 / W4A8
- ResNet18: 69.72 / 70.43 / 70.25 / 70.20 / 69.35 / 70.01
- MobileNetV2: 71.70 / 71.82 / 71.76 / 71.56 / 70.89 / 71.17
- HRNet: 81.05 / 81.27 / 81.20 / 81.14 / 81.06 / -
- DeeplabV3: 72.91 / 73.99 / 73.67 / 73.74 / 73.22 / 73.01
- SalsaNext: 55.80 / 55.0 / 55.3 / 55.7 / 55.2 / -
- BERT (GLUE avg): 83.06 / 83.26 / 81.20 / 83.74 / 83.91 / 82.64
"FP8-E4 is now never best" (caption). INT8 best for all 2D vision rows.

Table 3 (p.11) and Table 4 (p.15), FP8-E4 QAT then naive INT8 conversion; FP32 / INT8 PTQ / FP8 PTQ / FP8 QAT / -> INT8 / -> W8A16:
- Table 3 ResNet18: 69.7 / 69.55 / 68.49 / 69.58 / 69.84 / 69.68; MobileNetV2: 71.7 / 70.94 / 64.22 / 70.81 / 70.82 / 71.07.
- Table 4 ResNet18: 69.7 / 69.55 / 68.49 / 69.79 / 69.84 / 70.00; MobileNetV2: 71.7 / 70.94 / 64.22 / 71.59 / 71.59 / 71.54; BERT: 83.06 / 71.03 / 83.08 / 83.91 / 79.38 / 83.54; SalsaNext: 55.8 / 52.8 / 55.1 / 55.2 / 54.1 / 54.2.
- Tables 3 and 4 disagree for the same model on FP8 QAT and conversion columns (different experiments: Table 4 from-scratch FP8 QAT per Sec 6/App D). Table 2 also differs from Table 3 for FP8-E4 QAT: ResNet18 69.35 vs 69.58, MobileNetV2 70.89 vs 70.81. There is no single "FP8 QAT" number per model; cite each by table and do not reconcile.

Appendix E, Table 6 (AIMET INT8, FP32 -> INT8, method): Mnv2-ssd-lite 0.687 -> 0.686 QAT; Centernet (Res50) 0.218 -> 0.215 PTQ,INT8+W8A16; Yolov3 0.653 -> 0.649 PTQ,INT8+W8A16; Yolo-v5 0.557 -> 0.559 PTQ,INT8+W8A16; YoloR 0.700 -> 0.699 PTQ; MobileDet-EdgeTPU 0.281 -> 0.279 PTQ; Deeplabv3+ 78.600 -> 78.700 PTQ; Deeplabv3-Xception 79.700 -> 79.000 PTQ; FFNet 81.30 -> 80.70; HRNet-W48 81.04 -> 80.65; PSPNet101 81.84 -> 81.28; MaskRCNN 22.3 -> 22.1 (listed under segmentation). Table 7 (W4A8): DeeplabV3 (MNV2) 0.729 -> 0.721; DeepLabv3 (RN-50) 0.7607 -> 0.7591.

Toy-shape facts: FP8-E5 gives "around 4 or 5 bits of information" on Gaussian layers; on student-t, E4 is best with "only 5 bits of accuracy"; uniform + INT8 reaches 8 bits (Sec 4.1, Fig 5). INT8 "can exactly represent roughly 90% of the range covered by the FP8-E4 format" (Sec 6, Fig 9). Fig 5 distributions are uniform, normal, student-t (not Laplace).

## 6. Outline feedback
- ch-guide "Honesty: no primary source here measures YOLO-family INT8 deltas": wrong. Table 1 has YoloV5 COCO mAP 56.3 with INT8 naive PTQ -1.8 (E4 -2, E3 -0.3); Table 6 has Yolov3 0.653 -> 0.649 and Yolo-v5 0.557 -> 0.559 (AIMET PTQ with some W8A16 layers). Caveat: simulated quantization on Qualcomm tooling, not TensorRT; YOLOv5 variant/size and input resolution not stated.
- ch-fp8 viz-fp8 toy distributions: the outline uses Gaussian / Laplace / Gaussian + outliers. To reproduce a sourced ordering, match Fig 5 (uniform, normal, student-t): INT8 best on uniform, E2 then INT8 on normal, E4 best on student-t. If Laplace stays, label its result "our computation".
- ch-fp8: state that van Baalen's FP8-E4 is a variant (one reserved code, 255 values, flexible scale per channel), so its MSE numbers are not for the exact E4M3 code set.
- ch-guide: DeeplabV3 here is MobileNetV2 backbone on Pascal VOC (Appendix C.1.4); Wu and Nagel DeepLabV3 rows are other setups.
- The hardware claim has three textual forms (53%/183%, "50-180%", "50% to 180%"); the page must use one form verbatim with its loc.
