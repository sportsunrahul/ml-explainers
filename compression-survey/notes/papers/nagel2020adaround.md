# nagel2020adaround -- Up or Down? Adaptive Rounding for Post-Training Quantization (BRANCH)

## Identity
- arXiv 2004.10568v2 (read). v1 submitted 2020-04-22; v2 last revised 2020-06-30 (abs.html). Comments: ICML 2020.
- Authors: Markus Nagel (first) ... Tijmen Blankevoort (last). Qualcomm AI Research.

## Key idea (two sentences)
Rounding each weight to the nearest grid point is not optimal for the task loss; a second-order Taylor expansion
turns the choice of rounding up or down into a per-layer quadratic problem, which under further assumptions becomes
"reconstruct this layer's output". AdaRound relaxes the binary choice with a rectified sigmoid h(V) plus an annealed
regularizer that pushes h to 0 or 1, and solves it layer by layer with SGD on a few unlabeled images.

## Defining equation
Eq. 25 (Sec. 3, p5), final objective, TeX from src/sections/03_method.tex:
`\argmin_{\mat{V}} \norm*{ \func{f_{a}}{\mat{W} \vec{x}} - \func{f_{a}}{\matsq{W} \vec{\hat{x}}}}^2_F + \lambda \func{f_{reg}}{\mat{V}},`
with soft weights (Eq. 22) `\matsq{W} &= \ct{s} \cdot clip \left(\floor*{\frac{\mat{W}}{\ct{s}}} + \func{h}{\mat{V}}, \ct{n}, \ct{p} \right).`,
rectified sigmoid (Eq. 23) h(V) = clip(sigmoid(V)(zeta - gamma) + gamma, 0, 1) with zeta = 1.1, gamma = -0.1, and
regularizer (Eq. 24) f_reg(V) = sum_{i,j} 1 - |2h(V_ij) - 1|^beta, beta annealed. x-hat = input with all preceding
layers quantized; f_a = activation function. Same equation as Nagel 2021 Eq. 35.

## Key results
- Table 1 (p3), ResNet18 with only the first layer at 4 bits, ImageNet top-1: nearest 52.29; ceil 0.10; floor 0.10;
  stochastic 52.06 (std 5.52 over 100 samples); best stochastic 63.06. 48 of 100 stochastic roundings beat nearest (Sec. 2).
- Table 7 (p8), ImageNet top-1, per-layer symmetric weight scale set by MSE: ResNet18 FP32 69.68, nearest W4/A32 23.99,
  AdaRound W4/A32 68.71 (std 0.06), AdaRound W4/A8 68.55; MobilenetV2 FP32 71.72, nearest W4/A32 8.09, AdaRound W4/A32
  69.78, W4/A8 69.25 (MobileNetV2 with CLE preprocessing).
- Table 9 (p9), DeeplabV3+ (MobileNetV2 backend) Pascal VOC mIoU: FP32 72.94; DFQ W8/A8 72.33; nearest W4/A8 6.09;
  DFQ W4/A8 14.45; AdaRound W4/A32 70.89; AdaRound W4/A8 70.86 (512 images for this model).
- Setup (Sec. 5, p6): 1024 unlabeled ImageNet training images, Adam default hyper-parameters, 10k iterations, batch 32;
  ResNet18 takes 10 minutes on a single Nvidia GTX 1080 Ti. With 256 images it stays within 2% of FP32 (Sec. 5, Fig. 4).

## Outline feedback
- The stochastic-rounding numbers (52.29 nearest, 63.06 best) belong to AdaRound Table 1; Nagel 2021 cites them only in
  prose (">10%") and Fig. 7. The adaround_intro table in the Nagel source is not in the compiled white paper.
- Table 9 gives a W4A8 segmentation row (70.86 vs 72.94) usable in the guide's evidence table.
