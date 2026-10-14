# li2021brecq -- BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction (BRANCH)

## Identity
- arXiv 2102.05426v2 (read). v1 submitted 2021-02-10; v2 last revised 2021-07-25 (abs.html). ICLR 2021 (PDF header).
- Authors: Yuhang Li (first) ... Shi Gu (last). Code: github.com/yhhhli/BRECQ (abstract).
- Macros: `\E` = mathbb{E} (math_commands.tex); `\transpose` = mathsf{T}; `\FracPartial{a}{b}` = da/db; `\mypm` = gray +-.

## Key idea (two sentences)
AdaRound reconstructs one layer at a time and ignores dependencies between layers, while reconstructing the whole
network overfits the small calibration set; BRECQ reconstructs one building block (for example a residual bottleneck)
at a time, which keeps the intra-block cross-layer dependency of the Hessian at low generalization error. The block
output error is weighted by a diagonal Fisher approximation of the pre-activation Hessian (squared output gradients),
with AdaRound for weights and LSQ step sizes for activations.

## Defining equation
Eq. 10 (Sec. 3.3, p5), TeX from src/iclr2021_conference.tex:
`\min_{\hat{\mathbf{w}}} \E\left[\Delta\mathbf{z}^{(\ell),\transpose}\mathbf{H}^{(\mathbf{z}^{(\ell)})}\Delta\mathbf{z}^{(\ell)}\right] =  \min_{\hat{\mathbf{w}}} \E\left[ \Delta\mathbf{z}^{(\ell),\transpose} \mathrm{diag}\left((\FracPartial{L}{\mathbf{z}^{(\ell)}_1})^2, \dots, (\FracPartial{L}{\mathbf{z}^{(\ell)}_a})^2\right) \Delta\mathbf{z}^{(\ell)}\right].`
Delta z^(l) = change of the block output z^(l) caused by quantizing the block's weights.

## Key results
- Table 5 (p8), COCO box mAP, activations 8-bit, 256 COCO calibration images, block reconstruction only in the backbone:
  Faster R-CNN ResNet-50: FP 38.55; bias correction W8A8 38.25, W4A8 0.25; AdaRound W4A8 37.58, W2A8 19.63;
  BRECQ W8A8 38.54, W4A8 38.29, W2A8 34.23. RetinaNet ResNet-50: FP 36.82; BRECQ W4A8 36.65, W2A8 34.75; ZeroQ 4-bit
  mixed precision 33.7. MobileNetV2 backbones: Faster R-CNN FP 33.44 -> BRECQ W4A8 33.18; RetinaNet FP 32.63 -> 32.31.
  (All 6 rows in the claims file.) Bias correction at W4A8 collapses for ResNet backbones (0.04-0.84 mAP).
- Table 3 (p7), ImageNet, activations 4-bit: ResNet-18 FP 71.08, BRECQ W4A4 69.60, W2A4 64.80; MobileNetV2 FP 72.49,
  W4A4 66.57, W2A4 53.34.
- Table 4 (p7): ResNet-18 W4A4 BRECQ 0.4 GPU hours with 1024 images vs LSQ (QAT) 71.1 with 1.2 M images and 100 GPU hours.
- Setup: first and last layer kept 8-bit (Sec. 4.2); per block 2x10^4 iterations, batch 32, Adam lr 1e-3 (App. B.4.4).

## Outline feedback
- Table 5 is the only source in this group with W4A8 and W2A8 PTQ rows for Faster R-CNN and RetinaNet; add it to the
  ch-guide evidence table and as the "what came after AdaRound" line in ch-ptq.
