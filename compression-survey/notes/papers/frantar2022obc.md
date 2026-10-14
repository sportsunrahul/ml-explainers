# frantar2022obc -- Optimal Brain Compression / OBQ (BRANCH)

## Identity
- Title: "Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning"
- arXiv 2208.11580, read v2. v1 2022-08-24; v2 (read) 2023-01-08 (abs.html).
- Authors: Elias Frantar (first) ... Dan Alistarh (last); with Sidak Pal Singh. Venue: NeurIPS 2022 (abs.html Comments).
- Source: `src/neurips.tex`.

## Key idea (two sentences)
The layer-wise problem argmin ||W X - W-hat X||^2 splits into independent rows that share one Hessian H = 2 X X^T,
so the classic Optimal Brain Surgeon (OBS) formulas are exact here; ExactOBS applies them one weight at a time and
updates H^-1 by one Gaussian-elimination step (Lemma 1), at O(d_row * d_col^3) time and Theta(d_col^2) memory. OBQ
reuses the same machinery for quantization by replacing the OBS target 0 with quant(w_p), unifying pruning and
quantization (Sec. 5).

## Defining equation (`eq:obs-quant`, PDF Eq. 7, Sec. 5), verbatim
```
w_p = \text{argmin}_{w_p} \, \frac{(\text{quant}(w_p) - w_p)^2}{[\mathbf{H}^{-1}]_{pp}}, \quad \boldsymbol{\delta_p} = - \frac{w_p - \text{quant}(w_p)}{[\mathbf{H}^{-1}]_{pp}} \cdot \mathbf{H}^{-1}_{:, p}.
```
Gloss: pick the weight whose rounding costs least (scaled by the inverse-Hessian diagonal), round it, and move all
remaining weights by delta_p to compensate. If quant() always returns 0 this is OBS pruning (`eq:obs`, PDF Eq. 3).

## Cost
- "prune e.g. all layers of a medium-sized model such as ResNet50 in a bit more than one hour on a single NVIDIA
  RTX 3090 GPU" (Sec. 4).
- ResNet-50 4-bit quantization runtime (App. Table 6): BitSplit 124m, AdaRound 55m, AdaQuant 17m, BRECQ 53m, OBQ 65m.
- ExactOBS runtimes (App. Table 7): YOLOv5s Quant 7m, Unstr 6m, 4-block 10m, 2:4 3m, Quant 2:4 4m; ResNet50 65m /
  64m / 61m / 31m / 35m; BERT 111m / 103m / 142m / 51m / 56m.
- Calibration: 1024 random training samples for every task; ImageNet adds flip/crop augmentation x10 (Sec. 6).

## Key rows
- Quantization, ImageNet top-1, asymmetric per-channel weight quantization of all layers (Table 4; columns 4 / 3 /
  2 bit; FP RN18 69.76, RN50 76.13):
  - AdaRound: RN18 69.34 / 68.37 / 63.37; RN50 75.84 / 75.14 / 71.58
  - AdaQuant: RN18 68.12 / 59.21 / 00.10; RN50 74.68 / 64.98 / 00.10
  - BRECQ: RN18 69.37 / 68.47 / 64.70; RN50 75.88 / 75.32 / 72.41
  - OBQ: RN18 69.56 / 68.69 / 64.04; RN50 75.72 / 75.24 / 70.71
- Detection (Table 1; unstructured pruning to 2x / 3x / 4x FLOP reduction; YOLOv5l COCO mAP@0.5, dense 66.97):
  GMP 65.83 / 62.30 / 55.09; L-OBS 66.21 / 64.47 / 61.15; AdaPrune 66.00 / 64.88 / 62.71; ExactOBS 66.14 / 65.35 /
  64.05. Same table ResNet50 (76.13): ExactOBS 75.64 / 75.01 / 74.05.
- Joint quantization + 2:4 on YOLOv5m/l (Fig. 3b, Fig. 7) is plotted only (mAP@0.5 vs BOP reduction); text: at
  ~2.5% relative drop, 12-14x BOP reduction for ResNets and 7-8x for YOLO and BERT (Sec. 6). Symmetric per-channel
  weights, asymmetric per-tensor activations in that GPU scenario.

## Outline feedback
- `ch-guide` honesty line: OBC does measure YOLO (v5l/v5m/v5s) on COCO, but for pruning and joint W4A4/W8A8 + 2:4
  curves, not an INT8 PTQ table. Reword the line: "no primary source here gives a YOLO INT8 PTQ delta".
- `viz-tree` edge OBS/OBC -> GPTQ: GPTQ Sec. 3 calls OBQ its base and cites a "more than three orders of magnitude"
  speedup over it.
