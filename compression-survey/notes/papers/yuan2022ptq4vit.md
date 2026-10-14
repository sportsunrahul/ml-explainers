# yuan2022ptq4vit -- PTQ4ViT (branch)

## Identity
- Title: "PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization".
- arXiv 2111.12293, version read: v3 (2024-06-23). v1 submitted 2021-11-24; v2 2022-07-27 (abs.html).
- Authors: Zhihang Yuan ... Guangyu Sun (Yuan, Xue, Chen, Wu, Sun). Peking University, Houmo AI.
- Venue: not stated in abs.html. Our reading: ECCV 2022, from the camera-ready block of the ECCV-22 template in src/main.tex.
- Repo: https://github.com/hahnyuan/PTQ4ViT (Sec 1 footnote).

## Key idea (two sentences)
Post-softmax values pile up near 0 with a few near 1, and post-GELU values are asymmetric (large unbounded positives, a tiny negative range), so one uniform grid fits neither. Twin uniform quantization gives each such tensor two ranges with two scales (constrained to differ by a power of two so a bit-shift aligns them), and the scales are picked with a Hessian-guided metric (output error weighted by squared dL/dO) instead of MSE or cosine distance (Sec 3.2-3.3, p6-8).

## Defining equation (Eq 4, Sec 3.2, p6; TeX verbatim from src/3_method.tex)
```
\text{T}_k(x,\Delta_{\text{R1}},\Delta_{\text{R2}})=\begin{cases}\Psi_{k-1}(x,\Delta_{\text{R1}}), x\in\text{R1} \\ \Psi_{k-1}(x,\Delta_{\text{R2}}), otherwise\end{cases}.
```
Gloss: Psi_{k-1} = symmetric uniform quantizer with k-1 bits (Eq 2: clamp(round(x/Delta), -2^{k-1}, 2^{k-1}-1)); one bit flags the range. Softmax: R1 = [0, 2^{k-1} Delta_R1^s) with a small searched step, R2 = [0, 1] with Delta_R2^s fixed to 1/2^{k-1}. GELU: R1 = negatives with fixed step, R2 = positives with searched step. Constraint Delta_R2 = 2^m Delta_R1 (Eq 5). Hessian-guided metric is Eq 7.

## Key row (Table 1, p11; ImageNet top-1; drops are absolute points)
DeiT-S/224: FP32 79.80 | Base PTQ W8A8 77.65 (2.14), W6A6 72.26 (7.53) | PTQ4ViT W8A8 79.47 (0.32), W6A6 76.28 (3.51).
Base PTQ = layer-wise alternating search of Delta_A, Delta_B with cosine distance (Sec 3.1). Setup: 32 random ImageNet train images for calibration; symmetric uniform, one scale per matrix operand (Sec 3.1), separate scales per attention head and for W^Q, W^K, W^V; all FC layers and both attention matmul inputs quantized; softmax and normalization layers not quantized (Sec 4.1, p10). W8A8 drop < 0.5 points on all 11 ViT/DeiT/Swin models; W6A6 average drop 9.8 (base) vs 2.1 (PTQ4ViT) (Sec 4.2).

## Outline feedback
- ch-outliers says "ViTs and DETR-style detectors have softmax/GELU outliers too (PTQ4ViT)". PTQ4ViT measures ImageNet classification only; its detection table is commented out in the source. It does not study DETR.
- Its problem is distribution shape (post-softmax mass near 0, post-GELU asymmetry), not per-channel outlier ridges like LLMs. Suggested line: "ViT post-softmax and post-GELU activations are far from bell-shaped; PTQ4ViT gives them two scales and keeps DeiT-S W8A8 within 0.32 points of FP32."
