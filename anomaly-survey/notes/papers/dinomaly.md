# Dinomaly (2405.14325v5)

## 1. Identity
- Title: "Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection".
- arXiv 2405.14325. Version read: v5 (2025-04-02). v1: 2024-05-23 (v2 2024-05-29, v3 2024-10-31, v4 2024-11-14).
- Authors: Jia Guo ... Hongen Liao (6 authors; Tsinghua, Beijing Institute of Technology, Shanghai Jiao Tong).
- Venue: CVPR 2025 (abs.html comments; LaTeX footnote "Accepted by CVPR 2025").
- Repo: https://github.com/guojiajeremy/Dinomaly (abstract).
- Sources read: `src/sec/*.tex` (all), Fig. 2 and Fig. 4 PNGs, PDF pages 7-8 for rendered Tables 2 and 4.
- Note: the v5 LaTeX abstract says "(1) Scalable foundation Transformers"; the abs.html abstract says "(1) Foundation
  Transformers". Cite the LaTeX (v5).

## 2. Core problem
Multi-class unsupervised anomaly detection (MUAD): one model trained on the normal images of all categories at once
(15 for MVTec AD, 12 for VisA, 30 for Real-IAD). Multi-class models lagged far behind one-model-per-class models. Goal:
close that gap with a minimal pure-Transformer reconstruction model.

## 3. What prior methods did and why that hurt (per the paper)
- Reconstruction models trained on diverse multi-class normals "generalize" to anomalies too: the "identity mapping" /
  "identical shortcut" (Sec. 1, 2). Dinomaly reframes it as "over-generalization" (Sec. 3.2).
- Fixes so far were bespoke modules: UniAD neighbor-masked attention + feature jitter; HVQ-Trans vector quantization;
  LafitE / DiAD diffusion; OmniAL synthetic anomalies; ReContrast cross-reconstruction; MambaAD state space model
  (Sec. 2). Hand-crafted noise / pseudo anomalies "may not be universal" (Sec. 3.2).
- Layer-to-layer, point-by-point distillation lets the decoder mimic the encoder well even on anomalies (Sec. 3.4).
- Softmax attention and convolutions can both learn an identity (attention map = identity matrix) (Sec. 3.3).
- Prior AD backbones did not follow a scaling law (RD4AD: WRN50 > WRN101; ViTAD: ViT-S > ViT-B) (Sec. 4.4).

## 4. The one key idea
Frozen DINOv2-R ViT encoder -> MLP bottleneck with Dropout -> 8-layer Transformer decoder with linear attention,
trained to reconstruct grouped (summed) middle-layer encoder features with a hard-mining cosine loss. Each part makes
copying harder, so anomalies reconstruct badly even with one model for all classes.

## 5. Mechanism, step by step
The four components, named as in the v5 abstract: (1) Scalable foundation Transformers, (2) Noisy Bottleneck,
(3) Linear Attention ("Unfocused Linear Attention" in the section heading), (4) Loose Reconstruction. The ablations split
(4) into Loose Constraint (LC) and Loose Loss (LL) and test NB, LA, LC, LL. The conclusion names them "Foundation
Transformer, Noisy MLP Bottleneck, Linear Attention, and Loose Reconstruction".

1. Input: resize to 448 x 448, center-crop to 392 x 392 (App. B). Patch 14 -> feature map 28 x 28 (784 patch tokens).
2. Encoder (FROZEN; lock icon in Fig. 2, "frozen backbone Transformer" in App. C): ViT-Base/14 pretrained by DINOv2 with
   registers (DINOv2-R, 4 extra register tokens). 12 layers. Embedding dim not stated (our reading: 768).
3. Use the "middle 8 layers" of the 12 (App. B). Indices for ViT-B are not given in the text. Our reading from Fig. 2:
   2 blocks before the first group, 8 used, the deeper 2 discarded (indices 2..9 from 0). For ViT-L (24 layers) the
   paper gives [4, 6, 8, ..., 18] (index from 0).
4. Noisy Bottleneck (TRAINED): the 8 middle-layer feature maps are added (Fig. 2) and fed to an MLP
   (Linear -> GELU -> Linear) with Dropout before each Linear and at the output "if noisy" (Fig. 2). Dropout rate 0.2
   (0.4 on Real-IAD and Uni-Medical). Dropout acts as "pseudo feature anomaly", like a denoising autoencoder (Sec. 3.2).
5. Decoder (TRAINED): 8 Transformer layers; spatial mixer = Linear Attention, phi(x) = elu(x) + 1 (Sec. 3.3, Eq. 3).
   Linear attention "cannot focus", so each output draws on long-range context instead of copying its own position.
6. Loose Constraint: encoder layers grouped into 2 groups of 4 (low-semantic and high-semantic); each group's features
   are summed; the decoder's 8 layers are grouped and summed the same way; reconstruction is group-to-group (Fig. 4(e)),
   not layer-to-layer.
7. Loose Loss: hard-mining global cosine loss (Eqs. 4-6, borrowed from ReContrast). Per batch, the k% best-reconstructed
   feature points get their gradient shrunk to 0.1x (not fully stopped). k rises linearly 0% -> 90% over the first 1,000
   iterations (500 for class-separated). Total loss = average over the 2 group pairs.
8. Inference: anomaly map = per-point cosine distance between encoder and decoder group features, mean of the 2 groups,
   upsampled. Image score = mean of the top 1% pixels of the anomaly map (App. B). Localization evaluated at 256 x 256
   for all methods (App. C, "Input Size").
9. Gradient: from L_global-hm into the decoder and bottleneck only; the encoder is frozen. One model for all classes.

Delta vs UniAD (the chapter diagram; UniAD facts here are as Dinomaly describes them):
| | UniAD (per Dinomaly) | Dinomaly |
|---|---|---|
| Encoder | frozen pretrained encoder (CNN type not stated by Dinomaly; see ch-uniad notes) | frozen DINOv2-R ViT-B/14 |
| Noise | feature jitter (Gaussian noise on encoder features) | Dropout inside the MLP bottleneck |
| Anti-copy attention | neighbor-masked attention | linear attention (cannot focus); neighbor mask optional |
| Reconstruction target | last decoder layer to concatenated layers (Fig. 4(b) style) | 2 summed groups, group-to-group |
| Loss | point-by-point | global cosine with hard mining (easy points get 0.1x gradient) |
| Training | multi-class, one model | multi-class, one model |

## 6. Equations (LaTeX verbatim; no custom macros needed beyond standard ones)
Eq. (1):
```
\mathbf{Q} = \mathbf{X} \mathbf{W}^Q ~,
\mathbf{K} = \mathbf{X} \mathbf{W}^K ~,
\mathbf{V} = \mathbf{X} \mathbf{W}^V ~,
```
Gloss: X = N tokens of dim d; W^Q, W^K, W^V = learnable d x d projections.

Eq. (2):
```
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})= \text{Softmax}(\mathbf{Q}\mathbf{K}^T) \mathbf{V} ~.
```
Footnote: the sqrt(d) scale and multi-head are omitted "for narrative simplicity". Gloss: each output = weighted average
of values; softmax can put almost all weight on one token, including itself.

Identity examples (unnumbered `align*`): a 3x3 conv kernel with a single centered 1, and an identity attention map.
Paper: "both operations are in danger of forming identity mapping by over-concentrating on corresponding input
locations".

Eq. (3) Linear Attention:
```
\text{LA}(\mathbf{Q}, \mathbf{K}, \mathbf{V})= (\phi(\mathbf{Q}) \phi(\mathbf{K}^T)) \mathbf{V} = \phi(\mathbf{Q})(\phi(\mathbf{K}^T) \mathbf{V}) ~,
```
with "usually phi(x) = elu(x)+1"; complexity O(N d^2) instead of O(N^2 d). Paper: Linear Attention "cannot concentrate
on important regions related to the query ... This property, however, is exactly what the reconstruction decoder
favors". Gloss: no softmax, so no sharp peak; K^T V is a d x d summary of all tokens.

Eq. (4) Loose loss:
```
\mathcal{L}_{global-hm} = \mathcal{D}_{cos}(\mathcal{F}( f_{E} ), \mathcal{F}( \hat{f_{D}} )),
```
Eq. (5) `eq:ghm`:
```
\hat{f_{D}}(h,w) = \left\{ \begin{array}{l}
sg( f_{D}( h,w ) )_{0.1}, \text{ if } \mathcal{D}_{cos}( f_{D}, f_{E} ) < k\%_{batch}\\
f_{D}( h,w ), \text{ else} \\
\end{array} \right.
```
Eq. (6):
```
\mathcal{D}_{cos}(a, b) = 1 - \frac{{a}^{T} \cdot b}{\left\| {a} \right\| ~\left\| {b} \right\|},
```
Paper: hard-mining global cosine loss "detaches the gradients of well-restored feature points with low cosine distance";
sg(.)_0.1 = "shrink the gradient to one-tenth" (footnote: complete stop-gradient "causes optimization instability
occasionally"). Gloss: f_E, f_D = grouped encoder / decoder feature maps; F = flatten the whole map into one vector, so
the cosine is GLOBAL over the map, not per point; (h, w) = one feature point; the k% of points with the smallest
per-point cosine distance within the batch are the "easy" ones and get 0.1x gradient; k = 90% after warm-up (App. B).

## 7. Config / setup table
| Item | Value | Source |
|---|---|---|
| Encoder | ViT-Base/14, DINOv2-R (with registers), frozen | Sec. 3.1, App. B, Fig. 2 |
| Encoder layers used | middle 8 of 12 (ViT-B); [4,6,...,18] of 24 (ViT-L) | App. B |
| Groups | 2 (4 layers each, summed; our reading of Fig. 2 / 4(e)) | Sec. 4.1, Fig. 4 |
| Bottleneck | MLP, Dropout 0.2 (0.4 Real-IAD, Uni-Medical) | App. B, App. D |
| Decoder | 8 Transformer layers, Linear Attention | Sec. 3.1, App. B |
| Input | resize 448^2, center-crop 392^2; feature map 28^2 | Sec. 4.1 |
| Loss | L_global-hm, k% ramps 0 -> 90% over 1,000 iters; 0.1x gradient | Eq. 4-6, App. B |
| Optimizer | StableAdamW + AMSGrad, lr 2e-3, betas (0.9, 0.999), wd 1e-4, eps 1e-10 | App. B |
| Schedule | warm-up 0 -> 2e-3 over 100 iters, cosine anneal to 2e-4 | App. B |
| Iterations | 10,000 (MVTec AD, VisA), 50,000 (Real-IAD); 5,000 per class class-separated | App. B |
| Dinomaly-up | 20,000 / 20,000 / 100,000 iters | Table 1 caption |
| Training batch size | not stated | -- |
| Image score | mean of top 1% anomaly-map pixels | App. B |
| Pixel eval size | 256 x 256 | App. C |
| Seed | 1 (5 seeds in Table A12) | App. B |
| Hardware | NVIDIA GeForce RTX3090 (24GB); PyTorch 1.12.0 | App. B |
| Params / MACs / Im/s (ViT-B) | 148.0M / 104.7G / 58.1 (RTX3090, batch 16) | Table 4 |

## 8. Results worth remembering
Protocol unless noted: multi-class (one model for all classes), 392 px crop, ViT-B, %, seed 1.

Table 1 (multi-class), I-AUROC / P-AUROC / AUPRO:
| Method | MVTec AD | VisA | Real-IAD |
|---|---|---|---|
| RD4AD | 94.6 / 96.1 / 91.1 | 92.4 / 98.1 / 91.8 | 82.4 / 97.3 / 89.6 |
| SimpleNet | 95.3 / 96.9 / 86.5 | 87.2 / 96.8 / 81.4 | 57.2 / 75.7 / 39.0 |
| DeSTSeg | 89.2 / 93.1 / 64.8 | 88.9 / 96.1 / 67.4 | 82.3 / 94.6 / 40.6 |
| UniAD | 96.5 / 96.8 / 90.7 | 88.8 / 98.3 / 85.5 | 83.0 / 97.3 / 86.7 |
| ReContrast | 98.3 / 97.1 / 93.2 | 95.5 / 98.5 / 91.9 | 86.4 / 97.8 / 91.8 |
| DiAD | 97.2 / 96.8 / 90.7 | 86.8 / 96.0 / 75.2 | 75.6 / 88.0 / 58.1 |
| ViTAD | 98.3 / 97.7 / 91.4 | 90.5 / 98.2 / 85.1 | 82.7 / 97.2 / 84.8 |
| MambaAD | 98.6 / 97.7 / 93.1 | 94.3 / 98.5 / 91.0 | 86.3 / 98.5 / 90.5 |
| Dinomaly | 99.6 / 98.4 / 94.8 | 98.7 / 98.7 / 94.5 | 89.3 / 98.8 / 93.9 |
| Dinomaly-up (longer training) | 99.7 / 98.4 / 94.7 | 98.9 / 98.8 / 94.5 | 89.5 / 98.9 / 94.2 |
Baseline provenance (App. B, line 94): "Most results of compared MUAD SoTAs are directly drawn from a benchmark paper
ADer" (2406.03262). Per-class baseline tables (A14-A19) are "drawn from MambaAD" (2404.06564). Copied, not re-run.
Baselines run at their own default input sizes; the paper argues "optimum vs. optimum" (Table 5: RD4AD and ReContrast
degrade at larger inputs; RD4AD/ReContrast default 256^2).

Table 2 (class-separated vs multi-class), I-AUROC / P-AUROC / P-AUPRO:
| Row | MVTec AD | VisA | Real-IAD |
|---|---|---|---|
| Dinomaly (MUAD) | 99.6 / 98.4 / 94.8 | 98.7 / 98.7 / 94.5 | 89.3 / 98.8 / 93.9 |
| Dinomaly (class-separated) | 99.7 / 99.9 / 95.0 | 98.9 / 98.9 / 95.1 | 92.0 / 99.1 / 95.1 |
| RD4AD | 98.5 / 97.8 / 93.9 | 96.0 / 90.1 / 70.9 | 87.1 / n/a / 93.8 |
| PatchCore | 99.1 / 98.1 / 93.5 | 94.7 / 98.5 / 91.8 | 89.4 / n/a / 91.5 |
| SimpleNet | 99.6 / 98.1 / 90.0 | 97.1 / 98.2 / 90.7 | 88.5 / n/a / 84.6 |
The class-separated MVTec AD P-AUROC "99.9" is printed in both LaTeX and PDF. Our reading: likely a typo (next to
98.4 multi-class and AUPRO 95.0); copy as printed, do not use without a caveat. The source of the RD4AD / PatchCore /
SimpleNet class-separated rows is not stated. Text: on MVTec AD and VisA multi-class Dinomaly has "nearly no
performance drop"; on Real-IAD a "moderate performance drop" (89.3 vs 92.0 I-AUROC).

Ablation (Table 3 MVTec AD; Table A4 VisA), I-AUROC / AUPRO:
- Baseline (no NB, Softmax Attention, dense layer-to-layer, global cosine loss; "very similar to ViTAD"): MVTec 98.41 /
  92.95; VisA 95.81 / 93.43.
- +NB only: 99.06 / 93.71 (MVTec); 97.38 (VisA). +LA only 98.54; +LC only 98.35; +LL only 99.03 (MVTec).
- All four: MVTec 99.60 / 94.79; VisA 98.73 / 94.50.
- "LA and LC boost the performance with the presence of NB"; LC alone "makes the reconstruction too easy".

Component swaps (MVTec AD, I-AUROC / AUPRO):
- Noise (Table A8): no noise 99.19 / 93.33; feature jitter scale 20 99.59 / 94.40; Dropout 0.2 99.60 / 94.79.
- Spatial mixer (Table A7): ConvBlock 3x3 99.45 / 94.17; Softmax Attention 99.52 / 94.17; Softmax + neighbor mask
  n=3 99.56 / 94.50; Linear Attention 99.60 / 94.79; Linear + neighbor mask n=3 99.60 / 94.70.
- Constraint (Table A6): layer-to-layer dense 99.39 / 94.28; layer-to-cat-layer 99.48 / 93.16; 1 group 99.64 / 93.96;
  2 groups 99.60 / 94.79.
- Dropout rate (Table A5): 0 -> 99.19; 0.1 -> 99.54; 0.2 -> 99.60; 0.3 -> 99.65; 0.4 -> 99.64; 0.5 -> 99.56.
- Backbones at 392/448 crop (Table A1, ViT-B): DINOv2-R 99.60; DINOv2 99.55; iBOT 99.22; DINO 98.97; D-iGPT 98.75;
  MOCOv3 98.47; DeiT (supervised) 98.19; MAE 96.27.
- NB + LL bolted onto RD4AD (Table A9): 97.8 -> 98.5 I-AUROC.
- Seeds (Table A12): I-AUROC 99.60 (std 0.03), AUPRO 94.65 (std 0.09) over 5 seeds.

Scaling and cost:
- Table 4 (MVTec AD; Im/s on RTX3090, batch size 16): ViT-Small 37.4M params, 26.3G MACs, 153.6 Im/s, I-AUROC 99.26,
  AUPRO 94.36; ViT-Base 148.0M, 104.7G, 58.1, 99.60, 94.79; ViT-Large 275.3M, 413.5G, 24.2, 99.77, 95.09.
- Table A3 ViT-Large: VisA 98.85, Real-IAD 90.07 I-AUROC. Intro rounds the best to 99.8 / 98.9 / 90.1.
- Table A11 (Params / MACs / MVTec I-AUROC / VisA I-AUROC): DiAD 1331M / 451.5G / 97.2 / 86.8; ReContrast 154.2M /
  67.4G / 98.3 / 95.5; RD4AD 126.7M / 32.1G / 94.6 / 92.4; ViTAD 39.0M / 9.7G / 98.3 / 90.5; Dinomaly-Base-392 148M /
  104.7G / 99.6 / 98.7; Dinomaly-Base-280 148M / 53.7G / 99.6 / 97.8; Dinomaly-Small-392 37.4M / 26.2G / 99.3 / 97.9;
  Dinomaly-Small-280 37.4M / 14.5G / 99.3 / 96.5. Note: Small-392 MACs print 26.2G here but 26.3G in Table 4.
- Units are MACs, not FLOPs. Params include the frozen encoder (our reading: 148.0M is far above a bare ViT-B).
- Input size (Table 5 / A2): Dinomaly I-AUROC 99.6 at 280^2, 336^2, 392^2; AUPRO 93.6 -> 94.2 -> 94.8.

Other datasets (Table A13, multi-class I-AUROC): MPDD 97.2, BTAD 95.4, Uni-Medical 84.9.
Dataset counts (Sec. 4.1): MVTec AD 3,629 train / 1,725 test (467 normal, 1,258 anomalous); VisA 8,659 / 2,162 (962,
1,200); Real-IAD 30 objects, 36,465 / 114,585 (63,256, 51,329), 5 camera views each.

## 9. Limitations stated (App. F)
- ViTs are expensive; low-compute, high-speed settings are a barrier. Suggested: distillation, pruning, FlashAttention.
- Sensory AD only; not suitable for semantic AD.
- Zero-shot, few-shot, and noisy-training-set UAD not covered.
- Center crop (from PatchCore) "can also cause unreachable anomalies at the edge of images" (App. C).
- Later fixes: not stated in these sources.

## 10. Outline feedback
- Name the backbone exactly: DINOv2 with registers (DINOv2-R) ViT-B/14, frozen. List all four parts: Scalable foundation
  Transformers, Noisy Bottleneck (Dropout), Linear Attention, Loose Reconstruction (= Loose Constraint + Loose Loss).
  The outline lists only "a noisy bottleneck, loose reconstruction".
- Dinomaly's multi-class baseline rows are ADer's numbers. Reuse the same ADer values in the ch-dinov2 second tab as in
  ch-uniad; do not re-source them. Say that Dinomaly runs at 392 px while baselines run at their own defaults.
- "closes the gap to one-model-per-class" is supported on MVTec AD (99.6 vs 99.7) and VisA (98.7 vs 98.9), not fully on
  Real-IAD (89.3 vs 92.0). Table 2 MVTec P-AUROC 99.9 looks like a typo; avoid it on the page.
- Dinomaly's own ablation shows UniAD's neighbor mask and feature jitter roughly match Linear Attention and Dropout at
  their best settings (Tables A7, A8). Our reading: much of the gain over UniAD comes from the backbone and framework
  (the no-component baseline already scores 98.41 vs UniAD 96.5 on MVTec AD), then NB and LL. Good
  "less is more" talking point.
- Inference cost for ch-speed: 58.1 Im/s (ViT-B, RTX3090, batch 16), 153.6 Im/s (ViT-S).
