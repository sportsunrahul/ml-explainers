# MambaOut (branch, analysis lens)

## 1. Identity
- Title: "MambaOut: Do We Really Need Mamba for Vision?"
- arXiv 2405.07992, read **v3** (2024-05-20). v1 2024-05-13, v2 2024-05-14 (abs.html).
- Authors: Weihao Yu, Xinchao Wang. National University of Singapore.
- Venue: CVPR 2025 per README (code-only; the arXiv source is in NeurIPS 2024 style).
- Repo: https://github.com/yuweihao/MambaOut

## 2. Core problem
Visual Mamba models underperform strong conv and attention models. The paper asks which vision tasks need
the SSM at all.

## 3. Prior work (per the paper)
Vim, VMamba, LocalMamba, PlainMamba bring Mamba to vision but trail SOTA conv/attention models (Sec. 1).

## 4. Key idea (the hypothesis)
Mamba suits tasks that are (1) long-sequence and (2) need causal (autoregressive) token mixing, because the
SSM is an RNN with a fixed-size, lossy memory (Sec. 3.1, Fig. 2-3).
- Hypothesis 1: "It is not necessary to introduce SSM for image classification on ImageNet, as this task
  does not meet Characteristic 1 or Characteristic 2." (Sec. 3.4)
- Hypothesis 2: "It is still worthwhile to further explore the potential of SSM for visual detection and
  segmentation since these tasks align with Characteristic 1, despite not fulfilling Characteristic 2."

## 5. How tested (mechanism of the argument)
1. Long-sequence test (Sec. 3.2): Transformer block FLOPs (MLP ratio 4) = 24 D^2 L + 4 D L^2 (Eq. 6);
   ratio r_L = L / (6D) (Eq. 7). Long sequence if L > 6D. tau_small = 6 x 384 = 2304 (ViT-S),
   tau_base = 6 x 768 = 4608 (ViT-B). ImageNet 224^2 / 16^2 patches = 196 tokens: short.
   COCO 800 x 1280 and ADE20K 512 x 2048 give "approximately 4K" tokens: long.
2. Causal test (Sec. 3.3, Fig. 3b): turn ViT attention causal and retrain on ImageNet:
   ViT-Ti 72.2 -> 70.6; ViT-S 79.8 -> 78.9. Claim: causal mixing hurts understanding tasks.
3. Build MambaOut (Sec. 4.1): stack Gated CNN blocks = Mamba block minus SSM.
   Meta-block (Eq. 8-9): X' = Norm(X); Y = (TokenMixer(X' W_1) (.) sigma(X' W_2)) W_3 + X.
   TokenMixer_GatedCNN(Z) = Conv(Z) (Eq. 10); TokenMixer_Mamba(Z) = SSM(sigma(Conv(Z))) (Eq. 11).
   Conv = 7x7 depthwise on part of the channels (InceptionNeXt style); MLP ratio 8/3; 4 stages.
4. Compare on ImageNet (H1), COCO Mask R-CNN 1x and ADE20K UperNet (H2) against published visual Mamba rows.

## 6. Equations (src/3_method.tex, src/4_results.tex; numbers from PDF p. 5-6)
Eq. (6): `\mathrm{FLOPs} = 24D^2L + 4DL^2.`  L = tokens, D = channels.
Eq. (7): `r_L = \frac{4DL^2}{24D^2L} = \frac{L}{6D}.`  Ratio of the attention (quadratic) term to the
linear term. "If L > 6D, the computational load of the quadratic term in L surpasses that of the linear term."
Eq. (4) causal mode: `y_t = f(x_1, x_2, ..., x_t),`  Eq. (5) fully visible: `y_t = f(x_1, x_2, ..., x_t, ..., x_T),`
Eq. (10)-(11) (labels eq:gated_cnn, eq:mamba):
```
\mathrm{TokenMixer}_\mathrm{Gated CNN}(Z) &= \mathrm{Conv}(Z) \label{eq:gated_cnn} \\
\mathrm{TokenMixer}_\mathrm{Mamba}(Z) &= \mathrm{SSM}(\sigma (\mathrm{Conv}(Z))) \label{eq:mamba}
```

## 7. Measurement setup
- ImageNet-1k, 224^2, DeiT recipe without distillation (RandAugment, Mixup, CutMix, Random Erasing,
  color jitter, stochastic depth, label smoothing), AdamW, batch 4096, lr 0.004, TPU v3 (Sec. 4.2).
- COCO: Mask R-CNN 1x (12 epochs), AdamW lr 1e-4, batch 16, FP16, 4 x RTX 4090 (Sec. 4.3).
- ADE20K: UperNet, AdamW lr 1e-4, batch 16, 160K iterations, FP16, 4 x RTX 4090 (Sec. 4.4).
- MambaOut configs (Table 4): Tiny blocks (3, 3, 9, 3), channels (96, 192, 384, 576), 26.5M, 4.5G MACs;
  Small (3, 4, 27, 3), same channels, 48.5M; Base (3, 4, 27, 3), (128, 256, 512, 768), 84.8M; Femto 7.3M.
- Held fixed: nothing beyond "same task protocol". The SSM-vs-no-SSM comparison is across different models
  and recipes (MambaOut's own depths, widths, MLP ratio, batch 4096), not a same-architecture ablation
  (our reading).
- Falsifier (our reading): a controlled ablation where removing the SSM from a visual Mamba at matched
  compute and recipe drops ImageNet accuracy would refute H1. The paper names its own challenge: future work
  "will be necessary to develop visual Mamba models with token mixers of convolution and SSM to achieve
  state-of-the-art performance on ImageNet" (Sec. 4.2).

## 8. Results (ImageNet vs detection/segmentation)
Naming in MambaOut tables: "VMamba-T/S/B" = the original (vanilla) VMamba (22M/44M/75M, 82.2/83.5/83.7);
"VMambaV9-T/S/B" = the later VMamba design (31M/50M/89M).
- ImageNet top-1, Table 1: MambaOut-Femto 78.9 (7M); MambaOut-Tiny 82.7 (27M, 4.5G); MambaOut-Small 84.1
  (48M, 9.0G); MambaOut-Base 84.2 (85M, 15.8G).
  Visual Mamba rows there: Vim-Ti 76.1, Vim-S 80.5, VMamba-T 82.2, VMambaV9-T 82.5, LocalVMamba-T 82.7,
  VMamba-S 83.5, VMambaV9-S 83.6, LocalVMamba-S 83.7, VMamba-B 83.7, VMambaV9-B 83.9.
  Text: MambaOut-Small 84.1 is "0.4% higher than that of LocalVMamba-S, while requiring only 79% of the MACs".
  Note (our reading): at Tiny, MambaOut-Tiny 82.7 **ties** LocalVMamba-T 82.7; "surpasses all" holds at S/B.
  VMamba v4's own final VMamba-T is 82.6 (VMamba Table 1), still 0.1 below MambaOut-Tiny.
- COCO Mask R-CNN 1x, Table 2 (AP^b / AP^m): MambaOut-Tiny 45.1 / 41.0 vs VMambaV9-T 47.4 / 42.7 and
  VMamba-T 46.5 / 42.1. MambaOut-Small 47.4 / 42.7 vs VMambaV9-S 48.7 / 43.7. MambaOut-Base 47.4 / 43.0
  vs VMambaV9-B 49.2 / 43.9. Text: MambaOut-Tiny "trails VMamba-T by 1.4 AP^b and 1.1 AP^m" -- that is the
  vanilla VMamba-T row (46.5 / 42.1), not VMambaV9-T (gap 2.3 / 1.7, our reading).
- ADE20K UperNet, Table 3 (mIoU SS / MS): MambaOut-Tiny 47.4 / 48.6 vs VMambaV9-T 48.3 / 48.6 and
  LocalVMamba-T 47.9 / 49.1. MambaOut-Small 49.5 / 50.6 vs VMambaV9-S 50.6 / 51.2. MambaOut-Base 49.6 / 51.0
  vs VMambaV9-B 51.0 / 51.6. Text: LocalVMamba-T beats MambaOut-Tiny "by 0.5 mIoU in both" SS and MS.
- Both SSM and no-SSM models trail conv-attention hybrids (TransNeXt-Tiny 84.0 top-1, 49.9 AP^b, 51.1 mIoU SS).

## 9. Limitations stated
"Due to computational resource limitations, this paper only verifies the Mamba concept for visual tasks"
(Sec. 5). No same-architecture SSM ablation (our reading). Note that both Vim (2 directions) and VMamba
(4 routes) already approximate fully-visible mixing; MambaOut says "the issue persists within each branch" (Sec. 1).

## 10. Outline feedback
- ch 9 MambaOut box is accurate. Add the computable rule L > 6D (2304 / 4608 vs 196 and ~4K tokens); it fits
  ch 0's cost framing and can be a live readout in `viz-scan2d` or the box.
- Name which VMamba row a gap refers to; MambaOut tables carry both old VMamba and VMambaV9.
- Causal-ViT ablation (72.2 -> 70.6, 79.8 -> 78.9) is a clean one-line claim for "why one causal scan is not enough".
