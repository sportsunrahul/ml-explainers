# DINO (branch)

- Title: Emerging Properties in Self-Supervised Vision Transformers. arXiv 2104.14294, read v2 (2021-05-24); v1 2021-04-29.
- Authors: Mathilde Caron ... Armand Joulin. abs.html comment: "21 pages" (no venue stated).
- Key idea: self-distillation with no labels: a student matches the softmax output of a momentum (EMA) teacher across views (multi-crop); collapse is avoided by centering and sharpening the teacher output instead of negatives or Sinkhorn.
- Defining equation (`eq:loss`, src/approach.tex):
```
\min_{\theta_s} \sum_{x \in \{x^{g}_1, x^{g}_2\}} \quad \sum_{\substack{x' \in V\\x'\neq \, x}} \quad H(P_t(x), P_s(x')).
```
  H(a, b) = -a log b; P_s = softmax(g_s(x)/tau_s), tau_s = 0.1; tau_t warmed 0.04 -> 0.07 over 30 epochs; teacher sees only the 2 global crops. Center update `c \leftarrow m c + (1-m) \frac{1}{B} \sum_{i=1}^B g_{\theta_t}(x_i)` (`eq:center_update`).
- Key numbers (Table 2, ImageNet val top-1, frozen features): ViT-B/8 80.1 linear, 77.4 k-NN; ViT-S/8 79.7 linear, 78.3 k-NN; ResNet-50 75.3 linear, 67.5 k-NN; ViT-S/16 77.0 linear, 74.5 k-NN.
  Epochs are not in Table 2; iBOT Table 1 lists DINO ViT-S/16 at 3200 and ViT-B/16 at 1600 effective epochs.
- Table 2 RN50 rows as printed by DINO: SCLR 69.1, MoCov2 71.1, InfoMin 73.0, BarlowT 73.2, OBoW 73.8, BYOL 74.4, DCv2 75.2, SwAV 75.3. Note SimCLR 69.1 (own paper 69.3) and BYOL 74.4 (own paper 74.3).
