# DINOv2 (branch)

- Title: DINOv2: Learning Robust Visual Features without Supervision. arXiv 2304.07193, read v2 (2024-02-02); v1 2023-04-14.
- Authors: Maxime Oquab ... Piotr Bojanowski. Venue: TMLR (01/2024) (PDF header).
- Key idea: scale DINO + iBOT style self-distillation (with Sinkhorn-Knopp centering, KoLeo regularizer) on a curated 142M-image dataset (LVD-142M), train a 1B ViT-g and distill smaller models; features work frozen across tasks.
- Equation: none new defining; objective = DINO image-level loss + iBOT patch loss (approach.tex). Teacher centering uses "the Sinkhorn-Knopp (SK) batch normalization of SwAV", 3 iterations.
- Key number: ViT-g/14 86.5 ImageNet-1k linear top-1 (frozen, 224), k-NN 83.5; ViT-L/14 86.3 (Table 4). Fine-tune ViT-g/14: 88.5 at 224, 88.9 at 448 (Table 5).
- Dense headline: frozen features + linear head, ADE20k mIoU ViT-g/14 49.0 (lin.) vs OpenCLIP ViT-G/14 39.3, DINO ViT-B/8 31.8, iBOT ViT-L/16 44.6 (Table 10). Abstract: surpasses OpenCLIP "on most of the benchmarks at image and pixel levels".
