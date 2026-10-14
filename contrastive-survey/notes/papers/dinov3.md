# DINOv3 (branch)

- Title: DINOv3. arXiv 2508.10104, read v1 (2025-08-13).
- Authors: Oriane Simeoni ... Piotr Bojanowski (accent dropped for ASCII). Technical report; venue not stated.
- Key idea: scale self-distillation to a 7B ViT on curated web data, and fix dense features that degrade in long training with "Gram anchoring": keep the student's patch-similarity (Gram) matrix close to that of an earlier "Gram teacher".
- Defining equation (`eq:gram-loss`, chapters/gram.tex):
```
\mathcal{L}_{\text{Gram}} = \left \| \mathbf{X}_{S} \cdot \mathbf{X}_{S}^\top - \mathbf{X}_{G} \cdot \mathbf{X}_{G}^\top  \right \|_\text{F}^2.
```
  X_S, X_G: P x d L2-normalized patch features of student and Gram teacher; applied on global crops, starting after 1M iterations. Refinement objective: L_Ref = w_D L_DINO + L_iBOT + w_DK L_DKoleo + w_Gram L_Gram.
- Key number: ViT-7B/16 88.4 ImageNet val linear top-1, frozen, resolution adapted to 1024 patch tokens (512x512) (Table 7). Same table lists DINOv2 g/14 87.3 (at 448; DINOv2's own paper prints 86.5 at 224), SigLIP 2 g/16 89.1, PEcore G/14 89.3.
- Dense headline: ADE20k linear mIoU 55.9 vs DINOv2 g/14 49.5, SigLIP 2 42.7, PEcore 38.9, AM-RADIOv2.5 53.0 (Table 3). Text: "outperforms the self-supervised baselines by more than 6 mIoU points, and the weakly supervised baselines by more than 13 points".
