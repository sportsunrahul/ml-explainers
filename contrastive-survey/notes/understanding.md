# Understanding: contrastive learning for vision (synthesis)

Per-paper detail lives in `notes/papers/<name>.md` (46 files; equations verbatim from the LaTeX, or from the
PDF for the four non-arXiv papers). Numbers live in `notes/claims/*.json` (896 claims, 6 uncertain).
Extraction corrections are logged in `notes/outline-feedback.md` and folded into `outline.md`.
This file is the story the page tells, with the lens chosen for each paper.

## Core problem
Labels are expensive. Can an encoder learn features that transfer (linear probe, fine-tune, retrieval,
zero-shot) from unlabeled images alone? The signal: two augmented views of one image belong together.

## Why the naive version fails
Minimizing only the distance between partner views has a trivial optimum: a constant encoder (collapse).
Every method is a way to make collapse expensive.

## The one key idea
Score the anchor against one positive and many negatives and train a softmax to pick the positive
(InfoNCE). Negatives push the embedding apart; the positive pulls the pair together.

## Lineage, each paper as a delta (lens in brackets)
- Chopra 2005 [method]: siamese network, distance-based energy; margin only as a design condition (its loss,
  Eq. 9, has no hinge).
- Hadsell 2006, DrLIM [method]: the margin contrastive loss, Eq. 4: pull similar pairs (Y=0), hinge-push
  dissimilar pairs closer than m (Y=1). No experimental m is stated.
- Gutmann 2010, NCE [theory]: estimate a density by classifying data vs noise; origin of the name.
- FaceNet 2015 [method]: triplet loss, Eq. 3, margin alpha = 0.2.
- Sohn 2016, N-pair [method]: one positive vs N-1 negatives through a softmax, batch reused as negatives.
- Wu 2018 [method]: instance discrimination; non-parametric softmax over every image, approximated with
  NCE (m = 4096 noise samples), memory bank of 128-d vectors for all 1.28M images, tau 0.07.
- CPC 2018 [method + theory]: InfoNCE (Eq. 4); predicting future latents; I >= log N - L (unnumbered, Sec.
  2.3); no temperature; image setup = 7x7 grid of 64x64 patches, ResNet-v2-101; ImageNet linear 48.7.
- CMC 2019, CPC v2 2019, Arora 2019, Poole 2019 [branches]: views as channels; scaled-up CPC; first
  theory (Arora's tau = class collision probability); exact proof of the InfoNCE bound.
- Tschannen 2019 [analysis]: tighter MI bounds can give worse features; the encoder/critic inductive bias
  matters; InfoNCE = log K - N-pair loss (Sec. 4).
- MoCo 2019 [method]: queue of K keys (65536) from a momentum encoder (m 0.999; ablation at K 4096);
  only the query encoder gets gradient; shuffling BN. v2 adds MLP head, blur, cosine (tau 0.2). v3: ViT, no
  queue at batch 4096, momentum kept (0.99) + prediction head, frozen random patch projection for stability.
- SimCLR 2020 [method]: NT-Xent (Eq. 1) over 2N views; crop + color is the key augmentation (Fig. 5,
  asymmetric protocol); nonlinear head g improves h; big batch and long training help; LARS lr 4.8, batch
  4096; ResNet-50 linear 69.3. No default tau stated (Table 5 best 0.1).
- Wang & Isola 2020 [analysis]: L_align and L_uniform (unnumbered, Sec. 4.1); Theorem 1 asymptotic split;
  vision results support direct optimization; text result (Table 4) does not.
- Wang & Liu 2020 [analysis]: gradient = hardness-aware weights (Eq. 4-5); temperature trades uniformity
  vs tolerance; sweep 0.05..1.0, best 0.2-0.3.
- Chuang 2020 [method]: debiased loss with class prior tau+ (0.1 CIFAR10, 0.01 ImageNet-100).
- Robinson 2020 [method]: hardness concentration beta (0..6, best 0.5-1) on top of debiasing.
- DCL 2021, HaoChen 2021 [branches]: drop the positive from the denominator; spectral loss guarantees.
- SwAV 2020 [method]: swapped prediction of Sinkhorn codes (eps 0.05, 3 iterations) over prototypes;
  multi-crop; 75.3 linear (Fig. 2; 70.1 without multi-crop at 400 ep).
- Barlow Twins 2021 [method]: cross-correlation to identity, lambda 0.005; 73.2 linear.
- Jing 2021 [analysis]: dimensional collapse in the 128-d embedding; trigger = augmentation variance >
  data variance along a direction (Thm 1) and implicit regularization; representation before the projector
  stays healthy; DirectCLR.
- BYOL, SimSiam, VICReg, DINO, iBOT, DINOv2, DINOv3, Web-SSL [branches]: the no-negative and
  self-distillation line; see the JEPA survey.
- SupCon 2020 [method]: all same-class samples are positives; L_out beats L_in (gradient argument; v5 fixes
  a Jensen sign); triplet = special case (Supp. Sec. 8); one positive 69.3 vs all 78.5 (Supp. Table 5).
- CLIP 2021 [method]: symmetric InfoNCE over an N x N image-text matrix with learnable temperature; zero-
  shot 76.2 (L/14@336px), 59.6 (RN50).
- SigLIP 2023 [method]: independent sigmoid per pair with learnable t and b (t' = log 10, b = -10); no
  global normalization; better at small batch (App. B Table 5); SigLiT 84.5 on 4 TPUv4 chips in 2 days.
- ALIGN, OpenCLIP scaling, MetaCLIP, SigLIP 2, PE, DenseCL, DetCon [branches]: one line each.

## Wording traps (from extraction; see also the fact-check logs)
- Never present linear-probe numbers from different papers as one controlled comparison.
- tau: CPC has none; Chuang and Robinson write t for temperature and tau+ for the prior; Arora's tau is a
  probability. The page uses tau for temperature only and tau+ for the prior.
- SigLIP equations are unnumbered; the sign of b follows Algorithm 1.
- Wang & Isola L_align/L_uniform are unnumbered; cite sections.

## Fact-check notes-level corrections (2026-10-13; claims already fixed)
- SwAV: 1.2x slower than SimCLR, ~1.4x slower than MoCo v2 (Sec 4.3); codes are not treated as a target in
  the paper's wording; ablations use crop scale (0.08, 1). The "71.8 = SimSiam's reproduction" attribution
  is our reading (value match only).
- Jing: repo github.com/facebookresearch/directclr; Fig 3 singular values drop sharply but stay nonzero.
- InfoMin "InfoMin Aug." numbers include a JigSaw branch: not a pure-augmentation result.
- MoCo v1 GPU type is inferred from MoCo v2. MoCo v3 fine-tune 83.2 (150 ep) vs 84.1 (100 ep): give schedules.
- SimCLR Table 5 batch is not stated (caption: contrastive distribution over 4096 examples).
- DrLIM Fig 1 omits the 1/2 factor of Eq. 4: do not copy the plot.
- SupCon one-positive 69.3 also removes the other positives from the denominator.
- CLIP Sec. 6 limitation is compute, not big batches. "Saturates at log N" is our reading of CPC.
