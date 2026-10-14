## Group B (2026-10-13)
- viz-queue: momentum ablation ran at K=4096 (not 65536); includes m=0.9999 (58.9); m=0 is "fail", not a number.
- viz-mechs: end-to-end curve stops at K=1024 and uses K-1 negatives; MoCo v1 negatives are queue keys only.
- SimCLR tau: default never stated; Table 5 best 0.1 of {0.05,0.1,0.5,1}; do not print as config.
- SimCLR head hidden width not stated; only h=2048, z=128 sourced.
- viz-augment: Fig. 5 uses an asymmetric protocol (transform on one branch only); crop->color 56.3, crop alone 33.1.
- MoCo v2: adds blur (not SimCLR's stronger color); cosine +0.2; tau 0.2 vs 0.07 in v1.
- MoCo v3: keeps momentum encoder (m=0.99 ViT) + prediction head; drops queue at batch 4096; symmetric loss x 2 tau; ViT-B 76.7 is 600-ep (300-ep 76.5).
- ch-step: 8192 views, 8192x8192 matrix diag masked; 16382 negatives stated only for batch 8192; 8190 at 4096 is our reading. LARS lr 4.8, wd 1e-6.
## Group C (2026-10-13)
- InfoNCE gradient printed: Wang & Liu Eq. 4 (Sec 3.1, wrt similarities), Eq. 5 r_i relative share of push -> viz-negatives bars; DCL Eq. 3 Prop. 1 full gradient incl. coupling factor.
- Wang & Isola v10: L_align, L_uniform unnumbered (Sec 4.1.1/4.1.2); Eq. 1 = contrastive loss, Eq. 2 = Theorem 1 limit (Sec 4.2). Cite by section.
- W&I Table 4 (BookCorpus): align+uniform 3-4 pts worse than contrastive -> claim card restricted to vision (Tables 1,2,3,5).
- No printed CIFAR10/STL10 tables in Chuang (plots + 4.26% STL10 gain); Robinson no ImageNet-100. Use Chuang Table 1 (ImageNet-100), Robinson STL10 hard x debias grid (87.44/84.41/84.26/80.15), Robinson tinyImageNet Table 1.
- Wang & Liu ImageNet100 tau=1.0: 63.57 main Table 1 vs 65.49 supp Table 1; do not reconcile.
- Wang & Liu tau 0.05..1.0, peak 0.2-0.3; uniformity reported as -L_uniform; memory-bank negatives.
- Chuang/Robinson use t for temperature, tau+ for class prior: page renames consistently.
- "about 1/C false negatives" is our arithmetic; tau+ values used 0.1 (CIFAR10), 0.01 (ImageNet-100).
- Robinson beta in {0,0.5,1,2,6}, best 0.5-1; CIFAR10 gain vanishes at large N.
## Group A (2026-10-13)
- DrLIM margin: no experimental m value; Fig 1 illustration 1.25 (uncertain). Y=0 similar.
- CPC images: 256x256 crop of 300x300, 7x7 grid of 64x64 patches, 32 px overlap (stride 32 our reading), random 60x60 subcrop per patch, ResNet-v2-101 -> 7x7x1024, 32 GPUs batch 16, predict up to 5 rows. No temperature; negatives count for images not stated -> walkthrough uses labelled choice or CPC v2's stated negatives source.
- Wu 2018: bank 128-d x 1.28M (~600 MB), tau 0.07, NCE m=4096; bank momentum 0.5 code-only.
- MI bound: unnumbered display after Eq. 5 in CPC Sec 2.3; derivation App A.1 Eq 6-11 (Eq 8 approximation); Poole Sec 2.3 Eq 10 proves it without the approximation.
- FaceNet triplet Eq. 3, alpha 0.2.
- Chopra 2005: margin only in a design condition; actual loss Eq. 9 has no hinge; DrLIM adds the hinge.
- Tschannen: hedged ("can result in worse"); claim card mentions fixed-bound encoder test, invertible-encoder test, MNIST halves.
- Arora tau = probability two sampled classes coincide (not temperature): warn.
- CPC v2 Table 1 "MoCo [2]" row looks like a citation typo: do not repeat.
- NCE uses equal data and noise. Sohn N-pair = bridge ch-compare -> ch-infonce (Tschannen Sec 4: InfoNCE equivalent to N-pair).
## Group E (2026-10-13)
- SigLIP v4 equations unnumbered: cite Sec 3.1 (softmax), Sec 3.2 (sigmoid).
- SigLIP sign of b: printed loss and Algorithm 1 disagree; with b=-10 only Algorithm 1 (logits = t x.y + b) starts every pair near "no match" -> build toy from Algorithm 1; say so.
- SigLIP headline: 84.5 (g/14, 4 TPUv4, 2 days); B/8 4 chips 1 day 79.8 table vs 79.7 text: avoid.
- SigLIP small batch: Fig 2 unlabeled; App B Table 5 e.g. batch 512, 3B examples: sigmoid 51.5 vs softmax 47.7.
- SupCon mCE in Fig 3 (left) not table; cite v5 (Jensen sign fix).
- SupCon toy anchor: Supp Table 5: 1 positive 69.3, no cap (13 expected positives) 78.5. Supp Sec 8: triplet is special case with margin 2 tau -> link ch-compare.
- CLIP vocab 49,152 text vs 49408 table: do not print. CLIP: 76.2 (L/14@336 zero-shot), 59.6 (RN50 zero-shot), 85.4 (linear probe) separate.
- Lint: SigLIP claim values spaced ("16 k", "32 k", "98 k", "268 M").
## Group D (2026-10-13)
- DirectCLR d0=360 only in commented LaTeX table (62.104 vs Table 1 62.7): uncertain. Jing spectrum cliff ~95 of 128 read off plot. SwAV batch for 400-ep no-multicrop run not stated.
- Linear table: 7 rows (add SimSiam, VICReg). SwAV 75.3 is Fig 2. "SwAV w/o multi-crop 71.8" in Barlow/VICReg = SimSiam's reproduction (SimSiam Table 4, 800 ep); SwAV own no-multicrop 70.1 at 400 ep.
- Baselines differ across papers: SimCLR own 69.3; SwAV prints 70.0; DINO 69.1; DINO prints BYOL 74.4 vs BYOL own 74.3.
- Jing: collapse measured in 128-d embedding; 2048-d representation healthy with projector -> ties to SimCLR head. Trigger: augmentation variance > data variance along a direction (Thm 1). Fig 3 toy: 16-d, 8 augmented dims, k 0.1..2.
- viz-barlow: lambda 0.002..0.02, default 0.005; 8x8 toy has 56 off-diag entries -> scale or explain.
- viz-sinkhorn: eps 0.05, 3 iterations (rows to 1/K then columns to 1/B); 1 iteration fails; codes get no gradient.
