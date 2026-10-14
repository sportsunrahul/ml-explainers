# Fact-check of claims C.json + D.json, and notes for C/D papers

Checked against sources/<name>/ (LaTeX, paper.pdf, abs.html). 319 claims (C 163, D 156).

## Summary
| group | claims | OK | WRONG | WRONG-LOC | MISLEADING | UNVERIFIABLE |
|---|---|---|---|---|---|---|
| wangisola | 46 | 42 | 0 | 4 | 0 | 0 |
| wangliu | 55 | 54 | 0 | 0 | 1 | 0 |
| robinson, chuang, dcl, haochen | 62 | 62 | 0 | 0 | 0 | 0 |
| swav | 53 | 52 | 0 | 0 | 1 | 0 |
| barlow, jing | 59 | 56 | 0 | 3 | 0 | 0 |
| simsiam, dino, dinov2, dinov3, byol, vicreg, webssl, ibot, mocov2, simclr | 44 | 43 | 0 | 0 | 1 | 0 |
| total | 319 | 309 | 0 | 7 | 3 | 0 |

No value is wrong. All quoted equations match the LaTeX (macros expanded). All dates, versions and venues match abs.html.

## Non-OK claims
- wangisola-n-stl10 | WRONG-LOC | The "We optimize a total of 304 STL-10 encoders..." paragraph is in Sec. 5 on PDF p.7, BEFORE Tables 1-2 (p.8). It follows Table 2 only in source order. | loc -> "Sec. 5, p.7, paragraph 'We optimize a total of ...'; also App. B".
- wangisola-n-nyu | WRONG-LOC | same paragraph, p.7 | same fix.
- wangisola-n-in100 | WRONG-LOC | same paragraph, p.7; also App. B.2 | same fix.
- wangisola-n-bookcorpus | WRONG-LOC | same paragraph, p.7; also App. B.3 | same fix.
- wangliu-in100-hard-negatives | MISLEADING | Sec 5.1: "alpha is set to 0.0819, 0.0819, 0.0315 and 0.034 for CIFAR10, CIFAR100, SVHN, and ImageNet100 (4095 negative samples)". The parenthetical covers the list, and 0.0819 x 50000 = 4095 (CIFAR). Sec 5.5 "nearest 4095 features" is for HardSimple on all datasets. | Drop "on ImageNet100"; say the hard/HardSimple losses keep 4095 nearest negatives.
- swav-multicrop-gain | MISLEADING | Sec 5.2 says "2-4%", but Fig 3 left prints +2.4 (SimCLR), +4.6 (SeLa-v2), +4.1 (DeepCluster-v2), +4.0 (SwAV). | Quote "2-4%" as the paper's wording; the printed range is +2.4 to +4.6.
- barlow-ablation-baseline | WRONG-LOC | 71.4 is in Sec 4 intro and the Baseline row of Table 5 "Loss function explorations" (4_ablations.tex). PDF Table 3 is transfer classification. | loc -> "Sec 4 intro; Table 5".
- barlow-only-invariance | WRONG-LOC | 57.3 is in Table 5 (4_ablations.tex l.31). | loc -> "Table 5".
- barlow-only-redundancy | WRONG-LOC | 0.1 is in Table 5 (4_ablations.tex l.32). | loc -> "Table 5".
- vicreg-coeffs | MISLEADING (low) | Sec 4.2: "Coefficients lambda and mu are 25 and nu is 1". The claim text "lambda = mu, nu" with value 25 reads as nu = 25. | Claim "lambda = mu" = 25; add nu = 1 as a separate claim.

## Uncertain and "our reading" items
- wangisola-n-total (521): sound, our sum. The paper prints no total. The v10 changelog removed 2 STL-10 encoders from Fig. 5, but the text still says 304. Do not call 521 "encoders plotted".
- wangisola-collapse-ratio (4): the paper says "e.g., < 4" (Sec. 5, p.8). Present it as an example bound, not a measured threshold.
- wangisola-in1k-mocov2 (67.5): the number comes from the official MoCo v2 code (Table 5 caption), not a matched rerun. The claim is faithful.
- swav Sinkhorn-iteration and fixed-prototype claims: the setting (400 ep, 2x160+4x96) is inferred from the shared 73.9 baseline (App C.1, C.3). Sound.
- swav-best-crops (2x224+6x96): inferred from Sec 5.1 "8 crops", App A.1 / Table 4 and the App D command. Sound.
- barlow-projdim-min (32), barlow-batch-min/max (128/4096): resolved from the included figures fig_projdim.pdf and fig_batch2_rev.pdf. Do not use the unused fig_batch2.pdf (starts at 256).
- barlow-t1-swav-nomc / SwAV 71.8 = SimSiam 800-ep "SwAV (repro.+)": sound but stated by no source. Barlow Table 1 gives no citation for the row. VICReg Table 1 cites caron2020swav, and SwAV prints no 71.8 for SwAV (71.8 in SwAV Fig 3 is SeLa-v2 with multi-crop). Keep "our reading".
- jing-spectrum-cliff (about 95): sound; the rendered plot drops at rank about 93-94 of 128.
- jing-directclr-d0 (360): sound only as "about". It comes from a commented-out table (360 -> 62.104) and Fig 12. The d0 for Table 1 (62.7) is never printed.
- robinson pseudocode: sound. The script's `reweight = (beta*neg)/neg.mean()` is linear in neg; method.tex uses e^{beta s}/Z_hat_beta. Treat the equations as the authority.
- robinson temperature: not stated anywhere in the paper (App. D.1 included). Unresolvable.
- chuang tau+ = 1/C: sound; the paper states only uniform rho(c) = tau+. Fig 4a plot readings (about 91 / 92 / 94) match the rendered figure.
- webssl table number: resolved, Table 3 (third table environment, tab:final_table).
- dino epochs via iBOT Table 1 (3200 / 1600 effective epochs): confirmed.
- imagenet-linear-table SwAV no-multi-crop batch 4096: sound. Better loc is App A.1 (general SwAV setup), not Sec 5.3.

## Notes-level problems
- wangisola.md | E2 says L_align is on p.5 | It is on PDF p.4 (Sec. 4.1.1) | Change to p.4.
- wangisola.md | E4 proof cited as "Appendix A" | It is subsection A.2 | Cite App. A.2.
- wangisola.md Sec. 7 | The ImageNet-100 recipe used the unofficial CMC MoCo code (supplementary.tex l.709); only ImageNet-1K used official MoCo v2 | State this next to 72.80 vs 74.60.
- wangisola.md Sec. 8 | "comparable or better ... across the board" is not a quote; the abstract has no "across the board" | Remove it or mark it as our wording. The criticism itself is sound (Table 4 is 3-4 points worse).
- wangliu.md Sec 7 | Ties 4095 negatives to ImageNet100 | Same fix as wangliu-in100-hard-negatives.
- wangliu.md Sec 8 | "Main Table 1 (p.7)" | Table 1 is on PDF p.8 | Change to p.8.
- robinson.md Sec. 7 | "temperature not stated in main text" | It is absent from the whole paper | "not stated anywhere".
- chuang.md Sec. 7 | "biased trained 500/600 epochs" | 500 comes only from the Fig 4c legend; caption and Sec 5.1 say 600 | Optional note.
- dcl.md | Paper-internal contradiction: Discussion (main.tex l.463) says q_B -> 0 at large batch, against Sec. 3 / Fig. 1b | Do not quote l.463. dcl claims "after Table 1" is true in tex order only; in the PDF the sentence is before Table 1; prefer "Sec. 4.2 text".
- swav.md Sec 9 | Wall-clock ratios swapped: notes say "slower than MoCov2/SimCLR (1.2x/1.4x)" | Sec 4.3: 1.2x vs SimCLR (2x224), about 1.4x vs MoCov2, for SwAV with 2x160+4x96 | Swap to "1.2x vs SimCLR, 1.4x vs MoCov2".
- swav.md Sec 5 step 4 | "The codes are a target" | method.tex l.11: "we do not consider the codes as a target, but only enforce consistent mapping between views" | Say codes are computed under no_grad; do not call them a target.
- swav.md Sec 5 step 1 and Config | Crop scale s=(0.14, 1) is the main-run setting; App A.7 says ablations use s=(0.08, 1) | Add this; it covers Fig 3 (70.1 / 74.1) and Tables 12-14.
- swav.md Sec 3 | Momentum encoder cited to Sec 3.1 | It is in Sec 4.3 and Table 3 | Fix loc.
- swav.md Sec 5 step 7 | "epoch 15 (Sec 4.3, App A.6)" | Epoch 15 is only in App A.6 | Fix loc. Also cite "codes from full-res crops only" to Sec 3.2.
- imagenet-linear-table.md | SimCLR "default setting, Sec 3" | It is Sec 2.3 (Evaluation Protocol) | Change to Sec 2.3.
- imagenet-linear-table.md | "SwAV's '(w/o multi-crop)' row is SwAV's own 400-epoch run" | Ambiguous: Barlow's "SwAV (w/o multi-crop)" row is 71.8, not SwAV's run | Say "our 'SwAV, no multi-crop' row (70.1) is SwAV Fig 3, 400 ep".
- imagenet-linear-table.md | SimCLR 70.0 row: "SwAV does not say where 70.0 comes from" | SwAV Fig 2 cites chen2020simple; it gives no epochs or batch | Reword.
- imagenet-linear-table.md | SwAV 71.8 source "SimSiam Table 4" | Value-match inference; VICReg cites SwAV, which lacks 71.8; SimSiam "+" means an improved reproduction (3-layer projection MLP) | Mark "(our reading)".
- barlow.md Sec 8 | "Loss ablations (Table 3)" | Table 5 | Change to Table 5.
- barlow.md Sec 8 | Batch "almost unaffected" | fig_batch2_rev.pdf: about -1.0 at 256, about -2.0 at 128 | Add the plot readings.
- jing.md Sec 1 | "Repo: none given" | Reproducibility Statement: github.com/facebookresearch/directclr (main.tex l.564) | Add the repo.
- jing.md Sec 5 | Fig 3 singular values "vanish" | They drop sharply but stay nonzero (about 0.15-0.25 at k=2); vanishing is Theorem 1's asymptotic claim | "shrink sharply".
- jing.md Sec 5 | "BYOL-style augmentations" | App D.1 lists the augmentations without a BYOL attribution | Drop "BYOL-style" or mark as our reading.
- simsiam.md | Calls all Table 4 baselines "repro.+" | BYOL is printed as "(repro.)" without "+" | Fix the BYOL label; optionally add the batch column (SimCLR/BYOL/SwAV 4096, MoCo v2/SimSiam 256).
- vicreg.md | "Table 1 prints the same baseline numbers as Barlow Twins" | Top-1 matches; MoCo v2 top-5 differs ("-" vs 90.1) | Say "same top-1 baseline numbers".
- ibot.md | Venue cited from src filename | The PDF header says "Published as a conference paper at ICLR 2022" | Cite the PDF header.
- byol.md | Quoted eq:cosine-loss drops a stray trailing "\cdot" from the source | Cosmetic; no fix needed.
- haochen.md, dino.md, dinov2.md, dinov3.md, webssl.md | No problems found.
