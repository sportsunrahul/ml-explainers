# ResNet-50 ImageNet linear-probe top-1, each method as reported by its OWN paper

All rows: ResNet-50 (1x), ImageNet-1k pretraining without labels, linear classifier on frozen 2048-d features, ImageNet val top-1.
These are not one benchmark run: epochs, batch, crops, optimizer and linear-eval recipe differ per paper.

| method | top-1 | pretrain epochs | batch | views / multi-crop | own-paper loc | arXiv read |
|---|---|---|---|---|---|---|
| SimCLR | 69.3 | 1000 | 4096 (default setting, Sec 3; Sec 6 says "trained for 1000 epochs" but does not restate batch) | 2 x 224, no multi-crop | Table 6, row SimCLR (ours), ResNet-50 | 2002.05709v3 |
| MoCo v2 | 71.1 | 800 | 256 | 2 x 224 (1-crop eval), no multi-crop | Table 2, row MoCo v2, 800 ep | 2003.04297v1 |
| SwAV | 75.3 | 800 | 4096 | 2 x 224 + 6 x 96 multi-crop | Fig 2 (left table), row SwAV; Sec 4.1; Sec 5.1 "800 epochs - 8 crops" | 2006.09882v5 |
| SwAV, no multi-crop | 70.1 | 400 | 4096 (our reading: Fig 3 and App A.7 do not state batch; Sec 5.3 epoch study uses 4096) | 2 x 224 | Fig 3 (left table), row SwAV, column 2x224 | 2006.09882v5 |
| BYOL | 74.3 | 1000 | 4096 | 2 x 224, no multi-crop | Table 1a, row BYOL (ours) | 2006.07733v3 |
| SimSiam | 71.3 | 800 | 256 | 2 x 224, no multi-crop | Table 4, row SimSiam, column 800 ep | 2011.10566v1 |
| Barlow Twins | 73.2 | 1000 | 2048 | 2 x 224, no multi-crop | Table 1, row Barlow Twins (ours) | 2103.03230v3 |
| VICReg | 73.2 | 1000 | 2048 | 2 x 224, no multi-crop | Table 1, row VICReg (ours) | 2105.04906v3 |

SimCLR epochs/batch are confirmed by MoCo v2 Table 2 row "SimCLR, 1000, 4096, 69.3".
SwAV's "(w/o multi-crop)" row is SwAV's own 400-epoch run; SwAV never prints an 800-epoch no-multi-crop number.

## Where papers disagree about the same baseline
| baseline | own paper | other papers print | source of the other number |
|---|---|---|---|
| SimCLR | 69.3 (SimCLR Table 6) | 70.0 in SwAV Fig 2 | SwAV does not say where 70.0 comes from (no epochs/batch given) |
| SimCLR | 69.3 | 69.1 in DINO Table 2 (row SCLR) | not explained in DINO |
| SimCLR | 69.3 | 70.4 in SimSiam Table 4 (800 ep, repro.+) | SimSiam's improved reproduction |
| SimCLR | 69.3 | 68.2 in SwAV Fig 3 (400 ep, 2x224) | SwAV's own reimplementation |
| SimCLR | 69.3 | 61.9 (200 ep, batch 256) and 66.6 (200 ep, batch 8192) in MoCo v2 Table 2; 61.9 also in SwAV Table 3 | shorter runs, not the headline |
| MoCo v2 | 71.1 | 72.2 in SimSiam Table 4 (800 ep, repro.+) | SimSiam's improved reproduction |
| SwAV (no multi-crop) | 70.1 at 400 ep (SwAV Fig 3) | 71.8 in Barlow Table 1 "SwAV (w/o multi-crop)" and VICReg Table 1 "SwAV" | SimSiam Table 4 reproduction, 800 ep, batch 4096, 2 x 224 |
| SwAV | 75.3 | VICReg prints 75.3 as "SwAV (w/ multi-crop)" and 71.8 as plain "SwAV" | labelling in VICReg Table 1 |
| BYOL | 74.3 | 74.4 in DINO Table 2 | not explained in DINO |
| BYOL | 74.3 | 74.3 in SimSiam Table 4 (800 ep repro) | coincidence of value; different epochs (800 vs 1000) |
| BYOL | 74.3 | 74.1 in Barlow Table 7 (baseline 4096-256 heads; Barlow-size 8192 heads give 72.3) | Barlow's reimplementation |

Consistent everywhere: MoCo v2 71.1 (MoCo v2, SwAV, BYOL, Barlow, VICReg, DINO), SimSiam 71.3 (Barlow, VICReg), Barlow 73.2 (VICReg, DINO), SwAV 75.3 (Barlow, VICReg, DINO).

## Page guidance (our reading)
- Show the own-paper number and put epochs, batch and crops in the same row. SwAV is the only multi-crop row; flag it.
- Do not present the spread (69.3 to 75.3) as a ranking of the losses: SwAV's 2x224 run at 400 ep (70.1) sits next to SimCLR's reimplementation at 400 ep (68.2) in SwAV's controlled Fig 3.
