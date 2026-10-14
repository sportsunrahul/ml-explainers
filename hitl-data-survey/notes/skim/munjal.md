# Munjal et al. -- Towards Robust and Reproducible Active Learning Using Neural Networks

Citation: Prateek Munjal, Nasir Hayat, Munawar Hayat, Jamshid Sourati, Shadab Khan. "Towards Robust and
Reproducible Active Learning Using Neural Networks." arXiv 2002.09564v3. CVPR 2022 (abs.html comment:
"Accepted at CVPR 2022"). Code: TorchAL (github.com/PrateekMunjal/TorchAL). Source: sources/munjal/src/arxiv.tex.

Lens: analysis/benchmark paper; the analysis leads. Claim: under one code base, tuned per iteration, AL methods
give inconsistent and small gains over random sampling, and strong regularization shrinks the gap further.
Measured by: test accuracy at 20/30/40% labeled data, 25 runs per method, Kruskal-Wallis + Games-Howell tests.
Falsified if: one AL method beat a strongly regularized random baseline (RSB-SR) by more than the run-to-run
spread, consistently across datasets, budgets, and architectures.

## Core problem and key idea
Published AL gains disagree, and published random baselines differ by up to approx. 13% at the same setting
(Sec 1: CIFAR10, 20% labels, Yoo 2019 vs BGADL). The key idea: re-run all methods in identical conditions, re-tune
hyperparameters at every AL iteration, repeat over initial sets and seeds, and add strong regularization. Then the
"AL beats random" signal mostly falls inside the noise.

## Methods compared (Sec 2; all inside one PyTorch toolkit)
- RSB: random sampling baseline.
- UC: softmax least confidence (Lewis & Gale 1994). Score max_j Phi(x_i), inline at arxiv.tex line 180. Note: the
  text says "descending order ... top k", but the stated intent is "least confident". Treat as least confidence.
- DBAL: MC-dropout max entropy (Gal 2017). Inline, arxiv.tex lines 186-188, unnumbered:
  `\argmax_i \mathbb{H} [P(\textbf{y}|x_i)]` with `P(\textbf{y}|x_i) = \sum_{j=1}^{T}\frac{1}{T} P(\textbf{y}|x_i, \phi_{j})`
  T = number of stochastic forward passes; phi_j = the j-th dropout mask. T value not given (UNSURE).
- BALD: `\argmax_i \mathbb{I}[P(\textbf{y},\Phi|x_i,L_0)]` (inline, line 191): mutual information between label and weights.
- Coreset: k-center greedy (not MIP), following Yoo 2019.
- VAAL: VAE + discriminator; query points the discriminator rates least likely "labeled".
- QBC: ensemble variation ratio, `v = 1-f_m/N` (inline, line 237). N = 5 VGG16 members with different inits;
  f_m = votes for the modal class. The ensemble only selects; a separate classifier is trained.

## Protocol (Sec 4, 5, 6.1; Supp. Alg. 1)
- Data: CIFAR10, CIFAR100; limited ImageNet. Split into train T_r, validation V, test T_s. Model selection on V.
- Initial labeled set 10%; budget b = 10% per iteration to 40% (also b = 5%, 10% -> 40%, Fig 3 / Supp. Table 6).
- Classifier VGG16 + batchnorm (penultimate FC = 512) unless stated. Retrained from random init every iteration on
  CIFAR. ImageNet: ResNeXt-50, fine-tuned from the previous iteration, 3 runs.
- Repeats: 5 initial sets L0..L4 "drawn with replacement", 5 random-init seeds each = 25 models per method.
- AutoML: Optuna random search, 50 trials at EACH AL iteration (200 models for 4 iterations). Best params from the
  first seed are reused for the other 4 seeds.
- AutoML ranges (Supp. "Auto-ML Hyper-parameters"): lr log [1e-5, 1e-2); wd log [1e-8, 1e-3); batch size
  {8,...,1024}; optimizer {SGD, Adam}; RandAugment N in 1..15, M in 1..8.
- Strong regularization (-SR suffix) = RandAugment (RA) + SWA in the main runs. SWA LR 5e-4, frequency 50 (CIFAR);
  SWA LR 1e-5, frequency 50 (ImageNet). Shake-Shake (SS) only in the Table 3 ablation (ResNeXt family only).
- Default validation-set size for CIFAR runs is not stated in the main text (UNSURE); Table 2 tests 2/5/10%.

## Variance findings (Sec 6.1, 6.2)
- "A standard deviation of 1-2% in accuracy among different AL methods" (Sec 6.1). SR models "exhibit
  appreciably-smaller variance across multiple runs" (Sec 6.4; also Sec 1, contribution 4).
- Concrete spread: Supp. Table 7 (CIFAR10, L_0^0), row UC, col 20%: 73.35 +- 4.84; same table RSB 20%: 77.65 +- 0.82.
- Base model (10% labels) differs by initial set alone: CIFAR10 69.16 / 68.02 / 70.34 / 68.19 / 67.19
  (Supp. Tables 7-11 captions); CIFAR100 34.73 / 32.73 / 34.66 / 30.44 / 34.85 (Supp. Tables 12-16).
- Stats (Sec 6.2): Kolmogorov-Smirnov rejects normality in all 6 settings; Levene rejects equal variance in 4/6;
  Kruskal-Wallis then Games-Howell at alpha = 0.05. Count of settings where a method > RSB: QBC 4/6, Coreset 2/6,
  DBAL 2/6, BALD 1/6. Table 1 (main), Kruskal-Wallis p-values: C10 20% 9.35e-4, C10 30% 2.67e-9, C10 40% 8.54e-19,
  C100 20% 1.13e-8, C100 30% 1.84e-12, C100 40% 5.63e-16. Table 1 rows with "< RSB": VAAL at C10 30% and C100 30%,
  UC at C10 40%.
- Rankings flip with budget size (Sec 6.3): CIFAR10 40%, b = 10% -> UC best; b = 5% -> BALD best.

## Key table values (copy exactly)
Table 3 (main): regularization ablation, best of 50 AutoML trials, 10% of training data, VGG16 (SS uses ResNeXt).
| Methods | CIFAR10 | CIFAR100 |
| RSB | 69.16 | 34.73 |
| + SWA | 74.6 | 38.06 |
| + RA | 76.36 | 38.01 |
| + Shake-Shake(SS) | 73.93 | 39.23 |
| + SWA + RA | 82.16 | 39.44 |
| + SS + SWA + RA | 84.45 | 48.92 |

Table 4 (main): CIFAR10 transferability; sets chosen on VGG16, trained on target nets; test acc at 20/30/40%.
| Method | VGG16 | WRN-28-2 | R18 | R18-SR |
| RSB | 77.34 80.91 82.05 | 81.3 81.98 85.68 | 75.73 79.69 80.69 | 88.69 90.44 91.8 |
| Coreset | 76.18 82.24 85.11 | 82.32 84.29 86.72 | 79.27 82.13 85.09 | 88.48 91.72 92.94 |
| DBAL | 78.08 82.42 85.17 | 80.79 84.54 87.87 | 75.05 80.69 82.95 | 89.41 92.23 93.34 |
| VAAL | 77.13 79.97 80.55 | 78.42 83.79 86.58 | 74.87 79.96 82.24 | 87.11 90.43 92.19 |
| QBC | 78.63 82.66 85.24 | 81.85 85.06 87.94 | 76.61 81.77 84.41 | 88.34 91.37 92.62 |
My arithmetic from Table 4: best AL minus RSB on R18 = +3.54 / +2.44 / +4.40 (Coreset); on R18-SR = +0.72 /
+1.79 / +1.54 (DBAL). Paper wording is "at par with or marginally inferior to" SOTA AL (Sec 8); "beats" holds only
across regimes: "RSB-SR model outperforms the best AL approach (coreset with ResNet18) ... though DBAL-SR appears
at the top" (Sec 6.6).

Supp. Table 1: RSB-SR(ours), VGG16, CIFAR10 at 10/20/30/40%: 82.16 / 85.09 / 89.43 / 91.16. RSB(ours) VGG16:
69.16 / 77.34 / 80.91 / 82.05. Reported coreset-paper random baseline: 60 / 68 / 71 / 74.
Conflict: Sec 6.4 text says RSB-SR at 40% = 90.87% (CIFAR10) and 59.36% (CIFAR100). Supp. Table 1 says 91.16.
Which one is the 25-run mean is UNSURE. Sec 6.4: RSB-SR at 20% is approx. 3% above RSB at 40% (CIFAR10).

Supp. Table 5: ImageNet, budget 5%, 3 runs, acc at 10/15/20/25%. Text cites "Tab 4 in suppl." (wrong ref).
| without RA+SWA | RSB 57.89 61.55 64.51 66.52 | Coreset 57.89 62.36 65.42 67.8 | VAAL 57.89 62.87 65.11 67.08 |
| with RA+SWA | RSB 60.1 64.59 67.14 69.2 | Coreset 60.1 64.98 67.97 70.12 | VAAL 60.1 64.88 67.45 69.37 |
RSB-SR at 20% (67.14) exceeds RSB and VAAL at 25%: "a saving of approx. 64000 images" (Sec 6.5).

Table 5 (main): noisy oracle, CIFAR10, labels of 10% / 20% of each random draw permuted; acc at 10/20/30/40%.
| Noise 10% | RSB 69.16 72.08 76.62 80.88 | RSB-SR 82.16 84.96 86.06 89.13 |
| Noise 20% | RSB 69.16 69.42 75.89 79.61 | RSB-SR 82.16 77.39 85.9 85.12 |

## What the figures show (redraw the idea)
- Fig 1 / Fig 2: CIFAR10 / CIFAR100 accuracy vs % labeled for each initial set L0..L4, plus the 25-run mean;
  the method curves cross and re-order between subplots.
- Fig 3: same with b = 5%; still no consistent winner.
- Fig 4: long-tailed CIFAR100 (power-law class counts, weighted CE); gap to RSB shrinks as labels grow.
- Fig 5: solid vs SR curves for Random/Coreset/VAAL/QBC; SR lifts all curves and squeezes them together.
- Fig 6: ImageNet, dashed SR lines above solid lines; gaps shrink.
- Fig 7: overlap between the sets different methods pick in the first CIFAR10 iteration.

## Limitations (authors) and follow-ups
- Image classification only (Sec 9). Compute heavy: months on 2 DGX-1 servers, AutoML at each step (Sec 5).
- Supp. "Unexplained performance degradation": accuracy sometimes drops when labels are added (UC -0.54% from
  20% to 30%, Supp. Table 8).
- Follow-ups: Beck et al. 2021 (DISTIL; with augmentation AL is still 2x to 4x label-efficient); Luth et al.
  2023 (light HP tuning on the starting budget; their models "generally outperform" Munjal's). TypiClust and
  ProbCover reuse the TorchAL framework for evaluation.

## Labeling decision
Before a team buys an AL pipeline for a new visual inspection app, they compare it to random selection trained
with strong augmentation and weight averaging over several seeds. If the gain is inside the seed spread
(1-2%), they spend effort on regularization and random labeling instead.

## Visual ideas
1. Seed cloud: 25 dots per method at 20/30/40%; a toggle adds RA+SWA and the clouds rise, shrink, and overlap.
2. Gap shrinker: Table 4 bars, R18 vs R18-SR; a slider for "regularization" morphs one into the other.
3. Ranking roulette: pick the initial set L0..L4 (Supp. Tables 7-11) and watch the leaderboard re-order.
