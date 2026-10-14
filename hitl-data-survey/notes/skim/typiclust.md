# Hacohen, Dekel, Weinshall -- Active Learning on a Budget (TypiClust)

Citation: Guy Hacohen, Avihu Dekel, Daphna Weinshall. "Active Learning on a Budget: Opposite Strategies Suit
High and Low Budgets." arXiv 2202.02794v4. ICML 2022 (PMLR 162). Code: github.com/avihu111/TypiClust.
Source: sources/typiclust/src/main.tex.

Lens: analysis leads, method follows. Claim: the best query strategy flips with budget size; typical points help
at low budget, atypical (uncertain) points help at high budget. Measured by: a mixture-model theory, then accuracy
minus random as a function of labeled count, in three training frameworks. Falsified if: uncertainty sampling
beat random at 1-10 labels per class, or typical-point selection still won at large budgets.

## Core problem and key idea
Deep AL fails at "cold start": with few labels, uncertainty methods do no better than random, or worse. The
paper argues this is not only bad uncertainty estimates; at low budget the right target is the opposite one.
Key idea: pick a diverse set of the most typical points (dense regions of a self-supervised embedding).

## Theory: the phase transition (Sec 2)
Setup: the data support splits into regions R1 (easy, probability p) and R2 (hard). Each region has its own
independent learner. Error score of a learner trained on m points, Def. 1 (unnumbered, main.tex line 331):
`E_{\Dd}(m) = \E_{\iX^{m}\sim\Dd^m}\left[\E_{\hD(\iX^{m})\sim\Dd_h}\left[Er\left(h\left(\iX^{m}\right)\right)\right]\right]`
Assumptions: efficiency (E strictly decreasing) and realizability (E -> 0).
Eq (1), main.tex line 344: `E_{\Dd}(m) = p\cdot E_{\Dd_\pR}(m_\pR) + (1-p)\cdot E_{\Dd_\rR}(m_\rR).`
- p = probability of R1; m_1, m_2 = labeled points that fall in R1, R2; E_D1, E_D2 = per-region error scores.
Def. 2 (order R1 < R2): E_D1(m) = E(m), E_D2(m) = E(alpha m), with alpha < p/(1-p).
Eq (2), main.tex line 361: `E_\Dd(m)= p \cdot E(m_\pR) + (1-p) \cdot E(\qhard m_\rR).`
- alpha (\qhard) = learning efficiency of R2 relative to R1; small alpha = R2 is slow to learn.
Biased sampling (unnumbered, line 373): `m_\pR =p\cdot m+\qDel, \qquad m_\rR =\left(1-p\right)\cdot m-\qDel.`
- Delta > 0 oversamples R1 (typical); Delta = 0 is random; Delta < 0 oversamples R2 (atypical).
Theorem 1, threshold test (unnumbered boxed display, lines 388-392):
`\frac{E'\left(pm\right)}{E'\left(\qhard\left(1-p\right)m\right)} > \frac{\qhard\left(1-p\right)}{p} => oversample R1; < => oversample R2`
- E' = derivative of the error score; the left side compares marginal gains in the two regions.
Def. 3/4: "undulating" = R1 preferred for all m < z1 and R2 for all m > z2; "SP-undulating" = z1 = z2 (one switch).
Thm 2: E is undulating if proper, certain limits exist, and `-\log(E(x))\in\omega\left(\log(x)\right)`.
Eq (3), line 505 (Cor. 1, sufficient bound): `E(x) \leq k e^{-\nu x} \qquad\forall x\in\psn,` (k, nu > 0 constants).
Cor. 2: E(x) = k e^{-nu x} is SP-undulating (exactly one transition).
Thm 4/5: a mixture of two least-squares linear classifiers has error bounded by k e^{-nu m_j}, so it is undulating.
App. D.2, Fig 11: log error of 100 VGG-16 nets on CIFAR-100 super-classes looks bounded by an exponential.
1-NN analysis (Sec 2.4.2, App. C): loss approx. (k-1)/k [1 - sum_i f_D(X_i) v]; so pick max-density points that
are far apart (max density + diversity). This is the bridge from theory to the method.
Toy in Fig 1a text: E(m) = e^{-m}, p = 0.8, alpha = 0.1, Delta = +-0.01. My derivation (not in the paper): the
switch for E = e^{-x} is at m0 = ln(p/(alpha(1-p))) / (p - alpha(1-p)) = ln(40)/0.78 = approx. 4.7.

## Typicality and the TypiClust algorithm (Sec 3)
Eq (4), main.tex line 668:
`Typicality(x)=\bigg(\frac{1}{K}\sum_{x_i\in K\text{-NN}(x)}||x-x_i||_2\bigg)^{-1}.`
- x = embedding of an image; K-NN(x) = its K nearest neighbors in embedding space; K = 20 (footnote 1).
- High typicality = small mean distance to neighbors = dense region. No labels needed.
Steps:
1. Representation: self-supervised training on U0 u L0; use the penultimate layer as the feature space.
2. Cluster into |L_{i-1}| + B clusters, so at least B clusters hold no labeled point ("uncovered clusters").
3. From the B largest uncovered clusters, query the most typical point of each.
Alg. 1 (initial pool, L0 empty): embed U; cluster into B; for each cluster add argmax Typicality.
Variants: TPC_DC = SCAN deep clustering (first step only, no pseudo-label step). TPC_RP = SimCLR (CIFAR,
TinyImageNet) or DINO (ImageNet) + K-means.

## Hyperparameters a figure needs (App. F)
- SimCLR: ResNet18, MLP head to 128-d, 500 epochs, SCAN code; SGD momentum 0.9, lr 0.4 cosine, batch 512,
  wd 1e-4; embedding = L2-normalized 512-d penultimate layer.
- DINO: official ViT-S/16 ImageNet weights, L2-normalized penultimate layer.
- Clusters K = min(|L_{i-1}| + B, max_clusters); max_clusters = 500 (CIFAR-10/100), 1000 (TinyImageNet, ImageNet).
- scikit-learn KMeans if K <= 50, else MiniBatchKMeans. Typicality uses min{20, cluster_size} neighbors; drop
  clusters with < 5 points; each pick goes to the largest cluster among those with the fewest labeled points.
- Fully supervised eval: Munjal et al. TorchAL framework; ResNet18, SGD 0.9 Nesterov, lr 0.025 cosine, crop+flip;
  weights re-initialized each round; 5 AL rounds of B = M or B = 5M (M = number of classes); L0 empty;
  10 repetitions (CIFAR), 3 (ImageNet).
- Linear eval: one d x C layer on the embedding, lr 2.5, epochs x2. Semi-supervised: FlexMatch, 3 repetitions;
  CIFAR-10 WRN-28-2, 400k iterations, lr 0.03, batch 64, wd 5e-4.

## When uncertainty wins or loses vs random
- Fig 1b (CIFAR-10, accuracy minus random vs number of labels, log axis; file
  graphs/visulizations/all_budget_margin_scan_avihu_comparison_v2.png): Margin and BADGE are below random at the
  smallest budgets and above random from roughly 10^3 labels; TPC_DC is above random (peak approx. +2.6) until
  roughly 10^3 and below random after (approx. -1.8 near 4000-8000). Read from plot: approximate, UNSURE.
  Caption says "2 basic uncertainty-based strategies"; the legend shows Margin and BADGE.
- Sec 4.2.1, Fig 4: with 1-5 labels per class per round, "all other baseline AL methods perform on par with random
  selection or worse". Semi-supervised regime studied: "only 0.02%~1% of the data is labeled" (Sec 4.2.3).
- Sec 4.3.4, Fig 8: even oracle uncertainty fails at low budget. Uncertainty = margin of a VGG-19 trained on all of
  CIFAR-10; |L0| = 10, B = 10, 100 models. Read from plot (approx., UNSURE): at 10 labels Random approx. 18%,
  Oracle Margin approx. 14.8%, Oracle Reverse Margin (most certain) approx. 22.8%.

## Headline numbers (bar labels in the authors' figure PNGs)
Fig 6a, FlexMatch, CIFAR-10, 10 labels, 3 reps (graphs/semi_supervised/cifar10_10samples_ssl_comparison.png):
| Random 53.8 | Balanced 55.9 | Coreset 56.6 | TPC_RP 80.5 | TPC_DC 93.2 |
Abstract confirms 93.2% and "+39.4% over random" (93.2 - 53.8).
Fig 6b, CIFAR-100, 300 labels: Random 38.3 | Balanced 41.2 | Coreset 38.6 | TPC_RP 58.6 | TPC_DC 46.
Fig 6c, TinyImageNet, 1000 labels: Random 12.5 | Balanced 12.9 | Coreset 12.1 | TPC_RP 18.9 | TPC_DC 14.3.
Note: TPC_DC wins on CIFAR-10; TPC_RP wins on CIFAR-100 and TinyImageNet.
Fig 9, gain over random (best variant), in points (graphs/semi_supervised/ssl_vs_fully_combined.png):
| Dataset (labels) | Semi-sup | Self-sup embedding | Fully sup |
| CIFAR-10 (10) | 39.4 | 27.5 | 6.4 |
| CIFAR-100 (300) | 20.3 | 5.7 | 3.6 |
| TinyImageNet (1000) | 6.4 | 2.4 | 1.7 |
| ImageNet-50 (50) | - | 18.4 | 2.6 |
| ImageNet-100 (100) | - | 21.9 | 2.2 |
| ImageNet-200 (200) | - | 18.7 | 1.5 |

## What the figures show (one line each)
- Fig 1a: schematic; typical-AL curve above zero then below, atypical-AL the mirror; they cross at the transition.
- Fig 2: t-SNE of 30 clusters with the densest point per cluster marked; picks are nearly class-balanced.
- Fig 3: (a) typical + diverse, (b) atypical per cluster, (c) typical without clustering = near-duplicates.
- Fig 4/5: accuracy vs cumulative budget; both TPC variants on top, the rest bunched at or below random.
- Fig 7a: TPC still wins with a random initial pool. 7b: lowest TV distance to the true class distribution.
- Fig 7c: TPC_Rand, TPC_Inv, TPC_NoClust all lose; both clustering and density matter.

## Limitations (authors) and follow-ups
- The transition point depends on task and data; finding it is left open (Sec 5).
- SSL often assumes balanced data; on imbalanced CIFAR-10 TypiClust wins at low budget, loses at high (Sec 4.3.5).
- FlexMatch too costly for ImageNet. Under-performs in the high-budget regime by design.
- Follow-up: ProbCover (Yehuda et al. 2022, same group) replaces clustering + typicality with max coverage of
  delta-balls. Counterpoint: Luth et al. 2023 report no cold start except Entropy once classifiers are tuned.

## Labeling decision
When a new inspection app or camera has almost no labels (1-10 per class), send annotators the most typical image
of each embedding cluster, not the most uncertain. Switch to uncertainty sampling once the labeled pool is large.

## Visual ideas
1. Phase-transition toy: sliders for p, alpha, m with E = e^{-m}; plot error change for Delta = +-0.01 and watch it flip.
2. 2D embedding playground: blobs of points, budget slider; compare TypiClust, margin, and random picks under 1-NN.
3. Typicality heatmap with a K slider and a cluster-count slider; show the densest point per cluster lighting up.
