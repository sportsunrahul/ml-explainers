# Confident Learning (CL)

Northcutt, Jiang, Chuang. "Confident Learning: Estimating Uncertainty in Dataset Labels". arXiv 1911.00068v6. JAIR vol. 70, 2021, pp. 1373-1411.

Lens: method paper first (inputs = out-of-sample predicted probabilities + noisy labels; output = a count matrix, a joint noise distribution, and a list of suspect examples; nothing new is trained). It is also an analysis paper: it claims label errors exist in "clean" benchmarks (ImageNet, MNIST). Falsifier for the method claim: a model-agnostic count that does no better than argmax disagreement (C_confusion) on Table 2/4.

Problem and key idea: given only noisy labels and a model's held-out probabilities, find which examples are mislabeled and as what. Key idea: count, per (given label i, likely true label j), the examples whose probability for j is above that class's own average self-confidence t_j. Per-class thresholds (not argmax) make the count robust to over-confident or rare classes.

## Mechanism
1. Get out-of-sample probabilities P_hat[k,i] by K-fold cross-validation (K=4 for all CIFAR/ImageNet runs, Sec 3.2 and Sec 5 intro; 5-fold for Amazon Reviews, Sec 5.3). A pretrained model that never saw the data also works (WebVision, Sec 5.4: 3 minutes on a laptop).
2. Per-class threshold t_j = mean self-confidence of examples labeled j (Eq 2).
3. Confident joint C: put each example in bin (given i, guessed j) if p(j) >= t_j; collisions go to the largest above-threshold class (Eq 1). Examples above no threshold are not counted.
4. Calibrate C to the joint Q_hat so row sums match observed label counts and all entries sum to 1 (Eq 3). Marginals give the latent prior, noise matrix, inverse noise matrix.
5. Rank and prune with one of five methods (below), then retrain on the kept data with class weights Q_hat_{y*}[i] / Q_hat[i][i].
"Theoretically, CL requires no hyper-parameters to find label errors"; K-fold is the only practical one (K=4 fixed, not tuned; Sec 3.2 "Learning with Noisy Labels").

## Equations (src/methods.tex unless noted; macros expanded from src/macros.tex; numbers verified in paper.pdf)
Eq (1), confident joint, methods.tex label `eqn_paper_confident_joint`:
```latex
\bm{C}_{\tilde{y}, y^*}[i][j] \coloneqq \lvert \hat{\bm{X}}_{\tilde{y}=i,y^*=j} \rvert \quad \text{where} \quad
\hat{\bm{X}}_{\tilde{y}=i,y^*=j} \coloneqq \left\{ \bm{x} \in \bm{X}_{\tilde{y}=i} : \hat{p}(\tilde{y}=j;\bm{x},\bm{\theta}) \ge t_j,\;
j = \operatorname*{argmax}_{l \in [m] : \hat{p}(\tilde{y}=l;\bm{x},\bm{\theta}) \ge t_l} \hat{p}(\tilde{y}=l;\bm{x},\bm{\theta}) \right\}
```
Eq (2), per-class threshold, label `eqn_paper_threshold`:
```latex
t_j = \frac{1}{|\bm{X}_{\tilde{y}=j}|} \sum_{\bm{x} \in \bm{X}_{\tilde{y}=j}} \hat{p}(\tilde{y}=j; \bm{x}, \bm{\theta})
```
Eq (3), calibration to the joint, label `eqn_calibration`:
```latex
\hat{\bm{Q}}_{\tilde{y}=i,y^*=j} = \frac{\frac{\bm{C}_{\tilde{y}=i,y^*=j}}{\sum_{j \in [m]} \bm{C}_{\tilde{y}=i,y^*=j}} \cdot \lvert \bm{X}_{\tilde{y}=i} \rvert}
{\sum_{i \in [m], j \in [m]} \left( \frac{\bm{C}_{\tilde{y}=i,y^*=j}}{\sum_{j' \in [m]} \bm{C}_{\tilde{y}=i,y^*=j'}} \cdot \lvert \bm{X}_{\tilde{y}=i} \rvert \right)}
```
Unnumbered (methods.tex "Label noise characterization"): `\hat{\bm{Q}}_{y^*=j} \coloneqq \sum_i \hat{\bm{Q}}_{\tilde{y}=i,y^*=j}`, `\hat{\bm{Q}}_{\tilde{y}=i \vert y^*=j} \coloneqq \hat{\bm{Q}}_{\tilde{y}=i,y^*=j} / \hat{\bm{Q}}_{y^*=j}`.
Unnumbered, normalized margin used for ordering (methods.tex "Which CL method to use?"): `\hat{p}(\tilde{y}=i;\bm{x},\bm{\theta}) - \max_{j \neq i} \hat{p}(\tilde{y}=j;\bm{x},\bm{\theta})`.
Eq (4), per-example diffracted noise, src/theory.tex label `eqn:piecewise_error` (p* = ideal probability):
```latex
\epsilon_{\bm{x},\tilde{y}=j} \sim \begin{cases}
\mathcal{U}(\epsilon_j + t_j - p^*_{\bm{x},\tilde{y}=j},\; \epsilon_j - t_j + p^*_{\bm{x},\tilde{y}=j}] & p^*_{\bm{x},\tilde{y}=j} \ge t_j \\
\mathcal{U}[\epsilon_j - t_j + p^*_{\bm{x},\tilde{y}=j},\; \epsilon_j + t_j - p^*_{\bm{x},\tilde{y}=j}) & p^*_{\bm{x},\tilde{y}=j} < t_j \end{cases}
```
Variables: m = number of classes; X_{y~=i} = examples whose given label is i; p_hat(y~=j; x, theta) = held-out model probability of class j; t_j = average self-confidence of class j; C = m x m integer counts (rows = given label, columns = guessed true label); |X_{y~=i}| = count of given label i; Q_hat = estimated joint p(y~, y*); epsilon_j = mean error of class j; p* = error-free probability.

## Rank and prune: the five methods (Sec 3.2)
- CL baseline 1, C_confusion: flag x_k if given label != argmax_j p_hat(j). This IS the "argmax" method; the paper has five methods, not six. Similar to one iteration of INCV.
- CL method 2, C_{y~,y*}: flag every example in an off-diagonal bin of Eq 1. Default method (matches Thm 2).
- CL method 3, Prune by Class (PBC): for class i remove the n * sum_{j != i} Q_hat[i][j] examples with lowest self-confidence p_hat(y~=i).
- CL method 4, Prune by Noise Rate (PBNR): for each off-diagonal (i,j) remove n * Q_hat[i][j] examples labeled i with largest margin p_hat(j) - p_hat(i).
- CL method 5, C+NR: remove only if both PBC and PBNR remove it (set intersection).
Ordering of found errors for display: normalized margin (ImageNet, WebVision, Amazon figures); MNIST Fig 6 is ordered by self-confidence.

Theory in one line (Sec 4): Thm 1, ideal probabilities and each diagonal of Q_{y~|y*} maximizes its row and column, gives exact errors; Cor 1, per-class diffracted (p_hat = eps1_j * p* + eps2_j), no collisions, diagonal maximizes its row, still exact (C_confusion fails here); Thm 2, per-example diffracted (Eq 4), still exact.

## Main results
Setting: CIFAR-10 train labels flipped by random class-conditional noise matrices; noise = 20/40/70%; sparsity = fraction of zero off-diagonals in Q (0, 0.2, 0.4, 0.6); ResNet-50; metric = test accuracy (%) on unaltered test set; CL rows mean of 10 trials (std in Table 3, none above 2%). Source: Table 2 (experiments.tex label `table:cifar10_benchmark`, uncommented version; an older commented-out table in the .tex has different numbers and was not used).

| Table 2 row | 20%: s=0 / 0.2 / 0.4 / 0.6 | 40%: s=0 / 0.2 / 0.4 / 0.6 | 70%: s=0 / 0.2 / 0.4 / 0.6 |
|---|---|---|---|
| CL: C_confusion | 89.6 / 89.4 / 90.2 / 89.9 | 83.9 / 83.9 / 83.2 / 84.2 | 31.5 / 39.3 / 33.7 / 30.6 |
| CL: PBC | 90.5 / 90.1 / 90.6 / 90.7 | 84.8 / 85.5 / 85.3 / 86.2 | 33.7 / 40.7 / 35.1 / 31.4 |
| CL: C_{y~,y*} | 91.1 / 90.9 / 91.1 / 91.3 | 86.7 / 86.7 / 86.6 / 86.9 | 32.4 / 41.8 / 34.4 / 34.5 |
| CL: C+NR | 90.8 / 90.7 / 91.0 / 91.1 | 87.1 / 86.9 / 86.7 / 87.2 | 41.1 / 41.7 / 39.0 / 32.9 |
| CL: PBNR | 90.7 / 90.5 / 90.9 / 90.9 | 87.1 / 86.8 / 86.6 / 87.2 | 41.0 / 41.8 / 39.1 / 36.4 |
| INCV | 87.8 / 88.6 / 89.6 / 89.2 | 84.4 / 76.6 / 85.4 / 73.6 | 28.3 / 25.3 / 34.8 / 29.7 |
| Mixup | 85.6 / 86.8 / 87.0 / 84.3 | 76.1 / 75.4 / 68.6 / 59.8 | 32.2 / 31.3 / 32.3 / 26.9 |
| SCE-loss | 87.2 / 87.5 / 88.8 / 84.4 | 76.3 / 74.1 / 64.9 / 58.3 | 33.0 / 28.7 / 30.9 / 24.0 |
| MentorNet | 84.9 / 85.1 / 83.2 / 83.4 | 64.4 / 64.2 / 62.4 / 61.5 | 30.0 / 31.6 / 29.3 / 27.9 |
| Co-Teaching | 81.2 / 81.3 / 81.4 / 80.6 | 62.9 / 61.6 / 60.9 / 58.1 | 30.5 / 30.2 / 27.7 / 26.0 |
| S-Model | 80.0 / 80.0 / 79.7 / 79.1 | 58.6 / 61.2 / 59.1 / 57.5 | 28.4 / 28.5 / 27.9 / 27.3 |
| Reed | 78.1 / 78.9 / 80.8 / 79.3 | 60.5 / 60.4 / 61.2 / 58.6 | 29.0 / 29.4 / 29.1 / 26.8 |
| Baseline | 78.4 / 79.2 / 79.0 / 78.2 | 60.2 / 60.8 / 59.6 / 57.3 | 27.0 / 29.7 / 28.2 / 26.8 |

Note: the text says best performance is by C_{y~,y*} "across all noise and sparsity settings", but in the table C+NR/PBNR win at 40% and 70%. Co-Teaching here is run by CL's authors on asymmetric sparse noise; not comparable to Co-teaching's own tables.
Finding errors, Table 4 (label `table:cifar10_label_error_measures`), CIFAR-10, cols noise 20% s=0, 20% s=0.6, 40% s=0, 40% s=0.6: C_{y~,y*} precision 67/70/78/77, recall 86/88/91/84; C_confusion precision 56/58/74/70, recall 98/97/97/90 (high recall, low precision). PBNR F1 77/79/85/80.
Joint estimation: RMSE .004 to true Q averaged over sparsities at 20%/40% noise; C_confusion normalized by Eq 3 gives .006 (Sec 5.1, Table in appendix). Fig 2: 40% noise, 60% sparsity, over 80% of Q entries within .005.
Time: CL under 3 hours vs INCV over 20 hours on CIFAR (Sec 5.1).
ImageNet (ILSVRC2012 train, ResNet-50, 4-fold CV):
- Table 5, top off-diagonal C entries: projectile->missile 645 (C_confusion 494, Q_hat 0.00050); tub->bathtub 539; breastplate->cuirass 476; green_lizard->chameleon 437; chameleon->green_lizard 435; missile->projectile 433; maillot->maillot (duplicated class n03710637 vs n03710721) 417; horned_viper->sidewinder 416; corn->ear 410; keyboard->space_bar 407.
- Human check of 500 random CL: PBNR issues: 58% were multi-label, ontological, or real errors (Sec 5.2).
- Fig 4/5: removing 20..100% of CL-flagged issues before training beats removing the same number of random examples; text: "reduce the size ... by 10% while still moderately improving" accuracy. Exact accuracy deltas only in the figure: UNSURE.
- Exact count of flagged ImageNet issues per method: only in commented-out LaTeX; not reported here (UNSURE).
Amazon Reviews (logistic regression on fastText, 5-fold): Table 7, N=1000K, test split "10th", 50 epochs: Baseline 84.4 +/- 0.04, CL: PBC/PBNR/C+NR 90.2.

## Figures (redraw ideas)
- Fig 1: flow from noisy labels + probabilities -> confident joint counts -> Q_hat -> pruned clean data.
- Fig 2: three 10x10 heatmaps: true Q, estimated Q_hat, absolute difference (40% noise, 60% sparsity).
- Fig 3: top-32 ImageNet issues (caption says 32, body text says "top 16"); red = error, green = ontological, blue = multi-label.
- Fig 4 (ResNet-18) / Fig 5 (ResNet-50): accuracy vs fraction of CL issues removed, CL vs random removal; panels all classes, 20 noisiest, foxhound, maillot.
- Fig 6: MNIST train label errors; Fig 7: WebVision issues.

## Limitations and follow-ups
Stated: assumes class-conditional noise p(y~|y*;x) = p(y~|y*) (not instance-dependent); theory needs "no label collisions" and diagonal-dominant noise; flagged items mix multi-label and ontology issues with errors; mean threshold chosen for theory, percentile left to future work; future work: CNWL benchmark, non-NN models, test-set errors (Northcutt 2021 "pervasive", in sources/pervasive), dynamic cleaning during training.
Follow-ups: CROWDLAB (sources/crowdlab) extends cleanlab to multiple annotators.

## Labeling decision
Before a training round, which already-labeled images go back to a reviewer queue: run 4-fold CV, compute the confident joint, send the off-diagonal examples (ordered by normalized margin) to review; the largest off-diagonal cells point at confusing class definitions to fix in the label guide.

## Visual ideas
- Threshold slider: a scatter of per-example probabilities for two classes; drag t_j and watch examples move between C cells; compare with argmax (C_confusion) under an over-confident class.
- Noise x sparsity grid: pick a cell, see the true Q heatmap and how many errors each of the five prune methods removes (precision/recall bar).
- Confident-joint builder: a 4x4 grid that fills one example at a time; click an off-diagonal cell to see its images and the calibrated Q_hat.
