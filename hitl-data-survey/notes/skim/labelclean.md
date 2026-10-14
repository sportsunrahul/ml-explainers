# Active label cleaning

Bernhardt, Castro, Tanno, Schwaighofer, Tezcan, Monteiro, Bannur, Lungren, Nori, Glocker, Alvarez-Valle, Oktay. "Active label cleaning for improved dataset quality under resource constraints". arXiv 2109.00574v2. Nature Communications 13, 1161 (2022).

Lens: method + benchmark paper; the method leads. Input = dataset with noisy (majority) labels, a relabelling budget B, a trained "selector" model; trained = the selector (vanilla CNN, co-teaching pair, or self-supervised encoder + linear head), optionally fine-tuned during cleaning; output = re-annotated dataset. Benchmark part: a simulation that draws new labels from each image's true label distribution, so relabel cost and annotator error are both modeled. Analysis claim: noisy validation labels can flip model selection (Table 3); falsifier = rankings that agree on noisy and clean labels.

Problem and key idea: experts can re-check only a few images; which ones first? Rank by "how wrong the current label looks" minus "how ambiguous the image is", so obvious errors (cheap to fix, one or two votes) come before hard cases that need many votes to reach a majority.

## Mechanism (Table 1 = the algorithm, src/paper_nat_comms.tex label `algo:sampleselection`; it is captioned as a table)
1. theta = TrainRobustModel(D) (selector).
2. All samples available; count = 0.
3. While count < B: pick j = argmax over available samples of Phi(x_i, l_i; theta) (Eq 2).
4. Repeat: draw one more one-hot label for j from its true distribution l_j (simulated annotator); count += 1; until a majority forms in j's label counts.
5. Move j to the cleaned set; update D.
6. Every b acquired labels, fine-tune theta on D (update helps: Supp. Table 3).
Each prioritized sample is reviewed "sequentially by different annotators until a majority is formed"; samples needing more votes cost more.

## Equations (src/paper_nat_comms.tex; numbers verified in paper.pdf)
Eq (1) objective, label `eqn:relabelling_objective`:
```latex
\max \; \frac{1}{N} \sum_{i=1}^N \mathbf{1}[\hat{y}_i = y_i] \quad \text{s.t.} \; \sum_{i=1}^N \lVert\hat{\mathbf{l}}_i\rVert_1 \leq B
```
Eq (2) scoring function (the ranking score), label `eqn:scoring_fn`:
```latex
\Phi(\mathbf{x}, \hat{\mathbf{l}}; \boldsymbol{\theta}) = \mathrm{CE}(\hat{\mathbf{l}}, p_{\boldsymbol{\theta}}) - \mathrm{H}(p_{\boldsymbol{\theta}})
```
Eq (3) noisiness term (unlabeled equation in .tex, numbered 3 in PDF):
```latex
\mathrm{CE}(\hat{\mathbf{l}}, p_{\boldsymbol{\theta}}) = -\mathbb{E}_{\hat{\mathbf{l}}/ \lVert\hat{\mathbf{l}}\rVert_1}\left[\log p_{\boldsymbol{\theta}}(\hat{y} \vert \mathbf{x})\right]
```
Eq (4) ambiguity term (numbered 4 in PDF):
```latex
\mathrm{H}(p_{\boldsymbol{\theta}}) = -\mathbb{E}_{p_{\boldsymbol{\theta}}(\hat{y} \vert \mathbf{x})}\left[\log p_{\boldsymbol{\theta}}(\hat{y} \vert \mathbf{x})\right]
```
Variables: N = dataset size; l_hat_i = vector of label counts per class collected so far for image i (starts with at least one label); y_hat_i = majority label (argmax of counts); y_i = true class; B = total number of annotations allowed; p_theta = selector's predicted class posterior; CE = cross-entropy from normalized label counts to the posterior (high = label disagrees with model = likely noisy); H = entropy of the posterior (high = ambiguous, needs many votes, so it LOWERS priority).
Sample difficulty in CIFAR10H (Methods): normalized Shannon entropy of the true label distribution above 0.3 = "difficult".

## Simulation of multiple annotators (Results, "Simulation of sequential relabelling"; Methods)
- New labels are sampled from each image's true label distribution (CIFAR10H: normalized histogram of about 51 crowd labels per image), so a relabel can itself be wrong and ambiguous images need more draws.
- Initial noisy labels on CIFAR10H: sample one label from a temperature-scaled version of that distribution; temperature tau controls the noise rate while keeping noise input-dependent.
- NoisyCXR: true label from RSNA radiologist adjudication; relabel distribution P(pneumonia opacity) = 1 or 0 for easy cases, 0.66 / 0.33 for difficult ones (low-confidence boxes, or readers who drew opacities on a negative case).
- Baselines: random selector (lower bound, what prior relabel campaigns did) and oracle (knows true distributions, picks easiest noisy samples first; upper bound); also a "minimal sampler" that always returns the true label. Budget B = expected relabels the oracle needs to clean all errors. 5 seeds.
- Metric: % of samples whose majority label is correct vs number of collected relabels; area under that curve (AUC).

## Datasets
- CIFAR10H: the CIFAR-10 test set (10k images) with on average 51.1 human labels per image (Methods); training noise eta = 15% (Fig 3b), 30% (Table 2), 50% (Supp).
- NoisyCXR (new benchmark): 26,684 NIH ChestXray images; noisy binary "pneumonia-like opacity" labels from NIH NLP-extracted labels (10% of consolidation/infiltration cases flipped to raise noise); true labels from the RSNA Kaggle adjudication; noise rate 12.6% (Methods, Table 4 label `table:xraylabels`: 1,296 + 2,056 wrong of 26,684). Fig 3a uses N = 13342 (the 50% training split), eta = 12.7%.

## Main results
- Headline (abstract): correct labels "up to 4x more effectively than typical random selection in realistic conditions". Which curve/budget gives 4x is not stated in the body: UNSURE.
- Body (Results, "Label cleaning can be done in a resource-effective manner"): the vanilla selector reaches 90% correct labels with 3x (NoisyCXR) or 2.5x (CIFAR10H) fewer re-annotations than random.
- Fig 3 legend AUCs. (a) NoisyCXR eta 12.7%: Oracle 0.9461, Random 0.8887, Vanilla 0.9125, Coteaching 0.9159, SSL+Coteaching 0.9194, Vanilla trained on clean labels ("model upper bound") 0.9256. (b) CIFAR10H eta 15%: Oracle 0.9504, Random 0.8772, Vanilla 0.9024, Coteaching 0.9161, Self-Supervision 0.9238. Entropy term not used for NoisyCXR in Fig 3a (binary labels, small effect).
- Table 2 (label `tab:classification_bald`), test accuracy (%) on a clean test set (N=50k, per the table footnote; presumably the CIFAR-10 train images, UNSURE) before / after relabelling 32.7% of a 5k training set with eta = 30%: (1) Vanilla selector, Eq 2, Vanilla classifier 64.1 / 68.4; (2) Vanilla selector, BALD score, Vanilla 64.1 / 68.3; (3) SSL selector, Eq 2, Vanilla 64.1 / 70.9; (4) SSL, Eq 2, SSL 78.7 / 80.3; (5) Vanilla, Eq 2, SSL 78.7 / 79.4; (6) Co-teaching, Eq 2, Co-teaching 66.5 / 68.8; (7) ELR only, no cleaning 67.0; (8) clean training Vanilla 73.6; (9) clean training SSL 80.7.
- Table 3 (label `tab:model_ranking_cifar`), evaluation-label noise (40%) on CIFAR10H eval set (50k), accuracy: SYM, M1 (co-teaching) True 73.32, Noisy 45.19, Cleaned 10% 54.37, Cleaned 20% 62.53; M2 (SSL) 80.16 / 48.93 / 58.45 / 67.42. IDN, M1 69.91 / 45.93 / 49.97 / 55.87; M2 65.76 / 46.50 / 49.90 / 55.28. Under IDN the noisy set ranks M2 first but true labels rank M1 first; at 10% cleaned M1 narrowly leads again (49.97 vs 49.90); at 20% the gap is clear (55.87 vs 55.28). After cleaning 10%, noise drops to 31.32% (IDN) and 30.25% (SYM); 86.8% (IDN) and 97.5% (SYM) of selected images were mislabeled.
- Text (Results, "Label noise undermines..."): 40% IDN noise, an SSL model measures 46.5% on noisy validation labels vs 65.8% on clean; a co-teaching model 45.9% noisy vs 69.9% clean, so the worse model would be chosen.

## Explicit link to active learning (Introduction; Results, "Sample scoring function impacts cleaning performance")
- Same act (prioritize samples for a human under a budget), different goal: active learning picks UNLABELED samples to raise model accuracy; active label cleaning re-annotates ALREADY-labeled samples to raise label correctness, useful for evaluation too (cites "re-active learning").
- The H term is the same quantity BALD-style active learning uses for aleatoric uncertainty, but here it is subtracted (deprioritize ambiguity); it is better estimated by marginalizing over an ensemble (Supp. Fig 6, Supp. Table 2).
- BALD as the ranking score gives similar test-accuracy gains (Table 2 rows 1-2: 68.4 vs 68.3) but much worse label-quality improvement (Supp. Fig 3), because BALD favors decision-boundary disagreement, not clear label noise.

## Hyperparameters a figure needs
Selectors: ResNet-50 encoder for all; SSL = BYOL pre-training, then frozen encoder + linear head ("SSL-Linear") with tanh logit clipping (alpha = 10) and label smoothing; co-teaching selector = average of both nets' posteriors. Difficulty threshold 0.3 (normalized entropy). Seeds: 5. Confusion matrices shown at temperature tau = 2 (Fig 6).

## Figures
- Fig 1: overview, noisy dataset sorted so clear mislabels come first under a fixed budget.
- Fig 2: ambiguity spectrum for CIFAR10H cat images; 2D sketch: clear noise sits among other-class neighbors, difficult cases sit near the decision boundary.
- Fig 3: % correct labels vs number of relabels, oracle/random/selectors, (a) NoisyCXR (b) CIFAR10H.
- Fig 4: CIFAR10H ranking by SSL-Linear, top-10% priority vs bottom-10% (ambiguous), each with CE and AMB scores and true label histograms.
- Fig 5: chest X-ray cases: correct catch, false flag (nodules mistaken for opacity), difficult case.
- Fig 6: noise models (X, Y, Y_hat, E dependence graphs) and CIFAR10H class confusion matrices, all vs difficult samples.

## Limitations and follow-ups
Stated: consistent (systematic) errors a model can learn may stay undetected; if noise is not bounded (more wrong than right), random may beat data-driven selection; majority vote assumes equal annotators (future: annotator expertise, choose next annotator, multi-label fusion); selector biases can leak into the ranking (mitigate with ensembles); when cleaning a test set, train the selector only on the test set and do not use it for evaluation; only imaging tested. Recommendation: clean evaluation data first, then training data.
Follow-up: CROWDLAB (sources/crowdlab) uses Active Label Cleaning as a baseline and adds per-annotator weights.

## Labeling decision
Given a fixed reviewer budget this sprint, which already-labeled images to send back for re-annotation, and in what order: highest CE minus entropy first; and spend budget on the validation set before the training set.

## Visual ideas
- Budget race: three curves (random, selector, oracle) of % correct labels vs relabels; a slider for budget B reads off how many fewer relabels the selector needs for a target accuracy.
- Score map: a 2D scatter of images by CE (x) and entropy (y); the Phi = CE - H contour shows the priority order; click a point to see its vote histogram and how many draws it takes to reach a majority.
- Noisy-validation trap: two models' accuracies on noisy vs cleaned validation labels; slide "% of eval set cleaned" and watch the ranking flip.
