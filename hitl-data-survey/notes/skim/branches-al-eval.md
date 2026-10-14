# Branch papers: AL evaluation, uncertainty definitions, low-budget and batch selection

Spine links: Munjal et al. 2022 (munjal.md) and TypiClust 2022 (typiclust.md). Chapters 2, 4, 5, 13.

## Lewis & Gale 1994 (root of uncertainty sampling)
Citation: David D. Lewis, William A. Gale. "A Sequential Algorithm for Training Text Classifiers." arXiv
cmp-lg/9407020 (24 Jul 1994). SIGIR 94, Springer-Verlag, pp. 3-12 (venue from the PDF header). PDF only.
Role: the origin. Replaces random sampling and relevance sampling (query the most likely positives) with
querying the examples whose class is least certain under one probabilistic classifier.
Definition used (Sec 4.3): score every unlabeled example with P(C|w) and select the ones with "estimated P(C|w)
values closest to 0.5, since 0.5 corresponds to the classifier being most uncertain of the class label." They
take the b/2 closest above 0.5 and the b/2 closest below 0.5, so at most half the batch can be exact duplicates.
Loop (Fig 1, PDF): create an initial classifier; while the teacher is willing: apply the classifier to each
unlabeled example; find the b least certain; have them labeled; retrain on all labels. (Sec 3 text calls this
"Figure 3"; the PDF labels it Figure 1.) Classifier: Eq (6), logistic regression on the summed log-likelihood ratio.
Setup: 319,463 AP titles train / 51,991 test; 10 rare categories (freq 0.0007-0.0049, Table 1); 3 random positive
seeds, then 249 iterations of b = 4 (999 labels); 10 runs per category.
Key number: up to 500-fold fewer labels than random for equal effectiveness (abstract, Sec 7). Table 2, row
"boxoffice" (ligature lost in PDF text), mean F1: 3+996 uncertainty .700; 3+7 random .222; 3+996 relevance .481;
full 319,460 .647.
Caveats in the paper: one classifier underestimates uncertainty and is biased by a nonrepresentative model (Sec 2);
the seed classifier decides how fast rare positives are found (Sec 3, 7); SDs reach 10% of the mean (Sec 7).
Relevance: a rare-defect class with random labeling yields almost no positives; this is the baseline rule for
chapter 2 ("send the images the model is closest to 50/50 on").

## Settles 2009 (survey; definitions and caveats)
Citation: Burr Settles. "Active Learning Literature Survey." Computer Sciences Technical Report 1648, University
of Wisconsin-Madison, 2009 (PDF says "Updated on: January 26, 2010"). PDF only.
Role: the standard reference that names the three uncertainty measures and lists when AL fails.
Definitions, Sec 3.1, pp. 12-13. The displays are unnumbered; transcribed from the PDF text:
- Least confident: `x^*_{LC} = \argmax_x 1 - P_\theta(\hat{y}|x)`, with `\hat{y} = \argmax_y P_\theta(y|x)`.
  Read as the expected 0/1 loss (the model's belief it will mislabel x).
- Margin (attributed to Scheffer et al. 2001): `x^*_{M} = \argmin_x P_\theta(\hat{y}_1|x) - P_\theta(\hat{y}_2|x)`,
  with y_1, y_2 the first and second most probable labels.
- Entropy: `x^*_{H} = \argmax_x - \sum_i P_\theta(y_i|x) \log P_\theta(y_i|x)`.
- For binary tasks all three equal "posterior closest to 0.5". Fig 5 (p. 14): heatmaps on a 3-class simplex.
  Empirical comparisons give "mixed results"; entropy suits log-loss, LC and margin suit classification error.
Caveats:
- Sec 3.6, Fig 7 (pp. 25-26): uncertainty sampling can query outliers on the boundary. Fix: information density,
  `x^*_{ID} = \argmax_x \phi_A(x) \times ( \frac{1}{U} \sum_{u=1}^{U} sim(x, x^{(u)}) )^\beta`.
- Sec 4.1 (p. 27), the sampling-bias caveat: the AL-built labeled set is "a biased distribution, not drawn i.i.d.
  from the underlying natural density" and is tied to the model class that chose it. The exact phrase "sampling
  bias" does not appear in the PDF text. Same section: Schein & Ungar 2007 saw AL need more labels than passive;
  Guo & Schuurmans 2008 saw myopic batch queries do worse than random.
- Sec 6.1 (p. 35): "Q-best" batches fail because the top items overlap in information.
- Sec 6.2 (p. 36): noisy oracles. Sec 6.6 (p. 43): if the model class may change, random sampling "may be more
  advisable" for pilot studies.
Relevance: chapter 2 definitions; Sec 4.1 and 6.6 are the classical form of chapter 5's warnings.

## Mittal et al. 2019 ("Parting with Illusions")
Citation: Sudhanshu Mittal, Maxim Tatarchenko, Ozgun Cicek, Thomas Brox. "Parting with Illusions about Deep
Active Learning." arXiv 1912.05361v1, 2019. No venue in abs.html (UNSURE).
Role: direct predecessor of Munjal. Adds modern augmentation (AutoAugment, cutout) and semi-supervised learning
(UDA) to the evaluation; also tests low budgets and semantic segmentation. WRN-28-2; 3 trials (2 for ensembles).
Key number: Sec 3.2 (augmentation subsection), Fig 2, CIFAR-10: without augmentation Learning Loss is +3.2% over
random; with augmentation "all the AL methods show similar performance within a range of 0.4%".
Also: SSL-Random with 5K labels nearly matches AL methods at 20K (Fig 4, 5). At 250/500 labels per cycle, no AL
method consistently beats random (Fig 6, 7).
Relevance: before ranking query rules, a team fixes augmentation and considers training on unlabeled frames.

## Beck et al. 2021 (DISTIL evaluation)
Citation: Nathan Beck, Durga Sivasubramanian, Apurva Dani, Ganesh Ramakrishnan, Rishabh Iyer. "Effective
Evaluation of Deep Active Learning on Image Classification Tasks." arXiv 2106.15324v3, 2021. Template is
neurips_data_2021 (Datasets and Benchmarks style); acceptance not stated in the PDF (UNSURE).
Role: follows Munjal with a unified toolkit (DISTIL) but reports labeling efficiency (labels saved vs random at
equal accuracy). With augmentation + SGD, BADGE no longer beats entropy or least confidence; BADGE wins when the
pool is made redundant; fewer examples per class means less room for AL; seed set and batch size matter little.
Key number: "AL techniques are 2x to 4x more label-efficient compared to RS with the use of data augmentation"
(abstract); "1.8x to 2x on CIFAR-10 but only 1.3x on CIFAR-100" (Sec 2, P6).
Relevance: redundant pools (video frames from fixed cameras) favor diversity-aware batch selection.

## Kirsch et al. (stochastic batch acquisition)
Citation: Andreas Kirsch, Sebastian Farquhar, Parmida Atighehchian, Andrew Jesson, Frederic Branchaud-Charron,
Yarin Gal. "Stochastic Batch Acquisition: A Simple Baseline for Deep Active Learning." arXiv 2106.12059v3.
TMLR (09/2023, PDF header).
Role: changes top-K batch selection (BALD, entropy) to sampling with Gumbel-perturbed scores. Soft-rank Eq (12),
softmax Eq (13), power Eq (14)-(15): `s^\text{power}(i) := \log s(i) + \epsilon_i`, epsilon_i ~ Gumbel(0, 1/beta),
equivalent to sampling p(i) proportional to s(i)^beta. beta = 1 by default. Rivals BatchBALD and BADGE.
Key number: Table 1, acquisition runtime in seconds at K = 10: top-K BALD 0.2 +- 0.0; PowerBALD 0.2 +- 0.0;
BatchBALD 566.0 +- 17.4; BADGE 9.2 +- 0.3. Header superscripts (mean accuracy, Repeated-MNIST): 80%, 90%, 90%, 86%.
Relevance: a platform that sends K images per labeling round can drop near-duplicate picks at no extra compute.

## Yehuda et al. 2022 (ProbCover)
Citation: Ofer Yehuda, Avihu Dekel, Guy Hacohen, Daphna Weinshall. "Active Learning Through a Covering Lens."
arXiv 2205.11320v3. NeurIPS 2022.
Role: successor to TypiClust (same group). Replaces "cluster + most typical" with greedy max coverage of
delta-balls in a self-supervised embedding. Eq (1) bounds 1-NN error by (1 - coverage) + (1 - purity(delta));
Eq (2): `\underset{L\subseteq X; |L|=b}{\text{argmax}} P(\bigcup_{x\in L}B_\delta(x))`. Dual view: Coreset fixes
coverage = 1 and minimizes delta (high budget). delta = largest value with purity >= alpha = 0.95, where purity
uses k-means pseudo-labels.
Key number: Fig 6a, FlexMatch, CIFAR-10, 10 labels (bar labels, figures/empirical/flexmatch/
cifar10_flexmatch_10_examples.png): Random 53.80, Coreset 56.57, TypiClust 93.19, ProbCover 94.10.
Relevance: a cold-start labeling plan with one tunable knob (purity alpha) instead of a cluster count.

## Luth et al. 2023 (pitfalls of AL evaluation)
Citation: Carsten T. Luth, Till J. Bungert, Lukas Klein, Paul F. Jaeger. "Navigating the Pitfalls of Active
Learning Evaluation: A Systematic Framework for Meaningful Performance Assessment." arXiv 2301.10625v3.
NeurIPS 2023.
Role: successor to Mittal, Munjal, and Beck. Names five pitfalls: P1 data distribution, P2 starting budget,
P3 query size, P4 classifier configuration, P5 alternative training paradigms. Replaces Munjal's per-iteration
AutoML with light tuning on the starting budget. Findings (Sec 4, 5): BADGE never falls below random; AL gains are
larger on class-imbalanced data; Self-SL pretraining helps; FixMatch fails to generalize to imbalanced roll-out
sets (MIO-TCD, ISIC-2019) and trains approx. 80x slower than standard training.
Key number: Sec 4, P4: tuning the classifier is worth approx. 5x more labels (approx. 44% at 200 samples vs the
same accuracy at 1k samples in Bengar et al.; dataset not named there, UNSURE).
Relevance: on imbalanced defect data, AL pays off most; BADGE is the safe default query rule.

## Ren et al. (survey of deep AL)
Citation: Pengzhen Ren, Yun Xiao, Xiaojun Chang, Po-Yao Huang, Zhihui Li, Brij B. Gupta, Xiaojiang Chen, Xin Wang.
"A Survey of Deep Active Learning." arXiv 2009.00236v2 (Dec 2021). Journal not printed in the source; ACM
template; ACM Computing Surveys is UNSURE.
Role: taxonomy of deep AL after Settles: query-strategy optimization, labeled-data expansion (pseudo-labels,
generative augmentation), generic frameworks, stopping rules, applications. Sec 3 names three challenges: softmax
is over-confident (softmax response "unreliable", can be worse than random), too few labels for deep nets, and
AL vs DL pipeline mismatch.
Key number: none headline; footnote: about 270 DBLP hits screened, references updated to November 2020.
Relevance: map for chapter 13; source for "softmax confidence is not trustworthy" in chapter 3.
