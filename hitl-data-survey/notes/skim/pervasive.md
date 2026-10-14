# Northcutt, Athalye, Mueller 2021 -- Pervasive Label Errors in Test Sets Destabilize ML Benchmarks

Citation: Curtis G. Northcutt, Anish Athalye, Jonas Mueller. "Pervasive Label Errors in Test Sets Destabilize Machine
Learning Benchmarks". arXiv 2103.14749v4 (7 Nov 2021). NeurIPS 2021 Datasets and Benchmarks Track.
Files: sources/pervasive/src/{identifying,validating,stability,expert,conclusion,appendix,macros}.tex.

## Lens
Analysis/benchmark paper (leads); the method (confident learning, CL) is reused from Northcutt et al. 2021 as a filter.
- Claim: test sets of 10 popular benchmarks contain at least 3.3% label errors on average; on corrected labels, smaller
  models can beat larger ones once the share of mislabeled test data rises a little.
- How measured: CL flags candidates -> 5 MTurk workers per candidate -> corrected test sets -> re-rank pre-trained models
  on original vs corrected labels, and simulate higher noise prevalence by subsampling benign examples.
- Falsifier: if humans rejected most CL flags (low validated %), or if model rankings on the correctable set matched the
  rankings on original labels, the "destabilize" claim would fail. Note Fig 3a: rankings on the full set do NOT change.

## Core problem and key idea
Everyone tunes and picks models by test accuracy, assuming test labels are correct. Key idea: let an algorithm (CL)
shortlist likely errors (Sec 3: "often as much as 90%" fewer items for humans to check), then show that errors are common and that larger models
fit the systematic errors, so the "best" model on noisy test labels can be the wrong one to deploy.

## Mechanism
1. Out-of-sample predicted probs P-hat (n x m): pre-train on train set, fine-tune on test set with cross-validation
   (or cross-validate on all data when no test split exists). Model per dataset in Table 1 col "Model".
2. CL estimates the joint Q of noisy label and true label; error fraction rho = 1 - trace(Q); flag top rho*n examples by
   normalized margin (Sec 3).
3. MTurk: each flagged example shown to 5 workers with the given label and the top CL-predicted label (order randomized),
   plus high-confidence examples of both classes (App B, Fig S1). Options: (1) given, (2) CL-predicted, (3) both, (4) neither.
4. "Error" if fewer than 3 of 5 workers agree the given label applies (agreement threshold = 3 of 5); else "non-error".
   Errors split into: correctable (majority picks CL label), multi-label (majority picks both), neither (majority picks
   neither), non-agreement (no majority) (Sec 4).
5. Sets (Sec 5): benign B, correctable C, unknown-label U, pruned P = D \ U = B u C. Noise prevalence N = |C|/|P|.
6. Re-rank 34 ImageNet and 13 CIFAR-10 pre-trained models on C (original vs corrected labels), then on reduced test sets
   with a fraction x of B removed, to find the N at which two models swap rank (Sec 5.2).
7. Expert check on ImageNet: 1 CL-flagged + 1 non-flagged image per class (934 + 1000 = 1934), >= 2 experts each (Sec 6).

## Equations
- Error fraction (identifying.tex, Sec 3, unnumbered):
  `\rho = 1 - \sum_{i \in [m]} \hat{p}(\tilde{y} \smalleq i , y^* \smalleq i )`; number of errors = rho * n.
  rho: fraction mislabeled; m: classes; y-tilde: given label; y*: true label; p-hat: estimated joint entry.
- Ranking score (Sec 3, unnumbered): `\hat{p}(\tilde{y} \smalleq i; \bm{x}) - \max_{j \neq i} \hat{p}(\tilde{y} \smalleq j; \bm{x})`
  (normalized margin; i = given label; lowest margin = most suspect).
- Confident joint (appendix.tex, App C, \nonumber; macro \cj = \bm{C}_{\tilde{y}, y^*}):
  `\bm{C}_{\tilde{y}, y^*} = \lvert \{\bm{x} \in \bm{X}_{\tilde{y} = i} : \,\, \hat{p} (\tilde{y} = j ;\bm{x}, \bm{\theta})  \ge t_j \} \rvert`
  X_{y~=i}: examples with given label i; theta: model; t_j: per-class threshold.
- Per-class threshold, Eq. (1) (App C): `t_j = \frac{1}{|\bm{X}_{\tilde{y}=j}|} \sum_{\bm{x} \in \bm{X}_{\tilde{y}=j}} \hat{p}(\tilde{y}=j; \bm{x}, \bm{\theta})`
  (average self-confidence of class j). Eq. (2) normalizes C into Q-hat so row sums match the given-label prior.
- Noise prevalence (stability.tex, Def., unnumbered): `N = \frac{\vert \mathcal{C} \vert}{\vert \mathcal{P} \vert}`;
  after removing fraction x of B: `N = \frac{|\mathcal{C}|}{|\mathcal{P}| - x|\mathcal{B}|}`.
- Expected accuracy on reduced set (Sec 5.2, unnumbered):
  `A(x; \mathcal{M}) = \frac{A_\mathcal{C}(\mathcal{M}) \cdot |\mathcal{C}| + (1-x) \cdot A_\mathcal{B}(\mathcal{M}) \cdot |B|}{|\mathcal{C}| + (1-x) \cdot |B|}`
  M: model; A_C, A_B: accuracy on correctable / benign sets (original or corrected); x in [0,1].

## Main results
Table 1 (identifying.tex), columns: Size | Model | CL guessed | MTurk checked | validated | estimated | % error
- MNIST 10,000 | 2-conv CNN | 100 | 100 (100%) | 15 | - | 0.15
- CIFAR-10 10,000 | VGG | 275 | 275 (100%) | 54 | - | 0.54
- CIFAR-100 10,000 | VGG | 2,235 | 2,235 (100%) | 585 | - | 5.85
- Caltech-256 29,780 | Wide ResNet-50-2 | 2,360 | 2,360 (100%) | 458 | - | 1.54 (whole dataset)
- ImageNet 50,000 (ILSVRC12 val) | ResNet-50 | 5,440 | 5,440 (100%) | 2,916 | - | 5.83
- QuickDraw 50,426,266 | VGG | 6,825,383 | 2,500 (0.04%) | 1870 | 5,105,386 | 10.12 (whole dataset)
- 20news 7,532 | TFIDF + SGD | 93 | 93 (100%) | 82 | - | 1.09
- IMDB 25,000 | FastText | 1,310 | 1,310 (100%) | 725 | - | 2.90
- Amazon Reviews 9,996,437 | FastText | 533,249 | 1,000 (0.2%) | 732 | 390,338 | 3.90 (whole dataset)
- AudioSet 20,371 | VGG | 307 | 307 (100%) | 275 | - | 1.35
- Average: "at least 3.3%" errors (abstract; mean of the 10 % values = 3.33, my arithmetic). "51% of the
  algorithmically-flagged candidates are indeed erroneously labeled, on average" (abstract).
Table 2 (validating.tex), non-errors | errors | non-agreement | correctable | multi-label | neither:
- CIFAR-10: 221 | 54 | 32 | 18 | 0 | 4.  ImageNet: 2524 | 2916 | 598 | 1428 | 597 | 293.
- CIFAR-100: 1650 | 585 | 210 | 318 | 20 | 37.  QuickDraw: 630 | 1870 | 563 | 1047 | 20 | 240.
Benchmark ranking on the correctable set (Sec 5.1, Table S1, top-1, threshold 3, |C| = 1428):
- resnet18 Acc@1 6.51, cAcc@1 82.42, rank 34 -> 1. nasnetlarge 19.61 -> 66.32, rank 1 -> 29. xception rank 2 -> 24.
  PyTorch resnet50 13.52 -> 73.74, rank 20 -> 2 (text says "20/24 -> 2/24", a typo for /34). alexnet 11.06 -> 58.96, rank 31 -> 34.
- Fig 3a: on the full ImageNet val set, removing errors does not change the ranking.
- CIFAR-10 (Table S2, |C| = 18): vgg11_bn Acc@1 27.78 -> cAcc@1 66.67, rank 11 -> 1; vgg19_bn 50.00 -> 38.89, rank 2 -> 11.
Noise-prevalence crossovers (Fig 4/5, threshold 3):
- ImageNet: base N = 2.9%. Caption (text-backed): "ResNet-50 and ResNet-18 ... cross at N = 9%", i.e. ResNet-18 wins
  after about +6%. Read from plot: the 9% pair is PyTorch resnet18 vs the Keras resnet50 line; 31% resnet18 vs Keras
  densenet169; 43% Keras nasnetlarge vs PyTorch resnet50/resnet18; original-label panel has one cross at 13%
  (resnet18 vs vgg11). (pairings UNSURE, inferred from line colors)
- CIFAR-10: base N = 0.2%. Abstract (text-backed): "VGG-11 outperforms VGG-19 if the prevalence ... increases by just
  5%". Read from plot: corrected-panel crosses at 6% (vgg11 vs vgg19), 7% (vgg11 vs vgg16), 12% (vgg11 vs vgg13);
  original panel cross at 2%. (pairings UNSURE; legend says vgg11..vgg19 while Table S2 lists *_bn variants)
Expert review (Table 3, ImageNet), non-errors | errors | correctable | multi-label | neither | time:
- CL (MTurk) 57.9 | 42.2 | 24.7 | 11.1 | 6.4 | 5 s.  CL (expert) 58.7 | 41.4 | 17.7 | 13.1 | 10.6 | 67 s.
- non-CL (expert) 84.0 | 16.0 | 3.2 | 9.1 | 3.7 | 67 s. CL-flagged is 2.6x as likely to be wrong; ImageNet val is
  "closer to 20% label errors" once missed non-flagged errors (~14%) are added (Sec 6).
- Quirk: MTurk row % appear to exclude the 103 non-agreement cases (205/831 = 24.7%), my arithmetic from Table S3.

## Settings a figure would need
5 workers per item; agreement threshold 3 of 5 (also 4 and 5 in App F: |C| = 1428 / 960 / 468); 34 ImageNet models
(PyTorch 1.0, Keras 2.2.4 checkpoints), 13 CIFAR-10 models; MTurk pay $7.20/h at ~5 s/image, total $1623.29 (App B).
Experts agreed on 77% of images first pass; time 13x MTurk (Sec 6) vs 14x (App G), an inconsistency in the source.

## What the figures show
- Fig 1: one example per category (correctable, multi-label, neither, non-agreement) for each image dataset.
- Fig 2: CL false positives (part-vs-whole crop, unusual viewpoint, ambiguous doodle).
- Fig 3: original vs corrected accuracy scatter: (a) full val set ranks unchanged; (b, c) correctable set ranks invert.
- Fig 4/5: accuracy vs noise prevalence lines per model, vertical lines at rank swaps; top = original, bottom = corrected.

## Practical recommendations (no section has this title; they are spread out)
- End of Sec 1: correct at least the test labels; use simpler/smaller models when your labels are noisier than benchmarks.
- End of Sec 5.2: separate corrected vs original accuracy; spend more annotation budget on test-label quality.
- Sec 6: with a limited review budget, use CL to prioritize which examples humans check.
- Sec 7: judge models on correctly labeled test sets; curate high-quality test labels even if train labels stay cheap.

## Limitations (stated) and follow-ups
- Error counts are lower bounds: only CL-flagged items were checked; experts estimate ImageNet val ~20% wrong (Sec 6).
- Corrected labels are not perfect even at 5 of 5 agreement (Sec 4.1); MTurk over-corrects "neither" cases (Sec 6).
- Cause of the high-capacity effect (train-noise fit, val-set tuning, or label shift) is open; how to split a verification
  budget between train and test is open (Sec 7). Follow-ups: ReaL labels (2006.07159) for multi-label ImageNet eval;
  active label cleaning (ch. 12) for budgeted review.

## Labeling decision
Which images to send for human re-review before trusting an eval set: rank by CL margin, review the top slice with 5
votes, and only then compare candidate models on the corrected eval set.

## Visual ideas
1. Prevalence slider: drag N from 2.9% to 50%; lines for ResNet-18 / ResNet-50 / NASNet re-rank live (built from A(x; M)).
2. Mini MTurk task: reader votes given / predicted / both / neither on 5 items; tally into the four outcome bins.
3. Funnel: 50,000 val images -> 5,440 CL flags -> 2,916 validated -> 1,428 correctable, with bar widths to scale.
