# al-random: chapter 5 (ch-random) and the active-learning half of the timeline

Claims file: notes/claims/al-random.json (246 claims). Claim ids are in brackets. Papers appear in time order.
Prose values in the claims file that contain math use $...$ for inline math. Equation values are bare LaTeX.

## Settles 2009 (branch; Sec 4.1 and Sec 6.6 only; version: UW-Madison CS TR 1648, updated 2010-01-26)
- Problem: an AL-built training set is chosen by one model, so it is not a random sample of the data.
- Sec 4.1, p. 27: the labeled instances are "a biased distribution, not drawn i.i.d. from the underlying natural
  density" [settles-biased-quote]. The set is tied to the model class that chose it. The exact words "sampling
  bias" do not occur.
- Sec 6.6, p. 43: if the model class can change, random sampling "at least for pilot studies" may be more advisable
  [settles-pilot-quote].
- Chapter 5 use: the classical form of reason (3). A selected set cannot be the evaluation set. Keep a random slice.

## Mittal et al. 2019, "Parting with Illusions" (branch; 1912.05361v1)
- Delta from the 2018-2019 AL papers: those papers trained with no augmentation or only flips. Mittal adds modern
  augmentation (AutoAugment, cutout, flip, crop) and semi-supervised training to the evaluation.
- Result, Sec 3.2.2 and Fig 2, CIFAR-10: without augmentation, Learning Loss is 3.2% above random
  [illusions-noaug-32]. With augmentation, "all the AL methods show similar performance within a range of 0.4%"
  [illusions-aug-04]. The ranking also changes on CIFAR-100.
- Lesson: augmentation lifts all methods and erases most of the gap between them.
- Limitation: 3 trials, one architecture (WRN-28-2), no venue stated [illusions-cite-venue]. Munjal repeats the
  test with 25 runs per method and with statistical tests.

## Munjal et al., "Towards Robust and Reproducible Active Learning" (spine; 2002.09564v3, CVPR 2022)
Core problem. Published AL papers disagree. The same method gets different accuracy in different papers. The
random baselines in those papers are also weak: the Core-set paper's random baseline on CIFAR10 with VGG16 is
60 / 68 / 71 / 74 at 10 / 20 / 30 / 40% labels [munjal-st1-coreset-rand-10 .. -40]. Munjal's own random baseline
is 69.16 / 77.34 / 80.91 / 82.05 [munjal-st1-rsb-vgg-10 .. -40].

What prior methods did, and why that hurt. Prior work tuned hyperparameters once, on the initial labeled set, and
fixed them for all rounds (Sec 4). It ran few repeats and used little regularization. So a small gain over random
could come from chance or from a weak baseline.

Key idea. Run every method in one code base, under the same conditions. Re-tune at every round. Repeat over initial
sets and seeds. Add strong regularization. Then test whether "AL beats random" survives.

Mechanism (protocol), step by step.
1. Split CIFAR into train, validation, test. Model selection uses the validation set (Sec 5).
2. Draw an initial labeled set of 10% [munjal-protocol-initial]. Each round adds 10% [munjal-protocol-budget],
   up to 40% [munjal-protocol-final].
3. Classifier: VGG16 with batchnorm [munjal-protocol-arch], retrained from random init every round
   [munjal-protocol-retrain].
4. At every round, run 50 trials of random hyperparameter search with Optuna [munjal-protocol-automl,
   munjal-protocol-optuna]. Ranges: [munjal-automl-ranges]. Four rounds = 200 models per method and set
   [munjal-protocol-models-per-run].
5. Repeat for 5 initial sets x 5 weight seeds = 25 runs per method [munjal-protocol-initsets,
   munjal-protocol-seeds, munjal-protocol-runs]. Seeds 2-5 reuse the hyperparameters found for seed 1
   [munjal-protocol-hp-reuse].
6. Methods: RSB (random), UC (least confidence), DBAL, BALD, Coreset, VAAL, QBC.
7. Statistics (Sec 6.2): Kolmogorov-Smirnov rejects normality in all 6 settings [munjal-stats-ks]; Levene rejects
   equal variance in 4 of 6 [munjal-stats-levene]. So the test is Kruskal-Wallis (rank-based ANOVA), then
   Games-Howell pairwise tests at alpha = 0.05 [munjal-stats-tests]. Kruskal-Wallis p-values: [munjal-t1-kw-*].
8. Strong regularization (SR) = RandAugment + SWA [munjal-protocol-sr]. Re-run with SR.

Equations. None numbered that the page needs. Two tests in words: Kruskal-Wallis asks "do the 7 methods' 25-run
accuracy distributions differ at all?" Games-Howell then asks "which pairs differ?" without assuming equal
variances.

Results worth remembering.
- Spread: "A standard deviation of 1-2% in accuracy among different AL methods, indicating that out of chance, it
  is possible to achieve seemingly better results" [munjal-std-quote]. No consistent winner [munjal-no-consistent-winner,
  munjal-stats-gh-result].
- Counts of the 6 settings where a method was significantly above RSB: QBC 4 out of 6, Coreset 2/6, DBAL 2/6,
  BALD 1/6 [munjal-beats-rsb-*]. Significantly below RSB: VAAL twice, UC once [munjal-t1-below-rsb].
- Table 3, 10% labels, VGG16: RSB 69.16 -> +SWA+RA 82.16 on CIFAR10; 34.73 -> 39.44 on CIFAR100
  [munjal-t3-*]. The paper calls the combined RA + SWA gain about 12% and 5% [munjal-t3-gain-quote]; Table 3
  itself gives 82.16 - 69.16 = 13.00 and 39.44 - 34.73 = 4.71. Compare: the largest Table 4 AL gain is +4.40
  [munjal-t4-gap-r18-40].
- SR also shrinks the seed spread [munjal-sr-variance-quote]. RSB-SR at 20% beats plain RSB at 40% by 3%
  [munjal-sr-20-vs-40].
- Table 4 (sets picked on VGG16, trained on other nets): on plain ResNet18, best AL minus RSB is +3.54 / +2.44 /
  +4.40 (Coreset). On ResNet18-SR it is +0.72 / +1.79 / +1.54 (DBAL) [munjal-t4-gap-*, derived]. These are
  best-model values with no spread [munjal-t4-* note]. This is the second panel of viz-budget.
- Direction of the headline: "well-regularized random baseline ... was either at par with or marginally inferior
  to the state-of-the-art AL methods" [munjal-par-quote]. AL is equal or slightly ahead; random is not better on
  average. Do not invert this on the page.
- Conflict: RSB-SR at 40%, CIFAR10 is 90.87% in Sec 6.4 text and 91.16 in Supp. Table 1
  [munjal-rsbsr-40-conflict]. Show both or write "about 91%".
- Noise definition (for the cross-paper noise table): labels of 10% and 20% of each random draw are randomly
  permuted [munjal-noise-def].

Claim card. Claim: under one code base, tuned every round, AL gives small and inconsistent gains over random;
strong regularization shrinks the gap further. Protocol: 25 runs per method, Kruskal-Wallis then Games-Howell.
Falsifier (ours): one AL method beats RSB-SR by more than the seed spread, in every dataset, budget and
architecture tested.

Limitations and follow-ups. Image classification only [munjal-limit-task]. Very costly (AutoML at every round).
Budgets start at 10% of CIFAR, which is a high budget. TypiClust shows the picture changes at very low budgets.
Beck re-measures the gap as label efficiency. Luth replaces per-round AutoML with light tuning on the starting
budget.

## Beck et al. 2021, DISTIL evaluation (branch; 2106.15324v3)
- Delta from Munjal: same unified-toolkit idea, but reports label efficiency (labels saved vs random at equal
  accuracy) instead of accuracy gaps.
- Result: "AL techniques are 2x to 4x more label-efficient compared to RS with the use of data augmentation"
  [beck-2x-4x]. Facet P6: "1.8x to 2x on CIFAR-10 but only 1.3x on CIFAR-100" [beck-c10-c100]. The source prints
  the multiplication sign; the claims file keeps it.
- Reading: a small accuracy gap at a fixed budget can still be a real label saving. Fewer examples per class
  means less room for AL (Sec 4.5).
- Venue not stated; the PDF says "Preprint. Under review." [beck-cite-venue].

## Hacohen, Dekel, Weinshall 2022, TypiClust (spine; 2202.02794v4, ICML 2022)
Core problem. Cold start. With very few labels, uncertainty-based AL does no better than random, or worse
[typiclust-baselines-quote].

What prior methods did, and why that hurt. Uncertainty methods pick atypical, confusing points. With a tiny labeled
set the model cannot use them yet; it has not even learned the common cases. Its uncertainty estimates are also
poor at that stage.

Key idea. The best query rule flips with the budget. At low budget, label typical points (dense regions); at high
budget, label atypical points. TypiClust picks one typical point per cluster of a self-supervised embedding.

The model (Sec 2), equations in plain words (all version 2202.02794v4).
- Definition 1 [typiclust-def1]: E_D(m) is the expected test error of a learner trained on m random points,
  averaged over the training sets and over the learner's randomness. D is the data distribution; X^m is the
  training sample; h(X^m) is the learned hypothesis; Er is its generalization error. Assumptions: E_D falls as m
  grows, and goes to 0 [typiclust-assumptions].
- Eq 1 [typiclust-eq1]: split the data into region R1 (probability p) and R2 (probability 1-p). Each region has
  its own learner. Total error = p x (error in R1 trained on m1 points) + (1-p) x (error in R2 trained on m2
  points).
- Definition 2 [typiclust-def2]: R1 is easier. Both regions share one error curve E, but R2 learns at rate
  alpha: E_D2(m) = E(alpha m), with alpha < p/(1-p).
- Eq 2 [typiclust-eq2]: E_D(m) = p E(m1) + (1-p) E(alpha m2).
- Biased sampling [typiclust-biased]: m1 = p m + Delta, m2 = (1-p) m - Delta. Delta = 0 is random sampling;
  Delta > 0 oversamples the easy (typical) region; Delta < 0 oversamples the hard (atypical) region.
- Theorem 1 [typiclust-thm1]: compare E'(pm) / E'(alpha(1-p)m) with alpha(1-p)/p. If the ratio is larger,
  oversample R1; if smaller, oversample R2. In words: put the next label where the error curve is still falling
  fastest, after weighting each region by its share of the data and its learning rate.
- Corollary 2 [typiclust-cor2]: for E(x) = k e^{-nu x} there is exactly one switch, at a budget m0. Budgets
  m <= m0 are the "low budget regime" [typiclust-lowbudget-def].

Worked check of the toy (derived; settings from Sec 2.2, "Example: exponentially decreasing function").
Settings: E(m) = e^{-m}, p = 0.8, alpha = 0.1, Delta = +-0.01 [typiclust-toy-E, -p, -alpha, -delta].
  Precondition: alpha = 0.1 < p/(1-p) = 0.8/0.2 = 4. It holds [typiclust-toy-precondition].
  E'(x) = -e^{-x}.
  Left side:  E'(pm) / E'(alpha(1-p)m) = e^{-0.8m} / e^{-0.02m} = e^{-0.78m}   (alpha(1-p) = 0.1 x 0.2 = 0.02)
  Right side: alpha(1-p)/p = 0.02 / 0.8 = 0.025
  Oversample R1 while e^{-0.78m} > 0.025, i.e. 0.78m < ln(40) = 3.6889, i.e. m < 4.729.
  So m0 = ln(40)/0.78 = 4.73 [typiclust-toy-m0, derived]. Below about 4.7 labels, typical points help; above it,
  atypical points help. A direct simulation of Eq 2 with Delta = +0.01 changes sign at m = 4.725, which agrees.
  General form for E = e^{-x}: m0 = ln(p / (alpha(1-p))) / (p - alpha(1-p)).
  The paper's Fig 1a has no axis numbers, so 4.73 is our number, not the paper's. viz-budget can show it live.

The method (Sec 3).
- Eq 4 [typiclust-eq4]: Typicality(x) = 1 / (mean L2 distance from x to its K nearest neighbors). x is an
  embedding; K = 20 [typiclust-K]. High typicality = dense neighborhood. No labels are needed.
- Step 1 [typiclust-step1]: train a self-supervised model on all images; use its penultimate layer as the space.
- Step 2 [typiclust-step2]: cluster into |L_{i-1}| + B clusters, so at least B clusters hold no labeled point.
- Step 3 [typiclust-step3]: from the B largest uncovered clusters, take the most typical point of each.
- Variants: TPC_DC (SCAN) and TPC_RP (SimCLR or DINO + K-means) [typiclust-variants].

Results worth remembering.
- Regime tested: "only 1-10 examples per class are labeled each round" [typiclust-budget-range]; the main
  supervised figure uses B = M and 5M, i.e. 1 and 5 per class [typiclust-fig4-budgets]. There, "all other baseline
  AL methods perform on par with random selection or worse" [typiclust-baselines-quote].
- FlexMatch, CIFAR-10, 10 labels (Fig 6a): Random 53.8, Balanced 55.9, Coreset 56.6, TPC_RP 80.5, TPC_DC 93.2
  [typiclust-f6a-*]. Abstract: 93.2% and +39.4% over random [typiclust-abstract-932, typiclust-abstract-394].
- Fig 9 gains over random, in points [typiclust-f9-*]: largest with semi-supervised training (CIFAR-10 39.4),
  smaller with a linear probe on the embedding (27.5), smallest with fully supervised training (6.4). Bar labels
  are printed in the figure image only, so most are flagged uncertain.
- Fig 1b (plot reading, uncertain): TPC_DC is about +2.6 at 20-40 labels and drops below random near 1,000-2,000
  labels; Margin and BADGE do the reverse [typiclust-f1b-*].

Claim card. Claim: the best query rule flips with budget; typical first, atypical later. Falsifier (ours):
uncertainty sampling beats random at 1-10 labels per class, or typical-point selection still wins at large budgets.

Limitations and follow-ups. The switch point depends on task and data; the paper does not say how to find it
[typiclust-limit-transition]. It loses at high budget by design. The cluster count is a knob. ProbCover replaces
clustering with coverage and has one purity knob.

## Yehuda et al. 2022, ProbCover (branch; 2205.11320v3, NeurIPS 2022)
- Delta from TypiClust: same group; replaces "cluster, then take the most typical point" with greedy maximum
  coverage of delta-balls in the self-supervised embedding [probcover-role].
- Fig 6a, FlexMatch, CIFAR-10, 10 labels: Random 53.80, Coreset 56.57, TypiClust 93.19, ProbCover 94.10
  [probcover-f6a-*]. Printed bar labels; TypiClust 93.19 agrees with its own paper's 93.2.

## Luth et al. 2023, pitfalls of AL evaluation (branch; 2301.10625v3, NeurIPS 2023)
- Delta from Mittal, Munjal and Beck: names five pitfalls and evaluates all five together [luth-p1 .. luth-p5]:
  P1 Lack of evaluated data distribution settings; P2 Lack of evaluated starting budgets; P3 Lack of evaluated
  query sizes; P4 Neglection of the classifier configuration; P5 Neglection of alternative training paradigms.
- Replaces Munjal's per-round AutoML with light tuning on the starting budget.
- BADGE result, exact words: "exhibits no failure modes (i.e. sub-Random performance)" [luth-badge-norandom].
  The paper does not say "never"; the page should say "did not fall below random in any tested setting".
- Chapter 5 use: the counterweight. Random is hard to beat, but a well-tuned BADGE did not lose to it.

## Ren et al., "A Survey of Deep Active Learning" (branch; 2009.00236v2; citation only)
- Cite for the timeline and the deeper-reading list only [rensurvey-cite-*]. Venue not stated in the source
  [rensurvey-cite-venue].

## Timeline facts (AL half, this group; first-submission dates)
| date | paper | delta, one line |
| 2009 (TR 1648, upd. 2010-01-26) | Settles survey | AL-built sets are biased; random for pilot studies |
| 11 Dec 2019 | Mittal, Parting with Illusions | augmentation shrinks AL gaps to 0.4% on CIFAR-10 |
| 21 Feb 2020 | Munjal (CVPR 2022) | 25 runs, per-round AutoML, RA+SWA: AL at par or slightly ahead of random |
| 30 Aug 2020 | Ren survey | taxonomy of deep AL (citation only) |
| 16 Jun 2021 | Beck (DISTIL) | gains as label efficiency: 2x-4x with augmentation |
| 6 Feb 2022 | Hacohen, TypiClust (ICML 2022) | low budget: label typical points; rule flips with budget |
| 23 May 2022 | Yehuda, ProbCover (NeurIPS 2022) | coverage of delta-balls replaces clusters + typicality |
| 25 Jan 2023 | Luth (NeurIPS 2023) | five evaluation pitfalls; BADGE had no sub-random result in their tests |
Each paper also has a *-cite-date-version claim for the version used. The timeline should use the first-submission
date for every paper (rule to share with the other groups).

## Gaps and conflicts
- "1-5 labels per class" is not in TypiClust. The text says 1-10 per class; Fig 4 uses 1 and 5.
- Munjal RSB-SR at 40%: 90.87% (Sec 6.4) vs 91.16 (Supp. Table 1).
- Munjal Table 4 reports best-model values without spread; the gap cannot be set against the 1-2% spread directly.
- Venues not stated: Mittal, Beck, Ren.
- Settles citation facts are left to the al-select group (it covers Settles Sec 3).
