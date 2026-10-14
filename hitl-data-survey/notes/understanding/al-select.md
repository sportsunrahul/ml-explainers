# Understanding: al-select (ch-uncertainty, ch-bald, ch-coreset, ch-round)

Claims file: `notes/claims/al-select.json` (claim ids in brackets). Order: chapter order, then date.
Spine papers: dbal, coreset. Branch papers: lewisgale, settles, houlsby, mcdropout, batchbald, badge, stochbatch.

## lewisgale -- Lewis & Gale 1994 (branch, root of uncertainty sampling; cmp-lg/9407020v2)
- Problem: labels are expensive and positives are rare. With 1 positive in 1000 and 500 labels, a random sample
  usually has no positives at all [lewisgale-rare].
- Prior: random sampling (finds no positives) and relevance sampling (label the most likely positives; redundant,
  and worse as the classifier improves).
- Key idea: label the examples whose class the current classifier is least sure of. For a probabilistic
  classifier: P(C|w) closest to 0.5 [lewisgale-rule].
- Mechanism: loop of score -> pick b least certain -> label -> retrain [lewisgale-loop]. They fill a batch with
  b/2 just above 0.5 and b/2 just below [lewisgale-b2], so at most half the batch can be exact duplicates
  [lewisgale-b2-why]. This is the first batch-diversity trick in the lineage.
- Setup: 3 positive seeds, 249 rounds of b = 4 (999 labels), 10 runs per category [lewisgale-setup-*].
- Results: up to 500-fold fewer labels than random (abstract, Sec 1, Sec 7) [lewisgale-500-abstract,
  lewisgale-500-intro, lewisgale-500-sec7]; boxoffice F1 .700 with 999 uncertainty labels vs .647 with the full
  set [lewisgale-t2-boxoffice-unc, lewisgale-t2-boxoffice-full].
- Limits (stated): one classifier underestimates uncertainty and is biased by an unrepresentative model
  [lewisgale-single-clf]; SDs reach 10% of the mean [lewisgale-sd]. The text calls the loop "Figure 3", the PDF
  labels it Figure 1.

## settles -- Settles survey (branch, definitions; TR 1648, 2009, updated 26 Jan 2010)
- Role: names the three uncertainty scores used in chapter 2 and lists when AL fails. PDF only; the displays
  are unnumbered and transcribed from the text layer.
- Least confident [settles-eq-lc]: 1 minus the top class probability. y-hat is the most probable label
  [settles-eq-lc-yhat]. Reading: the expected 0/1 loss [settles-lc-reading]. Flaw: looks at the top class only.
- Margin [settles-eq-margin]: gap between the top two probabilities; pick the smallest gap.
- Entropy [settles-eq-entropy]: minus sum of p log p over classes; pick the largest.
- With two classes all three are the same rule, "posterior closest to 0.5" [settles-binary-equiv], which is
  Lewis & Gale's rule. On the 3-class triangle all peak at the center [settles-fig5]; entropy differs on the
  edges [settles-fig5-edges]. Comparisons are mixed [settles-mixed]; entropy fits log-loss, margin fits
  0/1 error [settles-loss-match].
- Caveats used later: boundary outliers (Sec 3.6) and the information-density fix [settles-outlier,
  settles-eq-id]; AL-labeled data is "a biased distribution, not drawn i.i.d." (Sec 4.1) [settles-biased];
  top-Q batches overlap in information (Sec 6.1) [settles-qbest], and random batches generally beat Q-best
  [settles-qbest-random, settles-guo-batch].

## houlsby -- BALD (branch, root of chapter 3; 1112.5745v1)
- Problem: information gain about parameters, Eq (1) [houlsby-eq1], needs entropies in a huge parameter space.
- Key idea: Eq (1) is a mutual information [houlsby-mi], so rewrite it in label space, Eq (2) [houlsby-eq2]:
  entropy of the averaged prediction minus the average entropy of each parameter setting's prediction.
- Plain words: pick x where the model as a whole is unsure, but each plausible parameter setting is confident
  [houlsby-intuition]. The settings disagree, hence the name "Bayesian Active Learning by Disagreement"
  [houlsby-naming].
- Delta from max entropy (MES): MES cannot tell model uncertainty from observation noise [houlsby-mes] and
  fails on the noisy toy set [houlsby-mes-noisy]. This is the "ambiguous vs unfamiliar" split of chapter 3.
- Limit: built for GP classifiers and preference learning [houlsby-model]; a deep net needs a posterior, which
  mcdropout supplies.

## mcdropout -- Dropout as a Bayesian approximation (branch; 1506.02142v6, ICML 2016)
- Problem: deep nets give point estimates; softmax is not model confidence. "A model can be uncertain in its
  predictions even with a high softmax output" [mcdropout-softmax] (the chapter 2 over-confidence caveat).
- Key idea: dropout training approximates inference in a deep GP. Keep dropout on at test time; each pass
  samples one weight setting.
- Eq (6) [mcdropout-eq6]: predictive mean = average of T stochastic forward passes; W_i^t = layer-i weights
  with the t-th dropout mask [mcdropout-def].
- T values in the paper: 1000 for the drawings, T = 10 "reasonable" [mcdropout-T1000, mcdropout-T10]; 100 passes
  for the rotated-digit plot [mcdropout-T100].
- Cost: the model is unchanged; passes can run concurrently [mcdropout-cost]. The page's cost readout is N x T
  passes per round.

## dbal -- Deep Bayesian Active Learning with Image Data (spine; 1703.02910v1, 8 Mar 2017)
Source: `sources/dbal/src/example_paper.tex` (top level; the nested ICML2017 draft is ignored). The v1 we hold
names no venue [dbal-cite-venue].

Core problem. Active learning needs a model that learns from little data and knows what it does not know.
CNNs need much data and give no model uncertainty (Sec 1).

Prior methods and why that hurt. Kernel AL on images (MBR, RBF graph) did not use a CNN; even Random + CNN beats
it (Sec 5.3, Fig 3). Deterministic CNN softmax captures only data noise (aleatoric), not parameter uncertainty
(epistemic) [dbal-aleatoric-epistemic].

Key idea. Use a dropout CNN as a Bayesian CNN: MC dropout gives T posterior samples per image. Plug them into
classic acquisition functions (entropy, BALD, variation ratios, mean STD) [dbal-mc-dropout].

Mechanism, step by step (MNIST, Sec 5.1):
1. Start with 20 labeled images, random but class-balanced [dbal-mnist-init]; 100-image validation set for
   weight decay [dbal-mnist-val]; 10K test set; the rest is the pool [dbal-mnist-test].
2. Train the CNN: conv-relu-conv-relu-maxpool-dropout(0.25)-dense(128)-relu-dropout(0.5)-dense-softmax
   [dbal-mnist-arch]. Only 2 dropout layers [dbal-mnist-dropout-layers], although Sec 3 says "dropout before
   every weight layer".
3. For each pool image run T stochastic passes, get T softmax vectors [dbal-eq-phat]. T is NOT stated for
   MNIST [dbal-mnist-T].
4. Score with an acquisition function; take the top 10 [dbal-mnist-per-round, dbal-eq-select].
5. Label, add, retrain from scratch [dbal-retrain]. Repeat 100 rounds (1000 images) [dbal-mnist-rounds,
   dbal-mnist-total]; 3 repeats, averaged [dbal-mnist-repeats].
6. Test error uses standard dropout (weight scaling); only acquisition uses MC dropout [dbal-test-eval].

Equations (all unnumbered in v1; Sec 3-4):
- Predictive [dbal-eq-predictive]: p(y=c|x, D_train) = integral over weights omega of p(y=c|x, omega) times the
  posterior; replace the posterior with the dropout distribution q*_theta; approximate by the average over
  T draws omega-hat_t. c = class, x = image, T = passes.
- Max Entropy [dbal-eq-maxent]: H[y|x, D_train] = - sum_c p log p of the averaged prediction. High for
  ambiguous AND unfamiliar images.
- BALD [dbal-eq-bald]: I[y, omega|x, D_train] = H[y|x, D_train] - E_{p(omega|D_train)} H[y|x, omega]. First term:
  entropy of the mean prediction. Second: mean entropy of single passes. Gap = disagreement [dbal-bald-reading].
  Same as Houlsby Eq (2) [houlsby-eq2].
- MC estimator [dbal-eq-bald-mc]: p-hat_c^t = probability of class c on pass t. First term: entropy of the
  T-average. Second term: + (1/T) sum over c, t of p-hat log p-hat, i.e. minus the average per-pass entropy.
  This is the dome picture: dome height at the mean minus mean height of the samples.
- Variation Ratios [dbal-eq-varratio]: 1 - max_y p(y|x, D_train); Settles' least confident on the MC average.
- Mean STD [dbal-eq-meanstd]: per-class std of p over passes, averaged over C classes; "ad-hoc"
  [dbal-meanstd-adhoc].
- Random [dbal-eq-random]: a(x) = unif().

Results worth remembering:
- Table 1, images to reach 10% / 5% error: BALD 145/335, Var Ratios 120/295, Max Ent 165/355, Mean STD 230/695,
  Random 255/835 [dbal-t1-*]. Caption says "acquired images", body says "acquisition steps"; values exceed 100,
  so images [dbal-t1-units].
- The intro headline "295 vs 835" is the Var Ratios cell, not BALD [dbal-intro-295].
- Table 2, error after 1000 images (validation 5000): Random 4.66%, BALD 1.80%, Max Entropy 1.74%, Var Ratios
  1.64%; semi-supervised Ladder Gamma 1.53%, Ladder full 0.84%, DGN 2.40% [dbal-t2-*].
- Bayesian CNN beats deterministic CNN early and at convergence [dbal-bayes-vs-det]. Mean STD ~ Random
  [dbal-meanstd-random].
- BALD may skip noisy, label-conflicting images (large aleatoric uncertainty); a hypothesis, Sec 5.5
  [dbal-bald-noisy].

Limitations and follow-ups:
- Retrain to convergence each round is slow; not resetting risks local optima [dbal-retrain].
- Top-k with k = 10 scores images one by one, so a batch can hold near-duplicates. Fixed by BatchBALD (joint
  score), core-set (cover the space), stochastic batch acquisition (sample, not top-k).
- Table 1 cells come from 3 repeats; BatchBALD's reimplementation gives different (better) BALD numbers
  [batchbald-t1-bald-90, batchbald-outperform].

## coreset -- Active Learning for CNNs: A Core-Set Approach (spine; 1708.00489v4, ICLR 2018)
Source: `sources/coreset/src/active_learning.tex`; equation numbers from the PDF.

Core problem. CNNs must query in large batches: one image changes little and each round needs full retraining
[coreset-b1]. The paper blames the failure of uncertainty methods on "the correlation caused via batch
acquisition/sampling" [coreset-corr-intro].

Prior methods and why that hurt. Uncertainty scores (entropy, BALD, var ratios, on softmax or MC dropout) pick
the top b; the picks clump in one region [coreset-tsne]. In batch mode they lose to random
[coreset-batch-random]; Bayesian AL "does not scale ... because of batch sampling" [coreset-bayes-scale].

Key idea. Ignore uncertainty. Choose the batch so every unlabeled point is close, in feature space, to a labeled
point. A bound says the loss gap between the labeled subset and the whole set is controlled by that covering
radius delta. Minimizing delta is the k-Center problem.

Mechanism, step by step:
1. Train VGG-16 from scratch on the labeled set (RMSProp, lr 1e-3) [coreset-impl].
2. Embed every image: l2 distance between activations of the final fully-connected layer [coreset-feature].
   (BADGE calls its Coreset baseline "penultimate layer" [badge-coreset-baseline].)
3. Greedy k-Center (Alg 1): start from the labeled set; add the pool point farthest from its nearest chosen
   point; repeat b times [coreset-alg1-step].
4. Optional robust step (Alg 2): binary search delta between delta_2OPT/2 and delta_2OPT with a MIP that may
   leave a few outliers uncovered (Gurobi).
5. Label the b points, retrain from scratch, repeat.

Equations (PDF numbering):
- Eq (5), k-Center [coreset-eq5]: min over new batches s^1 of size <= b of max over points i of min over
  centers j in s^1 U s^0 of Delta(x_i, x_j). In words: shrink the worst-case distance from any point to its
  nearest labeled point [coreset-kcenter-words]. NP-hard [coreset-nphard].
- Theorem 1, as written [coreset-thm1]: |average loss over all n points - average loss over the labeled set
  s| <= delta (lambda^l + lambda^mu L C) + sqrt(L^2 log(1/gamma) / (2n)). delta = covering radius; lambda^l =
  Lipschitz constant of the loss; lambda^mu = Lipschitz constant of the regression function (called lambda^eta
  in the hypothesis); L = loss bound; C = number of classes; n = dataset size; 1 - gamma = confidence.
  Assumptions: Lipschitz, bounded loss, zero training loss on s [coreset-thm1-assume].
- Big-O form [coreset-fig1-bound]: gap <= O(delta_s) + O(sqrt(1/n)). The bound does not depend on how many
  labels you have; a label helps only if it shrinks the radius [coreset-thm1-reading].
- 2-OPT guarantee [coreset-2opt]: the greedy radius is at most twice the optimum.

Results worth remembering:
- All accuracy results are plots (Fig 3 weakly supervised, Fig 4 fully supervised; 5 random initial pools)
  [coreset-setup]. Read from plot, CIFAR-10 at labeled ratio 0.4: fully supervised ours ~78.5 vs Random ~74,
  DBAL ~73 [coreset-fig4-c10-*]; weakly supervised ours ~82.5 vs Random ~75, DBAL ~77 [coreset-fig3-c10-*].
  CIFAR-100 gain is smaller (~54.5 vs ~48.5) [coreset-fig4-c100-*]; the bound scales with C [coreset-c100-weaker].
- Table 1 run-time, b = 5k, |s0| = 10k: distance matrix 104.2 s, greedy 2 s, MIP iteration 7.5 s, MIP total
  244.03 s, total 360.23 s [coreset-t1-*]. Greedy is cheap; the distance matrix dominates.
- Fig 6, CIFAR-100, exact point labels: Greedy 0.29 / 0.43 / 0.52 / 0.63 (last point unlabeled); MIP
  0.30 / 0.45 / 0.54 / 0.64 / 0.65 at about 5k / 10k / 20k / 40k / 50k labels [coreset-fig6-*].

Limitations and follow-ups:
- Theory uses l2 loss and zero training error; experiments use cross-entropy [coreset-l2-vs-ce].
- No uncertainty; combining both is "an open problem" [coreset-no-unc]. BADGE combines them.
- Weaker with many classes; quality depends on the embedding (weak supervision helps a lot).
- In BADGE's 33-setting comparison Coreset averages worse than Random (6.9 vs 6.01) [badge-fig4-avg-coreset,
  badge-fig4-avg-rand]: diversity alone also loses.

## batchbald -- BatchBALD (branch, delta from DBAL; 1906.08158v2, NeurIPS 2019)
- Delta: DBAL's top-b BALD is a sum of single scores, Eq (3) [batchbald-eq3], which double counts shared
  information [batchbald-overcount]. BatchBALD scores the batch jointly, Eq (4) [batchbald-eq4]: mutual
  information between the joint labels y_1..y_b and the weights. Eq (8) [batchbald-eq8]: joint entropy minus
  expected conditional joint entropy (the second term splits into a sum because labels are independent given
  omega).
- Greedy selection is (1 - 1/e)-approximate (submodular) [batchbald-greedy]; at size 1 it equals BALD
  [batchbald-size1].
- Evidence: on Repeated MNIST (each image x3 plus noise) BALD does worse than random [batchbald-repeated,
  batchbald-repeated-result]. Table 1, points to 90% / 95%: BatchBALD 70/90/110 and 190/200/230; BALD
  reimplementation 120/120/170 and 250/250/>300; BALD as reported by DBAL 145 and 335 [batchbald-t1-*].
  The last row equals DBAL Table 1. MNIST uses 100 MC dropout samples [batchbald-mnist-T].
- Limit: cost. At K = 10, 566.0 s vs 0.2 s for top-K [stochbatch-t1-batchbald].

## badge -- BADGE (branch, delta from core-set; 1906.03671v2, ICLR 2020)
- Delta: core-set uses only diversity; BADGE embeds each image as the last-layer gradient of the loss under its
  own predicted label [badge-eq-grademb]. Small norm = confident, large norm = uncertain [badge-norm]. Then pick
  the batch with k-means++ seeding on those gradients [badge-kmeanspp]: far apart and large. No trade-off
  knob [badge-no-hyper].
- Evaluation: M = 100, B in {100, 1000, 10000} [badge-setup]; 3 batch sizes x 11 dataset-architecture pairs =
  33 settings [badge-settings]; t-test penalties [badge-penalty-protocol].
- Fig 4 column averages (lower is better): BADGE 0.66, Margin 1.56, ALBL 4.08, Conf 5.93, Rand 6.01, Coreset
  6.9, Entropy 8.01 [badge-fig4-avg-*]; average = column sum / 7 [badge-fig4-avg-check].
- Caveats: 4 of 7 datasets are non-image and one architecture is an MLP [badge-archs-data]; Coreset here is
  greedy furthest-first on the penultimate layer [badge-coreset-baseline]. At B = 10000 Margin degrades
  [badge-large-batch].

## stochbatch -- Stochastic batch acquisition (branch, delta from BatchBALD; 2106.12059v3, TMLR 09/2023)
- Delta: keep the single-point score (BALD, entropy) but sample the batch instead of taking the top-K.
- Power acquisition, Eq (14) [stochbatch-eq14]: add Gumbel noise to log s(i); take the top-K. Eq (15)
  [stochbatch-eq15]: same as sampling with probability proportional to s(i)^beta (Proposition 3.1)
  [stochbatch-gumbel]. beta = 1 by default; beta -> infinity is top-K, beta -> 0 is random [stochbatch-beta].
  Zero-score points get zero mass [stochbatch-zero].
- Table 1 (K = 10, synthetic 10,000-point pool, 20 parameter samples): top-K BALD 0.2 s, PowerBALD 0.2 s,
  BatchBALD 566.0 s, BADGE 9.2 s [stochbatch-t1-*]; Repeated-MNIST mean accuracy 80% / 90% / 90% / 86%
  [stochbatch-t1-acc-*]. Same cost as top-K, quality near BatchBALD.

## ch-round -- what the interlude can cite
- Embed + T passes: cost N x T forward passes per round; passes run in parallel [mcdropout-cost]. T values:
  DBAL not stated [dbal-mnist-T]; 10 is "reasonable" [mcdropout-T10]; BatchBALD 100 on MNIST
  [batchbald-mnist-T]; stochbatch Table 1 uses 20 [stochbatch-t1-setup].
- Score: chapter 2 and 3 equations. Pick: top-k (DBAL), k-center (core-set Alg 1), sampled (stochbatch).
  k-center cost: distance matrix 104.2 s at b = 5k, |s0| = 10k [coreset-t1-distance].
- Batch sizes in the sources: 4 (Lewis & Gale), 10 (DBAL), 100-10000 (BADGE), 5k (core-set Table 1).
- Retrain from scratch every round (DBAL, core-set) [dbal-retrain, coreset-impl].
- Evaluate on a random slice, because AL-picked data is biased [settles-biased].
- Not in this group: votes per image (k) and aggregation (label-noise papers).
