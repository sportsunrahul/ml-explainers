# Outline: Labels in the loop

Working title: "Labels in the loop: choosing what to label, and training when labels are wrong".
Reader: knows basic deep learning (a CNN, softmax, cross-entropy, SGD). Everything else is built on the page.
Pages: the long page (no narration; cue ids and player hooks kept so audio can be added later) and an atlas page.
Wording: generic. "A vision platform with a human feedback loop": a model scores a stream of images, some images go
to labelers, labels come back, the model retrains. No company names.

Badges on every figure: **toy** (made-up data, exact math), **computed** (our own run or the paper's own algorithm on
real data), **sourced** (the paper's published numbers).

Our runs (pod, CIFAR-10, the Co-teaching 9-layer CNN and schedule; job/run_noise.py):
- A: plain cross-entropy, 40% symmetric noise (39.69% flipped), 200 epochs, per-image loss every epoch.
- B: Co-teaching on the same labels, 200 epochs, label precision and test accuracy per epoch.
- C: 4-fold cross-validated probabilities on 40% asymmetric noise (cat<->dog, truck->automobile, bird->airplane,
  deer->horse; 20.13% of all labels flipped), the input for confident learning.

Color key, one meaning each, for the whole page: blue = the model (predictions, probabilities); yellow = the human
(labels, label requests, review); green = clean / kept; red = wrong label / loss; brown = the unlabeled pool.

## Story

Problem: labels are the scarce, expensive input; the loop must decide which images get a label and which labels to
believe. Insight, part one: a label is worth most where the model would learn the most from it, but "most uncertain"
is the wrong proxy when the image is ambiguous, when the batch is redundant, or when the budget is tiny. Insight,
part two: networks learn the consistent pattern before they memorize the exceptions, so early-training loss separates
clean labels from wrong ones; out-of-sample probabilities find the wrong ones; two networks or a two-bump loss fit
let training ignore them, and semi-supervised learning keeps their images. Close: the same ranking tools tell the
reviewer which labels to re-check first, and the evaluation set gets checked first of all.

## Hero (header)
A strip of image tiles flows past the model. Blue confidence bars fill under each tile. A few tiles get a yellow ring
and lift to a labeler; labels come back as tags; one tag is red (wrong). The red tile drops into the training set,
then a loss meter flags it and it returns to review. Ends static on the loop with two marked decision points:
"Which images get a label?" (part one) and "Which labels do we believe?" (part two). Replay button `btn-hero-replay`.

## 1. ch-loop: Two decisions in every feedback loop (~90 s)
Intuition: the label budget is the bottleneck; every round makes two decisions, what to send and what to trust.
Visual `viz-loop` (toy, SVG): the loop as a cycle: stream -> model -> select -> labeler -> label -> dataset ->
train. Buttons `btn-loop-select`, `btn-loop-trust`, `btn-loop-both` light the two decision points and list the
chapters that answer each (links). A small readout: "This round: N images seen, b sent, k votes each".
Decision: none yet; it frames the page. Prediction prompt: "If you could label only 1% of the stream, which 1%?"

## Part one: choosing what to label

## 2. ch-uncertainty: Ask about the coin flips (~120 s)
Intuition: the model's own probabilities say where it is unsure; label those images, not random ones.
Introduce: pool-based active learning, oracle, round, budget; softmax as confidence; three ways to score "unsure":
least confidence, margin, entropy (Settles 2009 Sec 3.1); Lewis & Gale 1994 (closest to 0.5).
Visual `viz-pool` (toy, simulated in node first): a 2D pool, three classes, a tiny classifier trained in the page.
Left: the pool with an uncertainty heatmap (measure toggle: `btn-pool-lc`, `btn-pool-margin`, `btn-pool-entropy`).
Right: the 3-class probability triangle; hovering a pool point drops its probability vector into the triangle.
Buttons `btn-pool-uncertain` (label 10 by uncertainty), `btn-pool-random` (label 10 at random), reset. A learning
curve runs both strategies on twin copies of the pool.
Equations: the three scores (Settles Sec 3.1), built term by term.
Sourced: Lewis & Gale: up to 500-fold fewer labels than random for rare text categories (abstract, Sec 7).
Decision: which images get a label request this round. Rare classes (rare defects) are where uncertainty sampling
pays most; random labeling rarely even sees them.
Caveat seeded for later: softmax confidence is often over-confident; the scores also cannot tell "ambiguous" from
"unfamiliar" (chapter 3), and top-k picks clump (chapter 4).

## 3. ch-bald: Ambiguous or unfamiliar? MC dropout and BALD (~150 s)
Intuition: two kinds of "unsure". An ambiguous image (blur, occlusion, a borderline defect) stays 50/50 however many
labels you buy. An unfamiliar image is one the model has not learned yet; a label fixes it. Entropy scores both high.
BALD scores only the second: it measures how much a set of plausible models disagree.
Introduce: many plausible weight settings fit the labels (posterior); dropout at test time samples one of them
(Gal & Ghahramani 2016); T passes give T predictions; mutual information.
Visual `viz-dome` (3D, pe3; exact math, toy samples): entropy as a dome over the 3-class triangle. T dropout samples sit
on the dome; their mean sits on the triangle; the dome height at the mean is H[mean]; the average height of the samples
is E[H]; the gap is BALD. Presets `btn-dome-ambiguous` (samples bunched at the center: high entropy, BALD near 0) and
`btn-dome-unfamiliar` (samples spread to corners: high entropy, high BALD), `btn-dome-confident`; sliders T and spread.
Prediction prompt: "both images score the same entropy; which one deserves the label?"
Equation build: BALD = H[y|x,D] - E_{p(w|D)} H[y|x,w] (DBAL Sec 4; Houlsby 2011 Eq 2), each term lights its part of the dome.
Sourced (small bar race): DBAL Table 1, acquired MNIST images to reach 10% / 5% test error: BALD 145/335, Var Ratios
120/295, Max Entropy 165/355, Mean STD 230/695, Random 255/835.
Decision: do not spend labels on images that are ambiguous to everyone; route them to a guideline decision (where is
the class boundary?) instead. Ask about the images where the model's own passes disagree.

## 4. ch-coreset: A batch of near-duplicates (~120 s)
Intuition: a labeling batch of the top-k most uncertain images is often k copies of one confusing moment (consecutive
frames). Cover the space instead: pick the batch so no unlabeled image is far from a labeled one.
Introduce: embeddings (penultimate-layer features), distance, covering radius delta, k-Center, greedy farthest-first.
Visual `viz-cover` (toy): a 2D embedding cloud with a dense burst (one moment, many frames) near the boundary.
Buttons `btn-cover-topk`, `btn-cover-greedy`, `btn-cover-random`, `btn-cover-step` (greedy one pick at a time);
covering circles with the live radius delta; readout of delta and of how many picks landed in the burst.
Equation: k-Center objective (core-set Eq 5); the bound in words (Theorem 1: loss gap <= O(delta) + O(sqrt(1/n))).
Sourced: core-set results are plots only (label any number "read from plot"); BADGE Fig 4 penalty column averages
over 33 settings (lower is better): BADGE 0.66, Margin 1.56, Conf 5.93, Rand 6.01, Coreset 6.9, Entropy 8.01. So
diversity alone also loses; the fix that held up mixes the two (BADGE), or scores the batch jointly (BatchBALD), or
samples instead of taking the top-k (stochastic batch acquisition, same cost as top-k).
Decision: how to fill a large batch from a feed of near-duplicate frames: deduplicate or spread the picks in feature
space; never take a plain top-k.

## 5. ch-random: Why random is hard to beat (~150 s)
Intuition: three reasons. (1) A well-tuned, well-regularized model gains more than clever selection does; the
remaining gains often sit inside the seed-to-seed spread. (2) With very few labels, the right pick is the opposite of
uncertain: typical images that represent each cluster. (3) A selected set is biased; it cannot be the evaluation set.
Visual `viz-budget` (computed from the paper's model, TypiClust Sec 2: E(m)=e^{-m}, two regions, p and alpha sliders):
the gain from oversampling typical vs atypical images as the budget m grows; the crossover moves with p and alpha.
Second panel (sourced, Munjal Table 4): best AL method minus random on ResNet-18 vs ResNet-18 with strong
regularization; toggle `btn-random-plain` / `btn-random-sr`.
Claim cards: Munjal (claim, protocol of 25 runs per method, Kruskal-Wallis, falsifier); Hacohen (low vs high budget
claim, falsifier). Also: Mittal (with augmentation, AL methods within 0.4% on CIFAR-10); Beck (2x to 4x label
efficiency with augmentation); Settles Sec 4.1 (AL-labeled data is "a biased distribution, not drawn i.i.d.").
Equation: TypiClust Eq 2 and the threshold test (Theorem 1) in words; typicality Eq 4.
Decision: cold start (a new camera or product, 1-10 labels per class): send the most typical image of each cluster.
Switch to uncertainty once the labeled pool is large. Before buying an AL pipeline, compare against random with
strong augmentation over several seeds. Always keep a random slice of the stream for the evaluation set.

## Interlude (not narrated): ch-round, one labeling round, end to end
Stepper `viz-round` over one round: pool -> embed + T dropout passes -> score -> pick batch (top-k / k-center /
typical) -> dedupe -> send to labelers (k votes each) -> aggregate -> split (train vs held-out random slice) ->
retrain -> evaluate on the random slice. Readout computes costs from an editable toy config: forward passes = N x T,
label requests = b x k, and so on. Links to the chapter where each step was derived.

## Part two: training when labels are wrong

## 6. ch-memorize: Clean first, wrong later (~120 s)
Intuition: a network fits the consistent pattern before it memorizes the exceptions. Early in training, wrong labels
have high loss; late in training, the network has memorized them and the loss no longer tells them apart.
Introduce: label noise models (symmetric, pair / asymmetric, instance-dependent), per-image loss.
Visual `viz-ridge` (3D, computed from run A): per-image loss histograms, clean (green) vs mislabeled (red), stacked
over epochs as a ridge; an epoch scrubber `btn-ridge-early`, `btn-ridge-peak`, `btn-ridge-late`; a 2D panel with test
accuracy per epoch (rises, then falls) and the fraction of mislabeled images the network now predicts as their wrong
label.
Claim card: Zhang 2017 (nets fit random labels: Inception on CIFAR-10 random labels train 100.0, test 9.78); Arpit
2017 (validation peaks before training accuracy saturates; dropout slows memorization). CIFAR-N: real human noise is
memorized more easily than synthetic noise at the same rate.
Prediction prompt: "At which epoch would you read the losses to find the wrong labels?"
Decision: when to stop training on suspect labels, and when per-image loss is a useful review signal (early, not late).
Training accuracy says nothing about label quality.

## 7. ch-benchmarks: How noisy are real datasets? (~120 s)
Intuition: even famous test sets carry label errors; on noisy test labels the "best" model can be the wrong one.
Visual `viz-prevalence` (sourced): bars of validated test-set error % for the 10 datasets (Pervasive Table 1), the
ImageNet funnel (50,000 -> 5,440 flagged -> 2,916 errors -> 1,428 correctable), and a noise-prevalence slider that
recomputes model accuracies with the paper's own formula A(x; M) for a few ImageNet models from Table S1; the rank
swap marker at N = 9% (ResNet-18 vs ResNet-50, text-backed).
Claim card (Pervasive): claim, MTurk protocol (5 workers, error if fewer than 3 of 5 accept the given label), falsifier.
Side evidence: CIFAR-10N human labels (aggregate 9.03%, single labeler about 17-18%, worst 40.21%); Beyer: original
ImageNet labels score about 90% against multi-label ReaL (read from plot).
Decision: audit the evaluation set before trusting a leaderboard; give evaluation images more votes than training
images; report accuracy on the corrected set.

## 8. ch-confident: Confident learning finds the wrong labels (~150 s)
Intuition: let a model that never saw an image's label vote on it; a vote counts only if it is at least as confident
as that class usually is; count (given label, likely true label) pairs; the off-diagonal counts estimate how labels
were flipped and how many; rank and send the worst to review.
Introduce: cross-validation for out-of-sample probabilities (a model defends labels it trained on); per-class threshold.
Visual `viz-cl` (computed from run C plus real CIFAR-10 thumbnails): per-class thresholds t_j as bars; the 10x10 confident
joint filling in; the estimated noise matrix next to the true one (we know it); the flagged list with thumbnails,
given label vs suggested label, precision of the flags. Buttons `btn-cl-threshold`, `btn-cl-joint`, `btn-cl-compare`,
`btn-cl-flags`.
Equation build: t_j (Eq 2), C (Eq 1), Q-hat (Eq 3), number of errors = n x off-diagonal mass.
Sourced: CL Table 2 (CIFAR-10, ResNet-50, 40% noise: CL 86-87% vs baseline about 60%); Table 4 precision / recall.
Decision: which existing labels go back to review, in what order, and how many; the largest off-diagonal cells name the
confusing class pairs to fix in the labeling guide.

## 9. ch-coteach: Two networks check each other (~120 s)
Intuition: small loss means probably clean (chapter 6). A network that picks its own small-loss images confirms its
own mistakes; two networks pick for each other, so one network's errors are filtered by the other.
Visual `viz-coteach` (computed from runs A and B): a mini-batch of real CIFAR-10 tiles sorted by loss for net f and net
g; the kept fraction R(T); the swap arrows; tiles revealed as clean or mislabeled. Curves: label precision of the kept
set per epoch (run B) and test accuracy, standard vs Co-teaching (runs A, B). Forget-rate slider for R(T).
Equation: R(T) = 1 - min{T tau / T_k, tau}; update rule (Alg 1).
Architecture walkthrough: forward (batch 128x3x32x32 -> two CNNs -> logits 128x10 -> per-image loss 128 -> keep
round((1-tau)128) -> swap) then backward (each net's gradient comes only from its peer's picks).
Sourced: Co-teaching Table 5 CIFAR-10 (Symmetry-50%: Standard 48.87, MentorNet 71.10, Co-teaching 74.02).
Decision: keep training while the review queue is still open; the images both networks keep dropping are the review queue.
The forget rate needs a noise-rate estimate; confident learning gives one.

## 10. ch-dividemix: Drop the label, keep the image (~150 s)
Intuition: fit two bumps to the loss histogram; the low bump is probably clean; for the high bump, throw away the
label but keep the image and let semi-supervised learning guess a label for it.
Introduce: Gaussian mixture, posterior probability; semi-supervised learning (pseudo-labels, sharpening, mixup).
Visual `viz-gmm` (computed from run A losses at a chosen epoch; EM runs in the page): histogram, two fitted bumps, the
clean probability w_i; threshold tau slider shows precision and recall of the "clean" set against ground truth; then a
co-refinement dial (toy): y-bar = w y + (1-w) p, sharpen with T.
Equation build: w_i, Eq 3, Eq 4, Eq 12.
Architecture walkthrough: two nets, each fits a GMM and splits data for the other; per batch: labeled + unlabeled,
M=2 augmentations, co-refine, co-guess, mixup, three losses; backward into the net being trained only.
Sourced: DivideMix Table 1 (CIFAR-10 sym 50%: Best 94.6, Last 94.4; CE 79.4 / 57.9) and Table 2 (asym 40%: 93.4 / 92.1);
the Best/Last caveat.
Decision: when a batch of labels is unreliable (crowd or auto-labels), keep the images; the images with w near 0.5 are
where a human label is worth the most.

## Closing

## 11. ch-cleaning: Which labels to re-check first (~120 s)
Intuition: re-checking labels is active learning on labeled data. Put clear errors first (cheap: one or two votes fix
them); put ambiguous images last (expensive: many votes, and they may stay ambiguous). Clean the evaluation set first.
Visual `viz-relabel` (computed from run C): fraction of wrong labels fixed vs number of reviews, for random order, CL
margin order, and the active-label-cleaning score Phi = CE - H (one always-right reviewer, a stated simplification);
sourced AUCs from Bernhardt Fig 3 beside it.
Then the decision table (static): decision -> method -> signal -> cost -> chapter.
Decision: the review queue order for this sprint.

## 12. ch-map: The field in numbers (~60 s)
Visual `viz-map` (computed from the arXiv scan, heuristic): monthly counts for the two queries (3,179 and 2,729
records, 66 in both), and a title-keyword split; every count is a data-claim whose loc is the query.

## Then (not narrated)
- ch-family: family tree / timeline of spine and branch papers (deltas, one line each).
- ch-summary: one paragraph.
- ch-deeper: paper, section, equation number per item.

## Atlas page (9 stages, no narration)
loop -> ask (uncertainty) -> disagree (BALD dome) -> cover (core-set) -> budget (random) -> memorize (ridge) ->
audit (benchmarks) -> find (confident joint) -> split (small loss + GMM). Each scene is written once and mounted in both pages.

## Checkpoint 2 preview
The hero, chapter 1 and chapter 2 built end to end in the long page; the atlas shell with its first two stages.
