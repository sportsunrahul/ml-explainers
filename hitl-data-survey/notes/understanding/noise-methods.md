# Understanding: label-noise methods (chapters 8-11 and the label-noise half of the timeline)

Claims file: notes/claims/noise-methods.json. Claim ids appear in square brackets. Every equation below is a
claim value that renders in KaTeX (checked with strict mode).

## Confident learning (ch-confident). Northcutt, Jiang, Chuang. 1911.00068v6, JAIR 2021

Core problem. We have noisy given labels and a trained model. Which labels are wrong, and what is the true class of
each? Nothing new is trained to answer this. [confident-cite-venue]

Prior methods and why that hurt. The closest prior method takes the argmax of the predicted probabilities as the true
label and flags every disagreement (the paper calls this baseline C_confusion; it is one iteration of INCV)
[confident-method-cconfusion]. Argmax fails when a class gets over-confident or under-confident probabilities, for
example under class imbalance: the over-confident class "steals" examples. Other prior work estimates the noise matrix
by averaging raw probabilities, so probability errors go straight into the estimate.

Key idea. A model vote for class j counts only if the probability for j is at least as high as the average
probability that the model gives to images that are labeled j. Per-class thresholds replace argmax
[confident-threshold-rationale]. Count the votes in a (given label, likely true label) table; the off-diagonal cells
are the suspected errors.

Mechanism.
1. Get out-of-sample probabilities with k-fold cross-validation, k = 4 (fixed, not tuned) [confident-kfold].
   A model that has trained on an image will defend that image's label, so the probabilities must come from a model
   that never saw the label. A pretrained model that never saw the data also works [confident-pretrained-alt].
2. Compute one threshold per class (Eq 2).
3. Fill the confident joint C (Eq 1). If an image is above the threshold of several classes, it goes to the class with
   the largest probability [confident-collision-rule]. An image above no threshold is not counted
   [confident-below-all-thresholds].
4. Calibrate C into a joint distribution Q-hat (Eq 3). Column sums give the latent prior; dividing gives the noise
   matrix [confident-eq-latent-prior] [confident-eq-noise-matrix].
5. Rank and prune with one of five methods [confident-n-methods]: C_confusion, C_{y~,y*} (default), PBC, PBNR, C+NR
   [confident-method-cconfusion] [confident-method-cj] [confident-method-pbc] [confident-method-pbnr]
   [confident-method-cnr] [confident-default-method]. Order the found errors by normalized margin
   [confident-eq-normalized-margin].
6. Retrain on the kept data with per-class weights [confident-retrain-weights].
"Theoretically, CL requires no hyper-parameters to find label errors." [confident-no-hyperparams-quote]

Equations (1911.00068v6).
- Eq 2 [confident-eq2-threshold]: t_j is the mean of p-hat(y~=j; x, theta) over all images x whose given label is j.
  X_{y~=j} = the images labeled j; p-hat = the held-out model probability; theta = the model.
- Eq 1 [confident-eq1-confident-joint]: C[i][j] counts the images with given label i whose probability for class j
  is at least t_j, where j is the highest-probability class among the classes that pass their thresholds.
  i = given (noisy) label; j = guessed true label; m = number of classes.
- Eq 3 [confident-eq3-calibration]: normalize each row of C to sum to 1, multiply by the number of images that carry
  label i, then divide by the grand total. Rows then match the observed label counts and the matrix sums to 1.
- Count of errors in cell (i, j) = n * Q-hat[i][j] [confident-eq-count-per-cell]; total = n times the off-diagonal
  mass [confident-eq-count-total] (our sum; derived).
- Normalized margin [confident-eq-normalized-margin]: probability of the given label minus the largest probability of
  any other class. Most negative = most suspicious.

Assumption. Class-conditional noise: the chance that a label flips depends only on the true class, not on the image
[confident-assumption-cnp] [confident-assumption-cnp-eq]. CL's synthetic noise is asymmetric, from random noise
matrices with a "sparsity" knob; it uses no symmetric noise [confident-noise-definition].

Results.
- Table 2, CIFAR-10, ResNet-50, 40% noise: the three best CL methods (C_{y~,y*}, C+NR, PBNR) 86.6 to 87.2 vs
  Baseline 57.3 to 60.8
  [confident-t2-cnr-n40-s0] [confident-t2-cj-n40-s0] [confident-t2-baseline-n40-s0]. Co-Teaching and MentorNet run by
  CL's authors sit near the baseline at 40% [confident-t2-coteaching-n40-s0] [confident-t2-mentornet-n40-s0]. At 70%
  noise every extracted row falls to 26.0 to 41.8 [confident-t2-pbnr-n70-s06].
- Table 4: argmax (C_confusion) has the highest recall (97 to 98 at 20 to 40% noise, sparsity 0) but the lowest
  precision (56 to 74); C_{y~,y*} trades recall for precision (67 to 78) [confident-t4-cconfusion-rec-n20-s0]
  [confident-t4-cconfusion-prec-n20-s0] [confident-t4-cj-prec-n40-s0].
- Table 5, ImageNet: projectile -> missile 645, tub -> bathtub 539, breastplate -> cuirass 476
  [confident-t5-top1-c] [confident-t5-top2-c] [confident-t5-top3-c]. These cells name class-definition problems, not
  only random slips; one is a duplicated class (maillot) [confident-imagenet-dup-class].
- 58% of 500 random ImageNet flags were real issues (errors, multi-label, or ontology) [confident-imagenet-human-check].

Conflict. The text says C_{y~,y*} is best "across all noise and sparsity settings"; Table 2 shows C+NR or PBNR ahead
at 40% and at most 70% settings [confident-t2-text-vs-table].

Limitations and follow-ups. The class-conditional assumption excludes image-dependent noise (blur, occlusion).
Flags mix real errors with multi-label and ontology issues. Mean thresholds are a choice; percentiles are left open.
Follow-ups: the "pervasive" test-set audit (noise-evidence group); CROWDLAB extends the scoring to many annotators;
active label cleaning orders the review queue by cost.

## Co-teaching (ch-coteach). Han et al. 1804.06872v3, NeurIPS 2018

Core problem. Deep networks first fit clean labels and later memorize wrong ones, so test accuracy rises and then
falls. Can training ignore the wrong labels before memorization starts?

Prior methods and why that hurt. MentorNet (self-paced version) and similar methods let one network pick its own
small-loss examples [mentornet-role]. A wrong pick then trains the same network, which makes the same wrong pick
more likely: selection errors accumulate. Decoupling updates two networks only where they disagree, but noisy labels
lie in the agreement region too, and it does not use the memorization effect. Loss correction (F-correction) needs
an accurate noise matrix and fails badly when the estimate is poor (6.61% on CIFAR-10 Pair-45%)
[coteaching-t5-pair45-fcorrection].

Key idea (delta from MentorNet). Keep the small-loss rule but cross the streams: each network picks its small-loss
fraction and hands it to its PEER for the gradient step. Two networks with different initializations make different
mistakes, so each filters some errors of the other.

Mechanism (Algorithm 1). Per mini-batch: each net ranks the batch by its own loss and keeps the fraction R(T)
(steps 4, 5); f is updated on g's pick and g on f's pick (steps 6, 7). At the end of each epoch R(T) shrinks (step 8).
Nothing is permanently removed; the pick is redone every batch.

Equations (1804.06872v3; the paper has no numbered equations [coteaching-no-numbered-eqs]).
- Step 4 [coteaching-alg1-step4]: D-bar_f is the smallest-loss subset of the mini-batch D-bar under net f, with size
  at least R(T) times the batch size. l(f, D') = f's cross-entropy on subset D'.
- Steps 6, 7 [coteaching-alg1-step6] [coteaching-alg1-step7]: one SGD step with learning rate eta; f's gradient
  comes only from g's pick, and g's only from f's pick.
- Step 8 [coteaching-alg1-step8]: R(T) = 1 - min(T tau / T_k, tau). T = epoch; tau = final drop rate; T_k = ramp
  length in epochs. R falls linearly from about 1 to 1 - tau over T_k epochs, then stays.
- Conditions [coteaching-rt-cond1] [coteaching-rt-cond2] [coteaching-rt-cond3]: R(T) in [tau, 1]; R(1) = 1; R
  non-increasing. Read literally, step 8 gives R(1) = 1 - tau/T_k; the official code starts with forget rate 0
  [coteaching-forget-schedule-code].

Setup. 9-layer CNN (LReLU 0.01, dropout 0.25, batch norm) [coteaching-arch]; Adam, momentum 0.9, lr 0.001
[coteaching-optimizer] [coteaching-lr]; batch 128 [coteaching-batch]; 200 epochs [coteaching-epochs]; T_k = 10
[coteaching-tk]; tau = epsilon, the noise rate, assumed known [coteaching-tau]; 5 repeats; accuracy = mean of the
last 10 epochs [coteaching-metric]. The lr decay is in the code only: linear decay to 0 from epoch 80, Adam beta1
0.9 -> 0.1 [coteaching-lr-decay-code] [coteaching-beta1-code].

Noise definitions. Symmetry flipping: keep the label with probability 1 - epsilon, else move it uniformly to one of
the OTHER n - 1 classes, so Symmetry-50% means half the labels are wrong [coteaching-noise-symmetry]. Pair flipping:
each class flips to the next class with probability epsilon [coteaching-noise-pair].
Label precision = clean labels / selected labels, per mini-batch [coteaching-label-precision-def].

Results.
- Table 5, CIFAR-10, Symmetry-50%: Standard 48.87, MentorNet 71.10, Co-teaching 74.02 [coteaching-t5-sym50-standard]
  [coteaching-t5-sym50-mentornet] [coteaching-t5-sym50-coteaching]. Pair-45%: 49.50 / 58.14 / 72.62
  [coteaching-t5-pair45-coteaching]. At Symmetry-20% F-correction wins (84.55 vs 82.32)
  [coteaching-t5-sym20-fcorrection] [coteaching-t5-sym20-coteaching].
- Table 4, MNIST Pair-45%: the table says 87.63, the Sec 4.1 text says 87.53 [coteaching-t4-pair45-coteaching].
- Dropping more than the noise rate helped on MNIST Pair-45%: 97.89 at tau = 1.25 epsilon [coteaching-t8-tau-pair45].

Limitations and follow-ups. The forget rate needs the noise rate (CL can estimate it). No theory. Synthetic noise
only. Over epochs the two networks converge and the method reduces to self-training; Co-teaching+ fixes this with
disagreement [coteachplus-consensus-claim]. Dropped images are wasted; DivideMix keeps them as unlabeled data.

## DivideMix (ch-dividemix). Li, Socher, Hoi. 2002.07394v1, ICLR 2020

Core problem. Small-loss selection throws away the images with suspect labels. Those images still carry
information about the input distribution.

Delta from Co-teaching. Keep two networks that select for each other, but (1) replace the per-batch top-k cut with a
two-component Gaussian mixture on per-sample loss, giving a soft clean probability w_i; (2) keep the suspect images
and drop only their labels; (3) train with semi-supervised learning (MixMatch [mixmatch-role]) on both sets.

Mechanism (Algorithm 1).
1. Warm up both nets on all data with cross-entropy (10 epochs CIFAR-10, 30 CIFAR-100) [dividemix-hp-warmup]. For
   asymmetric noise add a confidence penalty so losses stay spread out [dividemix-confidence-penalty-use].
2. Co-divide each epoch: net 1's per-sample losses -> GMM -> w_i for net 2, and the reverse [dividemix-codivide].
   Labeled set: w_i >= tau; unlabeled set: w_i < tau, tau = 0.5 [dividemix-hp-tau].
3. Per mini-batch, M = 2 augmentations of each labeled and unlabeled image [dividemix-hp-m].
4. Co-refine labeled targets (Eq 3), sharpen (Eq 4). Co-guess unlabeled targets with both nets, sharpen
   [dividemix-coguess].
5. MixMatch mixup across the combined batch (Eqs 5-8); loss Eqs 9-12; update only the net being trained.
6. Test with the average of both nets [dividemix-test-ensemble].

Equations (2002.07394v1).
- Eq 1 [dividemix-eq1-loss]: l_i = cross-entropy of sample i; y_i^c = one-hot given label; p_model^c = softmax output.
- w_i [dividemix-wi-def]: posterior probability that l_i belongs to the Gaussian with the smaller mean.
- Eq 3 [dividemix-eq3-corefine]: y-bar = w y + (1 - w) p. y_b = given one-hot label; p_b = this net's prediction
  averaged over the M augmentations; w_b = clean probability from the OTHER net. Trusted labels stay; distrusted
  labels move toward the prediction.
- Eq 4 [dividemix-eq4-sharpen]: raise each class probability to 1/T (T = 0.5 [dividemix-hp-t]) and renormalize; the
  target becomes more peaked.
- Co-guess [dividemix-coguess]: q-bar_b = mean of both nets' predictions over the M augmentations of unlabeled u_b.
- Eqs 5-8 [dividemix-eq5-beta] [dividemix-eq6-max] [dividemix-eq7-mixx] [dividemix-eq8-mixp]: lambda from
  Beta(alpha, alpha), alpha = 4 [dividemix-hp-alpha]; lambda' = max(lambda, 1 - lambda) keeps each mixed sample
  closer to its first source; mix inputs and targets with the same weight.
- Eq 9 [dividemix-eq9-lx]: cross-entropy on the mixed labeled batch X'. Eq 10 [dividemix-eq10-lu]: squared error
  between guessed target and prediction on the mixed unlabeled batch U'.
- Eq 11 [dividemix-eq11-lreg]: KL-style penalty that pushes the batch-average prediction toward uniform pi_c = 1/C.
- Eq 12 [dividemix-eq12-total]: L = L_X + lambda_u L_U + lambda_r L_reg; lambda_r = 1 [dividemix-hp-lambda-r];
  lambda_u from {0, 25, 50, 150} [dividemix-hp-lambda-u-grid] (25 at CIFAR-10 sym 50%, 0 at asym 40%).

CIFAR setup. PreAct ResNet-18, SGD momentum 0.9, weight decay 0.0005, 300 epochs, lr 0.02 divided by 10 after 150
[dividemix-hp-arch] [dividemix-hp-epochs] [dividemix-hp-lr] [dividemix-hp-lr-decay]. Batch size: the paper says 128
[dividemix-hp-batch-paper]; the official script defaults to 64 labeled + 64 unlabeled per step
[dividemix-hp-batch-code] (code-only; 64 + 64 may be the paper's 128).

Noise definitions. Symmetric = replace a fraction of labels with a random label from ALL classes, so the true label
can survive [dividemix-noise-symmetric]; "50%" flips about 45% on CIFAR-10 [dividemix-sym50-flipped] (derived).
Not comparable to Co-teaching's Symmetry-50%. Asymmetric = flips between similar classes [dividemix-noise-asymmetric].

Results. "Best" = best test accuracy over all epochs; "Last" = mean of the last 10 epochs [dividemix-best-def]
[dividemix-last-def]. Best selects an epoch on the test set, so read Last.
- Table 1, CIFAR-10 sym 50%: DivideMix 94.6 / 94.4 vs CE 79.4 / 57.9 [dividemix-t1-c10-sym50-dividemix-best]
  [dividemix-t1-c10-sym50-dividemix-last] [dividemix-t1-c10-sym50-ce-best] [dividemix-t1-c10-sym50-ce-last]. The CE
  gap between Best and Last (79.4 vs 57.9) is memorization in one number.
- Table 2, CIFAR-10 asym 40%: 93.4 / 92.1 vs CE 85.0 / 72.3 [dividemix-t2-asym40-dividemix-best]
  [dividemix-t2-asym40-ce-last].
- Table 3, Clothing1M: 74.76 vs CE 69.21 [dividemix-t3-dividemix] [dividemix-t3-ce].
- Table 5 ablation: self-divide (one net) loses 0.6 to 3.8 points of Best; no label refinement costs 5.7 points at
  asym 40% (93.4 -> 87.7); plain Divide + MixMatch is worst [dividemix-t5-no-cotrain-c10-asym40-best]
  [dividemix-t5-no-refine-c10-asym40-best] [dividemix-t5-divide-mixmatch-c10-asym40-best].

Limitations and follow-ups. Many moving parts (two nets, GMM, MixMatch, per-setting lambda_u chosen on a validation
set). Assumes clean and noisy losses form two modes. Best reports peek at the test set; ELR+ reports on a held-out
split and matches DivideMix on Clothing1M [elr-best-test-critique] [elr-t3-clothing].

## Active label cleaning (ch-cleaning). Bernhardt et al. 2109.00574v2, Nature Communications 2022

Core problem. Experts can re-check only a few labels. Which already-labeled images should they see first?

Prior methods and why that hurt. Relabeling campaigns pick at random. Pure loss ranking (CL margin, co-teaching loss)
puts clear errors first but also puts ambiguous images high; an ambiguous image needs many votes before a majority
forms, so it is expensive. Active learning scores (BALD) aim at model gain on unlabeled data, not at label
correctness [labelclean-bald-vs-phi].

Key idea (delta from CL and from active learning). Same act as active learning (spend a human budget on chosen
images), different target: already-labeled images, and the goal is correct labels. Rank by noisiness minus ambiguity.
The entropy term that active learning ADDS to find informative images is SUBTRACTED here, because ambiguity raises the
cost of a fix [labelclean-al-link] [labelclean-priority-rule].

Mechanism (Table 1, the algorithm) [labelclean-alg-steps]. Train a robust selector. While budget remains: take the
available image with the highest Phi; draw one label at a time from its true label distribution (simulated
annotators) until a majority forms [labelclean-majority-rule] [labelclean-relabel-sim]; mark it cleaned; every b
labels fine-tune the selector.

Equations (2109.00574v2).
- Eq 1 [labelclean-eq1-objective]: maximize the fraction of images whose majority label y-hat_i equals the true
  label y_i, subject to the total number of collected labels (sum of ||l-hat_i||_1) staying within the budget B.
  l-hat_i = vector of label counts per class for image i.
- Eq 2 [labelclean-eq2-score]: Phi = CE - H. x = image; theta = selector model.
- Eq 3 [labelclean-eq3-ce]: cross-entropy from the normalized label counts to the predicted posterior; high when the
  current labels disagree with the model.
- Eq 4 [labelclean-eq4-h]: entropy of the predicted posterior; high when the image is ambiguous.

Results.
- Abstract: "up to 4x more effectively" than random [labelclean-abstract-4x]; the body gives 3x (NoisyCXR) and 2.5x
  (CIFAR10H) fewer re-annotations to reach 90% correct labels [labelclean-body-3x-cxr]
  [labelclean-body-25x-cifar10h]. No body number supports 4x.
- Fig 3 AUCs, CIFAR10H 15% noise: Random 0.8772, Vanilla 0.9024, Coteaching 0.9161, Self-Supervision 0.9238, Oracle
  0.9504 [labelclean-fig3b-auc-random] [labelclean-fig3b-auc-vanilla] [labelclean-fig3b-auc-coteaching]
  [labelclean-fig3b-auc-ssl] [labelclean-fig3b-auc-oracle]. NoisyCXR: Random 0.8887 to Oracle 0.9461
  [labelclean-fig3a-auc-random] [labelclean-fig3a-auc-oracle].
- Table 3: with 40% IDN evaluation noise, noisy labels rank the SSL model above co-teaching (46.50 vs 45.93), but true
  labels say the reverse (65.76 vs 69.91); cleaning 20% restores the true order (55.28 vs 55.87)
  [labelclean-t3-rank-flip] [labelclean-t3-idn-m1-clean20] [labelclean-t3-idn-m2-clean20].
- Recommendation: clean the evaluation data first, then spend the rest on training data [labelclean-eval-first].

Noise definitions. SYM is not defined precisely [labelclean-sym-def]; IDN follows Xia et al. 2020 on ResNet-50
embeddings [labelclean-idn-def].

Limitations and follow-ups. Systematic errors that the model itself learns can stay hidden. If noise is not bounded
(wrong labels outnumber right ones in a class), random may beat the ranking. Majority vote treats annotators as equal;
CROWDLAB adds per-annotator weights and beats Active Label Cleaning at Lift@10 [crowdlab-t1-lift10].

## Branch papers (citation facts: <slug>-cite-*)

patrini (1609.03683v2, CVPR 2017 oral). Loss correction with a noise transition matrix T: backward multiplies the loss
by T^-1, forward passes predictions through T^T [patrini-forward-eq]; T is estimated from the noisy model alone. It
is the "F-correction" baseline in Co-teaching and DivideMix [patrini-role]. Clothing1M: forward 69.84 vs
cross-entropy 68.94 [patrini-t3-clothing]. Its CIFAR-10 class map (truck -> automobile, bird -> airplane, deer ->
horse, cat <-> dog) is the asymmetric scheme of our run C [patrini-asym-def]. Weak spot: a bad T estimate is
catastrophic (Co-teaching Table 5 Pair-45%, 6.61%).

mentornet (1712.05055v2, ICML 2018). A small network learns per-example weights from loss and training progress,
replacing a fixed self-paced threshold [mentornet-role]. Entire WebVision top-1 0.708 [mentornet-t5-webvision].
Co-teaching's predecessor and foil: one network choosing for itself accumulates its own selection errors.

gce (1805.07836v4, NeurIPS 2018). A loss between cross-entropy and MAE, L_q = (1 - f_j^q)/q, q = 0.7
[gce-lq-eq]; truncated L_q also prunes high-loss examples [gce-role]. CIFAR-10, 40% uniform noise: 87.13 vs CCE
81.88 [gce-t1-cifar10-u40]. Its uniform noise maps a label to a uniformly sampled random label
[gce-noise-uniform-def]. The cheapest knob: one line of code, no second network, no clean data.

coteachplus (1901.04215v3, ICML 2019). Delta from Co-teaching: the two nets drift into consensus, so Co-teaching
"reduces to the self-training MentorNet" [coteachplus-consensus-claim]; fix: select small-loss examples only among
disagreements [coteachplus-role]. Tiny-ImageNet Symmetry-50%: 41.19 vs 37.09 [coteachplus-t4-sym50]; at Pair-45%
plain Co-teaching wins. Disagreement between two models is itself a cheap review signal.

mixmatch (1905.02249v2, NeurIPS 2019). Semi-supervised learning, not a noise method; DivideMix's engine
[mixmatch-role]. CIFAR-10 with 250 labels: 11.08% error [mixmatch-250-labels]. Bridges the two halves of the page: a
suspect label can be dropped and the image still used.

elr (2007.00151v2, NeurIPS 2020). Regularize instead of select: a term keeps each output close to a running average of
the model's own early predictions [elr-eq6]; ELR+ adds two nets, weight averaging, mixup [elr-role]. Clothing1M
74.81 vs DivideMix 74.76 [elr-t3-clothing]. Critique: other methods' numbers "correspond to the best test performance
during training, whereas we use a separate validation set" [elr-best-test-critique]. Lesson for the page: report
Last, or pick checkpoints on held-out data.

crowdlab (2210.06812v2, NeurIPS 2022 HILL workshop). Confident-learning style scoring for multi-annotator data:
consensus = weighted ensemble of classifier prediction and each annotator's label; outputs consensus label, its
quality score, and per-annotator quality [crowdlab-role]. Hardest CIFAR-10H subset, Lift@10: 24.33 vs Active Label
Cleaning 17.03 [crowdlab-t1-lift10]. Decides which consensus labels to re-review and which labelers need feedback.

## Symmetric noise across papers (do not compare numbers across these)
- Co-teaching, Co-teaching+: the flipped label is never the true one [coteaching-noise-symmetry]
  [coteachplus-noise-def].
- DivideMix: the random label is drawn from all classes, so 50% means about 45% wrong on CIFAR-10
  [dividemix-noise-symmetric] [dividemix-sym50-flipped].
- GCE: "uniform" noise, a random label by uniform sampling; whether it can equal the true label is not stated
  [gce-noise-uniform-def].
- CL: no symmetric noise; random sparse asymmetric matrices [confident-noise-definition].
- Active label cleaning: SYM not defined precisely [labelclean-sym-def].
