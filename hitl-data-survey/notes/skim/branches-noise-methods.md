# Branch papers: label-noise methods

Read alongside confident.md, coteaching.md, dividemix.md, labelclean.md. "Symmetric X%" noise is defined differently across papers (true label kept or not); do not compare accuracies across papers.

## patrini
Patrini, Rozza, Menon, Nock, Qu. "Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach". arXiv 1609.03683v2. CVPR 2017 (oral).
- Role: loss correction with a noise transition matrix T, generalizing Natarajan et al. 2013 (binary unbiased losses) to multi-class deep nets. Backward: `T^{-1} \bm{\ell}(\hat{p}(\bm{y}|\bm{x}))` (Theorem 1). Forward: `-\log T^\top \hat{p}(\bm{y}|\bm{x})` (Sec 4.2; it dominates backward in practice). T estimated from the noisy model alone: for class i take the example with max (in practice 97th percentile) p(y~=i|x) and read its row (Eqs `eq:est1`, `eq:est2`). This "F-correction" is the transition-matrix baseline in Co-teaching and DivideMix.
- Key number: Table 3, Clothing1M, ResNet-50: cross-entropy on 1M noisy 68.94 (row #4); forward correction 69.84 (row #6); forward-corrected net fine-tuned on the 50k clean subset 80.38 (row #8, best).
- Relevance: if a small audited set gives the labelers' class-confusion matrix, the loss can be corrected without finding individual bad labels. Weak when T is hard to estimate (100 classes: Co-teaching Table 6 F-correction 1.60 at Pair-45%).

## mentornet
Jiang, Zhou, Leung, Li, Fei-Fei. "MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks on Corrupted Labels". arXiv 1712.05055v2. ICML 2018.
- Role: replaces the fixed self-paced curriculum (weight v_i = 1[loss_i <= lambda], Kumar et al. 2010) with a small learned network that outputs a weight per example from its loss and training state; predefined (PD) or data-driven (DD, trained on a small clean set) versions. Predecessor of Co-teaching's small-loss idea; Co-teaching's critique: one network selecting for itself accumulates selection errors.
- Key number: Table 5, entire WebVision, Inception-ResNet v2: MentorNet top-1 70.8% on WebVision val (0.708) and 62.5% on ILSVRC12 val (0.625); MentorNet* (with extra verification labels) 72.6% / 64.2%.
- Relevance: a small trusted subset can train the "which labels to trust" policy; it down-weights instead of deleting examples.

## gce
Zhang, Sabuncu. "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels". arXiv 1805.07836v4. NeurIPS 2018.
- Role: robust loss family between CCE and MAE (Ghosh et al. 2017): `\mathcal{L}_{q}(f(\boldsymbol{x}), \boldsymbol{e}_j) = \frac{(1-f_j(\boldsymbol{x})^q)}{q}` (Eq `eq:Lq`), CCE as q -> 0, MAE at q = 1; truncated L_q also drops examples whose loss exceeds L_q(k) (a pruning step, Algorithm 1). q = 0.7 throughout.
- Key number: Table 1, CIFAR-10, ResNet-34, uniform noise eta = 0.4: CCE 81.88 +/- 0.29, MAE 67.00, L_q 87.13, truncated L_q 87.62 +/- 0.26.
- Relevance: the cheapest knob, a one-line loss change with no second network or clean data; a baseline any platform can switch on before buying relabels.

## coteachplus
Yu, Han, Yao, Niu, Tsang, Sugiyama. "How does Disagreement Help Generalization against Label Corruption?" (Co-teaching+). arXiv 1901.04215v3. ICML 2019 (venue from the icml2019 style file in src/ and DivideMix's citation; abs page has no journal ref).
- Role: fixes Co-teaching's late-training failure, where the two networks converge to consensus and Co-teaching "reduces to the self-training MentorNet". Adds Decoupling's "update by disagreement": keep only examples where the two networks' predictions disagree, then cross-exchange the small-loss ones among those.
- Key number: Table 4, Tiny-ImageNet, average/max test accuracy over last 10 epochs, Symmetry-50%: Co-teaching+ 41.19/41.77 vs Co-teaching 37.09/37.60. Counter-case, same table, Pair-45%: Co-teaching 27.41/27.82 beats Co-teaching+ 26.54/26.87.
- Relevance: two-model disagreement is itself a cheap signal for which images to route to a human.

## mixmatch
Berthelot, Carlini, Goodfellow, Papernot, Oliver, Raffel. "MixMatch: A Holistic Approach to Semi-Supervised Learning". arXiv 1905.02249v2. NeurIPS 2019.
- Role: semi-supervised learning, not a noise method; it is DivideMix's training engine: guess labels for unlabeled images by averaging K augmentations, sharpen with temperature T, mixup labeled and unlabeled with lambda' = max(lambda, 1 - lambda), loss = CE on labeled + lambda_U * squared error on unlabeled (Eqs `eqn:l_x`, `eqn:l_u`, `eqn:l_combined`). DivideMix adds co-refinement, co-guessing, and the uniform-prior regularizer.
- Key number: Sec 4.2.2 text and Fig 2 (exact values in appendix Table 5): CIFAR-10 with 250 labels, error rate 11.08% vs VAT 36.03%; 6.24% with 4000 labels; fully supervised on 50000 labels 4.17%; lambda_U = 75 for CIFAR-10.
- Relevance: shows how much a few hundred labels plus unlabeled images buy; links the noise half (treat suspect labels as unlabeled) to the active-learning half (labeling budget).

## elr
Liu, Niles-Weed, Razavian, Fernandez-Granda. "Early-Learning Regularization Prevents Memorization of Noisy Labels". arXiv 2007.00151v2. NeurIPS 2020.
- Role: regularize instead of select. Proves early learning then memorization in a linear model; adds `\frac{\lambda}{n}\sum_{i=1}^n \log \left(1-\langle \mathbf{p}^{[i]},\mathbf{t}^{[i]} \rangle\right)` to CE (Eq 6 in PDF, src/main.tex label `eq:ELR`; macro \vq expanded to \mathbf{t}), p = model softmax output, t = target, a temporal running average of predictions (Eq 9). ELR+ adds two networks, weight averaging, mixup (borrowed from Co-teaching/DivideMix line).
- Key number: Table 3, Clothing1M: ELR+ 74.81 vs DivideMix 74.76. Table 4, mini WebVision top-1: ELR+ 77.78 vs DivideMix 77.32; ILSVRC12 top-1 DivideMix 75.20 vs ELR+ 70.29. Table 2 caption: ELR+ uses a held-out validation set, while DivideMix's numbers are best test accuracy during training.
- Relevance: formalizes "the model's early predictions are a usable second opinion on a label", the basis of loss- and margin-based review ranking; also a reminder to pick checkpoints on held-out data.

## crowdlab
Goh, Tkachenko, Mueller. "CROWDLAB: Supervised learning to infer consensus labels and quality scores for data with multiple annotators". arXiv 2210.06812v2. NeurIPS 2022 Human in the Loop Learning Workshop.
- Role: carries confident-learning style scoring (cleanlab) to multi-annotator data. Consensus probability = weighted average of the classifier's prediction and each annotator's label (Eq `eq:our_cqs`), weights from estimated classifier and annotator accuracy; outputs consensus label, its quality score, and a per-annotator quality score. Compared against Dawid-Skene and GLAD (generative, ignore image features) and Active Label Cleaning (sources/labelclean).
- Key number: Table 1, "Hardest" CIFAR-10H subset (starts from the 25 worst annotators), ResNet-18, Lift@10 for ranking incorrect consensus labels: CROWDLAB 24.33, Label Quality Score 19.46, Active Label Cleaning 17.03, GLAD 14.6, Dawid-Skene 12.89, Agreement 4.87.
- Relevance: when images get one to five labels from different reviewers, decide which consensus labels to re-review and which reviewers need feedback.

## songsurvey
Song, Kim, Park, Shin, Lee. "Learning from Noisy Labels with Deep Neural Networks: A Survey". arXiv 2007.08199v7. IEEE TNNLS, 2022.
- Role: the map. 62 methods in five groups (Sec III): robust architecture, robust regularization, robust loss function (GCE here), loss adjustment (loss correction such as Patrini, reweighting, label refurbishment, meta learning), sample selection (multi-network such as Co-teaching, multi-round, hybrid such as DivideMix). Also covers noise-rate estimation and evaluation practice.
- Key number: Sec I, real-world noise rates "range from 8.0% to 38.5%"; Table IV: ANIMAL-10N ~8.0, CIFAR-10N ~9.0/18.0/40.2, CIFAR-100N ~25.6/40.2, Food-101N ~18.4, Clothing1M ~38.5, WebVision ~20.0.
- Relevance: chapter 13 arXiv map taxonomy, and real noise levels for chapter 8.
