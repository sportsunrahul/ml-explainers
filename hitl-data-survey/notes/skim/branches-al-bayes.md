# Branches: Bayesian and batch active learning

## houlsby -- BALD
- **Citation.** Neil Houlsby, Ferenc Huszar, Zoubin Ghahramani, Mate Lengyel. "Bayesian Active Learning for Classification
  and Preference Learning". arXiv:1112.5745v1 (24 Dec 2011). arXiv only; no venue on the abs page.
- **Role.** Root of the family. Rewrites the information gain about parameters, Eq. (1) `H[theta|D] - E_y[H[theta|y,x,D]]`,
  as Eq. (2) `\argmax_{\x} \rmH[\y \vert \x, \data] - \E_{\param\sim p(\param|\data)} \left[ \rmH[\y \vert \x,\param] \right]`
  (`BALD_arxiv.tex` line 96), so entropies live in label space, not parameter space. Names it "Bayesian Active Learning by
  Disagreement". Applied to GP classifiers (closed-form approximation, Eq. labeled `eqn:BALD_GPC`), not neural nets.
  Separates model uncertainty from label noise, which plain max-entropy (MES) cannot.
- **Key number.** Fig. 5 (box plot of extra points needed vs BALD to reach 97.5% of full-pool performance): closest rival is
  Zhu et al., median increase 1.4 points, with zero inside its inter-quartile range (Discussion paragraph, line 264).
- **Relevance.** Gives the "disagreement" score used in chapter 3, and the argument for not wasting labels on
  inherently ambiguous images (high label noise, low BALD).

## mcdropout -- Dropout as a Bayesian Approximation
- **Citation.** Yarin Gal, Zoubin Ghahramani. "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep
  Learning". arXiv:1506.02142v6 (4 Oct 2016). ICML 2016 (per arXiv comment).
- **Role.** Supplies the posterior that BALD needs for deep nets. Casts dropout training as approximate inference in a deep
  Gaussian process. MC dropout = T stochastic forward passes, averaged, Eq. (6):
  `\mathbb{E}_{q(\y^* | \x^*)} (\y^*) \approx \frac{1}{T} \sum_{t=1}^T \widehat{\y}^*(\x^*, \Mh_{1}^t, ..., \Mh_{L}^t)`
  (`ICML_2016_deep_learning_uncertainty.tex` line 381). T=1000 used for drawing; T=10 shown as enough (line 514).
- **Key number.** Table 1, row "Boston Housing", Avg. Test RMSE: VI 4.32 +- 0.29, PBP 3.01 +- 0.18, Dropout 2.97 +- 0.19
  (Dropout bold, best or tied, on RMSE in 9 of 10 rows; PBP wins Yacht. v6 adds an "Update [October 2016]" table after a standard-error fix).
- **Relevance.** The practical trick that turns any dropout CNN on the platform into an uncertainty estimator at the cost of
  T extra forward passes per unlabeled image. No retraining needed.

## batchbald -- BatchBALD
- **Citation.** Andreas Kirsch, Joost van Amersfoort, Yarin Gal. "BatchBALD: Efficient and Diverse Batch Acquisition for Deep
  Bayesian Active Learning". arXiv:1906.08158v2 (28 Oct 2019). NeurIPS 2019 template (neurips_2019.sty); no venue on abs page.
- **Role.** Fixes DBAL's top-k BALD, which scores points one by one and so picks redundant points. Scores the batch jointly:
  `a_BatchBALD({x_1..x_b}) = H(y_1..y_b) - E_p(w)[H(y_1..y_b | w)]` (paraphrase of Eq. `eq:BB_simple`, `main_neurips_2019.tex` line 471).
  Greedy algorithm is a 1 - 1/e approximation (submodular). Repeated MNIST (each digit x3 plus noise) makes BALD worse than random (Fig. 2).
- **Key number.** Table 1 (MNIST, points to reach 90% / 95% accuracy, 25/50/75% quartiles): BatchBALD 70/90/110 and
  190/200/230; BALD reimplementation 120/120/170 and 250/250/>300; BALD [gal2017deep] 145 and 335.
  The last row equals DBAL Table 1 cells (10% / 5% error), a direct consistency check. MNIST run uses 100 MC dropout samples (line 665).
- **Relevance.** Answers "my labeling batch is 10 or more images per round; will top-k uncertainty waste labels on near-duplicates?" Yes; score the batch jointly.

## badge -- BADGE
- **Citation.** Jordan T. Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, Alekh Agarwal. "Deep Batch Active Learning by
  Diverse, Uncertain Gradient Lower Bounds". arXiv:1906.03671v2 (24 Feb 2020). ICLR 2020 (journal-ref on abs page).
- **Role.** Merges the two parents: core-set (diversity) and uncertainty. Embeds each pool point as the gradient of the loss
  w.r.t. the last layer under the model's own predicted label (magnitude ~ uncertainty), then picks the batch with k-means++
  seeding (diversity). No trade-off hyperparameter (Alg. 1, `algorithm.tex`). Initial M=100; batch B in {100, 1000, 10000}.
- **Key number.** Fig. 4 penalty matrix over 33 (dataset, batch, architecture) settings, column averages (lower is better):
  BADGE 0.66, ALBL 4.08, Coreset 6.9, Conf 5.93, Marg 1.56, Entropy 8.01, Rand 6.01. Coreset and Entropy average worse than Rand.
  At B=10000, Margin degrades while BADGE, ALBL, Coreset are best (Sec. 4, CDF paragraph; `experiments.tex` lines 208-210).
- **Relevance.** Default pick when the team does not know the batch size or the architecture in advance; robust across both.

## vaal -- VAAL
- **Citation.** Samarth Sinha, Sayna Ebrahimi, Trevor Darrell. "Variational Adversarial Active Learning". arXiv:1904.00370v3
  (28 Oct 2019). ICCV 2019 Oral (per arXiv comment).
- **Role.** Task-agnostic and model-free selection: a VAE learns a latent space on labeled + unlabeled images; a
  discriminator learns "labeled vs unlabeled"; query the points the discriminator is most sure are unlabeled.
  Replaces model uncertainty with "how unlike the labeled set does this look". Initial 10%, budget 5% per round (line 217).
- **Key number.** CIFAR-10, 40% labeled: VAAL 80.9% mean accuracy vs 90.16% with all data (Sec. 4.1 paragraph, Fig. 2).
  Sampling time per round, CIFAR-10 (Table 1): VAAL 10.59 s, Core-set 75.33 s, Ensembles w. VarR 20.48 s.
  Competitor claims: "DBAL and MC-Dropout perform below" random on ImageNet (line 251); on Caltech-256 DBAL ~ random and
  MC-Dropout below random (line 245). Budget ablation: b=5% slightly better than b=10% (Fig. 5, line 311). These are a
  method paper's claims about its baselines.
- **Relevance.** Selection that does not depend on the task head, so the same selector serves classification and
  segmentation projects; also evidence for chapter 5 that Bayesian uncertainty can lose to random at scale.

## learnloss -- Learning Loss for Active Learning
- **Citation.** Donggeun Yoo, In So Kweon. "Learning Loss for Active Learning". arXiv:1905.03677v1 (9 May 2019). CVPR 2019 Oral
  (per arXiv comment).
- **Role.** Replaces Bayesian uncertainty with a small learned module that predicts the target loss from intermediate
  features; query the top-K predicted losses. Trained with a pairwise ranking loss (Eq. 2, margin xi=1), so loss scale is ignored.
  Task-agnostic: classification, detection, pose. K=1,000 initial and per cycle. On CIFAR-10 it picks top-K from a random
  subset of M=10,000 (line 206), a trick credited to Beluch et al. and motivated by the same batch-overlap problem core-set describes.
- **Key number.** CIFAR-10 last cycle (Fig. 4): learn loss 0.9101, 0.42% above entropy, 0.91% above core-set (line 242).
  VOC07 detection last cycle (Fig. 6): 0.7338 mAP vs random 0.7117, entropy 0.7222, core-set 0.7171 (line 281).
- **Relevance.** Works for detection and pose, where a softmax entropy does not exist for box regression; a platform
  can attach the module to any production model and rank unlabeled images by predicted loss.
