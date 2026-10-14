# Core-set -- Active Learning for CNNs: A Core-Set Approach

**Citation.** Ozan Sener, Silvio Savarese. "Active Learning for Convolutional Neural Networks: A Core-Set Approach".
arXiv:1708.00489v4 (1 Jun 2018). ICLR 2018 (arXiv comment "ICLR 2018 Paper"). Source: `sources/coreset/src/active_learning.tex`.

**Lens.** Method paper leads (input: CNN features of all pool + labeled points; output: b points that best cover the
feature space; no labels or uncertainty used). An analysis claim rides along: uncertainty methods fail in batch mode
because the batch is correlated; random beats them. Falsifier: uncertainty sampling beating random at large b.

## Core problem and key idea
CNNs must query in large batches, and the top-b uncertain points tend to be near-duplicates. Key idea: pick the
batch so every unlabeled point is close (in feature space) to some labeled point. A bound shows the loss gap between
the labeled subset and the whole set is controlled by that covering radius delta, so minimize delta (k-Center).

## Mechanism
1. Train the CNN (VGG-16, from scratch) on the labeled set s0 (fully supervised, or weakly supervised with Ladder networks).
2. Embed every point: l2 distance between activations of the final fully-connected layer (Sec. 4.4, line 429).
3. Greedy k-Center (Alg. 1): repeatedly add the pool point farthest from its nearest labeled/selected point, b times.
4. Robust k-Center (Alg. 2): take the greedy radius delta_2OPT; binary search delta in [delta_2OPT/2, delta_2OPT] with a
   MIP feasibility check (Gurobi) that allows up to Xi outliers to stay uncovered. Return the centers.
5. Send the b points to the oracle; retrain from scratch; repeat.

## Equations (numbers from PDF; file `active_learning.tex`)
(1) AL objective, line 223: `\min_{\mathbf{s}^1 : |\mathbf{s}^1| \leq b} E_{\mathbf{x},y \sim p_\mathcal{Z}} [l(\mathbf{x},y; A_{\mathbf{s}^0 \cup \mathbf{s}^1})]`
- s0 = initial labeled indices; s1 = new batch; b = budget; A_s = learning algorithm trained on s; l = loss.

(3) Upper bound, lines 263-265 (population risk <= generalization error + training error + core-set loss):
`E_{\mathbf{x},y \sim p_\mathcal{Z}} [l(\mathbf{x},y; A_{\mathbf{s}})] \leq \underbrace{\left| E_{\mathbf{x},y \sim p_\mathcal{Z}} [l(\mathbf{x},y; A_{\mathbf{s}})] - \frac{1}{n}\sum_{i \in [n]} l(\mathbf{x}_i,y_i; A_{\mathbf{s}}) \right|}_{\text{Generalization Error}} + \underbrace{\frac{1}{|\mathbf{s}|}\sum_{j \in \mathbf{s}} l(\mathbf{x}_j,y_j; A_{\mathbf{s}})}_{\text{Training Error}} + \underbrace{\left| \frac{1}{n}\sum_{i \in [n]} l(\mathbf{x}_i,y_i; A_{\mathbf{s}}) - \frac{1}{|\mathbf{s}|}\sum_{j \in \mathbf{s}} l(\mathbf{x}_j,y_j; A_{\mathbf{s}}), \right|}_{\text{Core-Set Loss}}`
(alignment markup removed; stray comma kept as in source). Authors argue CNN training and generalization error are small, so the core-set loss is the target, giving (4), line 273.

Theorem 1, lines 289-300, copied as written:
`\left| \frac{1}{n}\sum_{i \in [n]} l(\mathbf{x}_i,y_i; A_{\mathbf{s}}) - \frac{1}{|\mathbf{s}|}\sum_{j \in \mathbf{s}} l(\mathbf{x}_j,y_j;A_{\mathbf{s}}) \right| \leq \delta (\lambda^l + \lambda^\mu LC)+ \sqrt{\frac{L^2 \log(1/\gamma)}{2n}}`
- Assumptions: loss lambda^l-Lipschitz in x and bounded by L; regression function eta_c(x)=p(y=c|x) is lambda^eta-Lipschitz;
  s is a delta_s cover; zero training loss on s. Holds with probability >= 1 - gamma. n = dataset size; C = number of classes.
- Inconsistency in source: the hypothesis says `\lambda^\eta`, the bound (and appendix proof, line 954) says `\lambda^\mu`. Same quantity.
- Reading (line 306): the bound does not depend on the number of labels; a label helps only if it shrinks the covering radius.
  Bound scales with C, which they use to explain weaker CIFAR-100 gains (line 493).
- Lemma 1 (line 325): l2-loss of a CNN with n_c conv and n_fc FC layers is `\frac{\sqrt{C-1}}{C} \alpha^{n_c+n_{fc}}`-Lipschitz
  (alpha = max sum of input weights per neuron). Theory uses l2 loss; experiments use cross-entropy.

(5) k-Center (min-max facility location), line 354:
`\min_{\mathbf{s}^1:|\mathbf{s}^1| \leq b} \max_i \min_{j \in \mathbf{s}^1 \cup \mathbf{s}^0} \Delta(\mathbf{x}_i,\mathbf{x}_j)`
- Delta = l2 distance in feature space. NP-hard; greedy gives `\max_i \min_j \Delta \leq 2 \times OPT` (line 361).

Alg. 1 greedy step (line 342): `u=\arg\max_{i \in [n] \setminus \mathbf{s}} \min_{j \in \mathbf{s}} \Delta(\mathbf{x}_i, \mathbf{x}_j)`

(6) Robust MIP, lines 378-381, copied with the stray comma in the source:
`Feasible(b,\mathbf{s}^0,\delta, \Xi): \sum_j u_j, = |\mathbf{s}^0|+ b, \sum_{i,j} \xi_{i,j} \leq \Xi, \sum_j \omega_{i,j} = 1 \forall i, \omega_{i,j} \leq u_j \forall i,j, u_i =1 \forall i\in \mathbf{s}^0, u_i \in \{0, 1\}, \omega_{i,j} = \xi_{i,j} \forall i,j \mid \Delta(\mathbf{x}_i,\mathbf{x}_j) > \delta`
- u_i = 1 if point i is a center; omega_ij = 1 if i is covered by j; xi_ij = 1 if i is an outlier covered by j beyond delta.
- Xi = outlier budget = 1e-4 x n, n = number of unlabeled points (line 434).

## Main results (all accuracy results are figures; no accuracy table)
Datasets: CIFAR-10, CIFAR-100, SVHN. 5 random initial pools; mean accuracy, std error bars. x-axis = ratio of labeled images.
CIFAR points at ratio 0.1, 0.2, 0.3, 0.4, 1.0; SVHN at about 0.0, 0.05, 0.15, 0.6, 1.0 (approx, read from plot).
Baselines: Random; Best Empirical Uncertainty (best of max-entropy/BALD/VarRatios on softmax); DBAL (MC dropout, best of the three);
Best Oracle Uncertainty (samples proportional to true loss); k-Median; BMDR; CEAL (weakly-supervised only).
- Fig. 4 (fully supervised), ratio 0.4 (approx, read from plot): CIFAR-10 ours ~78.5 vs Random ~74, DBAL ~73;
  CIFAR-100 ours ~54.5 vs Random ~48.5, DBAL ~50.5. SVHN at 0.6: ours ~93.5 vs Random ~91.
- Fig. 3 (weakly supervised), ratio 0.4 (approx): CIFAR-10 ours ~82.5 vs Random ~75, DBAL ~77; CIFAR-100 ours ~58 vs ~53.5.
  SVHN at ~0.15: ours ~94.5 vs others ~88-89.
- Fig. 6 (CIFAR-100, exact point labels), accuracy at ~5k/10k/20k/40k/50k labels: Greedy 0.29/0.43/0.52/0.63/0.65;
  Our Method (MIP) 0.30/0.45/0.54/0.64/0.65.
- Table 1, average run-time in seconds for b=5k, |s0|=10k: Distance Matrix 104.2; Greedy (2-OPT) 2; MIP (iteration) 7.5;
  MIP (total) 244.03; Total 360.23. Hardware: i7-5930K, 64GB.

## Batch-mode claims (quote-level)
- Intro, lines 107-111: ineffectiveness is due to "correlation caused via batch acquisition/sampling"; b=1 is infeasible
  because one point has no significant effect and each round needs full retraining.
- Lines 498-502: oracle and Bayesian uncertainty beat empirical uncertainty, "however, they are still not effective in the
  batch setting since random sampling outperforms them". tSNE (Fig. 5) shows uncertainty picks fail to cover the space.
- Line 142: Bayesian AL works for small data but does "not scale to large-scale datasets because of batch sampling".

## Budget facts (only what the source states)
x-axis ratios above; Fig. 5 caption: initial labeled pool 1000, 1000 chosen; Table 1: b=5k, |s0|=10k.
No single per-round budget is stated in the experimental text. Optimizer: RMSProp, lr 1e-3, TensorFlow, He init.

## Paper's figures
- Fig. 1: balls of radius delta_s around blue labeled points cover red unlabeled points; bound = O(delta) + O(sqrt(1/n)).
- Fig. 2: MIP variables on a 5-node toy: node 4 is a center, nodes 0,1,3 in its ball, node 2 an outlier.
- Fig. 3 / Fig. 4: accuracy vs labeled ratio, weakly / fully supervised, three datasets.
- Fig. 5: CIFAR tSNE; uncertainty oracle picks (green) clump; core-set picks spread evenly.
- Fig. 6: MIP vs greedy on CIFAR-100, MIP slightly higher.

## Limitations (stated) and follow-ups
- Theory assumes zero training error and l2 loss; experiments use cross-entropy.
- Weaker with many classes (CIFAR-100). Uses no uncertainty; combining is "an open problem" (line 496).
- Better features (weakly supervised) help a lot, so quality depends on the embedding.
- Naming: baseline list says "k-Median" (line 452), discussion says "k-Medoids" (line 496).
- Follow-ups: BADGE (gradient embeddings mix uncertainty + diversity; finds Coreset worse than Random overall, Fig. 4 penalty 6.9 vs 6.01);
  VAAL reports Core-set strong on classification but weak on high-resolution segmentation.

## Labeling decision
How to fill a large labeling batch (thousands of images per round): spread picks across feature space so no
region is far from a labeled example, instead of taking the top-k most uncertain, which clump.

## Visual ideas
1. Drag-the-batch: 2D point cloud; reader picks b; compare greedy k-Center vs top-b uncertainty; show covering radius delta live.
2. Greedy step-through: animate Alg. 1 (farthest point added each step) with the max-min distance shrinking; show the 2x bound band.
3. Outlier toggle: add a few far outliers; show plain k-Center wasting centers on them; slide Xi to let the robust version ignore them.
