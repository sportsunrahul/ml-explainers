# Co-teaching

Han, Yao, Yu, Niu, Xu, Hu, Tsang, Sugiyama. "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels". arXiv 1804.06872v3 (NIPS 2018 camera-ready). NeurIPS 2018.

Lens: method paper (input = noisy labeled data stream; trained = two identical-architecture networks f and g with different initializations; output = both networks, test with either). The analysis claim (small loss = likely clean, early in training) leans on Arpit et al. 2017 (sources/arpit), not re-proven here.

Problem and key idea: deep nets eventually memorize wrong labels, so train only on examples that look clean. Each network picks the small-loss fraction of every mini-batch, but hands that pick to its PEER for the gradient step. Because the two networks make different mistakes, one network's selection errors do not loop back into itself.

## Mechanism: Algorithm 1 line by line (src/5143_paper.tex, label `alg:Co-teaching`)
1. Input: w_f, w_g, learning rate eta, fixed tau, epochs T_k and T_max, iterations N_max.
   For T = 1..T_max:
2. Shuffle training set D (noisy).
   For N = 1..N_max:
3. Fetch mini-batch D_bar from D.
4. D_bar_f = argmin over D' with |D'| >= R(T)|D_bar| of loss(f, D'): network f keeps its R(T) fraction of small-loss instances.
5. D_bar_g: the same for network g.
6. w_f = w_f - eta * grad loss(f, D_bar_g): f learns from g's pick.
7. w_g = w_g - eta * grad loss(g, D_bar_f): g learns from f's pick.
8. (end of epoch) R(T) = 1 - min{ (T/T_k) tau, tau }.
9. Output w_f and w_g.
Selection is per mini-batch, by each network's own current loss; nothing is permanently discarded.

## Equations (no numbered equations in this paper; all from src/5143_paper.tex)
Alg 1 step 4:
```latex
\bar{\mathcal{D}}_f = \arg\min_{\mathcal{D}':|\mathcal{D}'|\ge R(T)|\bar{\mathcal{D}}|}\ell(f, \mathcal{D}')
```
Alg 1 step 8 (keep-rate schedule):
```latex
R(T) = 1 - \min\left\lbrace \frac{T}{T_k} \tau, \tau \right\rbrace
```
Sec 4.2 generalization: `R(T) = 1 - \tau \cdot \min \{ T^c / T_k, 1 \}`, c in {0.5, 1, 2}.
Variables: f, g = the two networks; w_f, w_g = their weights; D_bar = current mini-batch; l(f, D') = f's cross-entropy on subset D'; R(T) = fraction of the batch kept at epoch T; tau = final drop rate (set to the noise rate epsilon); T_k = epochs over which the drop rate ramps up linearly; eta = learning rate.
R(T) requirements (Sec 4.2): R(T) in [tau, 1]; R(1) = 1 (no dropping at the start, nets do not memorize noise early); R(T) non-increasing.
Noise transition matrices (Appendix "Definition of noise", label `app:noise`), Q_ij = Pr(y~ = j | y = i), rows = clean class, n classes:
```latex
\text{Pair: } Q = \begin{bmatrix} 1-\epsilon & \epsilon & 0 & \dots & 0\\ 0 & 1-\epsilon & \epsilon & & 0 \\ \vdots & & \ddots & \ddots & \vdots \\ 0 & & & 1-\epsilon & \epsilon \\ \epsilon & 0 & \dots & 0 & 1-\epsilon \end{bmatrix}
\qquad
\text{Symmetry: } Q_{ii} = 1-\epsilon,\; Q_{ij} = \frac{\epsilon}{n-1}\ (i \neq j)
```
Pair = fine-grained confusion with one neighbor class; harder: at eps = 45% the true class has only 10% more mass than the wrong one; at symmetry eps = 50% (5 classes) it has 37.5% more (Sec 4, Fig 2). In this symmetric scheme the flipped label is never the true one (eps/(n-1) goes only to other classes); other papers (e.g. DivideMix Table 1) use a different "symmetric" definition, so numbers are not comparable across papers.

## Why it works (Sec 3, Q1 and Q2)
- Q1, small loss + memorization effect: networks fit clean, easy patterns first (Arpit et al.), so early on small-loss examples are mostly clean. Keep everything at first (R=1), then drop more, before memorization starts.
- Q2, peer exchange: a single self-selecting network (like boosting/active learning, or self-paced MentorNet) feeds its own selection bias back into itself, so error accumulates. With two networks with different decision boundaries, each filters different errors; "error flow" zigzags between them (Fig 1). Analogy in text: peer review of an exam.
- vs MentorNet (sources/mentornet): self-paced MentorNet is one network selecting for itself (self-training), inheriting accumulated error from sample-selection bias.
- vs Decoupling (Malach and Shalev-Shwartz 2017): two networks update only where predictions disagree; noisy labels lie everywhere, so the disagreement region still contains noise; Decoupling does not use memorization. Its label precision equals Standard (Fig 4).
- vs Co-training: Co-training needs two views and is for semi-supervised learning; Co-teaching uses one view and targets noisy labels.

## Hyperparameters (Sec 4 "Network structure", "Experimental setup")
9-layer CNN with LReLU (slope 0.01), dropout 0.25, batch norm (Table 3); Adam, momentum 0.9, initial lr 0.001; batch 128; 200 epochs; two nets same architecture, different init; T_k = 10; tau = epsilon (noise rate assumed known; else estimate on validation); noise eps in {Pair-45%, Symmetry-50%, Symmetry-20%}; 5 repeats; accuracy = average over last 10 epochs.

## Main results: test accuracy (%), mean over last 10 epochs, 5 runs
| Table / setting | Standard | Bootstrap | S-model | F-correction | Decoupling | MentorNet | Co-teaching |
|---|---|---|---|---|---|---|---|
| T4 MNIST Pair-45% | 56.52 | 57.23 | 56.88 | 0.24 | 58.03 | 80.88 | 87.63 +/- 0.21 |
| T4 MNIST Symmetry-50% | 66.05 | 67.55 | 62.29 | 79.61 | 81.15 | 90.05 | 91.32 +/- 0.06 |
| T4 MNIST Symmetry-20% | 94.05 | 94.40 | 98.31 | 98.80 | 95.70 | 96.70 | 97.25 +/- 0.03 |
| T5 CIFAR-10 Pair-45% | 49.50 | 50.05 | 48.21 | 6.61 | 48.80 | 58.14 | 72.62 +/- 0.15 |
| T5 CIFAR-10 Symmetry-50% | 48.87 | 50.66 | 46.15 | 59.83 | 51.49 | 71.10 | 74.02 +/- 0.04 |
| T5 CIFAR-10 Symmetry-20% | 76.25 | 77.01 | 76.84 | 84.55 | 80.44 | 80.76 | 82.32 +/- 0.07 |
| T6 CIFAR-100 Pair-45% | 31.99 | 32.07 | 21.79 | 1.60 | 26.05 | 31.60 | 34.81 +/- 0.07 |
| T6 CIFAR-100 Symmetry-50% | 25.21 | 21.98 | 18.93 | 41.04 | 25.80 | 39.00 | 41.37 +/- 0.08 |
| T6 CIFAR-100 Symmetry-20% | 47.55 | 47.00 | 41.51 | 61.87 | 44.52 | 52.13 | 54.23 +/- 0.08 |
Locations: Table 4 (label `tab:mnist`), Table 5 (`tab:cifar10`), Table 6 (`tab:cifar100`); row = flipping rate, column = method. Discrepancy: Sec 4.1 text says "87.53% for Co-teaching vs. 80.88% for MentorNet"; Table 4 says 87.63 (use the table). F-correction wins at Symmetry-20% on every dataset; Co-teaching wins the hard settings.
Sensitivity (MNIST): Table 7 (R(T) shape, c x T_k): Pair-45% ranges 75.56 (c=0.5, T_k=5) to 88.43 (c=0.5, T_k=10). Table 8 (tau as multiple of eps): Pair-45% 0.5eps 66.74, 0.75eps 77.86, eps 87.63, 1.25eps 97.89, 1.5eps 69.47; Symmetry-50% best 98.62 at 1.25eps; Symmetry-20% best 99.39 at 1.5eps. Dropping somewhat more than the true noise rate helped; too much hurt.

## Figures
- Fig 1: error flow. Left: M-Net, one network, error loops back. Middle: Decoupling, two nets update on disagreement. Right: Co-teaching, each net's small-loss pick goes to the other (zigzag arrows).
- Fig 2: 5-class transition matrices, Pair 45% and Symmetry 50%.
- Fig 3 (MNIST) and Figs 5/6 top rows (CIFAR-10/100): test accuracy vs epoch; Standard rises then falls (memorization); Co-teaching stays high.
- "Label precision" = (# clean labels) / (# all selected labels) in each mini-batch, i.e. purity of the R(T) small-loss pick. Fig 4 (MNIST) and Figs 5/6 bottom rows: precision vs epoch for Standard, Decoupling, MentorNet, Co-teaching. Decoupling's precision equals Standard's; MentorNet and Co-teaching climb high; they tie on Symmetry-50%/20%; Co-teaching is higher on Pair-45%. Even with tied precision, Co-teaching gets higher test accuracy (credited to two networks).
- Figs 7-9 (appendix): full y-axis versions.

## Limitations and follow-ups
Stated: noise rate assumed known for tau; no theoretical guarantee (future work); tested only on synthetic noise (MNIST, CIFAR). Table 8 shows tau = eps is not optimal.
Follow-ups: Co-teaching+ (sources/coteachplus) says the two nets converge to consensus over epochs so Co-teaching reduces to self-training MentorNet; fix = select small-loss only among disagreement examples. DivideMix (sources/dividemix) keeps two nets but uses discarded examples as unlabeled data instead of throwing them away. CL's own CIFAR runs (confident.md Table 2) show Co-Teaching weak on sparse asymmetric noise.

## Labeling decision
During training on a freshly labeled batch with unknown quality, which examples the model should ignore for now: the large-loss tail per mini-batch, with the drop rate set from an estimate of the label error rate; the persistently dropped examples are candidates for a review queue.

## Visual ideas
- Memorization race: one network on a 2D toy set with 45% pair noise; play epochs; show test accuracy rising then falling; toggle "drop large-loss" with an R(T) slider (tau, T_k).
- Two-network relay: a mini-batch of 16 cards (clean or noisy, hidden), each net ranks by loss and passes its keep set across; show label precision per step for self-selection vs cross-selection.
- Transition-matrix toggle: switch pair vs symmetric at eps slider; show how far the true class leads the strongest wrong class (10% vs 37.5%).
