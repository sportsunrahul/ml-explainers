# frankle2019 -- The Lottery Ticket Hypothesis (SPINE, analysis lens)

## 1. Identity
- Title: The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks
- arXiv 1803.03635, read v5 (2019-03-04, "ICLR camera ready"). v1 submitted 2018-03-09 (abs.html).
- Authors: Jonathan Frankle ... Michael Carbin (MIT CSAIL).
- Venue: ICLR 2019. No repo cited in the source.
- Source read: src/*.tex (main, intro, mnist-oneshot, convolutional, dropout, larger, discussion,
  limitations, app-han, appendix-resnet, convolutional-appendix); paper.pdf for figure numbers and plots.

## 2. Core problem
Pruning removes over 90% of weights after training, but the pruned architectures were reported to be hard to
train from scratch. The paper asks whether sparse subnetworks that train well from the start exist.

## 3. What prior work said (per the paper)
- Li 2017: "Training a pruned model from scratch performs worse than retraining a pruned model" (intro footnote).
- Han 2015: better to keep the trained weights of surviving connections than to re-initialize (intro footnote).
- Random sparse subnetworks (Fig. 1, dashed): the sparser, the slower they learn and the lower the accuracy.

## 4. The one key idea
After pruning a trained network, reset the surviving weights to their ORIGINAL initial values theta_0 (not
random new values). These "winning tickets" then train as fast and as well as the dense network.

## 5. Mechanism (identifying winning tickets; Sec. 1, App. B "Strategy 1")
1. Randomly initialize f(x; theta_0), theta_0 ~ D_theta.
2. Train for j iterations -> theta_j.
3. Prune p% of the parameters in theta_j (smallest magnitudes, layer-wise for LeNet/Conv-2/4/6; global across
   conv layers for VGG-19/ResNet-18) -> mask m.
4. Reset survivors to theta_0 -> winning ticket f(x; m (.) theta_0).
5. Iterative pruning (used throughout): repeat steps 2-4 over n rounds; "each round prunes p^{1/n}% of the
   weights that survive the previous round" (Sec. 1, verbatim; in practice a fixed rate per round, e.g. 20%).
6. Control (random reinitialization): keep mask m, draw a new theta'_0 ~ D_theta, train f(x; m (.) theta'_0).
   3 reinitializations per trial x 5 trials = 15 per point (Sec. 2).
- Output layer pruned at half the rate of the rest for LeNet (Sec. 2). Ticks on the plots follow 20% per round:
  100, 51.3, 26.3, 13.5, 7.0, 3.6, 1.9 ... (LeNet, Fig. 4) and 100, 51.4, 26.5, 13.7, 7.1, 3.7, 1.9, 1.0
  (Conv, Fig. 5).
- Speed proxy: the iteration of minimum validation loss (early-stopping iteration) (Sec. 1 footnote).

## 6. Statement and formalization
Hypothesis, verbatim (src/intro.tex, `\begin{lth}`):
"A randomly-initialized, dense neural network contains a subnetwork that is initialized such that---when
trained in isolation---it can match the test accuracy of the original network after training for at most the
same number of iterations."

Formal version (Sec. 1, TeX verbatim fragments):
- Dense net `$f(x; \theta)$` with `$\theta = \theta_0 \sim \mathcal{D}_\theta$` reaches minimum validation loss
  `$l$` at iteration `$j$` with test accuracy `$a$`.
- Masked net `$f(x; m \odot \theta)$`, `$m \in \{0, 1\}^{|\theta|}$`, initialized `$m \odot \theta_0$`, reaches
  `$l'$` at `$j'$` with accuracy `$a'$`.
- Prediction: `$\exists~m$ for which $j' \leq j$` (commensurate training time), `$a' \geq a$` (commensurate
  accuracy), and `$\lVert m \rVert_0 \ll |\theta |$` (fewer parameters).
- Sparsity notation (Sec. 2): `$P_m  = \frac{\lVert m \rVert_0}{|\theta|}$`, "e.g., $P_m = 25\%$ when 75\% of
  weights are pruned". P_m is the fraction of weights REMAINING.
Macros: math_commands.tex (standard ICLR); `\kernel` = "winning ticket", `\kernels` = "winning tickets".

## 7. Measurement setup (Fig. 2 is a table printed as a figure)
| network | conv layers | FC layers | all / conv weights | iters / batch | optimizer | prune rate per round |
|---|---|---|---|---|---|---|
| LeNet (300-100) | - | 300, 100, 10 | 266K | 50K / 60 | Adam 1.2e-3 | fc 20% |
| Conv-2 | 64, 64, pool | 256, 256, 10 | 4.3M / 38K | 20K / 60 | Adam 2e-4 | conv 10%, fc 20% |
| Conv-4 | 64,64,pool,128,128,pool | 256, 256, 10 | 2.4M / 260K | 25K / 60 | Adam 3e-4 | conv 10%, fc 20% |
| Conv-6 | ...,256,256,pool | 256, 256, 10 | 1.7M / 1.1M | 30K / 60 | Adam 3e-4 | conv 15%, fc 20% |
| "Resnet-18" | 16, 3x[16,16], 3x[32,32], 3x[64,64] | avg-pool, 10 | 274K / 270K | 30K / 128 | SGD 0.1-0.01-0.001, mom. 0.9 | conv 20%, fc 0% |
| VGG-19 | 2x64,pool,2x128,pool,4x256,pool,4x512,pool,4x512 | avg-pool, 10 | 20.0M | 112K / 64 | SGD 0.1-0.01-0.001, mom. 0.9 | conv 20%, fc 0% |
- Datasets: MNIST (LeNet), CIFAR10 (all others). Convs 3x3; Gaussian Glorot init (Fig. 2 caption).
- Trials: 5 per point for tickets; 15 for reinit (Sec. 2, Fig. 5 caption).
- Baseline held fixed: same architecture, same optimizer and iteration budget; only mask and initialization differ.
- VGG-19 / ResNet-18: batchnorm, weight decay 0.0001, augmentation; VGG-19 160 epochs (112,480 iterations), lr /10
  at 80 and 120 epochs; ResNet-18 stages 20,000 / 5,000 / 5,000 iterations (App. I). Section 4 text says ResNet-18
  has 271,000 parameters; Fig. 2 says 274K (source inconsistency).

## 8. Results worth remembering (all accuracy deltas are absolute percentage points; speed is relative)
LeNet, MNIST (Sec. 2, Figs. 3-4):
- Tickets learn faster as P_m falls from 100% to 21%; at 21% early stopping is 38% earlier than the original.
- Test accuracy improves by more than 0.3 points at P_m = 13.5%; returns to the original level at P_m = 3.6%.
- At iteration 50,000: up to 0.35 points better while training accuracy is ~100% (better generalization).
- Random reinit: accuracy drops off from the original at P_m = 21.1%, vs 2.9% for winning tickets. At P_m = 21%
  the ticket reaches minimum validation loss 2.51x faster than when reinitialized and is half a point more accurate.
- One-shot pruning also finds tickets: earlier early-stopping for 67.5% >= P_m >= 17.6%; higher accuracy for
  95.0% >= P_m >= 5.17%; iterative finds smaller tickets.
Conv-2/4/6, CIFAR10 (Sec. 3, Fig. 5):
- Fastest early-stopping: 3.5x (Conv-2, P_m 8.8%), 3.5x (Conv-4, 9.2%), 2.5x (Conv-6, 15.1%).
- Best accuracy gain: 3.4 points (Conv-2, P_m 4.6%), 3.5 (Conv-4, 11.1%), 3.3 (Conv-6, 26.4%).
- All three stay above their original average test accuracy when P_m > 2%.
- Reinit: accuracy drops off faster; but for Conv-2 and Conv-4 reinit accuracy at moderate pruning first holds
  or improves ("structure of the winning tickets alone may lead to better accuracy").
- Dropout 0.5: initial accuracy +2.1, +3.0, +2.4 points; iterative pruning adds up to +2.3, +4.6, +4.7 (Sec. 3,
  Dropout paragraph, Fig. 6).
- Approximate baselines read from Fig. 5 (our reading, uncertain): Conv-2 ~0.65, Conv-4 ~0.75, Conv-6 ~0.78
  test accuracy unpruned. Not stated in text; do not quote as numbers.
VGG-19 and "Resnet-18", CIFAR10 (Sec. 4, Figs. 7-8):
- At lr 0.1 iterative pruning does NOT find winning tickets; pruned nets are no better than reinitialized.
- VGG-19 at lr 0.01: subnetworks within 1 point of the original while P_m >= 3.5% (not tickets: they do not match).
- VGG-19 with linear warmup k = 10000 at lr 0.1: unpruned improves by about 1 point; winning tickets for P_m >= 1.5%.
- ResNet-18 at lr 0.01: best tickets 89.5% for 41.7% >= P_m >= 21.9%, below the original's 90.5% at lr 0.1.
- ResNet-18 with warmup k = 20000 at lr 0.03: 90.5% at P_m = 27.1%; tickets for P_m >= 11.8%. No hyperparameters found
  for tickets at lr 0.1.
- Global vs layer-wise (App. I.1, VGG-19 lr 0.1 warmup 10000): tickets for P_m >= 1.5% global vs >= 6.9% layer-wise.
  VGG-19 first two conv layers have 1728 and 36864 parameters, the last 2.35 million (Sec. 4).
- Discussion (Sec. 5): Liu 2019 ("rethinking") found VGG-19 pruned up to 80% and reinitialized matches; LTH confirms at that
  sparsity but finds tickets up to 98.5% pruned where reinit is much worse.

### Claim card (part of section 8)
- Claim: dense random nets contain sparse subnetworks (10-20% of size or less) that, reset to their original
  init, train in isolation to at least the original test accuracy in at most as many iterations.
- Measurement: iterative magnitude pruning with reset to theta_0; LeNet-300-100 on MNIST, Conv-2/4/6 on CIFAR10
  (plus VGG-19, ResNet-18 with lr changes); compare early-stopping iteration and test accuracy vs the dense net
  and vs the same mask with a fresh random init (15 reinits per point).
- Falsifier: random reinitialization of the same mask matches the winning ticket (then structure alone explains
  it), or no mask at nontrivial sparsity matches the dense net. This is what happens at lr 0.1 on VGG-19 and
  ResNet-18 without warmup (Sec. 4) and what Liu 2019 Sec. 6 reports for structured pruning and for lr 0.1.
- Scale caveat: Frankle 2020 (LMC) shows IMP with reset to step 0 is not matching for ResNet-50 on ImageNet
  (73.7 vs 76.1); IMP subnetworks rewound to epoch 5 become stable, and the paper says stable and matching
  rewinding points "closely coincide" (Sec. 4.3, Fig. 7; see frankle2020lmc notes).

## 9. Limitations stated (Sec. 6) and later fixes
- Only small vision tasks (MNIST, CIFAR10); no ImageNet because iterative pruning needs training "15 or more
  times consecutively". -> LMC (Frankle 2020) reaches ImageNet with one-shot IMP + rewinding.
- Sparse (unstructured) pruning only: "the resulting architectures are not optimized for modern libraries or
  hardware". -> 2:4 structured sparsity (Mishra 2021) for hardware; Liu 2019 for structured pruning.
- Deeper nets need learning-rate warmup. -> LMC: rewinding to iteration k > 0 instead of 0.
- Naming: LMC Table 1 caption says Frankle & Carbin "mistakenly refer to ResNet-20 as ResNet-18". The LTH
  "Resnet-18" (274K params, 20 layers) is the CIFAR ResNet-20.

## 10. Outline feedback
- viz-imp "sourced accuracy-vs-sparsity points from Frankle Fig.": Figs. 3-5 are plots with no tabulated values.
  Use only the text-stated points above (LeNet: 21% / 38% earlier, 13.5% / +0.3, 3.6% regress, reinit drop-off
  21.1% vs 2.9%, 2.51x at 21%; Conv-2/4/6 speedups and gains with their P_m). Any figure-read curve must be
  labelled approximate.
- viz-imp "prune 20%": correct for LeNet FC, VGG-19, ResNet-18; Conv-2/4 prune conv layers 10% and Conv-6 15% per
  round (Fig. 2). The architecture table is Figure 2, not a Table.
- Claim card scale caveat: cite three sources: LTH Sec. 4 (no tickets at lr 0.1 without warmup; ResNet-18 only
  up to lr 0.03 even with warmup), LMC Table 2 + Sec. 4.3 (ResNet-50: not matching at k=0; stable from
  epoch 5, matching at about the same point), Liu 2019 Sec. 6 (at lr 0.1 the winning-ticket init is no better than random).
- Use "ResNet-20" (or "ResNet-18 [sic], a CIFAR ResNet-20") in the family tree / claim card.
- The intuition line "explains pruning more than it saves training" matches the paper's own limitation
  (15+ trainings to find a ticket).
