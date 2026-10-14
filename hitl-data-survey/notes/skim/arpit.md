# Arpit et al. 2017 -- A Closer Look at Memorization in Deep Networks

Citation: Devansh Arpit, Stanislaw Jastrzebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S. Kanwal,
Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, Simon Lacoste-Julien. "A Closer Look at Memorization in
Deep Networks". arXiv 1706.05394v2 (1 Jul 2017). ICML 2017.

## Lens
Analysis paper (leads). Nothing new is trained for use; the paper contrasts training dynamics on real vs noise data.
- Claim: DNNs trained with SGD fit simple, shared patterns (the clean part) first and memorize noise later; explicit
  regularizers (dropout most) slow memorization of noise without hurting learning on real data.
- How measured: randX (inputs replaced by Gaussian noise with matched mean/variance) and randY (labels replaced by random
  labels), at 100% or 20-80% of the training set; per-example difficulty, loss-sensitivity, time-to-convergence, and the
  critical sample ratio (CSR); train vs validation curves at each noise %.
- Falsifier: if validation accuracy on randY-corrupted data did not peak before train accuracy saturates, or if CSR and
  per-example difficulty looked the same on real and noise data, the "patterns first" claim would fail.

## Core problem and key idea
Zhang et al. (1611.03530) showed nets fit random labels, so capacity alone cannot explain generalization. Arpit et al. ask
whether nets also memorize real data the same way. Key idea: define memorization operationally as "what a net does on
noise data", then show the learning dynamics on real data differ, and with mixed data the clean part is learned first.

## Mechanism (experiments, step by step)
1. Easy vs hard examples (Sec 3.1, Fig 1): train an MLP 1 epoch from 100 random inits/shuffles; record each of 1000
   examples' misclassification rate. Real data: wide spread (some examples always right). randX: matches a Binomial(n=100).
   (Fig 1 legend reads "CIFAR-10", the text says MLP; UNSURE which dataset.)
2. Loss-sensitivity (Sec 3.2, Fig 3/4): gradient of later loss w.r.t. an earlier example, via unrolled SGD; Gini
   coefficient across examples. Real data: few examples have high sensitivity. Random data: almost all are high.
   Setup: FC net, 2 layers x 16 units, 1000 downscaled 14x14 MNIST digits, SGD.
3. Capacity (Sec 3.3.1, Fig 5): MNIST MLPs. With noise, the best validation accuracy needs more hidden units.
4. Time-to-convergence (Sec 3.3.2, Fig 6): epochs to reach 100% train acc, vs capacity (data fixed at 50000) and vs data
   size (capacity fixed at 4096). Noise data is hurt more by less capacity / more data. Curves are normalized; no
   absolute epoch counts are stated in the text.
5. CSR through training (Sec 4.2, Fig 7/8/9): fraction of validation points with an adversarial point within an
   L-inf box r. Higher CSR = denser decision boundaries = more complex hypothesis.
6. Noise % sweep (Sec 4.2, Fig 7 MNIST MLP, Fig 8 CIFAR-10 CNN): 20%-80% of training points get randX or randY.
7. Regularizers (Sec 5, Fig 10/11): CIFAR-10 CNN, train on randY, plot randY train acc vs best real validation acc.

## Equations (all in sources/arpit/src/main.tex)
Effective capacity (Sec 1, unnumbered):
  `EC(\mathcal{A}) = \{h \mid  \exists \mathcal{D} \text{ such that } h \in \mathcal{A}(\mathcal{D}) \}`
  - A: learning algorithm (model + training procedure); D: a dataset; A(D): set of hypotheses A can reach on D.
Loss-sensitivity (Sec 3.2, unnumbered): `g^t_\mathbf{x}=\|\partial \mathcal{L}_t/\partial \mathbf{x}\|_1`
  - L_t: loss after t SGD updates; x: an earlier training example; bar g_x = average of g^t_x over T steps.
Class-specific sensitivity (Sec 3.2, unnumbered):
  `\bar{g}_{i, j} =\mathbb{E}_{(x,y)}\sfrac{1}{T}\sum_t^T|\partial \mathcal{L}_t(y=i)/\partial x_{y=j}|`
  - L_t(y=i): cross-entropy term for class i; x_{y=j}: training examples of class j.
Critical sample, Eq. (1) (Sec 4.1):
  `\arg \max_{i} f_{i}(\mathbf{x}) \neq \arg \max_{j} f_{j}(\hat{\mathbf{x}}) \mspace{5mu} \mbox{s.t.  } \lVert \mathbf{x} - \hat{\mathbf{x}} \rVert_{\infty} \leq r`
  - f: network output vector over k classes; x: data point; x-hat: nearby point; r: box size. Uses predicted, not true, label.
CSR (Sec 4.1, unnumbered): `\frac{\# \text{critical samples}}{|\mathcal{D}|}`
LASS step (Algorithm 1, line 4): `\Delta = \alpha \cdot \mbox{sign}(\frac{\partial f_{k}(\mathbf{x})}{\partial \mathbf{x}}) + \beta \cdot \eta`
  - alpha: FGSM step; beta: noise scale; eta: standard normal noise; then clip each coordinate to the r-box around x.

## Main results (no tables in this paper; numbers are text-stated or read from figures)
Text-stated:
- Sec 4.2: "for randY datasets ... the network achieves maximum accuracy on the validation set before achieving high
  accuracy on the training set. Thus the model first learns the simple and general patterns of the real data before
  fitting the noise (which results in decreasing validation accuracy)." CSR "greatly increases" in the noise-fitting phase.
- Sec 4.2: final and maximum validation accuracies are lower for noisier datasets (both randX and randY).
- Sec 5: "dropout being the most effective"; "dropout, especially coupled with adversarial training, is best at hindering
  memorization without reducing the model's ability to learn."
- Sec 3.3.1: on CIFAR-10 randY, no relation between capacity and validation performance was found.
Read from figures (approx, eyeballed from src/fig_png):
- Fig 8b left (CIFAR-10 CNN, randY 0/0.2/0.4/0.6/0.8, 200 epochs): all train curves reach 1.0 by about epoch 25-65.
  Final validation acc approx 0.73 / 0.58 / 0.44 / 0.31 / 0.19. With 0.2 noise, validation peaks near 0.67 around
  epoch 20, then drops to about 0.58 as the noise is fit. (approx)
- Fig 7b left (MNIST MLP, randY, 100 epochs): validation peaks about 0.9-0.97 in the first ~5 epochs for all noise
  levels, then falls to about 0.89 / 0.75 / 0.55 / 0.33 for 0.2 / 0.4 / 0.6 / 0.8 noise. (approx) Clearest "peak then fall".
- Fig 9 (CIFAR-10 CNN, validation CSR, 140 epochs): real data plateaus near 0.2; randX near 0.76 by epoch 30; randY
  rises slowly to about 0.74. (approx)
- Fig 8b right (CIFAR-10 randY CSR; panel has no legend, noise mapping inferred from color order of Fig 8b left): final CSR approx 0.26 / 0.45 / 0.55 / 0.64 / 0.67 for noise 0 to 0.8. (approx)
- Fig 10 (CIFAR-10): dropout and adversarial+dropout keep best real validation acc near 0.76-0.77 while randY train
  acc is pushed down to about 0.1-0.2; weight decay holds about 0.72-0.74; input gaussian / gaussian / input binary mask
  lose real accuracy as they cut randY fitting. (approx)
- Fig 6 left (MNIST MLP, input-noise level 0 to 1): normalized time-to-convergence from 256 to 4096 units drops to
  about 0.82 for clean data and about 0.40 for 100% noise. (approx)

## Hyperparameters a figure would need (Sec 2, footnotes)
- MNIST: 2-layer ReLU MLP, 4096 units/layer, 1000 epochs, SGD lr 0.01 (default).
- CIFAR-10: small AlexNet-style CNN: Crop(2,2) -> Conv(200,5,5) -> BN -> ReLU -> MaxPool(3,3) -> Conv(200,5,5) -> BN ->
  ReLU -> MaxPool(3,3) -> Dense(384) -> BN -> ReLU -> Dense(192) -> BN -> ReLU -> Dense(#classes) -> Softmax.
  SGD momentum 0.9, lr 0.01 halved every 15 epochs.
- LASS: alpha = 0.25, beta = 0.2, eta ~ standard normal; box r = 0.3 on a 0-255 pixel scale.
- Regularizer ranges (Sec 5): dropout 0-0.9; input dropout 0-0.9; input Gaussian noise std 0-5; hidden Gaussian noise
  0-0.3; weight decay 0-1; adversarial training weight 0.2-0.7 with dropout 0.03-0.5.
- Easy/hard: 100 inits, 1 epoch, 1000 examples. Loss-sensitivity: 2x16 FC net, 1000 MNIST digits at 14x14.

## What the figures show (one line each)
- Fig 1: sorted per-example P(correct) after 1 epoch; real data spans 0-1, randX/randY stay flat near 0.1 (approx), like a binomial.
- Fig 2: first-layer filters, real CIFAR-10 vs randY; text says randY filters still show shared patterns (not viewed).
- Fig 3: Gini of loss-sensitivity over training; real data high inequality, random data low.
- Fig 4: class-by-class sensitivity heatmaps; real data more off-diagonal spread.
- Fig 5: validation acc vs hidden units for noisy MNIST; noise needs more units.
- Fig 6: normalized time-to-convergence vs capacity and vs dataset size, one line per input-noise level.
- Fig 7/8: train (solid) and validation (dotted) acc plus CSR per epoch, for randX and randY at 0-80% noise.
- Fig 9: validation CSR over epochs, real vs randX vs randY.
- Fig 10: per regularizer, randY train acc (x) vs best real validation acc (y); flat = good.
- Fig 11: train acc curves per regularizer on randY vs real data.

## Limitations (stated) and follow-ups
- Memorization is not formally defined; only operationalized via noise data (Sec 1).
- Small models (2-layer MLPs, small CNN), MNIST and CIFAR-10 only; synthetic, uniform label noise only.
- Why nets find generalizable solutions is left open (Sec 7 Conclusion).
- Follow-ups that use the "clean first" effect: Co-teaching / small-loss selection, early-learning regularization (ELR),
  DivideMix. CIFAR-N (2110.12088) shows human noise is memorized more easily than synthetic noise at the same rate.

## Labeling decision
When to stop training (or when to trust per-example loss) on a dataset with suspect labels: early in training, examples the
model still gets wrong are good candidates for human relabel review. Long training with noisy labels erases that signal.

## Visual ideas
1. Two-curve slider: pick noise % (0-80); animate train vs validation accuracy per epoch; mark the validation peak
   ("stop here") and show the gap growing as noise is memorized.
2. Decision-boundary toy: 2D points with a % of flipped labels; step an MLP through epochs; show the boundary stay smooth
   first, then wrap islands around flipped points; live CSR counter.
3. Regularizer knob: dropout rate slider; plot "fit on random labels" vs "accuracy on real data" as in Fig 10.
