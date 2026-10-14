# hinton2015 -- Distilling the Knowledge in a Neural Network (SPINE)

## 1. Identity
- Title: "Distilling the Knowledge in a Neural Network".
- arXiv 1503.02531, version read: v1 (the only version). Submitted 2015-03-09 (abs.html).
- Authors: Geoffrey Hinton, Oriol Vinyals, Jeff Dean (Google). Hinton and Vinyals marked equal contribution.
- Venue: abs.html comments "NIPS 2014 Deep Learning Workshop".
- Repo: none.
- Sources read: `src/*.tex` (all 9 section files), `paper.pdf` (9 pages) for page and table numbers.

## 2. Core problem
Ensembles and large regularized nets generalize well but cost too much to deploy. The paper asks how to move
that knowledge into a small model. The key barrier is "we tend to identify the knowledge in a trained model with
the learned parameter values" (Sec. 1, p. 1).

## 3. What prior methods did and why that hurt
- Caruana et al. (model compression): train the small model to match the big model's outputs. On easy tasks the
  big model is nearly one-hot, so the useful information lives in tiny probabilities that "has very little
  influence on the cross-entropy cost function during the transfer stage because the probabilities are so close
  to zero" (Sec. 1, p. 2).
- Caruana's fix: regress the logits with squared error. Hinton shows this is the high-T limit of distillation
  (Sec. 2.1), and argues lower T can be better because very negative logits "could be very noisy".
- Li et al. (speech): distill at T = 1 on unlabeled data; "only reduces the error rate of the small model by
  28% of the gap" (Sec. 4.1, p. 5).

## 4. The one key idea
Raise the softmax temperature of the trained "cumbersome" model until its outputs are soft, and train the small
model to match them at the same temperature, plus a lightly weighted hard-label term. The relative
probabilities of wrong answers ("dark knowledge"; the paper does not use that phrase) carry how the big model
generalizes.

Paper's example (Sec. 1, p. 2), verbatim: "An image of a BMW, for example, may only have a very small chance of
being mistaken for a garbage truck, but that mistake is still many times more probable than mistaking it for a
carrot."
Second example (Sec. 1, p. 2), verbatim: "one version of a 2 may be given a probability of $10^{-6}$ of being a
3 and $10^{-9}$ of being a 7 whereas for another version it may be the other way around."

## 5. Mechanism, step by step
1. Train the cumbersome model (an ensemble, or one big net with strong regularization such as dropout) the usual
   way. "Once the cumbersome model has been trained, we can then use a different kind of training, which we
   call 'distillation'" (Sec. 1, p. 1).
2. For each transfer-set case, run the cumbersome model with temperature T in its softmax -> soft targets p_i.
   For an ensemble, use "an arithmetic or geometric mean of their individual predictive distributions" (Sec. 1).
3. Run the small (distilled) model on the same case. Compute its softmax at the same T -> q_i. Loss 1 =
   cross-entropy between p and q (both at T).
4. If labels are known, compute the small model's softmax of the same logits at T = 1. Loss 2 = cross-entropy
   with the correct labels.
5. Objective = weighted average of loss 1 and loss 2, with "a condiderably [sic] lower weight on the second
   objective function" (Sec. 2, p. 3). Multiply the soft-target term by T^2 (see Eq. section).
6. Backprop into the small model only. After training, the small model runs at T = 1: "The same high temperature
   is used when training the distilled model, but after it has been trained it uses a temperature of 1"
   (Sec. 2, p. 3).
7. Transfer set: "could consist entirely of unlabeled data ... or we could use the original training set. We
   have found that using the original training set works well" (Sec. 1, p. 2).

Teacher is fixed: the paper never says "frozen". Anchors: (a) the intro sentence in step 1; (b) soft targets
are "produced by using the cumbersome model with a high temperature" (Sec. 2); (c) Eq. 2 differentiates only
w.r.t. the distilled model's logits z_i, with the teacher logits v_i given; (d) Table 5 caption: "The soft
targets are obtained by training on the full training set." Calling the teacher "frozen" with a stop-gradient
is `our reading` of these anchors.

## 6. Equations (TeX verbatim from `src/distillation11.tex`; numbers from the PDF)
No custom macros are used in Eq. 1-4. Note Eq. 1 writes bare `exp`, not `\exp`. `common.tex` defines `\bp`,
`\bq`, `\bz`, `\bx`, `\bs`, `\btheta` as bold letters (used in Sec. 4-5 only).

**Eq. 1 (p. 2), softmax with temperature**
```
q_i = \frac{exp(z_i/T)}{\sum_j exp(z_j/T)}
```
Paper: a softmax "converts the logit, $z_i$, computed for each class into a probability, $q_i$, by comparing
$z_i$ with the other logits ... where $T$ is a temperature that is normally set to $1$. Using a higher value for
$T$ produces a softer probability distribution over classes."
Gloss: z_i = logit for class i; T = temperature; q_i = probability for class i; sum over all classes j.

**Objective (Sec. 2, p. 3): prose only, no equation in the paper.** Verbatim:
"we found that a better way is to simply use a weighted average of two different objective functions. The first
objective function is the cross entropy with the soft targets and this cross entropy is computed using the same
high temperature in the softmax of the distilled model as was used for generating the soft targets from the
cumbersome model. The second objective function is the cross entropy with the correct labels. This is computed
using exactly the same logits in softmax of the distilled model but at a temperature of 1. We found that the
best results were generally obtained by using a condiderably lower weight on the second objective function."
("condiderably" is the typo in both TeX and PDF; quote with [sic].)

**Why T^2 (Sec. 2, p. 3), verbatim:**
"Since the magnitudes of the gradients produced by the soft targets scale as $1/T^2$ it is important to multiply
them by $T^2$ when using both hard and soft targets. This ensures that the relative contributions of the hard
and soft targets remain roughly unchanged if the temperature used for distillation is changed while
experimenting with meta-parameters."

**Eq. 2 (p. 3, label `deriv`), gradient of the soft-target cross-entropy**
```
\frac{\partial C}{\partial z_i} = \frac{1}{T}\left(q_i - p_i\right) = \frac{1}{T}\left(\frac{e^{z_i/T}}{\sum_j e^{z_j/T}} -\frac{e^{v_i/T}}{\sum_j e^{v_j/T}}\right)
```
Paper: "Each case in the transfer set contributes a cross-entropy gradient, $dC/dz_i$, with respect to each
logit, $z_i$ of the distilled model. If the cumbersome model has logits $v_i$ which produce soft target
probabilities $p_i$ and the transfer training is done at a temperature of $T$, this gradient is given by".
Gloss: C = soft-target cross-entropy at temperature T; z_i = student logit; v_i = teacher logit; q_i, p_i =
student and teacher softmax at T. `our reading`: KL(p||q) = CE(p,q) - H(p), and H(p) does not depend on the
student, so the KL form used by DeiT has the same gradient.

**Eq. 3 (p. 3, label `mess`), high-temperature approximation**
```
\frac{\partial C}{\partial z_i} \approx \frac{1}{T}\left( \frac{1+z_i/T}{N + \sum_j z_j/T} - \frac{1+v_i/T}{N + \sum_j v_j/T} \right)
```
Paper: "If the temperature is high compared with the magnitude of the logits, we can approximate".
Gloss: first-order expansion e^x ~ 1 + x; N = number of classes.

**Eq. 4 (p. 3, label `nice`), zero-meaned logits**
```
\frac{\partial C}{\partial z_i} \approx \frac{1}{NT^2}\left( z_i - v_i \right)
```
Paper: "If we now assume that the logits have been zero-meaned separately for each transfer case so that
$\sum_j z_j = \sum_j v_j = 0$ Eq. \ref{mess} simplifies to". Then: "So in the high temperature limit,
distillation is equivalent to minimizing ${1/2}(z_i-v_i)^2$, provided the logits are zero-meaned separately for
each transfer case." This is the "matching logits is a special case" claim (Sec. 2.1 title).
Gloss: gradient = logit difference scaled by 1/(N T^2); this 1/T^2 is the reason for the x T^2 factor.
Low-T trade-off, verbatim: "At lower temperatures, distillation pays much less attention to matching logits
that are much more negative than the average." And: "when the distilled model is much too small to capture all
of the knowledege [sic] in the cumbersome model, intermediate temperatures work best which strongly suggests
that ignoring the large negative logits can be helpful."

**Eq. 5 (p. 6, label `eq:kl`), specialist ensemble inference (not distillation)**
```
KL (\bp^g, \bq) + \sum_{m \in A_k} KL (\bp^m , \bq)
```
q minimized per image by gradient descent on logits z with q = softmax(z), T = 1.

## 7. Config / setup
| Item | MNIST (Sec. 3, p. 3-4) | Speech (Sec. 4, p. 4-5) |
|---|---|---|
| Teacher | 1 net, 2 hidden layers x 1200 ReLU; dropout + weight constraints; input jitter up to 2 px; all 60,000 training cases | 10 separately trained models, same arch as baseline, different random inits |
| Student | 2 hidden layers x 800 ReLU, no regularization | "Distilled Single model", same arch as baseline (not smaller) |
| Base arch | -- | 8 hidden layers x 2560 ReLU; softmax 14,000 labels (HMM states); input 26 frames x 40 Mel filterbank, 10ms advance, predict 21st frame; ~85M params |
| Data | MNIST 60,000 train; transfer set has no translations | ~2000 hours English, ~700M training examples; WER on 23K-word test set |
| Temperature | 20 (for the 74-error result); >=300 units: "all temperatures above 8 gave fairly similar results"; 30 units: 2.5 to 4 best | tried [1, 2, 5, 10]; 2 best (bold) |
| Hard-label weight | not stated | "relative weight of 0.5 on the cross-entropy for the hard targets" |
| Optimizer | not stated | "distributed stochastic gradient descent" |

## 8. Results worth remembering
MNIST (Sec. 3, p. 4; prose, no table):
- Teacher 67 test errors. Student 2x800 without regularization 146 errors. Same student "regularized solely by
  adding the additional task of matching the soft targets produced by the large net at a temperature of 20"
  -> 74 test errors.
- Omit digit 3 from transfer set: 206 test errors, 133 of them on the 1010 threes. Bias for class 3 raised by
  3.5 -> 109 errors, 14 on 3s; "98.6% of the test 3s correct despite never having seen a 3 during training".
- Transfer set of only 7s and 8s: 47.3% test errors; biases for 7 and 8 reduced by 7.6 -> 13.2% test errors.
Speech, Table 1 (p. 5), verbatim (columns: System | Test Frame Accuracy | WER):
| System | Test Frame Accuracy | WER |
|---|---|---|
| Baseline | 58.9% | 10.9% |
| 10xEnsemble | 61.1% | 10.7% |
| Distilled Single model | 60.8% | 10.7% |
- Prose (Sec. 4.1): "More than 80% of the improvement in frame classification accuracy achieved by using an
  ensemble of 10 models is transferred to the distilled model". Note Sec. 4 prose calls 58.9% / 10.9% the
  "development set" figures; Table 1 heads the column "Test Frame Accuracy". Both as printed.
JFT specialists (Sec. 5, Table 3 p. 7):
- JFT: 100 million labeled images, 15,000 labels; baseline trained ~6 months. 61 specialists x 300 classes
  (+ dustbin); specialists train in "a few days instead of many weeks".
- Table 3: Baseline 43.1% conditional / 25.0% test; "+ 61 Specialist models" 45.9% / 26.1%. Prose: "4.4%
  relative improvement in test accuracy overall" (relative).
- Specialists were NOT distilled back: "We have not yet shown that we can distill the knowledge in the
  specialists back into the single large net" (Sec. 8).
Soft targets as regularizer, Table 5 (p. 8), columns Train Frame Accuracy | Test Frame Accuracy:
| System & training set | Train | Test |
|---|---|---|
| Baseline (100% of training set) | 63.4% | 58.9% |
| Baseline (3% of training set) | 67.3% | 44.5% |
| Soft Targets (3% of training set) | 65.4% | 57.0% |
- 3% = "about 20M examples". Hard-target run needed early stopping; soft-target run "simply 'converged' to 57%".

## 9. Limitations stated
- Specialists not distilled back into one net (Sec. 8). Specialists use a dustbin class; soft targets to stop
  specialist overfitting are "currently exploring" (Sec. 6.1).
- Very small students prefer intermediate T; T is a tuned meta-parameter (Sec. 3, Sec. 2.1).
- `our reading` (not stated by the paper): classification only. Detectors: Wang 2019 shows plain logit KD gives +0.9 mAP on a toy detector (wang2019fgfi
  Table 1, row 0.25x-D).

## 10. Outline feedback
- `ch-distill` "Sourced: Hinton MNIST and speech results (Tables 1-2)" is wrong. MNIST is prose (Sec. 3);
  Table 1 = speech; Table 2 = JFT cluster examples; Table 5 = the 3% regularizer result.
- The loss "(1-a) CE(y, q^S) + a T^2 KL(q^T || q^S)" is not written in Hinton; it is our transcription of
  Sec. 2 prose. The written form matches DeiT Eq. 2 (touvron2021deit). Hinton says cross-entropy, not KL (same
  gradient, `our reading`). Label it so.
- "a" must be large (soft weight high). Hinton: "condiderably lower weight" on the hard term; speech used 0.5 on
  the hard term (relative). DeiT uses lambda = 0.1 on the soft term, the opposite convention; flag it if both
  appear.
- Intuition example: the outline's truck/bus/cat is ours. The paper's is BMW / garbage truck / carrot; use it
  or label ours.
- viz-temp slider 1..20 matches the paper's range (MNIST T = 20; speech best T = 2; small-net best 2.5 to 4).
- Speech result distills a 10-model ensemble into ONE model of the same size, not a smaller student. Say so.
- "Frozen teacher" in viz-distflow is `our reading` (see Sec. 5 anchors); FGFI code freezes it explicitly.
