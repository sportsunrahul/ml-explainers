# DBAL -- Deep Bayesian Active Learning with Image Data

**Citation.** Yarin Gal, Riashat Islam, Zoubin Ghahramani. "Deep Bayesian Active Learning with Image Data".
arXiv:1703.02910v1 (8 Mar 2017). Written in the ICML 2017 template (icml2017.sty); the arXiv v1 page states no venue.
Source used: `sources/dbal/src/example_paper.tex` (top level). It matches the v1 PDF (Turing affiliation,
"We continue with a comparison", "second place winner"). The nested `ICML2017 (arxiv)/` copy is an older draft; ignore it.

**Lens.** Method paper leads (trace: labeled set -> dropout CNN -> T stochastic passes -> acquisition score -> top-k
to oracle -> retrain from scratch). A small empirical claim rides along: Bayesian uncertainty beats deterministic softmax.

## Core problem and key idea
Active learning needs a model that learns from little data and knows what it does not know. Deep CNNs had neither.
Key idea: train a normal CNN with dropout, keep dropout ON at test time, and treat the T different predictions
as samples from a posterior. Plug those samples into classic acquisition functions (entropy, BALD, variation ratios).

## Mechanism
1. Start: 20 labeled MNIST images (random but class-balanced), 100-point validation set (tunes weight decay), 10K test set.
   Pool = all remaining MNIST training points. No pool subset is described (tex lines 405-407).
2. Train the CNN with dropout on the labeled set.
3. For every pool image, run T stochastic forward passes (MC dropout), get T softmax vectors p^t.
4. Score each image with an acquisition function; take the top 10.
5. Oracle labels them; add to training set; retrain. Repeat 100 times (so 1000 acquired images). 3 repeats, averaged.
6. Test error is measured "using the dropout approximation at test time" (line 406), while acquisition uses MC dropout.
   My reading: standard dropout weight scaling at test, not T-sample averaging.

## Equations (all unnumbered in source; file `sources/dbal/src/example_paper.tex`)
Predictive distribution via MC dropout (Sec. 3, lines 217-223):
`p(y=c | \x, \D_\train) \approx \int p(y=c | \x, \bo) q_\theta^*(\bo) \td \bo \approx \frac{1}{T} \sum_{t=1}^T p(y=c | \x, \boh_t)`
- `\bo` = all weight matrices {W_1..W_L}; `q_\theta^*` = dropout distribution fitted by training; `\boh_t` = one dropout mask draw; T = number of passes.

Acquisition functions, as written (Sec. 4, lines 243-290):
- Max Entropy: `\bbH [y | \x, \D_\train] := - \sum_c p(y=c | \x, \D_\train) \log p(y=c | \x, \D_\train)`
- BALD: `\bbI[y, \bo | \x, \D_\train] = \bbH [y | \x, \D_\train] - \E_{p(\bo | \D_\train)} \big[ \bbH[y | \x, \bo] \big]`
- Variation Ratios: `\varration[\x] := 1 - \max_y p(y | x, \D_\train)`
- Mean STD: `\sigma_c = \sqrt{\E_{q(\bo)}[p(y=c | \x, \bo)^2] - \E_{q(\bo)}[p(y=c | \x, \bo)]^2}`, `\sigma(\x) = \frac{1}{C} \sum_c \sigma_c` (authors call it "more of an ad-hoc technique")
- Random: `a(\x) = \text{unif}()`, a uniform draw on [0, 1].
- Selection rule (line 237): `x^* = \text{argmax}_{x \in \D_\text{pool}} a(x, \mathcal{M})`

BALD MC estimator (lines 330-336):
`\widehat{\bbI} = - \sum_c \bigg(\frac{1}{T} \sum_t \ph_c^t \bigg) \log \bigg(\frac{1}{T} \sum_t \ph_c^t \bigg) + \frac{1}{T} \sum_{c,t} \ph_c^t \log \ph_c^t`,
with `\bph^t = [\ph_1^t, ..., \ph_C^t] = \softmax(\f^{\boh_t}(\x))`
- `\ph_c^t` = probability of class c on pass t; first term = entropy of the averaged prediction; second = minus the average per-pass entropy.
- Plain reading (line 268): high BALD = model is unsure on average, but individual passes are confident and disagree.

## MC dropout details
- Theory text (line 214): "training a model with dropout before every weight layer, and ... dropout at test time as well".
- MISMATCH: the actual MNIST net (line 364) has dropout in only two places: conv-relu-conv-relu-maxpool-**dropout(0.25)**-dense(128)-relu-**dropout(0.5)**-dense-softmax.
  32 conv kernels, 4x4 kernels, 2x2 pooling (Keras MNIST example). No dropout before the conv layers.
- T for MNIST acquisition: **not stated in the paper.** The only explicit T is 20 MC samples for melanoma test logging (line 638).
  BatchBALD (its tex line 665) says it "follow[s] the setup of gal2017deep" with 100 MC dropout samples; that is BatchBALD's choice, not DBAL's.
- Dropout is a variational approximation: q is "a mixture of two Gaussians with small variances and the mean of one ... fixed at zero" (line 216).

## Main results
Table 1 (tex lines 428-433), MNIST, number of acquired images to reach a test error. Caption says "acquired images";
body text (line 420) says "acquisition steps", but values exceed 100 steps, so they are images.

| % error | BALD | Var Ratios | Max Ent | Mean STD | Random |
|---|---|---|---|---|---|
| 10% | 145 | 120 | 165 | 230 | 255 |
| 5% | 335 | 295 | 355 | 695 | 835 |

Note: eight of ten values end in 5 and are not multiples of the acquisition size 10 (120 and 230 are); how they were read off is not stated (UNSURE).
Cross-check: BatchBALD Table 1 quotes "BALD [gal2017deep]" as 145 (90% acc) and 335 (95% acc), the same cells.

Table 2 (lines 504-532), MNIST test error with 1000 labeled images, validation set 5000:
Random 4.66%, BALD 1.80%, Max Entropy 1.74%, Var Ratios 1.64%; semi-supervised Ladder Gamma-model 1.53%, Ladder full 0.84%, DGN 2.40%.
Intro claim (line 176): 5% error with 295 labels vs 835 random.

Melanoma (ISIC 2016, Part 3B), Sec. "Cancer diagnosis", lines 586-652:
- 900 images, 727 benign / 173 malignant. Test set 100+100 balanced; two random test splits; 3 repeats each.
- Initial train 80 neg + 20 pos; acquire 100 per step; pool reshuffled each repeat. Positives augmented by flips.
- Model: VGG16 pretrained on ImageNet, top replaced by 2-way output; two FC-4096 layers each followed by dropout p=0.5.
- 100 epochs per step, batch size 8, weight decay `(1 - p) l^2 / N` with p=0.5, l^2=0.5. Model reset to pretrained weights each step.
- Metric AUC (average precision rejected: predicting all "benign" beat the winner). MC dropout with 20 samples at test.
- Only BALD vs uniform. Var Ratios "fails": nearly all pool points get identical scores (line 642).
- Full-pool model AUC: 0.71 +- 0.003 (split 1), 0.75 +- 0.01 (split 2), both worse than BALD after 4 steps (line 652).
- Fig. 5 split 1, BALD at step 4 about 0.74-0.75 vs uniform about 0.69 (approx, read from plot).
- Step count UNSURE: arithmetic (700 non-test - 100 initial = 600 pool / 100) gives 6 steps, but Fig. 5 x-axis shows 0..4.

## Hyperparameters a figure needs
initial 20 (balanced); acquire 10 per round; 100 rounds; val 100 (Table 2: 5000); test 10K; dropout 0.25 / 0.5;
T not stated for MNIST; melanoma: initial 100, acquire 100/step, T=20 at test, batch 8, 100 epochs. Retrain from scratch every round.

## Paper's figures
- Fig. 1: MNIST test accuracy vs acquired images (0-1000); BALD / VarRatios / MaxEnt cluster high, MeanSTD tracks Random.
- Fig. 2: Bayesian (red) vs deterministic (blue) CNN for BALD, VarRatios, MaxEnt; Bayesian higher later and at convergence.
- Fig. 3: two-digit MNIST vs MBR (Zhu 2003, RBF graph); even Random+CNN beats MBR.
- Fig. 4: four example lesions (2 benign, 2 malignant).
- Fig. 5: AUC and number of acquired positives vs step, two test splits; BALD acquires more malignant cases than uniform.

## Limitations (stated)
- Retraining to convergence after every acquisition is slow (20 hours per melanoma experiment); not resetting may fall into local optima (Sec. 5).
- Small-data evaluation is noisy: AUC differs a lot between the two test splits.
- Not stated by authors but fixed later: top-k BALD picks redundant points in a batch -> BatchBALD (1906.08158), core-set (1708.00489).

## Labeling decision
Which unlabeled images to send to annotators next, when the model is a dropout CNN: rank by disagreement across
T stochastic passes and send the top ones. Also: BALD found rare positives (malignant) faster than random.

## Visual ideas
1. Dropout dial: one image, slider for T; show T softmax bars, their mean, entropy, and BALD as two stacked bars (total vs average-per-pass).
2. Two-image contrast: "noisy label" image (all passes agree on 50/50: high entropy, low BALD) vs "unknown" image (passes disagree confidently: high BALD).
3. Race chart: replay Table 1 as horizontal bars for 10% and 5% error; toggle Bayesian vs deterministic (Fig. 2 idea).
