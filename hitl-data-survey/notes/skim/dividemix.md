# DivideMix

Li, Socher, Hoi. "DivideMix: Learning with Noisy Labels as Semi-supervised Learning". arXiv 2002.07394v1. ICLR 2020.

Lens: method paper (input = noisy labeled training set; trained = two networks theta(1), theta(2) plus a 2-component GMM refit each epoch per network; output = average of both networks' predictions at test time). The "Best/Last" reporting choice matters for reading its numbers (see Results).

Problem and key idea: sample selection (Co-teaching) throws away suspected-noisy examples, wasting their images. DivideMix splits the data each epoch into a likely-clean labeled set and a likely-noisy set whose labels are dropped, then trains with semi-supervised learning (MixMatch) on both. Two networks divide the data for each other, so a network never trains on its own split.

## Mechanism (Alg 1 in src/table/Alg1.tex, label `alg:dividemix`; Sec 3 in src/sec_method.tex)
1. Warm up both nets on all data with cross-entropy for a few epochs (+ confidence penalty for asymmetric noise).
2. Each epoch, co-divide (Alg 1 lines 4-8): compute each sample's CE loss under theta(1); fit a 2-component GMM by EM; w_i = posterior of the low-mean component = "clean probability". This W(2) splits data for theta(2) (and vice versa): labeled set X = {w_i >= tau}, unlabeled set U = {w_i < tau}.
3. Train each net k in turn (other net fixed). Per mini-batch of B labeled (x_b, y_b, w_b) and B unlabeled u_b, make M augmentations of each.
4. Label co-refinement (lines 17-18): p_b = net k's own prediction averaged over the M augmentations of x_b; mix with the given label using w_b, which came from the OTHER net's GMM (Eq 3); then sharpen with temperature T (Eq 4).
5. Label co-guessing (line 20): q_b = average of BOTH nets' predictions over the M augmentations of u_b; sharpen.
6. MixMatch: mixup across the combined labeled + unlabeled batch with lambda' = max(lambda, 1 - lambda) (Eqs 5-8), so mixed labeled samples stay closer to labeled inputs.
7. Loss = CE on mixed labeled + lambda_u * MSE on mixed unlabeled + lambda_r * uniform-prior regularizer (Eqs 9-12); SGD step.
Two nets stay diverged through different init, different splits, different mini-batch order, different targets.

## Equations (src/sec_method.tex; numbers verified in paper.pdf)
Eq (1) per-sample loss: `\ell(\theta)=\big\{\ell_i\big\}_{i=1}^N=\big\{-\sum_{c=1}^C y_i^c \log(\mathrm{p_{model}^c}(x_i;\theta))\big\}_{i=1}^N`
Clean probability (unnumbered, Sec 3.1): w_i = p(g | l_i), g = the Gaussian component with the smaller mean. The GMM is fit to the loss (the paper reports BMM on max-normalized loss, as in M-correction, gives flat fits and fails on asymmetric noise).
Eq (2) entropy for the confidence penalty (label `eqn:penalty`); the warm-up loss adds -H, i.e. maximizes entropy:
`\mathcal{H} = -\sum_{c} \mathrm{p^c_{model}}(x;\theta) \log(\mathrm{p^c_{model}}(x;\theta))`
Eq (3) label co-refinement: `\bar{y}_b=w_b y_b+(1-w_b){p}_{b}`
Eq (4) sharpening: `\hat{y}_b=\mathrm{Sharpen}(\bar{y}_b,T)={\bar y_b^c}^{\frac{1}{T}} \bigg/ \sum_{c=1}^C {\bar y_b^c}^{\frac{1}{T}},~\text{for}~ c=1,2,...,C`
Co-guessing (Alg 1 line 20, unnumbered): `\bar{q}_{b}=\frac{1}{2M}\sum_{m}\big(\mathrm{p_{model}}(\hat{u}_{b,m};\theta^{(1)})+\mathrm{p_{model}}(\hat{u}_{b,m};\theta^{(2)})\big)`, then q_b = Sharpen(q_bar_b, T).
Eqs (5)-(8) mixup: `\lambda \sim \mathrm{Beta}(\alpha,\alpha)`, `\lambda' = \max(\lambda,1-\lambda)`, `x'=\lambda' x_1+(1-\lambda')x_2`, `p'=\lambda' p_1+(1-\lambda')p_2`
Eq (9): `\mathcal{L}_\mathcal{X} = -\frac{1}{|\mathcal{X}'|} \sum_{x,p\in\mathcal{X}'} \sum_{c} p_c \log(\mathrm{p_{model}^c}(x;\theta))`
Eq (10): `\mathcal{L}_\mathcal{U} = \frac{1}{|\mathcal{U}'|} \sum_{x,p\in\mathcal{U}'} \left\lVert {p-\mathrm{p_{model}}(x;\theta)} \right\rVert^2_2`
Eq (11): `\mathcal{L}_\mathrm{reg}= \sum_c \pi_c \log({\pi_c}\bigg/ \frac{1}{|\mathcal{X}'|+|\mathcal{U}'|} \sum_{x \in\mathcal{X}'+\mathcal{U}'} \mathrm{p_{model}^c}(x;\theta))`
Eq (12): `\mathcal{L}=\mathcal{L}_\mathcal{X}+\lambda_u \mathcal{L}_\mathcal{U}+\lambda_r \mathcal{L}_\mathrm{reg}`
Variables: y_i = one-hot given label over C classes; p_model^c = softmax prob of class c; w_b = clean probability from the other net's GMM; p_b = own-net prediction averaged over M augmentations; T = sharpening temperature; q_b = guessed label for unlabeled u_b; X', U' = mixed labeled/unlabeled batches; pi_c = 1/C uniform prior; lambda_u, lambda_r = loss weights; alpha = Beta parameter.

## Hyperparameters (Sec 4.1; Appendix B; Table 7)
CIFAR: PreAct ResNet-18; SGD momentum 0.9, weight decay 0.0005, batch 128, 300 epochs, lr 0.02 divided by 10 after 150 epochs; warm-up 10 epochs (CIFAR-10), 30 (CIFAR-100); M = 2, T = 0.5, alpha = 4; tau = 0.5, except 0.6 at 90% noise; lambda_r = 1; lambda_u from {0, 25, 50, 150} via a small validation set. Table 7 lambda_u: CIFAR-10 asym 40% 0, sym 20% 0, 50% 25, 80% 25, 90% 50; CIFAR-100 sym 20% 25, 50/80/90% 150.
Clothing1M / WebVision: M = 2, T = 0.5, tau = 0.5, lambda_u = 0, alpha = 0.5; SGD momentum 0.9, wd 0.001, batch 32, warm-up 1 epoch. Clothing1M: ResNet-50 ImageNet-pretrained, 80 epochs, lr 0.002 / 10 after 40, 1000 class-balanced mini-batches per epoch. WebVision: Inception-ResNet v2, first 50 classes of Google subset, 100 epochs, lr 0.01 / 10 after 50.
Confidence penalty: used during warm-up for asymmetric noise only.

## Main results (test accuracy %)
Convention (Sec 4.2, Table 1): "Best" = best test accuracy across all epochs; "Last" = average test accuracy over the last 10 epochs. "Best" peeks at the test set; ELR (sources/elr, Table 2 caption) flags this and reports its own numbers with a held-out validation set instead.
Symmetric noise here = replace a fraction of labels with a random label from ALL classes (true label can be kept). Table 6 (appendix) uses the stricter variant where the true label cannot be kept. Not comparable to Co-teaching's "Symmetry" numbers.
Table 1 (src/table/cifar.tex, label `tbl:cifar`), PreAct ResNet-18, cols CIFAR-10 20/50/80/90% then CIFAR-100 20/50/80/90%; * = re-implementation:
| Method | Best C10: 20 / 50 / 80 / 90 | Last C10 | Best C100: 20 / 50 / 80 / 90 | Last C100 |
|---|---|---|---|---|
| Cross-Entropy | 86.8 / 79.4 / 62.9 / 42.7 | 82.7 / 57.9 / 26.1 / 16.8 | 62.0 / 46.7 / 19.9 / 10.1 | 61.8 / 37.3 / 8.8 / 3.5 |
| Bootstrap | 86.8 / 79.8 / 63.3 / 42.9 | 82.9 / 58.4 / 26.8 / 17.0 | 62.1 / 46.6 / 19.9 / 10.2 | 62.0 / 37.9 / 8.9 / 3.8 |
| F-correction | 86.8 / 79.8 / 63.3 / 42.9 | 83.1 / 59.4 / 26.2 / 18.8 | 61.5 / 46.6 / 19.9 / 10.2 | 61.4 / 37.3 / 9.0 / 3.4 |
| Co-teaching+* | 89.5 / 85.7 / 67.4 / 47.9 | 88.2 / 84.1 / 45.5 / 30.1 | 65.6 / 51.8 / 27.9 / 13.7 | 64.1 / 45.3 / 15.5 / 8.8 |
| Mixup | 95.6 / 87.1 / 71.6 / 52.2 | 92.3 / 77.6 / 46.7 / 43.9 | 67.8 / 57.3 / 30.8 / 14.6 | 66.0 / 46.6 / 17.6 / 8.1 |
| P-correction* | 92.4 / 89.1 / 77.5 / 58.9 | 92.0 / 88.7 / 76.5 / 58.2 | 69.4 / 57.5 / 31.1 / 15.3 | 68.1 / 56.4 / 20.7 / 8.8 |
| Meta-Learning* | 92.9 / 89.3 / 77.4 / 58.7 | 92.0 / 88.8 / 76.1 / 58.3 | 68.5 / 59.2 / 42.4 / 19.5 | 67.7 / 58.0 / 40.1 / 14.3 |
| M-correction | 94.0 / 92.0 / 86.8 / 69.1 | 93.8 / 91.9 / 86.6 / 68.7 | 73.9 / 66.1 / 48.2 / 24.3 | 73.4 / 65.4 / 47.6 / 20.5 |
| DivideMix | 96.1 / 94.6 / 93.2 / 76.0 | 95.7 / 94.4 / 92.9 / 75.4 | 77.3 / 74.6 / 60.2 / 31.5 | 76.9 / 74.2 / 59.6 / 31.0 |
Table 2 (cifar_an.tex, `tbl:cifar_an`), CIFAR-10, 40% asymmetric (deer->horse, dog<->cat style), Best/Last: Cross-Entropy 85.0/72.3; F-correction 87.2/83.1; M-correction 87.4/86.3; Iterative-CV 88.6/88.0; P-correction 88.5/88.1; Joint-Optim 88.9/88.4; Meta-Learning 89.2/88.6; DivideMix 93.4/92.1. (40% chosen because above 50% some classes become indistinguishable.)
Table 3 (clothing.tex, `tbl:clothing`), Clothing1M test acc: Cross-Entropy 69.21; F-correction 69.84; M-correction 71.00; Joint-Optim 72.16; Meta-Cleaner 72.50; Meta-Learning 73.47; P-correction 73.49; DivideMix 74.76.
Table 4 (webvision.tex, `tbl:webvision`), mini WebVision, top-1 (top-5) on WebVision val / ILSVRC12 val: F-correction 61.12 (82.68) / 57.36 (82.36); Decoupling 62.54 (84.74) / 58.26 (82.26); D2L 62.68 (84.00) / 57.80 (81.36); MentorNet 63.00 (81.40) / 57.80 (79.92); Co-teaching 63.58 (85.20) / 61.48 (84.70); Iterative-CV 65.24 (85.34) / 61.60 (84.98); DivideMix 77.32 (91.64) / 75.20 (90.84).
Ablation, Table 5 (ablation.tex), Best, CIFAR-10 sym 80% / asym 40% / CIFAR-100 sym 80%: full 93.2 / 93.4 / 60.2; theta(1) test only 93.0 / 92.7 / 58.3; w/o co-training (self-divide) 92.6 / 91.9 / 56.7; w/o label refinement 93.0 / 87.7 / 58.7; w/o augmentation 92.2 / 89.5 / 58.2; Divide + plain MixMatch 89.7 / 86.5 / 55.3.
Training time on CIFAR-10, one V100 (Table 8): Co-teaching+ 4.3 h, P-correction 6.0 h, Meta-Learning 8.6 h, DivideMix 5.2 h.

## Figures
- Fig 1: framework; nets A and B; each fits a GMM to its per-sample loss and hands the labeled/unlabeled split to the other; per mini-batch, co-refinement + co-guessing + MixMatch.
- Fig 2: loss histograms, CIFAR-10 40% asym, clean vs noisy samples: (a) plain CE warm-up, everything near zero loss (overconfident); (b) with confidence penalty, spread out and separable; (c) after 10 more DivideMix epochs, clean losses low, most noisy losses high.
- Fig 3: AUC of clean/noisy classification from one GMM over the first 100 epochs on CIFAR-10; rises with training even at high noise.
- Fig 4: Clothing1M images flagged noisy; given (red) vs co-guessed (blue) labels. Fig 5: t-SNE after 200 epochs at 80% noise; 10 clusters follow true labels.

## Limitations and follow-ups
Stated: future work, more SSL ideas, other domains (NLP). Implicit: many moving parts (two nets, GMM, MixMatch, per-setting lambda_u); "Best" uses test set for selection. Assumes clean and noisy losses form two modes. Follow-up ELR+ (sources/elr) matches or beats it on Clothing1M (74.81) and WebVision top-1 (77.78) with held-out validation.

## Labeling decision
When a dataset has a known chunk of unreliable labels (e.g. crowd or auto-labels), whether to drop the suspected-wrong ones or keep their images: DivideMix keeps them as unlabeled data, and its per-example clean probability w_i is a ready-made ranking for which images to send to human review.

## Visual ideas
- Loss histogram + GMM: per-sample loss histogram colored by (hidden) clean/noisy; fit two Gaussians live; slide tau and see precision/recall of the labeled set; toggle confidence penalty for asymmetric noise.
- Co-divide swap: two panels (net A, net B) each with its own split; animate the hand-off and show what goes wrong in self-divide mode.
- Label co-refinement dial: one image with given label y, model prediction p and w; drag w and T and watch y_bar and the sharpened y_hat.
