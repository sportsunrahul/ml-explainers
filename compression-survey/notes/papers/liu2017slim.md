# liu2017slim -- Network Slimming (BRANCH)

## Identity
- Title: Learning Efficient Convolutional Networks through Network Slimming
- arXiv 1708.06519, read v1 (2017-08-22, only version) (abs.html).
- Authors: Zhuang Liu ... Changshui Zhang (arXiv listing). Venue: ICCV 2017 (abs.html comments).

## Key idea (two sentences)
Reuse each BatchNorm layer's per-channel scale gamma as a channel-importance score and push it toward zero with
an L1 penalty during training. After training, prune channels whose gamma falls below a global percentile
threshold and fine-tune; the result is a thinner dense network that needs no sparse kernels.

## Defining equation (Eq. 1, label `lambda`, src/method.tex)
`L = \sum_{(x, y)}{l(f(x, W), y)} + \lambda \sum_{\gamma\in {\Gamma}} {g(\gamma)}`
with `g(s) = |s|` (L1), optimized by subgradient descent. (x, y) = training input and target; W = trainable
weights; l = the normal task loss; Gamma = all BN scale factors; lambda balances the two terms.
BN (Eq. 2): `\hat{z} = \frac{z_{in} - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}; \ \ z_{out} = \gamma \hat{z} + \beta`.
Settings (Sec. 4.3): lambda = 1e-4 for VGGNet, 1e-5 for ResNet and DenseNet (CIFAR/SVHN); 1e-5 for VGG-A on
ImageNet. Gamma initialized to 0.5. Global percentile threshold, e.g. 70% (Sec. 3).

## Key row (Table 1(a), CIFAR-10)
VGGNet baseline 6.34% test error, 20.04M params, 7.97e8 FLOPs -> 70% channels pruned: 6.20% error, 2.30M params
(88.5% pruned), 3.91e8 FLOPs (51.0% pruned).
Abstract: multi-pass slimming on VGGNet gives 20x smaller model and 5x fewer computing operations.

## Outline feedback
- Fits ch-lottery "network slimming by BN gamma". For a detector, the gamma criterion needs BN after every conv
  being pruned (Sec. 3 argues scaling layers without BN are not meaningful) (our reading).
- Pre-activation ResNet/DenseNet need a "channel selection" layer to realize savings (Sec. 3); speed is reported
  as FLOPs, not time, in Table 1.
