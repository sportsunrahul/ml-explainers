# li2017filters -- Pruning Filters for Efficient ConvNets (BRANCH)

## Identity
- Title: Pruning Filters for Efficient ConvNets
- arXiv 1608.08710, read v3 (2017-03-10). v1 submitted 2016-08-31 (abs.html).
- Authors: Hao Li ... Hans Peter Graf (abs.html).
- Venue: ICLR 2017 (abs.html comments).

## Key idea (two sentences)
Rank each conv filter by the L1 norm of its weights and remove the smallest filters together with their output
feature maps and the matching input kernels of the next layer. The result is a smaller dense network, so normal
BLAS/cuDNN kernels run it faster with no sparse library.

## Defining rule (Sec. 3.1)
`s_{j} = \sum_{l=1}^{n_i} \sum|\mathcal{K}_l|` -- the sum of absolute kernel weights of filter F_{i,j}
(its L1 norm). Sort by s_j, prune the m smallest, rebuild layers i and i+1, copy remaining weights, retrain.
Pruning m filters of layer i removes m/n_{i+1} of the computation of layers i and i+1 (Sec. 3).

## Key rows
- Table 1, VGG-16 on CIFAR-10: error 6.75 -> 6.60 (VGG-16-pruned-A); FLOP 3.13e8 -> 2.06e8 (34.2% pruned);
  params 1.5e7 -> 5.4e6 (64.0% pruned). Scratch-trained pruned-A: 6.88.
- Table 1, ResNet-110 on CIFAR-10: 6.47 -> 6.70 (pruned-B), FLOP 2.53e8 -> 1.55e8 (38.6%); scratch-train 7.06.
  ResNet-56-pruned-B: 6.96 -> 6.94, FLOP 27.6% pruned; scratch-train 8.69.
- Table 3 (Appendix, measured): VGG-16 CIFAR-10 inference over the 10,000-image test set 1.23 s -> 0.73 s
  (40.7% saved); ResNet-110 2.38 -> 1.86 s (21.8%); ResNet-56 1.31 -> 0.99 s (24.4%); ResNet-34 ImageNet val
  (50,000 images) 36.02 -> 22.93 s (28.0%). Hardware: Titan X (Pascal), cuDNN v5.1, Torch7, mini-batch 128.
  "the saved inference time is close to the FLOP reduction".

## Outline feedback
- Outline calls it "Li 2016"; the version read is the ICLR 2017 paper (v3, 2017-03-10). Use Li 2017.
- Sec. 3.1 ("Relationship to pruning weights") gives a direct statement for ch-prune: weight pruning "generates sparse convolutional kernels which
  can be hard to accelerate given the lack of efficient sparse libraries, especially for the case of
  low-sparsity". Abstract: filter pruning "does not need the support of sparse convolution libraries".
- Table 1's scratch-train rows (worse than fine-tuned) are the claim Liu 2019 later overturns with a longer
  training budget (Liu 2019 Table 1: Scratch-B 93.78 vs fine-tuned 93.41 for VGG-16-A).
