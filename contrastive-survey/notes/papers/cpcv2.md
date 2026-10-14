# CPC v2: Data-Efficient Image Recognition with Contrastive Predictive Coding (branch)

## Identity
- arXiv 1905.09272, read v3 (2020-07-01); v1 2019-05-22. Authors: Olivier J. Henaff ... Aaron van den Oord. Venue: ICML 2020 (icml2020 accepted style).

## Key idea
Re-engineer CPC for images (bigger ResNet-161 encoder, layer norm, more patch augmentation, larger
patches, predictions in more directions) and test it on linear eval, few-label classification and detection.

## Equation
Same InfoNCE as CPC; negatives "are taken from other locations in the image and other images in the mini-batch" (Sec 2.1).

## Setup (Appendix, pretraining)
300x300 resize, 260x260 random crop, 6x6 grid of 80x80 patches with stride 36; ResNet-161 -> [6,6,4096];
PixelCNN context; Adam lr 0.0004, 200 epochs, batch 512 over 32 workers.

## Key numbers (Table 1, linear, single crop, full ImageNet labels)
- CPC v2 ResNet-161 (305M params): 71.5 top-1 / 90.1 top-5.
- CPC v2 ResNet-50 (24M): 63.8 top-1 / 85.3 top-5.
- Lists CPC v1 at 48.7 / 73.6 (28M params) and Instance Discr. (Wu) 54.0.
- Few labels: 78.3 top-5 with 1% of labels (Sec 4.2 text).

## Outline feedback
Table 1 row "MoCo [2] 375M 68.6" cites [2] (Local Aggr.); likely a source typo for [3]. Do not repeat that row.
