# Branches: evidence on label noise (memorization and noisy benchmarks)

## zhang -- Understanding deep learning requires rethinking generalization
Citation: Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals. arXiv 1611.03530v2 (26 Feb 2017).
ICLR 2017 (arXiv comment: "Published in ICLR 2017"). src is a PDF wrapper; text read from paper.pdf.
Role: the root of the memorization branch. It showed standard nets fit random labels; Arpit et al. (1706.05394) then
asked how the fitting differs on real vs noise data.
Key statements (exact):
- Sec 1.1 (box): "Deep neural networks easily fit random labels." "when trained on a completely random labeling of the
  true data, neural networks achieve 0 training error."
- Sec 1.1, implication 2: "Even optimization on random labels remains easy. In fact, training time increases only by
  a small constant factor compared with training on the true labels."
- Sec 2.1: no learning-rate schedule change is needed; "once the fitting starts, it converges quickly"; ImageNet with
  random labels reaches 95.20% top-1 train accuracy (Inception V3, Table 2 in appendix).
- Table 1 (CIFAR-10), row "Inception (fitting random labels)": train 100.0, test 9.78 (true labels, no aug/no wd:
  train 100.0, test 85.75).
- Fig 1b: relative "time to overfit" vs label corruption 0 to 1 for Inception, AlexNet, MLP 1x512; y-axis spans 1.0 to
  4.0; exact per-model factor at full corruption not stated in text (UNSURE, read from plot only).
Relevance: a big model will fit wrong labels to 0 training error, so training accuracy says nothing about label quality;
label review must use held-out or early-training signals.

## cifarn -- Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations (CIFAR-N)
Citation: Jiaheng Wei, Zhaowei Zhu, Hao Cheng, Tongliang Liu, Gang Niu, Yang Liu. arXiv 2110.12088v2. ICLR 2022.
Role: replaces synthetic CIFAR noise (used by Arpit, Co-teaching, DivideMix) with real MTurk labels: 3 labels per
CIFAR-10 train image, 1 coarse + 1 fine label per CIFAR-100 train image.
Key numbers (Sec 3.1, src/src/data_intro.tex; Table 1):
- CIFAR-10N noise rates: Aggregate (majority vote of 3) 9.03%, Random 1 17.23%, Random 2 18.12%, Random 3 17.64%,
  Worst 40.21%. 60.27% of train images got a unanimous label from 3 labelers.
- CIFAR-100N (Sec 3.2): fine 40.20%, coarse 25.60%.
- Real vs synthetic: human noise is feature-dependent (instance-dependent), not class-dependent; two-sided t-test vs a
  synthetic noise with the same expected transition matrix T gives p = 1.8e-36 (Sec 4.2).
- Memorization (Sec 5.2, Fig 6, ResNet-34, eta = 0.95): "deep neural nets memorize features more easily when learning
  with real-world human annotations than synthetic ones."
- Table 3 (gap = synthetic acc - human acc, same T): CE (Standard) gaps 4.35 (Aggregate), 6.01 (Random 1), 9.00 (Worst).
Relevance: crowd label errors cluster on ambiguous images and similar classes (truck vs automobile ~25-30% flips,
Sec 4.1), so "learn clean first" helps less than synthetic benchmarks suggest; aggregating 3 votes cuts noise ~2x.

## beyer -- Are we done with ImageNet?
Citation: Lucas Beyer, Olivier J. Henaff, Alexander Kolesnikov, Xiaohua Zhai, Aaron van den Oord. arXiv 2006.07159v1
(2020). Preprint ("Under review", NeurIPS template); not a published NeurIPS paper as of this version.
Role: relabels the ImageNet val set as multi-label ("ReaL" labels) using 19 models' proposals (6-model subset) plus 5
human raters per task; Pervasive (2103.14749) later does single-label error finding across 10 datasets.
Key numbers:
- Sec 3.2-3.3: 24,889 of 50,000 images needed re-rating (all models agreed on the rest); 37,988 tasks; result 57,553
  labels for 46,837 images; 3,163 images got no label and are dropped. Dawid-Skene aggregation (majority vote: 90.8%
  precision, 71.8% recall).
- Headline (Sec 4, Fig 4): ImageNet acc predicts ReaL acc with slope 0.86 for the earlier half of models but only 0.51
  for the recent half (p < 0.001). Several recent models (BiT-L, Noisy Student) beat the original ImageNet labels on
  ReaL accuracy; the original labels score about 90% ReaL (red point in Fig 4, read from plot, approx).
- Ensemble of 3 best models: 89.0% original ImageNet accuracy vs 91.20% ReaL accuracy (Sec 4).
- Sec 5: top models still err ~11% by ImageNet labels and ~9% by ReaL labels.
Relevance: when an image holds several valid classes, single-label eval punishes correct predictions; a platform should
allow multi-label answers in review and measure "top-1 in the accepted set".

## relabel -- Re-labeling ImageNet: from Single to Multi-Labels, from Global to Localized Labels
Citation: Sangdoo Yun, Seong Joon Oh, Byeongho Heo, Dongyoon Han, Junsuk Choe, Sanghyuk Chun. arXiv 2101.05022v2.
CVPR 2021.
Role: Beyer fixed only the val set; ReLabel fixes the training set with a machine annotator (EfficientNet-L2 pre-trained
with extra data, Noisy Student weights, 475x475 input, 15x15x1000 label map; App B) that outputs dense per-pixel label maps; LabelPooling (RoIAlign over the random crop) gives each crop
its own soft multi-label.
Key number (Table 3, row "ResNet-50 / ReLabel" vs "Original"): ImageNet 78.9 vs 77.5 (+1.4 pp), ImageNetV2 80.5 vs 79.0,
ReaL 85.0 vs 83.6, Shankar et al. 86.1 vs 85.3. Label smoothing gives +0.5 pp and label cleaning (Beyer) +0.6 pp on
ImageNet (Sec 4.1). With CutMix: 80.2% (abstract).
Relevance: a strong model can relabel training data at scale (machine labels instead of human re-labels), and labels
must match the crop the model sees; the cost is one forward pass of a big teacher per image, stored once.

## webvision -- WebVision Database: Visual Learning and Understanding from Web Data
Citation: Wen Li, Limin Wang, Wei Li, Eirikur Agustsson, Luc Van Gool. arXiv 1708.02862v1 (9 Aug 2017). Tech report
(UNSURE of a formal venue for this version).
Role: a large real-noise training set (2.4M+ images from Google and Flickr queried with the 1,000 ILSVRC 2012 synsets;
label = query synset, no human labeling on train); a standard real-noise benchmark for later methods (e.g. DivideMix,
Co-teaching follow-ups report "mini-WebVision").
Key number (Sec 3, Fig 4; 200K images, 200 per class, 3 AMT votes each): about 20% are "0 vote" (true noise) and
inliers ("3 votes" + "2 votes") are only 66%. Inconsistency in source: the val/test protocol says inlier "if more than
two users agree", the noise analysis counts 2 or 3 votes as inlier.
- Sec 4.2: with 1.28M images, WebVision-trained AlexNet beats ILSVRC 2012-trained AlexNet on WebVision val top-1
  (55.11% vs 52.58%);
  quantity offsets noise.
Relevance: scraped or query-derived labels can be ~20-34% wrong (0 votes to non-inlier), but more cheap data can still
win; the decision is whether to spend budget on more weak labels or on cleaning fewer.
