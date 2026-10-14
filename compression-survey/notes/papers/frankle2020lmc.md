# frankle2020lmc -- Linear Mode Connectivity and the Lottery Ticket Hypothesis (BRANCH)

## Identity
- Title: Linear Mode Connectivity and the Lottery Ticket Hypothesis
- arXiv 1912.05671, read v4 (2020-07-18). v1 submitted 2019-12-11 (abs.html).
- Authors: Jonathan Frankle ... Michael Carbin (src/instability-icml-submission.tex).
- Venue: ICML 2020 (abs.html comments; subsumes arXiv 1903.01611, "Stabilizing the LTH" / "LTH at Scale").

## Key idea (two sentences)
IMP with reset to initialization (k = 0) fails to find matching subnetworks for standard VGG/ResNet on CIFAR-10
and for ResNet-50/Inception-v3 on ImageNet, but rewinding the surviving weights to their values at an early
iteration k > 0 does find them. IMP subnetworks are matching exactly when they are "stable to SGD noise"
(two copies trained with different data order end in the same linearly connected minimum).

## Defining procedure (Algorithm 2, "IMP rewinding to step k and N iterations"; Sec. 4.2)
Train W_0 to W_k; then N times: train m (.) W_k to W_T, prune the lowest-magnitude remaining weights of W_T
(globally), update mask m. Return (W_k, m). Iterative (20% per round) for CIFAR/MNIST; one-shot for ImageNet.
Terms: "matching" = accuracy drop < 0.2%; "stable" = instability < 2% (Fig. 9 caption).

## Key numbers
- ResNet-50, ImageNet (Table 1): 25.5M params, 90 epochs, batch 1024, lr 0.4, 5 epochs warmup, full accuracy
  76.1 +/- 0.1%; pruned density 30%, one-shot.
- At k = 0 (Table 2): ResNet-50 full 76.1, IMP 73.7, random prune 73.1, random reinit 73.4, delta 2.4 -> not
  matching. Inception-v3: 78.1 / 75.7 / 75.2 / 75.5, delta 2.4, not matching.
- Rewinding (Sec. 4.3): IMP subnetworks of ResNet-50 become stable at epoch 5 (5.5% into training) and
  Inception-v3 at epoch 6 (3.5%); unpruned networks only at epoch 18 (20%) and 28 (16%) (Sec. 3). Stable
  rewinding points "closely coincide" with matching ones.
- CIFAR-10: IMP subnetworks of ResNet-20 stable at iteration 500 (0.8%), VGG-16 at 1000 (1.6%).
- Table 2, CIFAR-10 at k = 0: ResNet-20 standard (lr 0.1) 91.7 vs IMP 88.5 (not matching); ResNet-20 warmup 89.7
  vs 89.6 (matching); VGG-16 standard 93.7 vs 90.9 (not matching); VGG-16 warmup 93.4 vs 93.2 (matching).
- Naming fix (Table 1 caption): Frankle & Carbin "mistakenly refer to ResNet-20 as ResNet-18".

## Outline feedback
- The LTH claim card's scale caveat is correct; quote "IMP subnetworks of ResNet-50 (ImageNet, 30% density,
  one-shot) become stable when rewound to epoch 5 of 90 (5.5% of training)", and that matching "closely
  coincides" with stability (Fig. 7 shows it; no exact matching epoch is stated in text). Note these rewound subnetworks are no longer "winning
  tickets" in the LTH sense (Sec. 4.2 calls them "matching").
- Still unstructured, magnitude-based: no speed claim in this paper.
