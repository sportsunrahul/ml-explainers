# liu2019rethink -- Rethinking the Value of Network Pruning (BRANCH)

## Identity
- Title: Rethinking the Value of Network Pruning
- arXiv 1810.05270, read v2 (2019-03-05). v1 submitted 2018-10-11 (abs.html).
- Authors: Zhuang Liu ... Trevor Darrell (arXiv listing). Venue: ICLR 2019 (abs.html comments).

## Key idea (two sentences)
For structured pruning (channels/filters or larger), training the pruned architecture from random
initialization gives the same or better accuracy than fine-tuning the inherited "important" weights, provided
it gets a fair training budget. So the value of structured pruning is the architecture it finds, not the
weights it keeps.

## Scope conditions (as stated)
- "Scratch-E" = same epochs as the large model; "Scratch-B" = same computation budget (on ImageNet, if the pruned
  model saves more than 2x FLOPs, epochs are doubled) (Sec. 4).
- Holds for all structured methods examined: L1-norm filter pruning, ThiNet, regression-based feature
  reconstruction, Network Slimming, Sparse Structure Selection (Tables 1-5).
- Unstructured (Han 2015) is different (Table 6): from-scratch matches on CIFAR at prune ratios <= 80%, but at
  95% fine-tuning can win, and on ImageNet Scratch-B is "mostly worse than fine-tuned result by a noticable
  margin".
- Training setup: momentum SGD with initial lr 0.1. LTH comparison (Sec. 6): winning-ticket init helps only at
  lr 0.01, which gives lower accuracy than lr 0.1; for structured pruning it never beats random init.

## Key rows
- Table 1 (L1-norm filter pruning, accuracy %): VGG-16 CIFAR-10 unpruned 93.63; VGG-16-A fine-tuned 93.41,
  Scratch-E 93.62, Scratch-B 93.78. ResNet-34 ImageNet unpruned 73.31; ResNet-34-A fine-tuned 72.56, Scratch-E
  72.77, Scratch-B 73.03.
- Table 12 (Appendix, detection; Faster R-CNN on PASCAL VOC 07, mAP): ResNet-34 backbone unpruned 71.69.
  ResNet34-A: Prune-C 71.47, Prune-D 70.99, Scratch-E 71.64, Scratch-B 71.90. ResNet34-B: Prune-C 70.84,
  Prune-D 69.62, Scratch-E 71.68, Scratch-B 71.26. Prune-C = prune on classification weights then transfer;
  Prune-D = prune after transfer to detection; Scratch = train the small backbone from scratch on classification,
  then transfer.
- Table 6 (unstructured, ImageNet ResNet-50, unpruned 76.15): 30% pruned fine-tuned 76.06, Scratch-E 74.77,
  Scratch-B 75.70; 60%: 76.09 / 73.69 / 74.91.

## Outline feedback
- The detection row (Table 12) is the only detector result among the pruning branches besides Mishra 2021; it
  suits the guide ("distill or train a smaller backbone" vs "prune and fine-tune").
- State the scope on the page: structured pruning, standard lr 0.1 schedules, fair budget (Scratch-B). Do not
  extend the claim to unstructured pruning on ImageNet.
