# MoCo v2 (spine, delta)

## 1. Identity
- Title: Improved Baselines with Momentum Contrastive Learning
- arXiv 2003.04297v1, submitted 2020-03-09 (abs.html). Only version.
- Authors: Xinlei Chen ... Kaiming He (Chen, Fan, Girshick, He), FAIR. Technical note; no venue stated.

## 2. Core problem
SimCLR's gains came with 4k-8k batches on TPUs. Are its MLP head and stronger augmentation orthogonal to the
mechanism, so MoCo can use them with a batch of 256 on 8 GPUs?

## 3. Prior methods
- MoCo v1: fc head, weaker augmentation, step schedule; 60.6 linear.
- SimCLR: end-to-end, needs a large batch for negatives (Sec. 2 "Improved designs").

## 4. Key idea
Add SimCLR's 2-layer MLP head, blur augmentation, and cosine lr schedule to MoCo. Keep the queue and momentum encoder.

## 5. Mechanism
Same as MoCo v1 (queue, momentum encoder, shuffling BN, same codebase). Changes:
1. Head: fc -> 2-layer MLP (hidden 2048-d, ReLU). Used only in pretraining; dropped for linear probe/transfer.
2. tau: 0.07 -> 0.2 (searched for the MLP head).
3. aug+: add blur from SimCLR. Stronger SimCLR color distortion gave "diminishing gains" and is not added.
4. cos: cosine (half-period) lr schedule.

## 6. Equations
Eq. 1 (`\label{eq:infonce}`), Sec. 2:
```
\mathcal{L}_{q, k^+, \{k^-\}} = -\log \frac{\exp(q{\cdot}k^+ / \tau)}{\exp(q{\cdot}k^+ / \tau) + {\displaystyle\sum_{k^-}}\exp(q{\cdot}k^-  / \tau)}.
```
q: query; k^+: positive key (other augmentation of the same image); {k^-}: negative keys (queue); tau: temperature.
Same loss as MoCo Eq. 1, written with the positive separated out.

## 7. Config
| item | value | loc |
|---|---|---|
| encoder | standard ResNet-50 | Sec. 3 Settings |
| head | 2-layer MLP, hidden 2048-d, ReLU | Sec. 3 MLP head |
| tau | 0.2 | Sec. 3 MLP head |
| batch | 256 | Table 2 |
| epochs | 200 or 800 | Table 1, 2 |
| hardware | 8 V100 16G GPUs; 5.0G memory/GPU, 53 hrs per 200 epochs | Table 3 |
| eval | linear probe, frozen, 1-crop 224x224 top-1 val; VOC Faster R-CNN C4 fine-tune 24k iters | Sec. 3 Settings |
Other hyper-parameters: "same hyper-parameters (except when noted) and codebase as MoCo".

## 8. Results
Table 1 (R50, ImageNet linear top-1; VOC AP50/AP/AP75):
| case | MLP | aug+ | cos | epochs | IN acc | AP50 | AP | AP75 |
|---|---|---|---|---|---|---|---|---|
| supervised | | | | | 76.5 | 81.3 | 53.5 | 58.8 |
| MoCo v1 | | | | 200 | 60.6 | 81.5 | 55.9 | 62.6 |
| (a) | x | | | 200 | 66.2 | 82.0 | 56.4 | 62.6 |
| (b) | | x | | 200 | 63.4 | 82.2 | 56.8 | 63.2 |
| (c) | x | x | | 200 | 67.3 | 82.5 | 57.2 | 63.9 |
| (d) | x | x | x | 200 | 67.5 | 82.4 | 57.0 | 63.6 |
| (e) | x | x | x | 800 | 71.1 | 82.5 | 57.4 | 64.0 |

tau sweep (inline table, Sec. 3 MLP head), linear top-1:
| tau | 0.07 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|
| w/o MLP | 60.6 | 60.7 | 59.0 | 58.2 | 57.2 | 56.4 |
| w/ MLP | 62.9 | 64.9 | 66.2 | 65.7 | 65.0 | 64.3 |

Table 2 (MoCo vs SimCLR, R50, linear, 1-crop 224):
MoCo v1 200ep b256 60.6; SimCLR 200ep b256 61.9; SimCLR 200ep b8192 66.6; MoCo v2 200ep b256 67.5;
SimCLR 1000ep b4096 69.3; MoCo v2 800ep b256 71.1. SimCLR rows "from Fig. 9 in [SimCLR] (we thank the authors
for providing the numerical results)".

Table 3 (8 V100 16G): MoCo b256 5.0G, 53 hrs; end-to-end b256 7.4G, 65 hrs; end-to-end b4096 93.0G (estimate), n/a.

Finding: "linear classification accuracy is not monotonically related to transfer performance in detection" ((b) vs (a)).

## 9. Limitations
None stated. Short note. MoCo v3 later drops the queue for batch 4096 and moves to ViT.

## 10. Outline feedback
- "cosine schedule" adds only +0.2 (67.3 -> 67.5); most gain is MLP (+5.6 alone) and longer training. Say so.
- Outline says "stronger augmentation": v2 adds only blur; it does NOT adopt SimCLR's stronger color distortion.
