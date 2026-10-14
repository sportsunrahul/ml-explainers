# MoCo v3 (spine, delta)

## 1. Identity
- Title: An Empirical Study of Training Self-Supervised Vision Transformers (method name "MoCo v3").
- arXiv 2104.02057, read v4. v1 submitted 2021-04-05; v4 revised 2021-08-16 (abs.html).
- Authors: Xinlei Chen, Saining Xie, Kaiming He. Style file `iccv.sty` (venue not stated in abs.html).

## 2. Core problem
Swapping a ResNet for a ViT in contrastive frameworks is simple in principle, but training is unstable. The
instability hides as a small accuracy drop, not a divergence.

## 3. Prior methods
MoCo v1/v2 use a queue and ResNet. Contrastive/Siamese ViT training shows "dips" in kNN curves at large batch
or large lr (Sec. 4.1).

## 4. Key idea
(a) MoCo v3: drop the queue, use in-batch keys from a momentum encoder with a large batch, a symmetric loss, and a
BYOL-style prediction head. (b) Freeze the ViT patch projection at random init (stop-gradient after it); this
stabilizes training and raises accuracy.

## 5. Mechanism (Alg. 1)
1. x1, x2 = two augmentations of a minibatch of N images (two 224x224 crops).
2. q1, q2 = f_q(x1), f_q(x2): [N, C] each. f_q = backbone + projection MLP + prediction MLP.
3. k1, k2 = f_k(x1), f_k(x2): [N, C] each. f_k = backbone + projection MLP (no predictor), momentum copy.
4. loss = ctr(q1, k2) + ctr(q2, k1). ctr: logits = q k^T, [N, N]; labels = range(N) (positives on diagonal);
   CrossEntropy(logits/tau) times 2 tau.
5. Backprop updates f_q. f_k = m f_k + (1-m) f_q (excluding the prediction head).
- Positions graded: row i of the N x N matrix; column i positive (same image, other view, from f_k); the other N-1
  columns are negatives (other images' keys). No queue: "We abandon the memory queue ... diminishing gain if the
  batch is sufficiently large (e.g., 4096)" (Sec. 3).
- Random patch projection: patch embedding layer frozen at random init; "applying a stop-gradient operation right
  after this layer" (Sec. 4.2).

## 6. Equations
Eq. 1 (`\label{eq:infonce}`), Sec. 3:
```
\mathcal{L}_{q} = -\log \frac{\exp(q{\cdot}k^+ / \tau)}{\exp(q{\cdot}k^+ / \tau) + {\displaystyle\sum_{k^-}}\exp(q{\cdot}k^-  / \tau)}.
```
k^+: f_k's output on the same image as q. {k^-}: f_k's outputs on other images (same batch). q, k L2-normalized.
Symmetrized: ctr(q1, k2) + ctr(q2, k1); each ctr scaled by 2 tau (Sec. 5 Loss: "redundant ... absorbed by lr and wd").

## 7. Config
| item | value | loc |
|---|---|---|
| backbones | ViT-S 12 blocks/384/12 heads/22M; ViT-B 12/768/12/86M; ViT-L 24/1024/16/304M; ViT-H 32/1280/16/632M | Table 2 |
| patch | 16x16 or 14x14 -> 196 or 256 tokens for 224 input; sin-cos 2-D pos. emb.; [CLS] output | Sec. 5 ViT |
| proj head | 3-layer MLP, hidden 4096, output 256, BN in all layers | Sec. 5 MLP heads |
| pred head | 2-layer MLP, hidden 4096, output 256, BN | Sec. 5 MLP heads |
| optimizer | AdamW, batch 4096, lr = lr_base x BatchSize/256, 40-epoch warmup, cosine decay | Sec. 4.1, Sec. 5 |
| lr_base / wd | searched per case on 100-epoch runs; ablations use lr 1.0e-4 (and 1.5e-4); final values not stated | Fig. 1-5 captions, Sec. 5 |
| tau | 0.2 | Sec. 5 Loss |
| m | 0.99 (ViT default) | Sec. 6.2 momentum table |
| epochs | 300 default (ViT-S/B also 600) | Sec. 6.2 |
| aug | random resized crop, flip, color jitter, grayscale, blur, solarization; two 224x224 crops | App. A |
| BN | SyncBN; two crops forwarded as separate 4096 batches | App. A |
| hardware | TPU v3: ViT-B 256 TPUs, 2.1 h per 100 epochs; ViT-H 512 TPUs, 9.8 h per 100 ep | Table 3 |
| R50 variant | LARS, batch 4096, lr 0.3, wd 1.5e-6, tau 1.0, m 0.996 -> 1 cosine | App. A |
| linear probe | frozen, heads removed, SGD batch 4096, wd 0, 90 epochs, crop+flip only, single-crop top-1 | Sec. 5 |

## 8. Results
- Instability (ViT-B/16, 100 ep, AdamW): batch 1024/2048/4096/6144 -> 71.5/72.6/72.2/69.7 (Fig. 1 table);
  lr 0.5/1.0/1.5 e-4 -> 70.4/72.2/71.7 (Fig. 2); LAMB lr 3/5/6/8 e-4 -> 71.6/72.5/70.9/66.5 (Fig. 3).
- Gradient spikes appear first in the patch projection (first layer), later in last layers (Fig. 4).
- Random vs learned patch projection (Fig. 5, ViT-B/16, 100 ep, batch 4096): lr 0.5e-4 70.4 -> 70.8;
  1.0e-4 72.2 -> 72.8; 1.5e-4 71.7 -> 73.4 (+1.7). Also SimCLR 69.3 -> 70.1, BYOL 69.7 -> 71.0 (Fig. 6);
  SwAV 65.8 -> 66.4 (Sec. 4.2 text).
- Table 1 linear probe (ImageNet-1k, two 224 crops, no multi-crop): ViT-B 76.7, ViT-L 77.6, ViT-H 78.1,
  ViT-BN-H 79.1, ViT-BN-L/7 81.0. Epochs: ViT-B 76.7 matches the 600-ep row of the training-length table;
  ViT-L/H trained 300 epochs (Sec. 6.2).
- Table 4 (300 ep, linear): ViT-B MoCo v3 76.5, SimCLR 73.9, BYOL 73.9, SwAV 71.6; ViT-S 72.5/69.0/71.0/67.1;
  R-50 800-ep 73.8/70.4/74.3/71.8.
- R50 800-ep linear: MoCo v2 71.1, MoCo v2+ 72.2, MoCo v3 73.8 (Sec. 3 inline table).
- Momentum (ViT-B 300 ep): m = 0 74.3; 0.9 75.6; 0.99 76.5; 0.999 75.0 (Sec. 6.2).
- Fine-tune (Table 5): ViT-B 83.2, ViT-L 84.1.

## 9. Limitations stated
Random patch projection "alleviates the issue, but does not solve it"; model can still be unstable if lr is too big
(Sec. 4.2 Discussions). Accuracy saturates with model size (Sec. 6.3).

## 10. Outline feedback
- Outline says "drops the queue for a large batch": correct. Also mention the prediction head and that momentum
  encoder stays (m = 0.99, not 0.999). Momentum m=0 gives 74.3 vs 76.5.
- Headline ViT-B number: use 76.7 (600 ep) or 76.5 (300 ep) and state the epochs.
