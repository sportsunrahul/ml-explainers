# DenseCL: Dense Contrastive Learning for Self-Supervised Visual Pre-Training

## Identity
- arXiv 2011.09157, version read: v2 (v1 Wed 18 Nov 2020; v2 Sun 4 Apr 2021). Xinlong Wang ... Lei Li (5 authors).
- Venue: CVPR 2021 (Oral) (abs.html comments).

## Key idea
Add a dense head to MoCo-v2: contrast S x S local feature vectors across two views, where each query's positive is its
best-matching (argmax cosine) location in the other view's backbone features. Loss = (1 - lambda) global + lambda dense.

## Defining equation (src/arXiv.tex, eq:dense_contra_loss, verbatim; the source omits /tau on the positive term in the denominator)
```
\mathcal{L}_{r} = \frac{1}{S^2} \sum_{s}
-\log \frac{\exp(r^s {\cdot} t^s_{+} / \tau)} {\exp(r^s {\cdot} t^s_{+}) + \sum_{t^s_-} \exp(r^s {\cdot} t^s_-  / \tau)},
\mathcal{L} = (1 - \lambda)\mathcal{L}_{q} + \lambda \mathcal{L}_{r}
```
S = 7 (7x7 grid), lambda = 0.5, dictionary (queue) 65536 for both global and dense. Positive matching: c_i = argmax_j sim(f_i, f'_j).

## Key number
VOC object detection (Faster R-CNN C4, fine-tuned trainval07+12, 2x schedule), ResNet-50 pretrained on ImageNet 200
epochs, batch 256, 8 GPUs: AP 58.7 vs MoCo-v2 (re-run) 57.0 (Table 1, rows "DenseCL IN" and "MoCo-v2 IN*", AP column).
