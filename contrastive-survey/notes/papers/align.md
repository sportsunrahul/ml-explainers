# ALIGN: Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision

## Identity
- arXiv 2102.05918, version read: v2 (v1 Thu 11 Feb 2021; v2 Fri 11 Jun 2021). Chao Jia ... Tom Duerig (10 authors).
- Venue: ICML 2021 (abs.html comments).

## Key idea
Skip expensive caption cleaning: take 1.8B noisy image alt-text pairs with only minimal frequency-based filtering
and train a dual encoder (EfficientNet + BERT) with a CLIP-style symmetric contrastive loss. Scale makes up for noise.

## Defining equation (src/align.tex, eq:i2t_loss and eq:t2i_loss; loss = sum of the two)
```
L_{i2t} = -\frac{1}{N}\sum_i^N\log{\frac{\exp(x_i^\top y_i / \sigma)}{\sum_{j=1}^{N} \exp(x_i^\top y_j / \sigma)}}
L_{t2i} = -\frac{1}{N}\sum_i^N\log{\frac{\exp(y_i^\top x_i / \sigma)}{\sum_{j=1}^{N} \exp(y_i^\top x_j / \sigma)}}
```
x, y = L2-normalized image/text embeddings; sigma = learnable temperature (init 1.0, shared); label smoothing 0.1.

## Config (Sec. 5 training details)
EfficientNet-L2 image tower at 289x289, BERT-Large text tower, max 64 tokens; LAMB, lr 1e-3, 1.2M steps (~12 epochs);
1024 Cloud TPUv3 cores x 16 positive pairs = effective batch 16384.

## Key number
Zero-shot ImageNet top-1 76.4 (CLIP 76.2 in the same table). Table 4 (PDF; tab:zs_imagenet_result), row ALIGN, col ImageNet.
Frozen-feature linear ImageNet 85.5 (Table 5, ALIGN EfficientNet-L2): a different protocol, keep separate.
