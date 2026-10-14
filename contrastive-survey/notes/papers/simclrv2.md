# SimCLRv2 (branch)

## Identity
- Title: Big Self-Supervised Models are Strong Semi-Supervised Learners
- arXiv 2006.10029, read v2. v1 submitted 2020-06-17; v2 revised 2020-10-26 (abs.html).
- Authors: Ting Chen ... Geoffrey Hinton (Chen, Kornblith, Swersky, Norouzi, Hinton). Style file `neurips_2020.sty`.
- Repo: github.com/google-research/simclr (footnote in abstract).

## Key idea (two sentences)
Pretrain a big ResNet (up to ResNet-152 3x + selective kernels) with SimCLR plus a 3-layer projection head and a
MoCo-style memory (64K, EMA 0.999), then fine-tune on 1% or 10% of labels. Distill the fine-tuned teacher into a
student on unlabeled images; the fewer the labels, the more a bigger network helps.

## Defining equation (Eq. 2, `\label{eq:distill}`, Sec. 2)
```
\mathcal{L}^{\mathrm{distill}} = - \sum_{\bm x_i\in \mathcal{D}}\bigg[\sum_y P^T(y | \bm x_i;\tau) \log P^S(y|\bm x_i;\tau)\bigg]
```
P^T: fixed teacher's softmax at temperature tau; P^S: student's. No labels used.

## Config (Sec. 3)
128 Cloud TPUs, batch 4096, global BN, 800 epochs, LARS, lr 6.4 = 0.1 x sqrt(BatchSize), warmup 5% of epochs,
cosine, wd 1e-4, 3-layer MLP head, memory buffer 64K, EMA 0.999. Same augmentations as SimCLR.

## Key numbers
- Linear eval R152 (3x+SK): 79.8 top-1 (Table 1). R50 (1x, no SK): 71.7 (Table 1).
- Semi-supervised (self-distilled R152 3x+SK): 76.6 (1%) / 80.9 (10%) top-1 (Table 3).
- Distilled to ResNet-50: 73.9 (1%) / 77.5 (10%) top-1 (abstract).
- Memory gives ~1% (Sec. 2, App. "The Impact of Memory").
