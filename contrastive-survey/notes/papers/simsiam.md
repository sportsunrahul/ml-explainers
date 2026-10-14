# SimSiam (branch)

- Title: Exploring Simple Siamese Representation Learning. arXiv 2011.10566, read v1 (2020-11-20).
- Authors: Xinlei Chen, Kaiming He. abs.html comment: "Technical report, 10 pages".
- Key idea: BYOL without the momentum encoder. A shared encoder plus a predictor on one side and stop-gradient on the other is enough to avoid collapse; no negatives, no large batch, plain SGD.
- Defining equation (`eq:loss_sym_stopgrad`, Eq. 4; `\dist` = D, `\p` = p, `\z` = z):
```
\mathcal{L}{=}\frac{1}{2}\dist(\p_1, \texttt{stopgrad}(\z_2)){+}\frac{1}{2}\dist(\p_2, \texttt{stopgrad}(\z_1)).
```
  with D(p1, z2) = negative cosine similarity (Eq. 1). Minimum -1.
- Config (Sec 3/4): projection MLP 3 layers, 2048-d with BN; predictor 2048 -> 512 -> 2048; SGD base lr 0.05 x batch/256, wd 1e-4, momentum 0.9; default batch 512.
- Key number: 71.3 ImageNet linear top-1, ResNet-50, 800 ep, batch 256, two 224x224 views (Table 4, row SimSiam, column 800 ep).
- Table 4 reproductions by SimSiam authors (all two 224 views, "repro.+" = improved vs original): SimCLR 66.5/68.3/69.8/70.4, MoCo v2 67.4/69.9/71.0/72.2, BYOL 66.5/70.6/73.2/74.3, SwAV 66.5/69.1/70.7/71.8 at 100/200/400/800 ep; SimSiam 68.1/70.0/70.8/71.3.
- Note: "SwAV (w/o multi-crop) 71.8" in Barlow Twins Table 1 and VICReg Table 1 is this SimSiam reproduction.
