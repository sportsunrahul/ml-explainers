# CPC: Representation Learning with Contrastive Predictive Coding

## 1. Identity
- Title: Representation Learning with Contrastive Predictive Coding
- arXiv 1807.03748, read v2 (v2 dated 2019-01-22). v1 submitted 2018-07-10 (abs.html).
- Authors: Aaron van den Oord ... Oriol Vinyals (3 authors: Oord, Li, Vinyals). DeepMind.
- Venue: arXiv preprint (nips_2018.sty in source; no venue stated on the paper).
- Repo: none stated.

## 2. Core problem
Learn useful features from unlabeled high-dimensional data (speech, images, text, RL) without a
generative model of every pixel. Predicting raw future observations wastes capacity on detail.

## 3. What prior methods did and why that hurt
- Unimodal losses (MSE, cross-entropy on x) are "not very useful" for high-dim targets; powerful
  conditional generative models p(x|c) "waste capacity" on detail and can ignore context c (Sec 2.1).
- Triplet / max-margin contrastive losses (Chopra 2005, FaceNet) existed; CPC frames the loss
  probabilistically instead (Sec 2.4).

## 4. Key idea
Predict the future in latent space, and score the prediction with a softmax over one true future
latent and N-1 negatives (InfoNCE). Minimizing it raises a lower bound on MI: I >= log N - L_N.

## 5. Mechanism (image version, Sec 3.2)
1. Resize/crop: a 256x256 crop is taken at random from a 300x300 image; horizontal flip p=0.5; greyscale.
2. Cut a 7x7 grid of 64x64 crops "with 32 pixels overlap" (stride 32 = our reading: (256-64)/32+1 = 7).
3. Each 64x64 crop: random 60x60 subcrop, padded back to 64x64.
4. Encoder g_enc = ResNet-v2-101 (not pretrained, no BatchNorm). Take third residual block output,
   spatial mean-pool -> one 1024-d vector per patch -> tensor 7x7x1024.
5. Context network g_ar: PixelCNN-style autoregressive model over rows (a row-GRU PixelRNN gave
   similar results) -> context c for each position, seeing only rows above.
6. Predict up to five rows below with a linear W_k per offset k; InfoNCE per patch in each target row.
7. Gradient reaches g_enc, g_ar and all W_k; nothing is frozen in pretraining.
8. Eval: freeze, mean-pool 7x7x1024 -> 1024-d, train a linear classifier.

## 6. Equations (method.tex, PDF numbers)
Eq. (1), label `mi`:
```
I(x;c) &= \sum_{x,c} p(x,c) \log \frac{p(x|c)}{p(x)}.
```
Mutual information between target x and context c.

Eq. (2), label `math:density_ratio`:
```
f_k(x_{t+k}, c_{t}) \propto \frac{p(x_{t+k}| c_t)}{p(x_{t+k})}
```
Score f_k models a density ratio, not a density.

Eq. (3) log-bilinear critic:
```
f_k(x_{t+k}, c_{t}) = \exp \Big(z_{t+k}^T W_k c_t\Big),
```
z_{t+k}: encoded future patch. c_t: context. W_k: one linear map per prediction step k.

Eq. (4), label `loss` (InfoNCE), Sec 2.3, PDF p.3:
```
\mathcal{L_{\text{N}}} &= - \mathop{{}\mathbb{E}}_{X}\left[\log \frac{f_k(x_{t+k}, c_{t})}{\sum_{x_j \in X} f_k(x_j, c_{t})}\right]
```
Paper: "the categorical cross-entropy of classifying the positive sample correctly".
- X = {x_1..x_N}: N samples, one positive from p(x_{t+k}|c_t), N-1 negatives from p(x_{t+k}).
- Numerator: score of the true future. Denominator: sum over all N candidates (positive included).
- Grades: for each context position and each k, the true patch k rows below is the positive;
  other patches in X are negatives.
- No temperature, no L2 normalization in CPC v1 (bilinear score).

Eq. (5) optimal posterior p(d=i|X,c_t) = ratio of density ratios; shows optimal f_k is
proportional to p(x|c)/p(x), "independent of the choice of the number of negative samples N-1".

MI bound, Sec 2.3 (unnumbered display, PDF p.3, just after Eq. 5):
```
I(x_{t+k}, c_t) \geq \log(N)-\mathcal{L_{\text{N}}},
```
"which becomes tighter as N becomes larger ... minimizing the InfoNCE loss L_N maximizes a lower
bound on mutual information." Derivation: Appendix A.1, Eqs. (6)-(11), PDF p.13. Step (8) is an
approximation ("\approx", replacing the negative sum by (N-1) times its expectation); Eq. (10) uses ">=".
Poole et al. 2019 (Sec 2.3, footnote) prove the bound without that approximation.
Consequence (our reading, consistent with Poole Sec 2.3): log N - L_N <= log N, since L_N >= 0.
Appendix A.1 also shows InfoNCE relates to MINE (Eqs. 12-15); MINE "became very unstable" for easy targets.

## 7. Config (images, Sec 3.2)
| item | value |
|---|---|
| data | ImageNet ILSVRC |
| input | 256x256 random crop of 300x300, flip 0.5, greyscale |
| patches | 7x7 grid of 64x64, 32 px overlap; 60x60 random subcrop padded to 64 |
| encoder | ResNet-v2-101, no BN, 3rd residual block, mean-pool -> 1024-d |
| context | PixelCNN-style autoregressive (row-GRU PixelRNN similar) |
| predictions | up to 5 rows ahead |
| optimizer | Adam, lr 2e-4 |
| batch / hardware | 32 GPUs, batch 16 each |
| negatives (images) | NOT STATED for images |
| epochs / steps (images) | NOT STATED |
| linear eval | SGD mom 0.9, lr 0.1/0.01/0.001 for 50k/25k/10k updates, batch 2048, 1 GPU, on mean-pooled 1024-d |
Audio (Sec 3.1) for contrast: 8 GPUs x minibatch 8, "from which the negative samples in the
contrastive loss are drawn"; 12 steps ahead; ~300,000 updates.

## 8. Results
- ImageNet top-1 linear, ResNet-v2-101, no BN: 48.7 (Table 3, row CPC). Previous best ResNet-V2: Colorization 39.6.
- ImageNet top-5: 73.6 (Table 4, row CPC); combination MS+Ex+RP+Col 69.3.
- Text: "improve upon state-of-the-art by 9% absolute in top-1" (Sec 3.2 text).
Protocol: linear classifier on frozen mean-pooled features, full ImageNet labels; pretraining epochs not stated.

## 9. Limitations
- Bound saturates at log N (stated in A.1: "useful to use large values of N").
- Derivation uses an approximation (Eq. 8); fixed by Poole 2019.
- Image numbers are far below supervised; CPC v2 (1905.09272) raises ResNet-50 linear to 63.8 and ResNet-161 to 71.5.

## 10. Outline feedback
- "CPC Eq. 4 and the bound (Sec. 2.3)": correct. Bound derivation is App. A.1 Eqs. 6-11.
- ArchFlow needs a negatives count: CPC v1 does not state it for images. Do not invent one; say
  "other patches in the batch" only if labelled our reading, or use CPC v2 which states "other
  locations in the image and other images in the mini-batch".
- CPC v1 uses no temperature and a bilinear score; the "page notation with sim(u,v)/tau" step must
  say tau comes from Wu 2018 / later work, not CPC.
- Stride 32 is our reading from "32 pixels overlap" on 64 px crops in 256 px.
