# CMC: Contrastive Multiview Coding (branch)

## Identity
- arXiv 1906.05849, read v5 (2020-12-18); v1 2019-06-13. Authors: Yonglong Tian, Dilip Krishnan, Phillip Isola. Venue: ECCV 2020 (stated by later papers; source does not print it -- not verified here).
- Code: github.com/HobbitLong/CMC (abstract).

## Key idea
Views need not be crops: different sensory channels of the same scene (L vs ab color channels, depth,
flow) are positives. One encoder per view; a symmetric contrastive loss pulls a scene's views together.

## Equation (Sec 3.2, PDF Eq. 2, label `eqn:contrastive_loss`)
```
\mathcal{L}_{contrast}^{V_1,V_2}  = - \mathop{{}\mathbb{E}}_{\{v_1^1, v_2^1, \ldots, v_2^{k+1}\}}\left[ \log \frac{h_{\theta}(\{v_1^1, v_2^1\})}{\sum_{j=1}^{k+1}h_{\theta}(\{v_1^1, v_2^j\})} \right]
```
with h = exp(cosine(f1(v1), f2(v2)) / tau); total loss L(V1,V2) = L^{V1,V2} + L^{V2,V1}. Bound
`I(z_i; z_j) \geq \log(k) - \mathcal{L}_{contrast}` (label `eqn:mi_lb`; k = number of negatives).

## Setup (Sec 4.1 "Implementation")
Memory bank as in Wu 2018: tau 0.07, memory momentum 0.5, 16384 negatives. Two encoders (half-width
networks split across channels, or one ResNet-50 per view).

## Key number (Table 1, linear, single crop, top-1/top-5)
- {L,ab}, ResNet-50 x1 per view: 64.0 / 85.5. ResNet-50 x2 per view: 68.3 / 88.2.
- {Y,DbDr} + RandAugment, ResNet-50 x2: 70.6 / 89.7.
Pretraining epochs for the ResNet runs: not found in the text.

## Outline feedback
"views = channels" is right. The params are per view (two encoders), so compare with care.
