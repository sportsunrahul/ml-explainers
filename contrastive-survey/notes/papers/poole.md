# Poole et al. 2019: On Variational Bounds of Mutual Information (branch)

## Identity
- arXiv 1905.06922, read v1 (2019-05-16). Authors: Ben Poole ... George Tucker (Poole, Ozair, van den Oord, Alemi, Tucker). Venue: ICML 2019 (icml2019 accepted style).

## Key idea
Puts the MI lower bounds (BA, NWJ/MINE, InfoNCE) in one framework. InfoNCE has low variance but high
bias: it is capped at log K. A new bound I_alpha interpolates between NWJ (alpha=0) and InfoNCE (alpha=1)
to trade bias for variance.

## Equation (Sec 2.3, Eq. 10, label `eq:cpc`; macro \cpc = \lbar{I}_{\text{NCE}})
```
I(X; Y) \ge \E\left[\frac{1}{K}\sum_{i=1}^K \log \frac{e^{f(x_i, y_i)}}{\frac{1}{K}\sum_{j=1}^K e^{f(x_i,y_j)}}\right]\triangleq \cpc,
```
"the expectation is over K independent samples from the joint distribution". Footnote: "The derivation by
oord2018representation relied on an approximation, which we show is unnecessary." Text: "I_NCE is upper
bounded by log K, meaning that this bound will be loose when I(X;Y) > log K." Optimal critic
f(x,y) = log p(y|x) + c(y).

## Key number
I_alpha is upper bounded by log(K/alpha) instead of log K (Sec 2.3).

## Outline feedback
Use Poole Sec 2.3 Eq. 10 as the rigorous source of "I >= log N - L, ceiling log N"; CPC App. A.1 uses an approximation.
