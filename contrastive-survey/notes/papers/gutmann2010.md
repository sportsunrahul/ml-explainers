# Gutmann & Hyvarinen 2010: Noise-contrastive estimation (NCE)

## 1. Identity
- Title: Noise-contrastive estimation: A new estimation principle for unnormalized statistical models
- Venue: AISTATS 2010, JMLR W&CP 9 (PDF footer). Non-arXiv; read sources/gutmann2010/paper.pdf (8 pages).
- Authors: Michael Gutmann, Aapo Hyvarinen (University of Helsinki).

## 2. Core problem
Estimate a density model whose normalizing constant Z(alpha) is intractable (energy-based models, MRFs).

## 3. Prior methods and why that hurt
MLE cannot treat Z as a free parameter ("the likelihood can be made arbitrarily large by making Z(alpha)
go to zero"). Contrastive divergence and score matching avoid Z but trade off cost and efficiency (Sec 1).

## 4. Key idea
Learn the density by logistic regression that tells data from artificial noise, with the model
log-density inside the classifier. The normalizer c = -ln Z becomes an ordinary parameter.

## 5. Mechanism
1. Model ln p_m(.;theta) = ln p0_m(.;alpha) + c, theta = {alpha, c}.
2. Draw T data points x_t and T noise points y_t from a known p_n.
3. Classifier G(u;theta) = ln p_m(u;theta) - ln p_n(u); h = sigmoid(G).
4. Maximize J_T (Eq. 3). Theorem 1: the maximum is at f = ln p_d, with no normalization constraint.

## 6. Equations (PDF p.2, Sec 2.1)
Eq. (3)-(5):
```
J_T(\theta) = \frac{1}{2T} \sum_t \ln[h(x_t;\theta)] + \ln[1 - h(y_t;\theta)],
h(u;\theta) = \frac{1}{1 + \exp[-G(u;\theta)]},
G(u;\theta) = \ln p_m(u;\theta) - \ln p_n(u).
```
- x_t: observed data. y_t: noise samples from p_n. h: probability that u is data, not noise.
- In this paper the noise and data sets are the same size (T each; P(C=1) = P(C=0) = 1/2, Sec 2.2).
  The "m (or nu) noise samples per data sample" form is later work, used by Wu 2018 Eq. 6-7.
Sec 2.2: "the idea behind noise-contrastive estimation is 'learning by comparison'."

## 7. Setup (measurement)
ICA model in 4D (Sec 3.1): noise = Gaussian with the data's mean and covariance; compared to MLE,
MLE with importance sampling, contrastive divergence, score matching. Then natural-image models
(two-layer model, MRF).

## 8. Results
Conclusion: NCE "offers the best trade-off between computational and statistical efficiency" among the compared estimators (Sec 4).

## 9. Limitations
Noise must be nonzero wherever data is (Theorem 1). Efficiency depends on the noise choice.

## 10. Outline feedback
- Outline names NCE as "the ancestor of the name": correct. Binary data-vs-noise logistic loss,
  not a softmax over N; CPC's InfoNCE is the multi-class version. Note that difference on the page.
- The quote "learning by comparison" (Sec 2.2) fits ch-compare.
