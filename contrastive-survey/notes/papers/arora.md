# Arora et al. 2019: A Theoretical Analysis of Contrastive Unsupervised Representation Learning (branch)

## Identity
- arXiv 1902.09229, read v1 (2019-02-25). Authors: Sanjeev Arora ... Nikunj Saunshi (Arora, Khandeparkar, Khodak, Plevrakis, Saunshi). Venue: ICML 2019 (source is the ICML 2019 template).

## Key idea
Assume positives come from the same latent class and negatives from the class marginal. Then a low
contrastive loss provably gives a low supervised loss for the mean classifier on downstream tasks over
those classes, minus a penalty for "class collision" (a negative from the positive's class).

## Equation (Sec 2, Eq. 5, label `eq:QT`; macros \gD etc. are calligraphic letters from math_commands.tex)
```
L_{un}(f)\coloneqq\mathop{\E}\left[ \ell \left(\left\{f(x)^T\big(f(x^+)-f(x^-_i)\right)\right\}_{i=1}^k \right) \Big]
```
Theorem 4.1 (Sec 4.1): `L^{\mu}_{sup}(\widehat f)\le\frac{1}{(1-\tau)}(L_{un}(f)-\tau) + \frac{1}{(1-\tau)}Gen_M`,
where tau "is the probability that two classes sampled independently from rho are the same" (not a temperature).

## Key number (Table 1, CIFAR-100, avg-2 binary tasks, trained output layer)
Unsupervised 93.2 vs supervised 97.2.

## Outline feedback
Warn readers: Arora's tau is the class-collision probability, not the softmax temperature tau of ch-infonce.
