# bengio2013 (BRANCH) -- Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation

Sources read: `sources/bengio2013/src/arxiv-dcc1.tex`, `paper.pdf` (section/equation numbers), `abs.html`.

## Identity
- arXiv 1308.3432v1 (only version), submitted 2013-08-15. Authors: Yoshua Bengio, Nicholas Leonard, Aaron
  Courville. Venue: none stated (comment: "substantial text overlap with arXiv:1305.2982").

## Key idea (two sentences)
The paper compares four ways to get a gradient through stochastic binary (hard-threshold) neurons, for
conditional computation (sparse gaters). One of them, which the paper names the straight-through (ST)
estimator, backpropagates through the threshold as if it were the identity.

## Exact definition (Sec. 4 "Straight-Through Estimator", PDF p. 7)
Quote: "Another estimator of the expected gradient through stochastic neurons was proposed by
\citet{Hinton-Coursera2012} in his lecture 15b. The idea is simply to back-propagate through the hard
threshold function (1 if the argument is positive, 0 otherwise) as if it had been the identity function. It
is clearly a biased estimator, but when considering a single layer of neurons, it has the right sign (this is
not guaranteed anymore when back-propagating through more hidden layers). We call it the {\bf
straight-through} (ST) estimator."

Eq. 13 (no label), with h_i from Eq. 3 (`eq:stochastic-binary-neuron`):
```
h_i = f(a_i, z_i) = \one_{z_i > \sigm(a_i)}      (Eq. 3; z_i ~ U[0,1])
g_i = \frac{\partial L}{\partial h_i}.           (Eq. 13)
```
Custom macros: `\one` = indicator, `\sigm` = sigmoid 1/(1+exp(-u)). g_i is the ST estimate of dL/da_i, where
a_i is the pre-sigmoid activation. A variant multiplies by the sigmoid derivative; "Better results were
actually obtained without multiplying by the derivative of the sigmoid" (Sec. 4; also App. "Hyperparameters").

Important for ch-qat: Bengio's STE is the plain identity, for a binary threshold, with no range clipping.
The "1 inside [min, max], 0 outside" STE used for fake quantization is later: Krishnamoorthi 1806.08342 Eq.
14-15 (credits Courbariaux et al. 2015), Wu 2004.09602 Eq. 13, Esser 1902.08153 Eq. 5.

Clipped form for the page, verbatim from Wu 2004.09602v1, Sec. 5.2, Eq. 13 (`eq:ste`, Figure 4 "3-bit fake
quantization forward and backward pass with STE derivative approximation"); the conditions use `y` where `x`
is meant, as printed:
```
 \frac{d\hat{x}}{dx} =
\begin{cases}
    0, & y < \beta \\
    1, &  \beta \le y \le \alpha \\
    0, & y > \alpha
\end{cases}
```
Wu Eq. 12 (`eq:fake`): `\hat{x} = \dequantize(\quantize(x, b, s), b, s)` (macros are `\DeclareMathOperator`).
[beta, alpha] = representable range (Wu Sec. 3.1). Wu cites Bengio 2013 for the STE.

## One key number
Figure 2 (right, a table inside the figure, not a numbered Table): MNIST classification error (%), conditional
computation gaters. Straight-through: train 3.3e-3, valid 1.42, test 1.39 (best valid and test among stochastic
gaters). Baseline Rectifier: valid 1.66, test 1.60.

## Outline feedback
- ch-qat line 125-126 cites "STE dx_hat/dx = 1 inside the range, 0 outside (Bengio 2013; ...)". Bengio defines
  only the unclipped identity pass-through (Sec. 4, Eq. 13), credited to Hinton's Coursera lecture 15b. Cite
  Bengio for the name and idea; cite Krishnamoorthi Eq. 15 / Wu Eq. 13 / Esser Eq. 5 for the clipped form.
