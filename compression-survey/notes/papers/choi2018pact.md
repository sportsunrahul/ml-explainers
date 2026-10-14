# choi2018pact (BRANCH) -- PACT: Parameterized Clipping Activation for Quantized Neural Networks

Sources read: `sources/choi2018pact/paper.pdf` (pages 3-4 viewed as images for the equations), `abs.html`.
`src/main2.tex` only includes the PDF (`\includepdf`), so there is no TeX for the equations; the TeX below is
transcribed from the PDF.

## Identity
- arXiv 1805.06085v2 (read v2; v1 2018-05-16, v2 2018-07-17). Authors: Jungwook Choi ... Kailash
  Gopalakrishnan (IBM Research AI). Venue: none stated on arXiv (source file name suggests an ICLR 2018
  submission; not a claim).

## Key idea (two sentences)
Replace ReLU with a clipped ReLU whose clip level alpha is a trainable per-layer parameter, then quantize
[0, alpha] uniformly to k bits. alpha gets a gradient through the STE and an L2 regularizer, so each layer
learns its own activation range instead of a fixed or EMA range.

## Equations (transcribed from PDF p. 3-4, Sec. 4)
Eq. 1:
```
y = PACT(x) = 0.5(|x| - |x - \alpha| + \alpha) =
\begin{cases} 0, & x \in (-\infty, 0) \\ x, & x \in [0, \alpha) \\ \alpha, & x \in [\alpha, +\infty) \end{cases}
```
Eq. 2:
```
y_q = round(y \cdot \frac{2^k - 1}{\alpha}) \cdot \frac{\alpha}{2^k - 1}
```
Eq. 3 (STE estimates dy_q/dy as 1, "(Bengio et al. (2013))"):
```
\frac{\partial y_q}{\partial \alpha} = \frac{\partial y_q}{\partial y}\frac{\partial y}{\partial \alpha} =
\begin{cases} 0, & x \in (-\infty, \alpha) \\ 1, & x \in [\alpha, +\infty) \end{cases}
```
So only clipped inputs push alpha. Setup (Sec. 4.2, 5): one alpha per layer; init large then L2-regularize;
first and last layers not quantized; weights quantized with the DoReFa scheme; all networks trained from
scratch. Fig. 2: CIFAR10-ResNet20 alpha starts at 10 and converges to much smaller values.

## Key result row
Table 1 (top-1, fraction as printed; weights via DoReFa, activations via DoReFa or PACT; k bits for both):
| network | FullPrec | DoReFa 2b | 3b | 4b | 5b | PACT 2b | 3b | 4b | 5b |
|---|---|---|---|---|---|---|---|---|---|
| ResNet50 | 0.769 | 0.671 | 0.699 | 0.714 | 0.714 | 0.722 | 0.753 | 0.765 | 0.767 |
| ResNet18 | 0.702 | 0.626 | 0.675 | 0.681 | 0.684 | 0.644 | 0.681 | 0.692 | 0.698 |
Sec. 6 (simulated accelerator, 64 GBps): 16 -> 4 bits gives 4x peak FLOPs but 4.5x performance.

## Outline feedback
- viz-ste's learned-range toggle could show PACT next to LSQ: PACT's alpha gradient is 0 inside the range and
  1 above it (Eq. 3); LSQ's step gradient also depends on distance to each rounding transition (Esser Fig. 2).
