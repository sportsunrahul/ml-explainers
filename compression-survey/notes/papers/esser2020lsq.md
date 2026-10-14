# esser2020lsq (BRANCH) -- Learned Step Size Quantization

Sources read: `sources/esser2020lsq/src/LSQ.tex`, `paper.pdf` (equation/table numbers), `abs.html`.

## Identity
- arXiv 1902.08153v3 (read v3; v1 2019-02-21, v3 2020-05-07). Authors: Steven K. Esser ... Dharmendra S.
  Modha (IBM). Venue: ICLR 2020 (abs.html comment and PDF header).

## Key idea (two sentences)
Make each layer's quantizer step size s a trainable parameter and give it a gradient that comes from the STE
through the round, so the gradient grows as a value nears a transition point between bins. Scale that
gradient by g = 1/sqrt(N Q_P) so step-size updates have a size comparable to weight updates.

## Equations (verbatim TeX; macros: `\sfrac` from xfrac, `\tb` = `\textbf`)
Eq. 1-2 (`eq:quantization1`, `eq:quantization2`), quantizer:
```
\bar{v} = \lfloor clip( \sfrac{v}{s}, -Q_N, Q_P) \rceil,
\hat{v} = \bar{v} \times s.
```
b bits: activations (unsigned) Q_N = 0, Q_P = 2^b - 1; weights (signed) Q_N = 2^{b-1}, Q_P = 2^{b-1} - 1.

Eq. 3 (`eq:lsq`, Sec. 2.1), the step-size gradient:
```
\frac{\partial{\hat{v}}}{\partial{s}} =
\begin{cases}
-\sfrac{v}{s} + \lfloor \sfrac{v}{s} \rceil			& \text{if $-Q_N < \sfrac{v}{s} < Q_P$} \\
-Q_N									& \text{if $\sfrac{v}{s} \le -Q_N$} \\
Q_P									& \text{if $\sfrac{v}{s} \ge Q_P$} \\
\end{cases}
```
Paper: "derived by using the straight through estimator \citep{bengio2013estimating} to approximate the
gradient through the round function as a pass through operation ... and differentiating all other operations
... normally." (One commented-out TeX line inside the cases block omitted.)

Gradient scale (Sec. 2.2 text, no equation number): "for weight step size $g=\sfrac{1}{\sqrt{N_WQ_P}}$ and for
activation step size $g=\sfrac{1}{\sqrt{N_FQ_P}}$", N_W = number of weights in the layer, N_F = number of
features. Motivation Eq. 4: ratio R of relative update sizes of s and w "should on average be near 1".

Eq. 5 (`eq:xgrad`, Sec. 2.3), STE for data:
```
\frac{\partial \hat{v}}{\partial v} =
\begin{cases}
1 & \text{if $-Q_N < \sfrac{v}{s} < Q_P$} \\
0 & \text{otherwise},
\end{cases}
```
Init: s = 2<|v|>/sqrt(Q_P) on initial weights or first activation batch. One step size per weight layer and
per activation layer (per-tensor). First and last layers always 8-bit. Fine-tuned from full-precision; 90
epochs for 2/3/4-bit (lr 0.01), 1 epoch for 8-bit (lr 0.001); cosine decay, momentum 0.9; PyTorch.

## Key result row
Table 1 (ImageNet top-1, pre-activation ResNet-50, full precision 76.9; weights and activations at the bit
width shown):
| method | 2 | 3 | 4 | 8 |
|---|---|---|---|---|
| LSQ | 73.7 | 75.8 | 76.7 | 76.8 |
| PACT (as cited in this table) | 72.2 | 75.3 | 76.5 | -- |
Table 4 (LSQ + same-architecture KD, ResNet-50 top-1): 2-bit 74.6, 3-bit 76.9, 4-bit 77.6, 8-bit 76.8, fp32
76.9 ("3-bit networks reaching the accuracy of full precision").
Table 3 (2-bit ResNet-18, gradient-scale ablation): g = 1/sqrt(N Q_P), lr 0.01 -> 67.6; g = 1/sqrt(N) -> 67.3;
g = 1, lr 0.01 -> "Did not converge"; g = 1, lr 0.0001 -> 64.2.

## Outline feedback
- ch-qat "LSQ step-size gradient (Esser Eq. 3)": correct. g is in Sec. 2.2 text, not an equation.
- For viz-ste: Fig. 2 of the paper (s = 1, Q_N = 0, Q_P = 3) plots LSQ vs QIL vs PACT step-size gradients; the
  toy can reuse these parameters. PACT's gradient is "zero everywhere below the clip point".
