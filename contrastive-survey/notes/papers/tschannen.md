# Tschannen et al. 2019/2020: On MI maximization for representation learning (analysis lens)

## 1. Identity
- Title: On Mutual Information Maximization for Representation Learning
- arXiv 1907.13625, read v2 (2020-01-23). v1 submitted 2019-07-31 (abs.html).
- Authors: Michael Tschannen ... Mario Lucic (Tschannen, Djolonga, Rubenstein, Gelly, Lucic). Google Research, Brain Team.
- Venue: ICLR 2020 (PDF header "Published as a conference paper at ICLR 2020").
- Code: github.com/google-research/google-research/tree/master/mutual_information_representation_learning (Sec 1 footnote).

## 2. Core problem
Many methods (CPC, CMC, DIM, AMDIM) say they maximize MI between views. MI is hard to estimate and
invariant to any invertible map, so why do these methods learn good features?

## 3. Claim
"the success of these methods cannot be attributed to the properties of MI alone, and ... they strongly
depend on the inductive bias in both the choice of feature extractor architectures and the
parametrization of the employed MI estimators" (abstract). "maximizing tighter bounds on MI can result
in worse representations" (Sec 1). Alternative view: InfoNCE is a metric-learning (N-pair / triplet) loss (Sec 4).

## 4. The four experiments (Sec 3) and one theory result (Sec 4)
1. Sec 3.1 Invertible encoders (RealNVP, 30 coupling layers, MLP shifts 2x512): true MI is constant
   for every parameter value, yet I_EST and linear accuracy both rise during training (Fig. 1a,b).
   An invertible encoder trained adversarially reaches worse-than-pixels accuracy while still
   maximizing MI (Fig. 1c). Thought experiment: PNG compression keeps MI but not useful geometry.
2. Sec 3.1 Non-invertible MLP (4 hidden layers, skip connections, init near identity), coupled noise
   std 0.05: I_EST and accuracy rise while the Jacobian condition number grows (Fig. 2): estimators
   favor hard-to-invert encoders.
3. Sec 3.2 Critics: bilinear vs separable (phi MLPs, 100 hidden, 100 out, 40k params) vs MLP critic
   (200 hidden, 40k params) on MLP encoders (2x300 hidden -> 100-d). MLP critic gives the tightest
   I_NWJ but "barely outperform[s] the baseline on pixel space"; bilinear critic is loosest and best (Fig. 3).
4. Sec 3.3 Encoders at a fixed bound: minimize L_t = |I_EST - t| for t = 2, 4. After 7k iterations
   both MLP (238k params) and ConvNet (220k params) match t, but the ConvNet gives clearly higher
   linear accuracy (Fig. 4a,b). Supervised, both reach about 94%.
5. Sec 4: with dependent (non-i.i.d.) negatives InfoNCE is "in general neither a lower nor an upper
   bound on the true MI" (proof in appendix; Fig. 4c on 2D Gaussians). CPC v2 draws negatives from the
   same image, which violates independence.

## 5. Equations (main.tex; macros \infonce=I_\texttt{NCE}, \inwj=\mine=I_\texttt{NWJ}, \iest=I_\texttt{EST}, lines 50-53)
Eq. (2) multi-view objective: `\max_{g_1 \in \mathcal{G}_1, g_2 \in \mathcal{G}_2} \quad \iest\left(g_1(X^{(1)}); g_2(X^{(2)})\right),`
Eq. (3) InfoNCE bound, label `eqn:infonce`:
```
I(X ; Y) \geq \mathbb{E}\left[\frac{1}{K} \sum_{i=1}^{K} \log \frac{e^{f\left(x_{i}, y_{i}\right)}}{\frac{1}{K} \sum_{j=1}^{K} e^{f\left(x_{i}, y_{j}\right)}}\right] \triangleq \infonce(X;Y),
```
K: batch size; f: critic; the 1/K inside the denominator is why the value tops out at log K.
Eq. (4) NWJ bound: `I(X ; Y) \geq \mathbb{E}_{p(x, y)}[f(x, y)]-e^{-1} \mathbb{E}_{p(x)}[\mathbb{E}_{p(y)} e^{f(x,y)}] \triangleq \inwj(X;Y).`
Sec 4 rewrite (unnumbered):
```
\infonce = ... = \log K -\mathbb{E}\!\left[\frac{1}{K} \sum_{i=1}^{K} \log \left(1 +  \sum_{j\neq i} e^{f\left(x_{i}, y_{j}\right) - f\left(x_{i}, y_{i}\right)}\right)\!\right]
```
Eq. (5) K-pair-mc loss (Sohn 2016 Eq. 7): with f(x,y) = phi(x)^T phi(y), maximizing I_NCE "is thus
equivalent to metric learning based on (5)".

## 6. Measurement setup
| item | value |
|---|---|
| data | MNIST: x_top (392 px) = view 1, x_bottom (392 px) = view 2; CIFAR10 repeats in appendix ("conclusions are analogous") |
| metric | linear classifier (SAGA, scikit-learn) on g_1(x_top), all training labels, test accuracy |
| baselines | linear on pixels of x_top about 85% (about 84% with added noise); supervised MLP/ConvNet about 94% |
| estimators | I_NCE, I_NWJ |
| default critic | bilinear x^T W y |
| batch / lr / optimizer | 128 / 1e-4 / Adam (batch up to 512 tried, same conclusions) |
| I_NCE ceiling | log(128) about 4.85 (footnote) |
| repeats | averaged over 20 runs |
| held fixed | per experiment: encoder (Sec 3.2), bound value t (Sec 3.3), critic (Sec 3.3 bilinear) |

## 7. What would falsify the claim (our reading)
- If, across critics with matched encoders, a tighter I_NWJ always gave higher linear accuracy (Sec 3.2 shows the opposite).
- If MLP and ConvNet encoders at the same bound t gave the same accuracy (Sec 3.3 shows ConvNet higher).
- If invertible encoders (constant true MI) showed no accuracy change during training (Sec 3.1 shows a rise).
Scope limit: MNIST/CIFAR10 halves, small nets, linear protocol; not ImageNet scale.

## 8. Results worth remembering
- Numbers are mostly figure-only. Text numbers: pixel baseline about 85%, supervised about 94%,
  noise baseline about 84%, ceiling log 128 about 4.85, t in {2, 4}, 7k iterations, 20 runs.
- "the testing I_NCE value is close to the theoretically achievable maximum value for all critics" (Sec 3.2) -- so I_NCE cannot rank critics; the critic result rests on I_NWJ.

## 9. Limitations (stated)
Linear evaluation only (Sec 5 discusses other protocols). Small-scale setup by design.

## 10. Outline feedback
- Outline: "tighter MI estimates do not give better features". The paper's wording is hedged:
  "can result in worse representations" / "can lead to better". Suggest "a tighter MI bound can give
  worse features" and keep "can".
- Claim card "what was measured: bound tightness vs linear accuracy" -- correct for Sec 3.2; add the
  fixed-bound encoder test (Sec 3.3) and the invertible-encoder test (Sec 3.1); say MNIST halves.
- Sec 4 ties InfoNCE to Sohn's N-pair loss: good bridge back to ch-compare.
