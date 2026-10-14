# Chuang et al. 2020 -- Debiased Contrastive Learning

## 1. Identity
- Title: "Debiased Contrastive Learning".
- arXiv 2007.00224, read v3 (2020-10-21). v1 published 2020-07-01 (abs.html).
- Authors: Ching-Yao Chuang ... Stefanie Jegelka (with Joshua Robinson, Lin Yen-Chen, Antonio Torralba). MIT.
- Venue: NeurIPS 2020 (PDF p.1 footer). Code: github.com/chingyaoc/DCL (Sec. 5). Not to be confused with "DCL" = Decoupled Contrastive Learning (2110.06848).

## 2. Core problem
Negatives are drawn uniformly from unlabeled data, so some share the anchor's class ("sampling bias"). These false negatives get pushed away and cost accuracy (Fig. 2: CIFAR-10 biased vs unbiased gap).

## 3. Prior methods and why that hurt
Standard loss (Eq. 1) samples x^- from p(x). Lemma 1: the biased loss upper-bounds the ideal unbiased loss plus a term that grows as the unbiased loss shrinks, so minimizing it can give "very different learned representations".

## 4. Key idea
Write the data distribution as a mix of same-class and other-class parts, p = tau+ p+ + tau- p-, and solve for the negative distribution: p- = (p - tau+ p+)/tau-. Estimate the "true negative" term by subtracting tau+ times the positive score from the mean negative score.

## 5. Mechanism
1. Assume uniform latent classes: rho(c) = tau+ ; tau- = 1 - tau+. A random x^- from p has the anchor's class with probability tau+ (Sec. 3.1). With C equal classes this is 1/C (our reading; paper states tau+ = rho(c) uniform).
2. Ideal loss uses x^- ~ p_x^- (Eq. 2, "unbiased", needs labels).
3. Take N -> inf (Lemma 2, Eq. 5-6) to get a loss with only expectations over p and p_x^+.
4. Estimate those with N samples u_i ~ p and M positives v_i ~ p_x^+ (Eq. 7, estimator g), clipped below at e^{-1/t}.
5. Plug N*g into the denominator (Eq. 8). Error O(N^{-1/2} + M^{-1/2}) (Theorem 3, Eq. 9).

## 6. Equations (src/method.tex, src/intro.tex; numbers from PDF; macro `\xm`=x^{-})
Note: here t is temperature (embedding on sphere of radius 1/t; t=1 in theory), tau+ is the class prior. Do not confuse with tau elsewhere on the page.

Eq. (1), `eq_bias` (standard/biased loss, Sec. 1):
```
\mathbb{E}_{x,x^+,\{x^-_i\}_{i=1}^N} \left [-\log \frac{e^{f(x)^T f(x^+)}}{e^{f(x)^T f(x^+) } + \sum_{i=1}^N e^{f(x)^T f(x_i^-)} } \right ].
```
Eq. (2), `eq_unbias_f`, unbiased loss with x_i^- ~ p_x^- and weight Q/N.

Eq. (6), `eq_pu_loss` (limit as N -> inf, Lemma 2, denoted \widetilde L^Q_Debiased):
```
\mathbb{E}_{\substack{x \sim p \\ x^+ \sim p_x^+ }} \left [-\log \frac{e^{f(x)^T f(x^+)}}{e^{f(x)^T f(x^+)} + \frac{Q}{\tau^-}\big(\mathbb{E}_{x^- \sim p} [e^{f(x)^T f(x^-)}] - \tau^+ \mathbb{E}_{v \sim p_x^+} [e^{f(x)^T f(v) }]\big)} \right ]
```
Eq. (7), `eq_g`, the estimator:
```
g(x, \{u_i\}_{i=1}^N, \{v_i\}_{i=1}^M) = \max \Big\{  \frac{1}{\tau^-} \Big (\frac{1}{N}\sum_{i=1}^N e^{f(x)^T f(u_i)} - \tau^+ \frac{1}{M}\sum_{i=1}^M e^{f(x)^T f(v_i)} \Big ),\;\;  e^{-1/t} \Big \}
```
Eq. (8), `eq_prac_obj`, THE DEBIASED OBJECTIVE:
```
L_{\substack{\textnormal{Debiased}}}^{N,M}(f) = \mathbb{E}_{\substack{x \sim p;\, x^+ \sim p_x^+ \\ \{u_i\}_{i=1}^N \sim p^N \\ \{v_i\}_{i=1}^N \sim {p_x^+}^M }} \left [-\log \frac{e^{f(x)^T f(x^+)}}{e^{f(x)^T f(x^+)} + Ng\Big(x, \{u_i\}_{i=1}^N, \{v_i\}_{i=1}^M\Big)} \right ]
```
(Source typo: `\{v_i\}_{i=1}^N` under the expectation; M is meant.) Gloss: u_i the N in-batch negatives (some false); v_i extra positives (M=1: v_1 = x^+); tau+ class prior; floor e^{-1/t} prevents log of a negative number.

Fig. 3 pseudocode (M=1):
```
standard_loss = -log(pos / (pos + neg))
Ng = max((-N * tau_plus * pos + neg) / (1-tau_plus), N * e**(-1/t))
debiased_loss = -log(pos / (pos + Ng))
```
Theorem 3, Eq. (9): |L~ - L^{N,M}| <= (e^{3/2}/tau-) sqrt(pi/(2N)) + (e^{3/2} tau+/tau-) sqrt(pi/(2M)).

## 7. Setup
- CIFAR10, STL10: SimCLR, ResNet-50, Adam lr 0.001, wd 1e-6, temperature t=0.5, latent dim 128, 400 epochs, linear probe (Adam lr 0.001). N = 2(BatchSize - 1); N plotted = 30, 62, 126, 254, 510. tau+ in {0.05, 0.1, 0.12} (Fig. 4a,b). M in {1,2,4,8} with N=256, tau+=0.1 (Fig. 4c), vs biased trained 500/600 epochs.
- Unbiased (supervised negatives) only on CIFAR10.
- ImageNet-100: CMC official code, temperature 0.07, M=1; tau+ in {0.005, 0.01}. When g < e^{-1/t}, they fall back to the biased loss (App. B).
- Also sentences (QT, M=2) and RL (CURL) -- out of scope (vision only).

## 8. Results
- Table 1 (ImageNet-100, CMC, linear top-1 / top-5): Biased (CMC) 73.58 / 92.06; Debiased (tau+=0.005) 73.86 / 91.86; Debiased (tau+=0.01) 74.6 / 92.08.
- STL10: "we improve the accuracy of SimCLR on STL10 by 4.26%" (Sec. 5.1, M=1). CIFAR10/STL10 values otherwise only in plots (Fig. 4): no printed table. Read from plot (approximate, do not quote as exact): CIFAR10 N=510 biased ~91, debiased ~92, unbiased ~94.
- Fig. 5: t-SNE CIFAR10, tau+=0.1.

## 9. Limitations stated
Assumes uniform class prior; tau+ must be estimated or tuned; positives come from augmentations (surrogate p+). Robinson et al. 2010.04592 (same group) add hardness weighting beta; beta=0 recovers this method.

## 10. Outline feedback
- Outline: "With C equally likely classes about 1/C of them are" -- consistent with tau+ = rho(c) uniform (Sec. 3); the "1/C" is our arithmetic, not printed. Note values used are 0.1 (CIFAR10, 10 classes) and 0.01 (ImageNet-100, 100 classes), consistent with 1/C.
- "Sourced rows ... CIFAR10, STL10": Chuang has no printed CIFAR10/STL10 table, only Fig. 4 curves plus the 4.26% STL10 sentence. Use ImageNet-100 Table 1 and the 4.26% number; or use Robinson's printed STL10 heatmap.
- Symbol clash: Chuang uses t for temperature and tau+ for class prior; page must rename consistently.
