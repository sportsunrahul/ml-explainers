# Robinson et al. 2021 -- Contrastive Learning with Hard Negative Samples

## 1. Identity
- Title: "Contrastive Learning with Hard Negative Samples".
- arXiv 2010.04592, read v2 (2021-01-24). v1 published 2020-10-09 (abs.html).
- Authors: Joshua Robinson ... Stefanie Jegelka (with Ching-Yao Chuang, Suvrit Sra). MIT.
- Venue: ICLR 2021 (PDF page header). Code: github.com/joshr17/HCL (App. D.1).

## 2. Core problem
How to pick negatives without labels so they are both true negatives and hard (close to the anchor), at no extra compute.

## 3. Prior methods and why that hurt
Negatives are sampled from the marginal p (Eq. 1, q = p). Most are easy and far away. Plain hard mining without labels selects false negatives, because the closest points are the most likely to share the label (Sec. 3, 6.1).

## 4. Key idea
Sample negatives from q_beta^-: true negatives (debiased, as in Chuang) re-weighted by exp(beta * similarity). Implement it by importance weights inside the batch, not by changing the sampler.

## 5. Mechanism
1. Principle 1: sample only true negatives. Principle 2: prefer negatives the embedding thinks are similar.
2. Define q_beta(x^-) proportional to e^{beta f(x)^T f(x^-)} p(x^-) (unnormalized vMF with concentration beta), and q_beta^- = q_beta conditioned on a different class.
3. PU split (Eq. 2): q_beta = tau- q_beta^- + tau+ q_beta^+; solve for q_beta^-.
4. Insert q_beta^- into the N -> inf loss (Eq. 3) to get Eq. (4).
5. Estimate the two expectations by importance sampling with in-batch samples: weights e^{beta s}/Z_beta, Z_beta estimated by the batch mean.
6. beta = 0 recovers Chuang's debiased loss; tau+ = 0 gives hard sampling without debiasing; beta -> inf approaches a worst-case-negative objective (Prop. 3, Eq. 5).

## 6. Equations (src/method.tex, src/background.tex; numbers from PDF)
Eq. (1), `eqn: practical objective` (Sec. 2): standard NCE-style loss with negatives x_i^- ~ q and weight Q/N (Q=N for finite N).

q_beta (Sec. 3.1, unnumbered display):
```
q_\beta^-(x^-) := q_\beta(x^- | h(x) \neq h(x^-)),\quad\text{where}\quad q_\beta(x^-) \propto e^{\beta f(x)^\top f(x^-)}\cdot  p(x^-),
```
for beta >= 0. Paper: "The exponential term in $q_\beta$ is an unnormalized von Mises--Fisher distribution with mean direction $f(x)$ and ``concentration parameter'' $\beta$".

Eq. (2): `q_\beta(x^-) =   \tau^-  q_\beta^-(x^-) + \tau^+ q^+_\beta(x^-)`.

Eq. (4), `eqn: main objective`, hardness-biased objective:
```
\mathbb{E}_{\substack{x \sim p \\ x^+ \sim p_x^+ }} \left [-\log \frac{e^{f(x)^T f(x^+)}}{e^{f(x)^T f(x^+)} + \frac{Q}{\tau^-}(\mathbb{E}_{x^- \sim q_\beta} [e^{f(x)^T f(x^-)}] - \tau^+ \mathbb{E}_{v \sim q_\beta^+} [e^{f(x)^T f(v) }])} \right ].
```
Final estimator (importance sampling, unnumbered, Sec. 3.1):
```
\mathbb{E}_{x^- \sim q_\beta} [e^{f(x)^T f(x^-)}] = \mathbb{E}_{x^- \sim p} [e^{(\beta+1) f(x)^T f(x^-)} /  Z_\beta],
\mathbb{E}_{v \sim q_\beta^+} [e^{f(x)^T f(v) }] = \mathbb{E}_{v \sim p^+} [e^{(\beta+1) f(x)^T f(v)} / Z_\beta^+],
\widehat{Z}_\beta = \frac{1}{N}  \sum_{i=1}^N e^{\beta f(x)^\top f(x_i^-)} , \qquad \widehat{Z}_\beta^+ = \frac{1}{M}  \sum_{i=1}^M e^{ \beta f(x)^\top f(x_i^+)}.
```
Gloss: each in-batch negative's term e^{s} gets weight e^{beta s}/Z_hat_beta (mean weight 1), so closer negatives count more. "only two extra lines of code ... with no additional computational overhead".

Pseudocode (App. D, file scripts/debiased-importance-objective.sh), as printed:
```
reweight = (beta*neg) / neg.mean()
Neg = max((-N*tau_plus*pos + reweight*neg).sum() / (1-tau_plus), e**(-1/t))
hard_loss = -log( pos.sum() / (pos.sum() + Neg))
```
Our reading: printed `reweight = (beta*neg)/neg.mean()` does not literally equal e^{beta s}/Z_beta (neg = e^{s/t}); the math (above) is authoritative. Not checked against the repo (no README in sources).

Eq. (5): inf_f sup_{q in Pi} L(f,q), Pi = distributions supported on different-class points; Prop. 3: L(f, q_beta^-) -> sup as beta -> inf.

## 7. Setup (Sec. 5.1, App. D.1)
- SimCLR, ResNet-50, 2048-d representation, 128-d projection, Adam lr 0.001, wd 1e-6, 400 epochs. N = 2(batch-1); batch 16, 32, 64, 128, 256 -> N = 30, 62, 126, 254, 510. M = 1 for all experiments.
- beta values in plots: Fig. 2 uses beta in {0.5, 1}; Fig. 4 sweeps beta in {0, 0.5, 1.0, 2.0, 6.0} (STL10 tau+=0.1; CIFAR100 tau+=0.05). tau+ values in Fig. 2: STL10 0.1, 0.12; CIFAR100 0.01, 0.05; CIFAR10 0.1, 0.12.
- MoCo-v2 (App. C.1): N = 65536 queue, 200 epochs, batch 128, CIFAR10.
- Temperature t for vision not stated in main text (our reading: SimCLR default; unverified).

## 8. Results
- Sec. 5.1 text: at N=510, absolute gains over SimCLR of 3% (CIFAR100) and 7.3% (STL10); over best debiased of 1.9% and 3.2%. CIFAR10: slight gain for small N, disappears at large N.
- Table 1 (tinyImageNet, linear top-1, tau+=0.01): SimCLR 53.4%, Debiased 53.7%, Hard (beta=1) 57.0%.
- Fig. 4 right (STL10 linear, printed in heatmap): Hard + Debiased 87.44%; Hard + not debiased 84.41%; not hard + debiased 84.26%; neither (SimCLR) 80.15%.
- App. C.1: CIFAR10 MoCo-v2 88.08% vs with hard negatives (beta=0.2, tau+=0) 88.47%.
- Sec. 6.3 / Fig. 11: kNN (k=200); STL10 beta=1 reaches SimCLR's 400-epoch accuracy in 60 epochs; CIFAR100 in 125 epochs.
- Sec. 6.1: larger beta does not always help; with true positives accuracy rises monotonically in beta. CIFAR100 best beta 0.5; beta=6 worse (App. C.2).
- No ImageNet-100 or ImageNet-1K result.

## 9. Limitations stated
Debiasing is only approximate because p+ comes from augmentations, so large beta selects false negatives; annealing beta helps robustness. tau+ > 0 caused numerical issues for sentences.

## 10. Outline feedback
- "Results: sourced rows from Chuang and Robinson (CIFAR10, STL10, ImageNet-100)": Robinson has no ImageNet-100. Best printed vision numbers: the STL10 2x2 heatmap (87.44 / 84.41 / 84.26 / 80.15) -- ideal for a "hard x debias" card -- and tinyImageNet Table 1.
- btn-neg-hard should reweight negatives by e^{beta s} (normalized by batch mean). Paper beta range 0-6; 0.5-1 best.
- Combination with debiasing: same formula as Chuang with e^{s} replaced by e^{(beta+1)s}/Z_hat_beta on both terms.
