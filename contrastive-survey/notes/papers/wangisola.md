# Wang & Isola 2020 -- alignment and uniformity on the hypersphere

## 1. Identity
- Title: "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere".
- arXiv 2005.10242, read v10 (2022-08-15). v1 published 2020-05-20 (abs.html).
- Authors: Tongzhou Wang, Phillip Isola. Venue: ICML 2020 (src uses `\usepackage[accepted]{icml2020}`).
- Code: github.com/SsnL/align_uniform, github.com/SsnL/moco_align_uniform. Project page ssnl.github.io/hypersphere.
- v10 changelog (main.tex "Major Changelog"): 8/24/2020 added full ImageNet + MoCo v2; 11/6/2020 corrected Theorem 1 rate to O(M^{-1/2}); 8/15/2022 removed two STL-10 encoders from Fig. 5 and the appendix table.

## 2. Core problem
Contrastive loss works, but the usual "it maximizes mutual information" story does not explain behavior (tighter MI bounds can give worse features). The paper asks what the loss itself optimizes on the unit sphere.

## 3. Prior view and why it hurt
InfoMax reading: L_contrastive is a lower bound on I(f(x); f(y)). Per Sec. 3 "The InfoMax principle", maximizing a tighter bound often leads to worse downstream performance (cites Tschannen et al.), so the bound view does not predict quality.

## 4. Key idea
With many negatives, the contrastive loss splits into two terms: alignment (positive pairs map close) and uniformity (all features spread evenly over S^{m-1}). Both can be measured and optimized directly with two short losses.

## 5. Mechanism
1. Encoder f maps input to l2-normalized feature on S^{m-1}.
2. L_align = mean of ||f(x)-f(y)||^alpha over positive pairs.
3. L_uniform = log of mean Gaussian potential exp(-t ||f(x)-f(y)||^2) over all pairs (in-batch).
4. Theorem 1: as M -> inf, L_contrastive - log M -> (alignment term) + (log-sum-exp uniformity term).
5. Experiments: train many encoders with weighted mixes of the three losses; show the two metrics track downstream accuracy and that align+uniform alone matches or beats contrastive.

## 6. Equations (macros: `\lcontr`=\mathcal{L}_\mathsf{contrastive}, `\lalign`=\mathcal{L}_\mathsf{align}, `\lunif`=\mathcal{L}_\mathsf{uniform}, `\distnpos`=p_\mathsf{pos}, `\distndata`=p_\mathsf{data}, `\trieq`=\triangleq, `\sphere`=\mathcal{S}, `\T`=^{\mathsf{T}}, `\iidsim`=\overset{\text{i.i.d.}}{\sim}; all in src/macros.sty)

E1. Contrastive loss, Eq. (1), label `eq:contrastive_loss`, Sec. 3 (src/text/preliminaries.tex):
```
\lcontr(f; \tau, M) \trieq \expectunder[\substack{(x, y) \sim \distnpos \\ \{x^-_i\}_{i=1}^M \iidsim \distndata}]{- \log \frac{e^{f(x)\T f(y) / \tau}}{e^{f(x)\T f(y) / \tau} + \sum_i e^{f(x^-_i)\T f(y) / \tau}}}
```
Gloss: (x,y) a positive pair; x^-_i are M negatives drawn from the data; tau > 0 temperature; f outputs unit vectors. Graded positions: one positive score f(x)^T f(y) in the numerator; M negatives f(x^-_i)^T f(y) only in the denominator.

E2. Alignment loss, Sec. 4.1.1 (p.5). UNNUMBERED in v10 (`equation*`):
```
\lalign(f; \alpha) \trieq \expectunder[(x, y) \sim \distnpos] {\norm{f(x) - f(y)}_2^\alpha}, \quad \alpha > 0.
```
Paper: "the expected distance between positive pairs". Gloss: alpha is the exponent on the distance (alpha in {1,2} swept; alpha=2 in all best rows).

E3. Uniformity loss, Sec. 4.1.2 (p.5). UNNUMBERED (`align*`):
```
G_t(u, v) \trieq e^{-t \norm{u - v}_2^2} = e^{2t \cdot u\T v - 2t}, \quad t > 0,
\lunif(f; t) \trieq \log \expectunder[\hspace{2pt}x, y \iidsim \distndata] {G_t(u, v)} = \log \expectunder[\hspace{2pt}x, y \iidsim \distndata] {e^{-t \norm{f(x) - f(y)}_2^2}}, \quad t > 0.
```
Paper: "the logarithm of the average pairwise Gaussian potential". Gloss: x, y two independent data points (not a pair); t sets how local the repulsion is. Prop. 1: uniform measure sigma_d is the unique minimizer of the expected potential; Prop. 2: N-point minimizers converge weak* to sigma_d.

E4. Theorem 1 (Asymptotics of L_contrastive), Eq. (2), label `eq:contrastive_loss_limit`, Sec. 4.2 "Limiting Behavior of Contrastive Learning", PDF p.6:
```
\lim_{M \rightarrow \infty} \lcontr(f; \tau, M) - \log M = -\frac{1}{\tau} \expectunder[(x, y) \sim \distnpos]{f(x)\T f(y)} + \expectunder[x \sim \distndata]{\log\expectunder[x^- \sim \distndata]{e^{f(x^-)\T f(x) / \tau}}}.
```
Statements: (1) first term minimized iff f perfectly aligned; (2) if perfectly uniform encoders exist, they are the exact minimizers of the second term; (3) absolute deviation from the limit decays in O(M^{-1/2}). Proof in Appendix A ("Proofs and Additional Results for Section 4.2"). "Relation with L_uniform": L_uniform "essentially pushes the log outside the outer expectation, without changing the minimizer". A weaker M=1 result (Theorem 2 in appendix) is also given.

E5. PyTorch code, Figure 6 (p.7): `lalign(x, y, alpha=2)`, `lunif(x, t=2)`, `loss = lalign(x, y) + lam * (lunif(x) + lunif(y)) / 2`. Defaults: alpha=2, t=2.

## 7. Measurement setup
- Encoders trained: 304 STL-10, 64 NYU-Depth-V2, 45 ImageNet-100, 108 BookCorpus (Sec. 5; total 521, our arithmetic). Swept: loss weights, tau, alpha in {1,2}, t in {1,...,8}, batch size, embedding dim, epochs, lr, init.
- STL-10: AlexNet-based encoder (archs follow CMC); linear and 5-NN on output and fc7; Table 1 pool = 150 encoders, 128-d output, batch 768.
- NYU-Depth-V2: CNN encoder, depth MSE on conv5/conv4; Table 2 pool = 64 encoders, 128-d, batch 128.
- ImageNet-100 (CMC's 100 classes): MoCo variants, ResNet-50, SGD batch 128, lr 0.03, 240 epochs, EMA 0.999; linear probe; Table 3 pool = 45 encoders, 128-d, batch 128.
- ImageNet-1K: MoCo v2 official code, ResNet-50, batch 256, lr 0.03, 200 epochs, EMA 0.999, queue 65536, 128-d; linear probe 100 epochs lr 30 (App. B).
- BookCorpus: Quick-Thought, bi-GRU, 1200-d, batch 400.
- Hardware: "1-4 NVIDIA Titan Xp, Titan X PASCAL, Titan RTX, or 2080 Ti GPUs" (App. B).
- In-batch losses: L_contrastive uses other in-batch samples as negatives (symmetric 2K terms); L_uniform over pairs within {x_i} and within {y_i}.
- Toy (Fig. 3, Sec. 4): CIFAR-10 on S^1 (m=2), AlexNet-based; contrastive with tau=0.5, M=256. Linear accuracies: random init 12.71%, supervised 57.19%, contrastive 28.60%.

## 8. Results worth remembering
- Table 1 (STL-10, val acc; cols Output+Linear / Output+5-NN / fc7+Linear / fc7+5-NN):
  - Best L_contrastive only, L_contrastive(tau=0.19): 80.46 / 78.75 / 83.89 / 76.33.
  - Best L_align and L_uniform only, 0.98 L_align(alpha=2) + 0.96 L_uniform(t=2): 81.15 / 78.89 / 84.43 / 76.78.
  - Best among all, L_contrastive(tau=0.5) + L_uniform(t=2): 81.06 / 79.05 / 84.14 / 76.48.
- Table 2 (NYU-Depth, MSE lower better; conv5 / conv4): best contrastive 0.5 L_contrastive(tau=0.1): 0.7024 / 0.7575; best align+uniform 0.75 L_align(alpha=2) + 0.5 L_uniform(t=2): 0.7014 / 0.7592 (also best overall).
- Table 3 (ImageNet-100, MoCo-based, linear top1 / top5): best contrastive L_contrastive(tau=0.07): 72.80 / 91.64; best align+uniform 3 L_align(alpha=2) + L_uniform(t=3): 74.60 / 92.74 (also best overall).
- Table 4 (BookCorpus, align+uniform LOSES): MR: contrastive(tau=0.075) 77.51 vs 0.9 L_align + 0.1 L_uniform(t=5) 73.76; CR: contrastive(tau=0.05) 83.86 vs 80.95.
- Table 5 (ImageNet-1K, MoCo v2, 200 epochs, linear top1): L_contrastive(tau=0.2) MoCo v2 67.5% +- 0.1% (5 runs, official impl.) vs 3 L_align(alpha=2) + L_uniform(t=3) 67.69%. Released checkpoint 67.694% (App. B).
- Fig. 7: weighting L_align much higher than L_uniform collapses all inputs to one point (exp L_uniform = 1); weight ratio < 4 stays good. Fig. 8: finetuning from a tau=2.5 encoder with only one of the two losses degrades accuracy; both improves it.

Falsifiable claim for a claim card: "Encoders with lower L_align and L_uniform on validation data have higher downstream accuracy; training with only L_align + L_uniform gives comparable or better representations than L_contrastive." Falsified if, under a matched sweep, align+uniform-only encoders lose consistently, or if low (L_align, L_uniform) encoders do not sit in the high-accuracy corner (Fig. 5/9). Note the paper's own Table 4 (BookCorpus) is a partial counterexample: align+uniform-only is 3-4 points worse there; the abstract says "comparable or better" across the board.

## 9. Limitations stated
- Perfect alignment and perfect uniformity cannot both hold on a finite augmented dataset (Sec. 4.2 "Realizability").
- Theorem 1 is asymptotic in M; the M=1 result is weaker.
- Discussion: why the hypersphere is a good space is not rigorously answered.
- Later: Wang & Liu 2012.09740 show excessive uniformity hurts (uniformity-tolerance dilemma); DCL 2110.06848 discusses "negative-negative coupling" in L_uniform (its Appendix 5).

## 10. Outline feedback
- Outline says "L_align (Wang & Isola Eq. 2) and L_uniform (Eq. 3)". Wrong for v10: both are unnumbered (Sec. 4.1.1 / 4.1.2). Eq. (1) is L_contrastive; Eq. (2) is the Theorem 1 limit. Cite by section, not number.
- Default alpha=2, t=2 come from the Fig. 6 code; best ImageNet rows use t=3 with weight 3 on L_align.
- Claim card should mention the BookCorpus exception or restrict to vision (page is vision only, so restricting is fine).
- The CIFAR-10-on-S^1 figure (Fig. 3) is a natural reference for viz-sphere.
