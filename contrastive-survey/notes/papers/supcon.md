# SupCon: Supervised Contrastive Learning

## 1. Identity
- Title: Supervised Contrastive Learning
- arXiv 2004.11362, version read: v5 (abs.html: v1 Thu 23 Apr 2020; v5 Wed 10 Mar 2021). The tex changelog
  calls this revision "Version 6 (2020-03-10)" (its own numbering and a year typo); cite as v5.
- Authors: Prannay Khosla ... Dilip Krishnan (9 authors; Google Research, Boston University, Snap Inc., MIT per PDF p.1).
- Venue: NeurIPS 2020 (PDF p.1 footer "34th Conference on Neural Information Processing Systems (NeurIPS 2020)").
- Code: TF https://t.ly/supcon ; PyTorch https://github.com/HobbitLong/SupContrast (abstract footnote).

## 2. Core problem
Cross-entropy is the default supervised loss but has known weaknesses (noise, poor margins). Self-supervised
batch contrastive losses gave strong features but have only one positive per anchor. With labels, every
same-class sample is a known positive; the self-supervised loss cannot use that.

## 3. Prior methods and why that hurt (per the paper)
- Self-supervised loss (Eq. 1): "incapable of handling the case where ... more than one sample is known to
  belong to the same class" (Sec. 3.2.2).
- Triplet loss: one positive, one negative, needs "computationally expensive hard negative mining" (Supp. Sec. 8).
- N-pairs: positive from another same-class image only, no augmentation view, no temperature; 57.4% top-1 in
  their framework at batch 6144 (Sec. 4.1; Supp. Sec. 9).

## 4. Key idea
For each anchor, treat ALL other same-label samples in the 2N-view batch as positives, average the log-softmax
over them (sum OUTSIDE the log: L_out), keep every other sample in the denominator.

## 5. Mechanism
1. Sample N image/label pairs; augment each twice -> "multiviewed batch" of 2N views; the two views share the label.
2. Encoder Enc (ResNet-50/101/200) -> r, final pooling layer, D_E = 2048, L2-normalized.
3. Projection Proj: MLP with one hidden layer of 2048 and output D_P = 128 (or a single linear layer to 128);
   output L2-normalized. Discarded after training.
4. Logits z_i . z_a / tau for every anchor i and every a != i (2N-1 entries per row; self is excluded).
5. Positives P(i) = all a != i with the same label (includes the other view of i). Loss L_out (Eq. 2).
   Positives are NOT removed from the denominator (A(i) = I \ {i}); only self is excluded.
6. After training: freeze encoder, drop Proj, train a linear classifier with cross-entropy (the reported top-1).

## 6. Equations (verbatim from src/method.tex; macros: `\bigcdot` = a centered bullet dot product, defined in
commands.tex)

Eq. 1 (self-supervised loss it generalizes; label eqn:self_loss):
```
\mathcal{L}^{self}
=\sum_{i\in I}\mathcal{L}_i^{self}
=-\sum_{i\in I}\log{
\frac{\text{exp}\left(\boldsymbol{z}_i\bigcdot\boldsymbol{z}_{j(i)}/\tau\right)}{\sum\limits_{a\in A(i)}\text{exp}\left(\boldsymbol{z}_i\bigcdot\boldsymbol{z}_a/\tau\right)}
}
```
Paper: "for each anchor i, there is 1 positive pair and 2N - 2 negative pairs. The denominator has a total of
2N - 1 terms." Gloss: I = {1..2N} views; j(i) = the other view of the same image; A(i) = I \ {i};
z = normalized projection output; tau = temperature.

Eq. 2 (L_out, SupCon; label eqn:supervised_loss):
```
\mathcal{L}_{out}^{sup}
=\sum_{i\in I}\mathcal{L}_{out,i}^{sup}
=\sum_{i\in I}\frac{-1}{|P(i)|}\sum_{p\in P(i)}\log{\frac{\text{exp}\left(\boldsymbol{z}_i\bigcdot\boldsymbol{z}_p/\tau\right)}{\sum\limits_{a\in A(i)}\text{exp}\left(\boldsymbol{z}_i\bigcdot\boldsymbol{z}_a/\tau\right)}}
```
Eq. 3 (L_in; label eqn:bad_supervised_loss):
```
\mathcal{L}_{in}^{sup}
=\sum_{i\in I}\mathcal{L}_{in,i}^{sup}
=\sum_{i\in I}-\log\left\{\frac{1}{|P(i)|}\sum_{p\in P(i)}\frac{\text{exp}\left(\boldsymbol{z}_i\bigcdot\boldsymbol{z}_p/\tau\right)}{\sum\limits_{a\in A(i)}\text{exp}\left(\boldsymbol{z}_i\bigcdot\boldsymbol{z}_a/\tau\right)}\right\}
```
"P(i) = {p in A(i): y_p = y_i} is the set of indices of all positives in the multiviewed batch distinct from i."
Gloss: L_out averages -log p(positive) over each positive separately; L_in takes -log of the average probability.
Expected extra positives per anchor: N/C (paper's words).

Eq. 4 (gradient, either loss; label eqn:gradient):
```
\frac{\partial\mathcal{L}_i^{sup}}{\partial\boldsymbol{z}_i} = \frac{1}{\tau}\left\{
\sum_{p\in P(i)}\boldsymbol{z}_p(P_{ip}-X_{ip})+
\sum_{n\in N(i)}\boldsymbol{z}_nP_{in}
\right\}
```
with P_ix = softmax prob of x in row i over A(i); N(i) = different-label indices.
Eq. 5: X_ip = exp(z_i.z_p/tau) / sum_{p' in P(i)} exp(z_i.z_p'/tau) for L_in; X_ip = 1/|P(i)| for L_out.
Eq. 6: if every z_p is set to the mean positive z-bar, X_ip^in reduces to 1/|P(i)| = X_ip^out.

Gradient argument (Sec. 3.2.2): Jensen gives L_in <= L_out (log concave). "For L_out, the positives
normalization factor (i.e., 1/|P(i)|) serves to remove bias present in the positives"; in L_in the same factor
sits inside the log, "contributes only an additive constant ... which does not affect the gradient", so L_in
gradients are "more susceptible to bias in the positives". Conclusion: "the stabilization due to using the
mean of positives benefits training." NOTE: v5 states L_in <= L_out; earlier versions had the inequality sign
flipped (tex changelog, final entry "Corrected sign mistake in Jensen's Inequality"; the old wording survives as
a comment in method.tex).

Hard-mining property (Supp. Sec. 7, Eqs. 20-24): gradient magnitude from a positive ~ |P_ip - X_ip|; large for
hard positives (z_i.z_p ~ 0), small for easy ones (~1); grows with the number of negatives, and for L_out also
with the number of positives. Requires the normalization layer.

Triplet as special case (Supp. Sec. 8): with one positive and one negative, the loss ~ ||z_a - z_p||^2 -
||z_a - z_n||^2 + 2tau: "a triplet loss with margin alpha = 2tau".

## 7. Config
| item | value | loc |
|---|---|---|
| Encoders | ResNet-50, -101, -200 (ResNet-v1) | Sec. 4 |
| Representation | final pooling, D_E = 2048, normalized | Sec. 3.1, 4 |
| Projection | MLP hidden 2048 -> D_P = 128 (or linear 128), normalized | Sec. 3.1 |
| Augmentation | AutoAugment best for R50; Stacked RandAugment best for R200 | Sec. 4 |
| Batch | 6144 (R50), 4096 (R200); "2048 suffice for most purposes" | Sec. 4.5 |
| Epochs | 700 (R200), 350 smaller models; "even 200 epochs is likely sufficient" | Sec. 4.5 |
| Temperature | tau = 0.1 for all results; loss rescaled by tau for stability | Sec. 4.5 |
| Optimizer | LARS (cosine decay) for pretraining; RMSProp (exp decay) for linear layer; SGD+momentum on CIFAR | Sec. 4.5, Supp. 14.1 |
| Linear stage | "as few as 10 epochs"; Table 8 runs used 350 | Sec. 4.5, Supp. Table 8 |
| Memory variant | MoCo-style memory 8192 x 128-d, batch 256, SGD, 8 Nvidia V100 -> 79.1% R50 | Sec. 4.1 |
| Hardware for main runs | not stated | - |

## 8. Results (ImageNet unless noted; protocol = linear classifier on frozen encoder, trained with labels)
- Table 1: L_out 78.7% vs L_in 67.4% (R50, batch 6144).
- Table 3 (top-1/top-5): R50 SupCon AutoAugment 78.7/94.3; CE baselines R50: MixUp 77.4, CutMix 78.6,
  AutoAugment 78.2 (published), our impl. 77.6. R200 SupCon Stacked RandAugment 81.4/95.9; CE R200 AutoAugment
  80.6 (published), our impl. 80.9. R101 SupCon 80.2/94.7.
- Table 2 (R50 top-1): CIFAR10 SimCLR 93.6 / CE 95.0 / Max-Margin 92.4 / SupCon 96.0; CIFAR100 70.7/75.3/70.5/76.5;
  ImageNet 70.2/78.2/78.0/78.7.
- CE with batch 12,288: 77.5%; CE 1400 epochs: 77.0% (Sec. 4.1). N-pairs batch 6144: 57.4%.
- Fig. 3 left (ImageNet-C, lower is better): mCE CE R50 68.6, R200 52.4; SupCon R50 67.2, R200 50.6.
  rel. mCE CE R50 96.2, R200 69.1; SupCon R50 94.6, R200 66.5. (tex label says table; rendered as Figure 3.)
- Fig. 4(a): box plot, top-1 across changes in augmentation, optimizer (LARS, SGD+momentum, RMSProp), lr:
  "significantly lower variance" for SupCon; same batch sizes for both. Fig. 4(b): SupCon better at every batch
  size swept. Fig. 4(c) epochs, 4(d) temperature. No numbers printed in Fig. 4.
- Supp. Table 5 (Sec. 11; R50, batch 6144, 350 epochs): max positives 1 -> 69.3, 3 -> 76.6, 5 -> 78.0,
  7 -> 78.4, 9 -> 78.3, no cap (13) -> 78.5. 13 = expected positives at batch 6144.
- Supp. Sec. 10: optimal tau "can improve performance by nearly 3%"; tau = 0.1 optimal for R50.
- Table 4 transfer (fine-tune, 12 datasets, mean): SimCLR-50 84.81, Xent-50 84.67, SupCon-50 84.27;
  Xent-200 85.77, SupCon-200 85.67 ("on par").

## 9. Limitations stated
- Needs large batches to include many positives and negatives (Supp. Sec. 8).
- Transfer not better than CE or SimCLR (Sec. 4.4). ECE worse with more tuning (changelog).
- Very low tau numerically unstable (Sec. 4.5).

## 10. Outline feedback
- mCE is in Figure 3 (left), not a table.
- Gradient argument is Eqs. 4-6 plus the Jensen remark; cite v5 (sign corrected).
- The toy `viz-supcon` has a sourced anchor: Supp. Table 5 (1 positive 69.3 -> no cap 78.5).
- Triplet = SupCon with one positive/negative, margin 2tau (Supp. Sec. 8): a free link to ch-compare.
