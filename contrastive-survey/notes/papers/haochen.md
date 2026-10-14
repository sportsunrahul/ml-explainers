# HaoChen et al. 2021 -- spectral contrastive loss [branch]

- Title: "Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss". arXiv 2106.04156, read v7 (2022-06-24); v1 2021-06-08 (abs.html).
- Authors: Jeff Z. HaoChen, Colin Wei, Adrien Gaidon, Tengyu Ma (Stanford). Venue: NeurIPS 2021, oral (abs.html comment).

Key idea: build the "population augmentation graph" (nodes = augmented images, edge weight = probability two views come from the same image). Minimizing the spectral contrastive loss is low-rank factorization of the graph's normalized adjacency, so the features are its top eigenvectors; if classes are poorly connected subgraphs, a linear probe on these features provably has small error (main result Sec. 3).

Defining equation, Lemma 3.2 (Spectral contrastive loss), PDF Eq. (6), label `equation:spectral_loss`, src/theory_framework.tex (macros: `\Loss{f}`=\mathcal{L}(f), `\Exp{#1}`=\mathbb{E}_{#1}):
```
\Loss{f} \triangleq -2\cdot \Exp{x, x^+} \big[f(x)^\top f(x^+) \big] + \Exp{x, x^-}\left[\left(f(x)^\top f(x^-) \right)^2\right]
```
Gloss: pull positives with a linear term; push random pairs with a squared dot product (no softmax, no normalization).

Key number: Table 1 (linear eval top-1, CIFAR-10, 800 epochs): Ours 92.07 vs SimCLR (repro.) 90.60, SimSiam (repro.) 91.40. ImageNet 100-epoch linear (Table 2, batch 384): Ours 66.97 vs SimCLR 66.5, SimSiam 68.1 (others from SimSiam paper).
