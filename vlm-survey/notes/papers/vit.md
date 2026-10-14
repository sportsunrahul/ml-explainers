# ViT (2010.11929v2) -- extraction for ch-patches

Source: LaTeX (03_method.tex, 04_experiments.tex, 06_appendix.tex). Equation numbers follow label order in the
align block: (1) embedding, (2) MSA residual, (3) MLP residual, (4) final representation.

## Patchify (Sec. 3.1)
Image x in R^{H x W x C} is reshaped to a sequence of flattened 2D patches x_p in R^{N x (P^2 C)}; (H,W) original
resolution, C channels, (P,P) patch resolution, N = HW/P^2 patches = the Transformer's effective sequence length.
Constant latent width D in all layers; a trainable linear projection E maps each flattened patch to D dims ("patch
embeddings"). A learnable [class] embedding is prepended (z_0^0 = x_class); its output state z_L^0 is the image
representation y. Classification head: MLP with one hidden layer at pre-training, single linear layer at fine-tuning.
Position embeddings: standard learnable 1D (no significant gain from 2D-aware, App. D.4). Pre-LN (LayerNorm before
every block), residual after every block, MLP = two layers with GELU. Sequence length = N + 1 (E_pos has N+1 rows).

## Eq. 1-4 verbatim (03_method.tex)
```
z_0 = [ x_class ; x_p^1 E ; x_p^2 E ; ... ; x_p^N E ] + E_pos,      E in R^{(P^2 C) x D},  E_pos in R^{(N+1) x D}      (1)
z'_l = MSA(LN(z_{l-1})) + z_{l-1},                                  l = 1...L                                          (2)
z_l  = MLP(LN(z'_l)) + z'_l,                                        l = 1...L                                          (3)
y    = LN(z_L^0)                                                                                                        (4)
```
TeX source: \mbf{z}_0 = [ \mbf{x}_\text{class}; \mbf{x}^1_p \mbf{E}; ... ; \mbf{x}^{N}_p \mbf{E} ] + \mbf{E}_{pos}.
Gloss: x_p^i = i-th flattened patch (length P^2 C, e.g. 768 for P=16, C=3); E = shared patch projection (one matrix for
all patches); E_pos = one learned vector per position incl. the class slot; z_0 = input token sequence; z_l = state after
layer l; L = number of layers; z_L^0 = final state of the class token; y = image representation fed to the head.
Self-attention (App. A): A = softmax(q k^T / sqrt(D_h)), A in R^{N x N} -- the N x N attention matrix is the quadratic cost.
"Note that the Transformer's sequence length is inversely proportional to the square of the patch size, thus models
with smaller patch size are computationally more expensive." (Sec. 4.1 Model Variants)

## Table 1 (Sec. 4.1, "Details of Vision Transformer model variants")
| Model | Layers | Hidden size D | MLP size | Heads | Params |
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |
Params are per variant, independent of patch size (patch only changes the E matrix and N). "Base"/"Large" are
directly from BERT; "Huge" added. Notation ViT-L/16 = Large with 16x16 input patches.

## Patch sizes and resolutions used
Variants trained: B/16, B/32, L/16, L/32, H/14 (Table 5 App.; main results L/16 and H/14). Pre-training res 224
("all training is done on resolution 224"). Fine-tuning ImageNet for Table 2: 512 for ViT-L/16, 518 for ViT-H/14;
Table 5 (appendix data-scaling) fine-tunes at 384. Higher-res fine-tuning keeps patch size, so N grows; pos embeddings
are 2D-interpolated. Token counts (computed by me, not stated in the paper, N=(H/P)^2, plus 1 class token):
224/16 -> 14x14 = 196 (+1 = 197); 224/32 -> 7x7 = 49; 224/14 -> 16x16 = 256; 384/16 -> 24x24 = 576; 512/16 -> 32x32 = 1024;
518/14 -> 37x37 = 1369. Paper itself mentions "14 x 14" spatial size for 224 input (App. D.4).

## Data-size finding (Sec. 4.3; Fig. 3, Fig. 4; Table 5 appendix)
Datasets (Sec. 4.1): ImageNet (ILSVRC-2012) 1k classes 1.3M images; ImageNet-21k 21k classes 14M images;
JFT 18k classes 303M high-res images (called JFT-300M).
Intro: on mid-sized datasets like ImageNet without strong regularization ViTs give "modest accuracies of a few
percentage points below ResNets of comparable size"; on 14M-300M images "large scale training trumps inductive bias".
Fig. 3 (transfer to ImageNet, three pre-training sets): "When pre-trained on the smallest dataset, ImageNet,
ViT-Large models underperform compared to ViT-Base models, despite (moderate) regularization. With ImageNet-21k
pre-training, their performances are similar. Only with JFT-300M, do we see the full benefit of larger models."
BiT CNNs beat ViT on ImageNet-only, ViT overtakes with larger datasets. (Fig. 3 is graph-only; exact numbers are in
Table 5 below.) Table 5, top-1 ImageNet after fine-tuning at 384 (B/16, B/32, L/16, L/32, H/14):
 ImageNet pretrain: 77.91, 73.38, 76.53, 71.16, -
 ImageNet-21k:      83.97, 81.28, 85.15, 80.99, 85.13
 JFT-300M:          84.15, 80.73, 87.12, 84.37, 88.04
Fig. 4 (linear 5-shot ImageNet vs pre-training subsets 9M, 30M, 90M, 300M JFT; same hyperparameters, early stopping):
"ViTs overfit more than ResNets with comparable computational cost on smaller datasets": ViT-B/32 (slightly faster than
ResNet50) is much worse on 9M, better on 90M+; same for ResNet152x2 vs ViT-L/16. "the convolutional inductive bias is
useful for smaller datasets, but for larger ones, learning the relevant patterns directly from data is sufficient,
even beneficial." Fig. 4 is graph-only; no table with the 9M/30M/90M numbers.
Headline (Table 2): ViT-H/14 JFT ImageNet 88.55 +- 0.04, L/16 JFT 87.76, L/16 I21k 85.30, BiT-L 87.54,
Noisy Student 88.4/88.5. Cost TPUv3-core-days: H/14 2.5k, L/16 (JFT) 0.68k, L/16 (I21k) 0.23k, BiT-L 9.9k, Noisy Student 12.3k.
(H/14 = 88.55 in Table 2 uses 518 res + Polyak averaging; 88.04 in Table 5 is 384 res without those.)
Scaling study (Sec. 4.4): ViT uses approx. 2-4x less compute than ResNets for the same performance.

## Outline assertions I could not confirm or that are wrong
1. "CLIP ViT-L/14 used 224 and 336": not in the ViT paper (CLIP paper's claim; another extractor). ViT has L/16 and H/14, no L/14.
2. "H/14" and "patch sizes B/16, L/16, H/14": confirmed; also B/32 and L/32 trained. The variant table does not
   depend on patch size.
3. "ViT underperforms ResNets on ImageNet-only pretraining": confirmed but qualified: only "a few percentage points below
   ResNets of comparable size" and before strong regularization; the paper regularises (weight decay 0.3, dropout 0.1,
   label smoothing, 300 epochs) to narrow it. BiT still wins on ImageNet-only pretraining (Fig. 3).
4. "ViT Eq. 1 (patch embedding + [class] + position embeddings)": confirmed; Eq. 1 is exactly that.
5. Fig. 3/4 exact values are not in the LaTeX (graphs); only Table 5 gives Fig. 3 numbers; Fig. 4 has none.
