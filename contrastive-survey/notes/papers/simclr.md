# SimCLR (spine)

## 1. Identity
- Title: A Simple Framework for Contrastive Learning of Visual Representations
- arXiv 2002.05709, read v3 (LaTeX `src/main.tex` + PDF). v1 submitted 2020-02-13; v3 revised 2020-07-01 (abs.html).
- Authors: Ting Chen ... Geoffrey Hinton (Chen, Kornblith, Norouzi, Hinton). Style file `icml2020.sty`.
- Repo: not in sources (no README); code-only values unavailable.

## 2. Core problem
Contrastive methods needed special architectures or a memory bank. Which simple components matter?

## 3. Prior methods (Sec. 3, App. C)
- CPC/AMDIM/DIM built the prediction task into the architecture (patchify, constrained receptive fields, context
  networks). SimCLR replaces this with random crop.
- Memory bank / MoCo for negatives; SimCLR uses in-batch negatives with a large batch instead.
- Local BN statistics leak information; SimCLR uses global BN (MoCo uses shuffling BN).

## 4. Key idea
Two augmented views per image, a standard ResNet, a small MLP head used only for the loss, and NT-Xent over all
2N views in a large batch. Findings: crop + color composition is crucial; nonlinear head helps the layer before it;
larger batch and longer training help contrastive learning more than supervised.

## 5. Mechanism (Alg. 1)
1. Sample minibatch {x_k}, k = 1..N (default N = 4096).
2. For each k draw t, t' ~ T; x~_{2k-1} = t(x_k), x~_{2k} = t'(x_k). Total 2N views (8192 at N = 4096).
3. h = f(x~) = ResNet output after average pooling (2048-d for ResNet-50). z = g(h) = W2 relu(W1 h), 128-d.
4. s_{i,j} = z_i^T z_j / (|z_i| |z_j|) for all i, j in 1..2N: a 2N x 2N cosine matrix.
5. l(i, j) per Eq. 1. Loss L = (1/2N) sum_k [l(2k-1, 2k) + l(2k, 2k-1)].
6. Update f and g. After training, discard g; use h downstream.
- Positions graded: row i of the 2N x 2N matrix. Diagonal k = i excluded by the indicator 1[k != i]. Positive:
  the partner view (2k-1 <-> 2k). Negatives: the other 2(N-1) views. Paper: batch 8192 gives 16382 negatives
  per positive pair (Sec. 2.2). For N = 4096: 8190 negatives (our reading, same formula).
- Gradient: f and g both trainable; all 2N views get gradient (no stop-gradient, no momentum encoder).
- Global BN: BN mean/variance aggregated over all devices (Sec. 2.2).

## 6. Equations
Eq. 1 (`\label{eq:loss}`), Sec. 2.1:
```
\ell_{i,j} = -\log \frac{\exp(\mathrm{sim}(\bm z_i, \bm z_j)/\tau)}{\sum_{k=1}^{2N} \one{k \neq i}\exp(\mathrm{sim}(\bm z_i, \bm z_k)/\tau)}~,
```
Macro: `\newcommand{\one}[1]{\mathbbm{1}_{[#1]}}` (main.tex line 48).
sim(u, v) = u^T v / (|u| |v|) (cosine). Paper: "The final loss is computed across all positive pairs, both (i, j) and
(j, i), in a mini-batch." Named NT-Xent (normalized temperature-scaled cross entropy).
- z_i, z_j: projections of the two views of one image. z_k: any of the 2N projections. 1[k != i]: drops self-similarity.
- tau: temperature.
Alg. 1 total loss:
```
\mathcal{L} = \frac{1}{2N} \sum_{k=1}^N \left[ \ell(2k\!-\!1, 2k) + \ell(2k, 2k\!-\!1)\right]
```
Table 2 also gives the NT-Xent gradient w.r.t. u (negatives weighted by exp(u^T v^- / tau)/Z(u)).

## 7. Config
| item | value | loc |
|---|---|---|
| encoder | ResNet-50 (1x, 2x, 4x width for SOTA) | Sec. 2.3, Sec. 6 |
| h dim | 2048 (R50) | Sec. 4.2, Fig. 8 caption, Table 3 caption |
| head g | MLP, one hidden layer, ReLU ("2-layer MLP"); hidden width not stated for default | Sec. 2.1, 2.3 |
| z dim | 128 | Sec. 2.3 Default setting |
| aug | random crop + resize to 224x224 (area 0.08-1.0, aspect 3/4-4/3) + flip p 0.5; color distortion (jitter p 0.8 with 0.8s/0.8s/0.8s/0.2s, grayscale p 0.2), s = 1; Gaussian blur p 0.5, sigma 0.1-2.0, kernel 10% of side | App. A |
| batch | 4096 default (sweep 256-8192; 32K in B.1) | Sec. 2.3, App. B.1 |
| optimizer | LARS, lr 4.8 = 0.3 x BatchSize/256, wd 1e-6, 10-epoch linear warmup, cosine decay no restarts | Sec. 2.3 |
| sqrt lr option | 0.075 x sqrt(BatchSize); equals 4.8 at batch 4096 | App. B.1 |
| epochs | 100 for ablations; 1000 for SOTA | Sec. 2.3, Sec. 6 |
| hardware | Cloud TPU, 32 to 128 cores; 128 TPU v3 cores ~1.5 h for R50, batch 4096, 100 epochs | Sec. 2.2 + footnote |
| tau | NOT stated for default/SOTA runs. Table 5 sweep (100 ep, batch 4096): 0.05/0.1/0.5/1 -> 59.7/64.4/60.7/58.0; CIFAR-10 optimum 0.5 (App. B.9) | Table 5, App. B.9 |
| linear eval | Nesterov momentum, batch 4096, lr 1.6 = 0.1 x B/256, 90 epochs (LARS alternative similar) | App. B.6 |
| semi-sup fine-tune | Nesterov, batch 4096, lr 0.8, no wd; 60 ep (1%), 30 ep (10%); center crop 224 | App. B.5 |

## 8. Results
- Table 1 (color distortion strength, R50, 100 ep, linear top-1 for SimCLR; supervised R50 90 ep):
  strength 1/8, 1/4, 1/2, 1, 1(+Blur), AutoAug -> SimCLR 59.6, 61.0, 62.6, 63.2, 64.5, 61.1;
  Supervised 77.0, 76.7, 76.5, 75.7, 75.4, 77.1.
- Fig. 5 (pairwise augmentations): values are data labels embedded in `src/figures/heatmap_da_pairwise.pdf`
  (pdftotext + render), not in LaTeX body. Asymmetric protocol: crop+resize always applied first, targeted
  transform(s) on ONE branch only, other branch identity; the paper says this "hurts the performance". Row = 1st
  transformation, column = 2nd. Order Crop, Cutout, Color, Sobel, Noise, Blur, Rotate, Average:
  - Crop: 33.1 33.9 56.3 46.0 39.9 35.0 30.2 | 39.2
  - Cutout: 32.2 25.6 33.9 40.0 26.5 25.2 22.4 | 29.4
  - Color: 55.8 35.5 18.8 21.0 11.4 16.5 20.8 | 25.7
  - Sobel: 46.2 40.6 20.9 4.0 9.3 6.2 4.2 | 18.8
  - Noise: 38.8 25.8 7.5 7.6 9.8 9.8 9.6 | 15.5
  - Blur: 35.1 25.2 16.6 5.8 9.7 2.6 6.7 | 14.5
  - Rotate: 30.0 22.5 20.7 4.3 9.7 6.5 2.6 | 13.8
- Fig. 6: color histograms of crops suffice to tell images apart (the shortcut); no numbers.
- Head ablation (Fig. 8, h = 2048, output dims 32-2048): bars only, no printed values. Text: nonlinear better than
  linear "(+3%)", much better than none "(>10%)"; h better than z = g(h) "(>10%)" (Sec. 4.2).
- Table 3 (predict transform from h vs g(h), both 2048-d): color vs gray 99.3 vs 97.4 (guess 80);
  rotation 67.6 vs 25.6 (25); orig vs corrupted 99.5 vs 59.6 (50); orig vs Sobel 96.6 vs 56.3 (50).
- Table 4 (loss, linear top-1): Margin 50.9, NT-Logi 51.6, Margin(sh) 57.5, NT-Logi(sh) 57.9, NT-Xent 63.9.
- Table 5 (l2 norm, tau; contrastive distribution over 4096): yes/0.05 59.7, yes/0.1 64.4, yes/0.5 60.7, yes/1 58.0,
  no/10 57.2, no/100 57.0.
- Batch x epochs (Fig. 9 plotted; exact values in Table B.1, linear / sqrt lr scaling, R50 top-1):
  | batch | 100 | 200 | 400 | 800 |
  |---|---|---|---|---|
  | 256 | 57.5 / 62.8 | 61.9 / 64.3 | 64.7 / 65.7 | 66.6 / 66.5 |
  | 512 | 60.7 / 63.8 | 64.0 / 65.6 | 66.2 / 66.7 | 67.8 / 67.4 |
  | 1024 | 62.8 / 64.3 | 65.3 / 66.1 | 67.2 / 67.2 | 68.5 / 68.3 |
  | 2048 | 64.0 / 64.7 | 66.1 / 66.8 | 68.1 / 67.9 | 68.9 / 68.8 |
  | 4096 | 64.6 / 64.5 | 66.5 / 66.8 | 68.2 / 68.0 | 68.9 / 69.1 |
  | 8192 | 64.8 / 64.8 | 66.6 / 67.0 | 67.8 / 68.3 | 69.0 / 69.1 |
  No exact values for 300/500/600/700/900/1000 epochs (plot only). MoCo v2 Table 2 quotes 61.9 (b256) and
  66.6 (b8192) at 200 ep "from Fig. 9", consistent with the linear-scaling column.
- Table 6 (linear, 1000 ep, batch 4096): R50 69.3 top-1 / 89.0 top-5; R50(2x) 74.2 / 92.0; R50(4x) 76.5 / 93.2.
  Broader augmentations (App. B.2): 70.0, 74.4, 76.8.
- Semi-supervised fine-tune top-1 (Table B.4) 1% / 10%: R50 48.3 / 65.6; R50(2x) 58.5 / 71.7; R50(4x) 63.0 / 74.4.
  Top-5 (Table 7): R50 75.5 / 87.8; 2x 83.0 / 91.2; 4x 85.8 / 92.6. Supervised R50 baseline top-1 25.4 / 56.4.
- Table 8 (R50 4x, transfer): linear beats/ties supervised on several of 12 datasets; fine-tuned "on par or better"
  on 10 of 12 (Sec. 1).

## 9. Limitations stated
Max performance not reached at 100 epochs (ablations, footnote). Performance seems to saturate with batch 8192,
longer training still helps (App. B.1). SimCLRv2 later adds bigger nets, 3-layer head, MoCo memory, distillation.

## 10. Outline feedback
- Default tau is not stated in the paper. Do not print "tau = 0.1" as SimCLR's config; say "Table 5 sweep: 0.1
  best of {0.05, 0.1, 0.5, 1}".
- Head hidden width is not stated for the default head. Do not label it 2048 in "Inside SimCLR"; only h = 2048,
  z = 128 are sourced.
- viz-augment: Fig. 5 values ARE printed in the figure; use them, but caption the asymmetric one-branch protocol.
  Crop+Color 56.3 (row Crop, col Color) and Color+Crop 55.8; Crop alone 33.1.
- Head ablation: Fig. 8 has no printed numbers; use the text deltas (+3%, >10%) and Table 3.
- Batch x epochs chart: use Table B.1 (100/200/400/800 epochs only).
- ch-step: batch 4096 -> 8192 views -> 8192 x 8192 matrix; diagonal masked; one positive per row; 8190 negatives
  per row (our reading). Linear-scaled lr 4.8; LARS; wd 1e-6.
