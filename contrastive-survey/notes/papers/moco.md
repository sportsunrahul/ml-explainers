# MoCo (spine)

## 1. Identity
- Title: Momentum Contrast for Unsupervised Visual Representation Learning
- arXiv 1911.05722, read v3 (LaTeX `src/moco.tex` + PDF). v1 submitted 2019-11-13; v3 revised 2020-03-23 (abs.html).
- Authors: Kaiming He ... Ross Girshick (5 authors: He, Fan, Wu, Xie, Girshick). Venue: CVPR style file (`cvpr.sty`); venue not stated in abs.html.
- Repo: not in sources (no README).

## 2. Core problem
Contrastive learning works better with a large dictionary of negative keys. The keys must also be encoded
by a consistent (slowly changing) encoder. Prior mechanisms gave one or the other, not both.

## 3. Prior methods and why that hurt (per the paper, Sec. 3.2 "Relations to previous mechanisms")
- End-to-end: keys are the current mini-batch. Keys are consistent, but dictionary size = batch size, limited
  by GPU memory (largest affordable batch on 8 Volta 32GB GPUs is 1024) and large-batch optimization is hard.
- Memory bank (Wu et al. 2018): stores one vector per dataset image. Large dictionary, but each stored key
  was computed at a different step over the past epoch, so keys are inconsistent. Memory-bank momentum is on
  the stored representations, not the encoder.

## 4. Key idea
Keep the dictionary as a FIFO queue of keys from recent mini-batches (size decoupled from batch size) and
encode keys with a momentum (moving-average) copy of the query encoder so keys stay consistent.

## 5. Mechanism (Alg. 1)
1. Load minibatch x of N images. Make two random augmentations x_q, x_k.
2. q = f_q(x_q): N x C. k = f_k(x_k): N x C, detached (no gradient).
3. l_pos = bmm(q, k): N x 1. l_neg = mm(q, queue): N x K (queue is C x K).
4. logits = cat([l_pos, l_neg]): N x (1+K). Labels are all 0 (positive is column 0). Cross-entropy on logits/t.
5. Backprop updates f_q only.
6. Momentum update f_k <- m f_k + (1-m) f_q (Eq. 2).
7. Enqueue the current k (N keys); dequeue the oldest minibatch.
- Positions graded: per query row, column 0 = its own key (positive); columns 1..K = queue keys from earlier
  batches (negatives). Other images in the current batch are NOT negatives in v1 (only the queue is).
- Gradient: only f_q. f_k and queue are no-gradient (brown).
- Shuffling BN (Sec. 3.3): multi-GPU BN per device. For f_k, shuffle the sample order before splitting across
  GPUs, shuffle back after. f_q order unchanged. Reason: intra-batch BN statistics leak information, so the model
  "cheats" the pretext task. Used for MoCo and the end-to-end ablation; not needed for the memory bank.
  Appendix A.9 / Fig. A.1: without it, pretext train accuracy goes > 99.9% and kNN val accuracy drops (curves only).

## 6. Equations
Eq. 1 (`\label{eq:infonce}`), Sec. 3.1:
```
\mathcal{L}_q = -\log \frac{\exp(q{\cdot}k_+ / \tau)}{\sum_{i=0}^{K}\exp(q{\cdot}k_i  / \tau)}
```
Paper: "the log loss of a (K+1)-way softmax-based classifier that tries to classify q as k_+". "The sum is over
one positive and K negative samples."
- q: encoded query, 128-D, L2-normalized. k_+: the key from the same image (other augmentation).
- k_i, i = 0..K: the positive plus K negative keys (queue). tau: temperature, 0.07.

Eq. 2 (`\label{eq:moco}`), Sec. 3.2:
```
\theta_\textrm{k} \leftarrow m \theta_\textrm{k} + (1 - m) \theta_\textrm{q}.
```
- theta_k: key encoder weights. theta_q: query encoder weights (trained by backprop). m in [0, 1), default 0.999.
- Paper: "Only the parameters theta_q are updated by back-propagation."

## 7. Config
| item | value | loc |
|---|---|---|
| encoder | ResNet-50 (also RX50, R50w2x, R50w4x) | Sec. 3.3, Table 1 |
| output | fc after global avg pool, 128-D, L2-normalized | Sec. 3.3 Technical details |
| tau | 0.07 | Sec. 3.3 |
| K | 65536 (main results); 4096 in momentum ablation | Sec. 4.1 |
| m | 0.999 default | Sec. 3.2, 4.1 |
| aug | 224x224 crop from randomly resized image, color jitter, horizontal flip, grayscale | Sec. 3.3 |
| optimizer | SGD, wd 0.0001, momentum 0.9 | Sec. 4 Training |
| IN-1M | batch 256 on 8 GPUs, lr 0.03, 200 epochs, lr x0.1 at 120 and 160, ~53 hours R50 | Sec. 4 Training |
| IG-1B | batch 1024 on 64 GPUs, lr 0.12, x0.9 every 62.5k iters, 1.25M iters (~1.4 epochs), ~6 days | Sec. 4 Training |
| GPU type | not stated for main run. MoCo v2 Table 3 gives 8 V100 16G, 53 hrs for this config | MoCo v2 Table 3 |
| linear probe | frozen features, fc+softmax on global avg pool, 100 epochs, lr 30, wd 0, 1-crop top-1 val | Sec. 4.1 |

## 8. Results
- Momentum ablation (inline table, Sec. 4.1, R50, K=4096): m = 0 fail; 0.9 -> 55.2; 0.99 -> 57.8; 0.999 -> 59.0;
  0.9999 -> 58.9. m=0: "training loss oscillates and fails to converge".
- Fig. 3 (three mechanisms vs K, R50, linear protocol): values are data labels embedded in the figure PDF
  (`src/fig/results_mechanisms.pdf`), extracted with pdftotext and confirmed by rendering; NOT in the LaTeX body.
  Negatives = K for memory bank and MoCo, K-1 for end-to-end. End-to-end stops at K=1024.
  | K | end-to-end | memory bank | MoCo |
  |---|---|---|---|
  | 256 | 54.9 | 50.0 | 54.7 |
  | 512 | 56.3 | 52.0 | 56.4 |
  | 1024 | 57.3 | 54.1 | 57.5 |
  | 4096 | - | 56.5 | 59.0 |
  | 16384 | - | 57.8 | 60.4 |
  | 65536 | - | 58.0 | 60.6 |
  Text independently confirms: memory bank 58.0 (K=65536, InfoNCE), "2.6% worse than MoCo", MoCo 60.6; 59.0 at
  K=4096 matches the momentum table. Footnote: memory bank with NCE and K=4096 reproduces 54.3 (orig. 54.0).
- Table 1 linear probe (IN-1M pretrain, 200 epochs, batch 256, K=65536, m=0.999, single crop):
  R50 (24M) 60.6; RX50 (46M) 63.9; R50w2x (94M) 65.4; R50w4x (375M) 68.6. InstDisc R50 54.0.
- Sec. 4.1 text: MoCo v2 R50 71.1 (up from 60.6).
- Table 3 (VOC, mechanisms, R50-C4 AP): end-to-end 54.6, memory bank 54.9, MoCo 55.9.
- Table 2 (VOC, R50-C4, AP): supervised IN-1M 53.5; MoCo IN-1M 55.9; MoCo IG-1B 57.2.

## 9. Limitations stated
- Large-batch end-to-end trend "questionable" to extrapolate; MoCo does not explore pretext task or augmentation
  (Sec. 4.1 last paragraph). Fixed by MoCo v2 (MLP head + blur + cosine, 71.1).
- Linear-probe lr 30 implies feature distribution differs from supervised (Sec. 4.1).

## 10. Outline feedback
- viz-queue momentum chips: the ablation has 0, 0.9, 0.99, 0.999, 0.9999 and was run at K=4096 (not 65536).
  Add 0.9999 (58.9) or say it is omitted; label the chart "K = 4096". m=0 is "fail", not a number.
- viz-mechs: Fig. 3 values ARE printed (as figure data labels); use them. Note end-to-end stops at 1024 and has
  K-1 negatives.
- In MoCo v1 the negatives are only the queue, not other current-batch keys (Alg. 1). Worth one sentence.
- Default tau is 0.07 in v1; v2 changes it to 0.2.
