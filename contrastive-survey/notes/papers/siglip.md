# SigLIP: Sigmoid Loss for Language Image Pre-Training

## 1. Identity
- arXiv 2303.15343, version read: v4 (abs.html: v1 Mon 27 Mar 2023; v4 Wed 27 Sep 2023).
- Authors: Xiaohua Zhai ... Lucas Beyer (4 authors; Google DeepMind, Zurich).
- Venue: ICCV 2023 (abs.html comments: "ICCV'23 Oral"; not printed in the PDF).
- Repo: https://github.com/google-research/big_vision
- Sources: PDF only (eprint is a PDF, no LaTeX). Equations transcribed from PDF p.3.

## 2. Core problem
CLIP's softmax loss normalizes every row and column over the whole batch. That needs all-gathers across devices,
a |B| x |B| matrix in memory, and an extra pass for numerical stabilization; it also ties the task definition to
the batch size.

## 3. Prior methods and why that hurt (per the paper)
Softmax contrastive loss "applied twice to normalize the pairwise similarity scores across all images, then all
texts"; naive softmax "is numerically unstable", stabilized by subtracting the max, "which requires another
pass over the full batch" (Sec. 1). Distributed: "expensive all-gathers and ... a memory-intensive |B| x |B|
matrix" (Sec. 3.3).

## 4. Key idea
Score every image-text cell independently with a sigmoid: binary "does this pair match?" (label +1 on the
diagonal, -1 elsewhere), plus a learnable bias b that starts strongly negative because almost all pairs are
negatives. No batch-wide normalization.

## 5. Mechanism
1. Image tower f (ViT) and text tower g (transformer); x_i = f(I_i)/||f(I_i)||_2, y_i = g(T_i)/||g(T_i)||_2.
2. logits = t * x.y^T + b over all |B| x |B| cells; t = exp(t'), t' and b learnable; init t' = log 10 (t = 10), b = -10.
3. Each cell is a binary logistic loss; sum over all cells, divide by |B|.
4. Chunked (Sec. 3.3, Fig. 1): per-device batch b = |B|/D. Each device computes its local b x b block (positives
   + local negatives), then texts are permuted to the neighbouring device and the next b x b block is added;
   repeat D times; final cross-device sum. No all-gathers; memory |B|^2 -> b^2; "D collective permutes is
   typically faster than two all-gathers between D devices". Enabled batch sizes over 1 million on few devices.
5. Stabilization: beta2 = 0.95 in Adam/AdaFactor for large batches (Sec. 4.7).

## 6. Equations (equations in v4 are UNNUMBERED; loc = Sec. 3.1 and Sec. 3.2, PDF p.3)
Softmax loss (Sec. 3.1), transcribed from PDF:
```
-\frac{1}{2|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\left(
\overbrace{\log\frac{e^{t\mathbf{x}_i\cdot\mathbf{y}_i}}{\sum_{j=1}^{|\mathcal{B}|}e^{t\mathbf{x}_i\cdot\mathbf{y}_j}}}^{\text{image}\rightarrow\text{text softmax}}
+\overbrace{\log\frac{e^{t\mathbf{x}_i\cdot\mathbf{y}_i}}{\sum_{j=1}^{|\mathcal{B}|}e^{t\mathbf{x}_j\cdot\mathbf{y}_i}}}^{\text{text}\rightarrow\text{image softmax}}
\right)
```
"The scalar t is parametrized as exp(t'), where t' is a global freely learnable parameter."

Sigmoid loss (Sec. 3.2), transcribed from PDF:
```
-\frac{1}{|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\sum_{j=1}^{|\mathcal{B}|}
\underbrace{\log\frac{1}{1+e^{z_{ij}(-t\mathbf{x}_i\cdot\mathbf{y}_j+b)}}}_{\mathcal{L}_{ij}}
```
"where z_ij is the label for a given image and text input, which equals 1 if they are paired and -1 otherwise."
Paper: "turning the learning problem into the standard binary classification on the dataset of all pair
combinations". Bias: "At initialization, the heavy imbalance coming from the many negatives dominates the loss
... we introduce an additional learnable bias term b ... We initialize t' and b to log 10 and -10 respectively.
This makes sure the training starts roughly close to the prior".

Algorithm 1 (pseudocode, p.2):
```
t = exp(t_prime)
zimg = l2_normalize(img_emb)
ztxt = l2_normalize(txt_emb)
logits = dot(zimg, ztxt.T) * t + b
labels = 2 * eye(n) - ones(n) # -1 with diagonal 1
l = -sum(log_sigmoid(labels * logits)) / n
```
Sign discrepancy (our reading): Algorithm 1 gives L_ij = log(1/(1 + e^{z_ij(-t x.y - b)})), i.e. the logit is
t x.y + b. The printed equation has "+b" inside the exponent, i.e. logit t x.y - b. With b init -10, only the
Algorithm 1 form starts every pair near "negative" (sigmoid(-10) ~ 0), which is what the paper says the bias is
for. The toy should use logit = t x.y + b.

Chunked loss (Sec. 3.3): -1/|B| sum_{d_i=1}^{D} [A: for all devices] sum_{d_j=1}^{D} [B: swap negs across
devices] sum_{i=b d_i}^{b(d_i+1)} sum_{j=b d_j}^{b(d_j+1)} L_ij [C: per device loss].

## 7. Config
| item | value | loc |
|---|---|---|
| SigLIP towers | B/16 ViT + B-size text, 224x224, 32k sentencepiece vocab (C4), max 16 text tokens | Sec. 4.2 |
| SigLIP data | WebLI, English pairs | Sec. 4.2 |
| Optimizer | ScalingViT-Adafactor default; beta2 0.95; lr 0.001, wd 0.0001 across batch 512 to 1024k | Sec. 4, 4.7, App. C |
| SigLiT | frozen precomputed ViT-g embeddings (LiT data), text tower trained; 288x288 resize only | Sec. 4.1, App. A |
| SigLiT 4-chip | frozen ViT-AugReg-B/8, 12-layer L text, LION, lr 1e-4, wd 1e-7, 6.5k warmup, 65 000 steps, batch 32k, < 1 day | Sec. 4.4 |
| Init | t' = log 10 (t = 10), b = -10 | Sec. 3.2, 4.9 |
| Scaled models | B, L, So-400m, 40B examples at batch 32k, 256 patches, 64 text tokens | Sec. 4.6 |
| Output dim d | 768 (B), 1024 (L), 1152 (So-400m) | App. F |

## 8. Results
- Table 1 (headline, ImageNet 0-shot): SigLiT B/8 + L*, 32k, 4 TPUv4, 1 day: 79.8 in the table row, but caption,
  intro and Sec. 4.4 say 79.7%. SigLiT g/14 + L, 20k, 4 TPUv4, 2 days: 84.5. SigLIP B/16 (pretrained, unlocked),
  16k, 16 TPUv4, 3 days: 71.0. From scratch B/16, 32k, 32 TPUv4: 2 days 72.1, 5 days 73.4.
- Comparison cost: CLIP "approx. 2500 TPUv3-days for 72.6%" (citing FLIP, Sec. 4.5); FLIP and CLIP "approximately
  5 and 10 days respectively on 256 TPUv3 cores" (Sec. 1).
- Batch size: sigmoid "significantly better than the softmax loss when the batch size is smaller than 16 k"
  (Sec. 1, 4.1). SigLiT saturates at 32k; 1M batch trained; best 84.7 (Table 8 1024k row, 18B examples; peak
  at 256k per Sec. 4.1). SigLIP best at 32k, softmax needs 98k and "still didn't outperform"; 307k hurts both (Sec. 4.2).
- Small-batch SigLIP vs CLIP-loss (App. B, Table 5, B/16, ImageNet 0-shot, 3B examples): 512: 51.5 vs 47.7;
  1k: 57.3 vs 53.2; 2k: 62.1 vs 59.3; 4k: 65.3 vs 63.8; 8k: 68.6 vs 66.6; 32k: 69.9 vs 69.9; 98k: 69.5 vs 69.7.
  9B examples: 32k 73.4 vs 72.9; 98k 73.0 vs 73.2; 307k 71.6 vs 72.6.
- SigLiT Table 8 (sigmoid vs softmax), 450M examples, batch 512: 72.5 vs 69.5.
- Memory: 4 TPU-v4 fit batch 4096 for Base SigLIP vs 2048 for CLIP (Sec. 4.2).
- Bias ablation Table 4 (B, 8k, 900M): b n/a, t' log10 -> 62.0; b -10, log10 -> 63.0; b 0, log1 -> 53.7.
- Negatives: at 16k, "268 M negative examples for only 16 k positive ones" (Sec. 4.8); keeping only hardest
  negatives "does almost maintain the quality"; easiest "does not work at all" (Fig. 6).
- Table 3: SigLIP So-400m, 729 patches: ImageNet 0-shot 83.2; SigLIP B/16 196 patches 76.2 vs OpenCLIP B 70.2.

## 9. Limitations stated
Batch-size gains saturate (~32k); very large batch hurts; large batches unstable without beta2 = 0.95; efficient
negative mining is "promising but is not trivial". SigLIP 2 (2502.14786) extends the recipe.

## 10. Outline feedback
- SigLIP equations are unnumbered in v4: cite "Sec. 3.1 (softmax) and Sec. 3.2 (sigmoid)", not Eq. 1/Eq. 2.
- Printed sigmoid equation and Algorithm 1 differ in the sign of b; build the toy from Algorithm 1.
- 79.7 vs 79.8 inconsistency for the SigLiT B/8 headline: prefer 84.5 (consistent everywhere) as the headline.
- The "small batch" result should use App. B Table 5 numbers, not Fig. 2 (no values printed in Fig. 2).
