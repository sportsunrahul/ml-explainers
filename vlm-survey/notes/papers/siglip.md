# SigLIP (2303.15343v4) -- extraction for ch-sigmoid

Source: PDF only. Equations transcribed from pdftotext and checked against rendered page 3.
IMPORTANT: none of the loss equations are numbered in the paper. Cite as "unnumbered, Sec. 3.1 / Sec. 3.2 / Sec. 3.3"
and "Algorithm 1" (pseudocode, p. 2). Do not print "Eq. 1 / Eq. 2".

## Notation
B = mini-batch {(I_1,T_1),(I_2,T_2),...}; |B| = batch size. f = image model, g = text model.
x_i = f(I_i)/||f(I_i)||_2, y_i = g(T_i)/||g(T_i)||_2 (L2-normalised embeddings).
t = exp(t'), t' a global freely learnable scalar. b = learnable bias (Sec. 3.2). NOTE: Sec. 3.3 reuses b for the
PER-DEVICE batch size (b = |B|/D); the bias is not in that formula's text but is inside L_ij.

## Softmax loss (unnumbered, Sec. 3.1, p. 3)
```
-\frac{1}{2|B|} \sum_{i=1}^{|B|} \Big( \log \frac{e^{t x_i \cdot y_i}}{\sum_{j=1}^{|B|} e^{t x_i \cdot y_j}}
   + \log \frac{e^{t x_i \cdot y_i}}{\sum_{j=1}^{|B|} e^{t x_j \cdot y_i}} \Big)
```
First term (overbrace "image->text softmax"): row i normalised over all texts j. Second ("text->image softmax"):
column i normalised over all images j. The paper says the normalisation is done twice, "across images and across texts",
because the softmax loss is asymmetric. Every row/column needs a sum over the FULL batch = global normalisation.
Intro: naive softmax is numerically unstable; usually stabilised by subtracting the max, "which requires another pass over the full batch".

## Sigmoid loss (unnumbered, Sec. 3.2, p. 3)
```
-\frac{1}{|B|} \sum_{i=1}^{|B|} \sum_{j=1}^{|B|} \underbrace{\log \frac{1}{1 + e^{z_{ij}(-t x_i \cdot y_j + b)}}}_{\mathcal{L}_{ij}}
```
z_ij = label: 1 if (I_i,T_j) is a matching pair (i=j), -1 otherwise. Each of the |B|^2 pairs is an independent
binary classification (logistic) term; sigma(z_ij (t x_i.y_j + b)) is what the log-sigmoid form equals:
log 1/(1+e^{-u}) = log sigmoid(u), u = z_ij (t x_i.y_j + b). Sum over all i,j, divide by |B| (NOT |B|^2, NOT 2|B|).
"processes every image-text pair independently, effectively turning the learning problem into the standard binary
classification on the dataset of all pair combinations".

## Algorithm 1 pseudocode (p. 2; the v4 fix of an earlier bug)
```
t = exp(t_prime)
zimg = l2_normalize(img_emb); ztxt = l2_normalize(txt_emb)
logits = dot(zimg, ztxt.T) * t + b
labels = 2 * eye(n) - ones(n)      # -1 with diagonal 1
l = -sum(log_sigmoid(labels * logits)) / n
```
Comment in paper: img_emb [n,dim], txt_emb [n,dim]; t_prime, b learnable; n mini-batch size.

## Initialisation (Sec. 3.2, p. 3; Sec. 4.9 Table 4)
"We initialize t' and b to log 10 and -10 respectively" => t = 10, b = -10. Reason: at init, the |B|^2-|B| negatives
dominate the loss ("heavy imbalance ... leading to large initial optimization steps attempting to correct this bias");
b=-10 starts training near the prior and avoids "massive over-correction". Sec. 4.9 default stated as b=-10 and t0=log 10 (t=10).
Table 4 (Base, 8k batch, 900M examples; INet-0 / Pet-0 / C100-0):
- b n/a, t'=log10: 62.0 / 81.8 / 59.9
- b -10, t'=log10: 63.0 / 82.4 / 61.0  (default)
- b -10, t'=log1: 61.0 / 80.0 / 60.4
- b 0, t'=log10: 61.7 / 79.9 / 59.0
- b 0, t'=log1: 53.7 / 73.2 / 53.8  (small temperature + zero bias hurts most)
Softmax reference: t = exp(t') with t' a global learnable parameter, no bias.

## Efficient "chunked" implementation (Sec. 3.3, p. 3-4; Fig. 1)
Problem: data-parallel training across D devices needs all embeddings gathered ("expensive all-gathers") and, "more
importantly, the materialization of a memory-intensive |B| x |B| matrix of pairwise similarities".
Formula (unnumbered, p. 3), per-device batch b = |B|/D:
```
-\frac{1}{|B|} \sum_{d_i=1}^{D} \sum_{d_j=1}^{D} \sum_{i=b d_i}^{b(d_i+1)} \sum_{j=b d_j}^{b(d_j+1)} \mathcal{L}_{ij}
```
Labels under the braces: A = "for all device d_i" (outer sum); B = "swap negs across devices" (sum over d_j);
C = "per device loss" (inner double sum over i = "all local positives", j = "negs from next device").
Algorithm in words: each device (1) computes the loss block for its own b image x b text embeddings, which contains the
positives (b positives + b-1 negatives per row block); (2) permutes text representations to the neighbouring device
(collective permute) and computes the loss block against those as negatives; (3) repeats until every image-text pair
has interacted (D steps); (4) a final cross-device sum (sum A) totals the losses. Works because each pair is an
independent term. Fig. 1 mock setup: 3 devices, global batch 12, 4 per device; "There are no all-gathers, and at any
point in time only the bright yellow square (size 4 x 4) is materialized in memory."
Memory claim: "the memory cost at any given moment is reduced from |B|^2 to b^2 (for sum C)". "Individual collective
permutes ... are fast (and indeed D collective permutes is typically faster than two all-gathers between D devices)".
"Usually b is constant as scaling |B| is achieved by increasing the number of accelerators." "This chunked approach
enabled training with batch sizes over 1 million on relatively few devices."
Related concrete memory datum (Sec. 4.2): four TPU-v4 chips fit batch 4096 with Base SigLIP but only 2048 with the
corresponding CLIP (softmax) model. For viz: full matrix at |B|=1M is 1e12 entries; per-device block b^2 with
b = 1M/D (e.g. D=... choose in viz; the paper does not state the SigLiT 1M device count).

## Batch-size results
Small batch (softmax loses): Sec. 1 "sigmoid loss performs significantly better than the softmax loss when the batch
size is smaller than 16 k. As the train batch size grows, the gap closes." Sec. 4.1: below 16k sigmoid beats softmax "by
a large margin"; softmax "quickly catches up and potentially slightly underperforms" at large batch.
SigLiT (Table 8, App. A; ImageNet 0-shot, sigmoid vs softmax), 18B examples: 2k 82.2 vs 81.2; 4k 83.0 vs 82.0;
8k 83.6 vs 83.1; 16k 84.2 vs 84.1; 32k 84.6 vs 84.4; 64k 84.7 vs 84.4; 128k 84.7 vs 84.6; 256k 84.7 vs 84.6;
1024k 84.7 vs "-". At 3B: 512 77.2 vs 74.6; 1k 79.6 vs 77.9; 2k 81.3 vs 80.1; 32k 84.2 vs 84.0. At 450M: 512 72.5 vs 69.5.
Fig. 2 left is this 18B curve (x-axis 2k..1024k).
SigLIP (Table 5, App. B; Base, ImageNet 0-shot sigmoid vs softmax): 3B examples: 512 51.5 vs 47.7; 1k 57.3 vs 53.2;
2k 62.1 vs 59.3; 4k 65.3 vs 63.8; 8k 68.6 vs 66.6; 32k 69.9 vs 69.9; 98k 69.5 vs 69.7.
9B examples: 4k 68.4 vs 66.6; 8k 70.6 vs 69.4; 16k 72.3 vs 71.7; 32k 73.4 vs 72.9; 98k 73.0 vs 73.2; 307k 71.6 vs 72.6.
Text: "SigLIP performs best at batch size 32 k, whereas the softmax loss required 98 k for optimal performance and still
didn't outperform the sigmoid based variant." "a larger batch size like 307 k hurts both losses." (Note: at 307k the
table has softmax 72.6 > sigmoid 71.6; the sentence "hurts both" is right but sigmoid drops more.)
32k sufficiency: Sec. 1 "a reasonable batch size, i.e. 32 k, is sufficient for image-text pre-training. This
conclusion also holds for multilingual SigLIP training on over 100 languages." Conclusion: "a relatively modest batch
size of 32 k yielded nearly optimal performance". mSigLIP (Table 2, 30B examples): INet-0 71.6 @16k, 73.2 @32k,
73.2 @64k, 73.2 @128k, 73.1 @240k; XM avg 34.8, 34.9, 34.4, 33.6, 32.7. Beyond 32k XM3600 retrieval gets worse.
1M experiment: Sec. 4.1 "We successfully trained an SigLiT model at one million batch size ... the performance
saturates at 32 k batch size, further scaling up the batch size only gives a minor boost, and the model peaks at 256 k
batch size." Best SigLiT with a B-sized text model: 84.7% ImageNet 0-shot vs original LiT 85.2% "with a 10 times larger
g-sized text model". Fig. 3: 262k batch beats 8k only if trained long enough (short schedules give few update steps).
Stated: "most of the existing studies stop at 64 k batch size". Batch range studied: 512 to 1M.

## Table 1 (p. 1; rendered page checked)
Caption: SigLiT frozen public B/8 (AugReg) on LiT data with 4 TPUv4 chips for one day -> 79.7% (caption + Sec. 1 + 4.4 text);
g/14 same setup two days -> 84.5%. SigLIP with public unlocked B/16 on WebLI, 16 TPUv4 chips, 3 days -> 71.0%.
Last two rows: randomly initialised.
| Model | Image | Text | BS | #TPUv4 | Days | INet-0 |
| SigLiT | B/8 (frozen) | L* | 32k | 4 | 1 | 79.8 |
| SigLiT | g/14 (frozen) | L | 20k | 4 | 2 | 84.5 |
| SigLIP | B/16 (unlocked, pretrained) | B | 16k | 16 | 3 | 71.0 |
| SigLIP | B/16 | B | 32k | 32 | 2 | 72.1 |
| SigLIP | B/16 | B | 32k | 32 | 5 | 73.4 |
* variant of the L text model with 12 layers.
INTERNAL INCONSISTENCY: the table cell says 79.8 but caption/Sec. 1/Sec. 4.4 all say 79.7. Use 79.7 for prose and
flag it; do not print both without a note.
Other: SigLiT B/8+L*: LION, peak LR 1e-4, wd 1e-7, 6.5k warmup, 65,000 steps total at batch 32k ("just under one day").
g/14: batch 20k, 107k steps, "under two days". SigLIP from scratch 73.4% "in 5 days with 32 TPUv4 chips" vs CLIP
approx. 2500 TPUv3-days for 72.6% (per [30]); intro says FLIP/CLIP need approx. 5 and 10 days on 256 TPUv3 cores.

## Other items a page might use
- Sec. 4.8: |B| positives and |B|^2-|B| negatives per batch; at 16k, 268M negatives vs 16k positives. Imbalance not
  a concern (Fig. 6); keeping hardest negatives nearly maintains quality; random masking hurts; easiest fails.
- Sec. 4.7: beta2 0.999 -> 0.95 stabilises large-batch training (used everywhere).
- Sec. 4.10 / Fig. 7: sigmoid more robust to injected image/text/batch corruption (M-scale, batch 16384, 3.6B examples).
- SigLIP recipe: WebLI English, B/16 image + B text, 224x224, 32k sentencepiece vocab, max 16 text tokens.
- Table 3: SigLIP B/16@224 (196 patches) INet 76.2; SO400M 729 patches 83.2.

## Outline assertions I could not confirm or that are wrong
1. "SigLIP Eq. (1) (softmax) vs Eq. (2)?": the paper NUMBERS NO EQUATIONS. Label them by section (Sec. 3.1 softmax, Sec. 3.2
   sigmoid, Sec. 3.3 chunked) or as Algorithm 1. Sec. 3 has no "Eq." numbers anywhere.
2. "init t'=log 10, b=-10": confirmed (t=10).
3. "SigLiT 84.5% 4 TPUv4 2 days": confirmed (g/14 frozen, batch 20k, Table 1). The B/8 row is 79.8 in the table but 79.7 in text.
4. "32k sufficiency": confirmed, but the peak differs: SigLIP peaks at 32k, softmax at 98k (9B examples), SigLiT saturates ~32k and peaks 256k.
5. "1M batch experiment": exists for SigLiT (frozen ViT-g embeddings, B text tower) only, 18B examples: 84.7 at 1024k.
   It is NOT a SigLIP from-scratch experiment; SigLIP/mSigLIP went to 307k/245k.
6. "small-batch advantage numbers (Fig./Table)": Fig. 2 is graph-only; exact numbers are in App. Table 8 (SigLiT) and
   App. Table 5 (SigLIP), listed above.
7. Chunked memory: paper states |B|^2 -> b^2 (per-device), not a numeric factor. The device count for the 1M run is not given.


## Fact-check note (sign of b)
The printed Sec. 3.2 loss log 1/(1+e^{z_ij(-t x_i.y_j + b)}) has the opposite sign on b from Algorithm 1 (logits = t x.y + b). With b initialised at -10 only the Algorithm 1 form acts as a prior toward "negative"; the page uses the Algorithm 1 form and says so.
