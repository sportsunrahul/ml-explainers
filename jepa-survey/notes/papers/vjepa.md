# V-JEPA (notes for ch-vjepa)

Lens: architecture; delta from I-JEPA. Sources: sources/vjepa/{abs.html, paper.pdf, src/paper.tex, README.md}.
Table/figure numbers and pages are verified against paper.pdf. Appendix sections print without letters in the
PDF, so they are cited by title + page + LaTeX label.

## 1. Identity
- Title: Revisiting Feature Prediction for Learning Visual Representations from Video
- arXiv: 2404.08471v1 (version read). abs.html dateline: "[Submitted on 15 Feb 2024]" -> published 2024-02-15,
  recorded as given (the 2404 id prefix implies an April 2024 listing; abs.html says 15 Feb 2024).
- Authors: Adrien Bardes (first) ... Nicolas Ballas (last); 8 authors; Assran and Ballas marked joint last.
- Venue: not stated in abs.html or in the LaTeX (\date{\today}, no \metadata[Venue]).
- Repo: https://github.com/facebookresearch/jepa (LaTeX \metadata[Code]; README.md read).
- PDF: 23 pages. Main text pp. 1-10, references pp. 11-14, appendix pp. 15-23.

## 2. Core problem
Is feature prediction, on its own, enough to learn visual representations from video? The paper trains with a
feature-prediction objective only ("without the use of pretrained image encoders, text, negative examples,
reconstruction, or other sources of supervision", abstract) and asks how far a frozen backbone gets on both
motion tasks (SSv2) and appearance tasks (K400, IN1K).

## 3. What prior methods did and why that hurt (per the paper)
- Pixel prediction (VideoMAE, OmniMAE, Hiera): "must dedicate significant model capacity and compute to capture
  all the low-level detail in the visual input" (Sec 2, Feature Prediction versus Pixel Reconstruction). Their
  schedules are long: 400K-1500K iterations, 410M-2400M samples seen (Table 5, cols Iter. and #Samples Seen).
- Slow-feature / temporal-invariance methods (SFA, SSA, noise-contrastive variants): only enforce invariance
  between adjacent frames; "Our exploration in this paper goes beyond temporal invariance" (Sec 2, Slow Features).
- Earlier predictive-feature methods used frozen encoders, supervised forecasting losses, or contrastive
  negatives with distant clips, often with small convolutional encoders (Sec 2, Predictive Features).
- I-JEPA / data2vec did latent masked prediction on images and audio; V-JEPA extends it to video (Sec 2).

## 4. The one key idea
Mask 3D blocks that span every frame of the clip (so nothing can be copied from a neighbouring frame), and train
a predictor to regress the EMA target encoder's embeddings of the masked tubelets from the visible ones with an
L1 loss. No pixels are reconstructed; collapse is prevented by the stop-gradient + EMA target encoder.

## 5. Mechanism, step by step
Shapes from Appendix "Extended Description of V-JEPA" (p16, label appendix:vjepa_extended_description) and
Fig 7 (p16) unless noted.
1. Sample a clip: 16 frames with temporal stride 4 (64 raw frames). Resize to 224 x 224 -> tensor 16 x 224 x
   224 x 3. (For ViT-H/16_384 the spatial size is 384.)
2. Tokenize with a 3D conv: d filters of size 2 x 16 x 16, temporal stride 2, spatial stride 16 ->
   8 x 14 x 14 x d. Add absolute 3D sin-cos positional embeddings. Flatten -> L = 1568 tokens of width d.
   (our reading: at 384 res the grid is 8 x 24 x 24 = 4608 tokens; the paper never states this number.)
3. Sample 2 masks per clip (Appendix "Pretraining details", Masking, p18; Sec 3.2, p4):
   - short-range: union of 8 blocks, each covering 15% of the frame (spatial_scale 0.15);
   - long-range: union of 2 blocks, each covering 70% of the frame (spatial_scale 0.7);
   - aspect ratio of every block sampled in (0.75, 1.5);
   - each block is "repeat[ed] ... across the entire temporal dimension of the video" (Sec 3.2), i.e. the mask
     is a tube through all 8 temporal token slots; the appendix masking study calls this "temporal ratio of
     100%" (p21).
   - The paper states "an average masking ratio of ~90%" (Sec 3.2; LaTeX `$\sim90\%$`).
   x = complement of the mask (visible tokens). y = masked tokens.
4. x-encoder E_theta (ViT-L/16 or ViT-H/16, no [cls] token, p17) processes only the N = L - M visible tokens
   x_N -> z_N = (z_{j_1}, ..., z_{j_N}).
5. Predictor P_phi: narrow ViT, 12 blocks, embedding dim 384, same head count as the backbone (p17). Input =
   z_N concatenated with M mask tokens; each mask token = one shared learnable vector + the absolute 3D sin-cos
   positional embedding of that masked position. Output = one d-dimensional vector per mask token,
   \hat s_M = P_phi(z_N, m_M). (The paper does not state the projections between encoder width and 384.)
6. y-encoder \overline{E}_theta = EMA of the x-encoder processes the FULL clip x_L (all 1568 tokens) ->
   s_L; targets are s_k at the masked indices. Masking is applied "at the output of the y-encoder to construct
   contextualized targets" (Sec 3.3).
7. Loss = mean L1 over the M masked positions between \hat s_k and sg(s_k) (eq. 2). Gradients flow to theta
   (x-encoder) and phi (predictor) only. The y-encoder is updated as an EMA with momentum 0.998 -> 1.0,
   linear, initialised equal to the x-encoder (p17).
8. Multi-mask amortisation: per clip the x-encoder and predictor run once per mask (2 masks), the y-encoder
   runs once (Appendix, Multi-Mask Prediction, p17).
9. Evaluation: the target encoder \overline{E}_theta is frozen; an attentive probe (cross-attention with a
   learnable query, residual, 2-layer MLP with GeLU, LayerNorm, linear classifier; 12 heads of dim 12) is
   trained on top (Appendix "Evaluation details", p18). Images are evaluated by duplicating the image into a
   16-frame still clip (p18). K400 uses 8 clips x 16 frames and 3 spatial views (16x8x3); SSv2 uses 2 clips
   (16x2x3).

Delta from I-JEPA, as V-JEPA itself describes it: same JEPA (encoder, EMA target, predictor conditioned on
positions z <- Delta_y); the loss "is similar to the loss of (Assran et al. 2023) used for image pretraining,
but we modify it to use an l1 regression, which we found to be more stable" (Sec 3.1, p3). New: tubelet
tokenizer, 3D sin-cos positions, masks that extend through time, two masks per clip, attentive-probe
evaluation. The conditioning variable is positions only; no transformation parameters (contrast with IWM).

## 6. Equations
Custom macro in the preamble: `\putalg` = `{\small V-JEPA}\xspace` (not used inside the equations).

(a) Naive objective, unnumbered display, Sec 3.1 (p3):
```
\text{minimize}_{\theta,\phi}\quad \lVert P_\phi(E_\theta(x), \Delta_y) - E_\theta(y) \rVert_1,
```
Paper: "would admit a trivial solution, where the encoder outputs a constant representation".

(b) Equation (1), label eq:loss, Sec 3.1 (p3). Quoted as in the source; note the delimiters are reversed
(`\rVert ... \lVert_1`) in the .tex:
```
\text{minimize}_{\theta,\phi}\quad \rVert P_\phi(E_\theta(x), \Delta_y) - \text{sg}(\overline{E}_\theta(y)) \lVert_1,
```
Paper: "sg(.) denotes a stop-gradient operation, which does not backpropagate through its argument, and
\overline{E}_theta(.) is an exponential moving average of the network E_theta(.)".
Gloss: theta = encoder weights; phi = predictor weights; x = visible region; y = masked region; Delta_y =
spatio-temporal positions of y (the conditioning variable z); E_theta = x-encoder; \overline{E}_theta = EMA
target encoder; sg = stop gradient (no gradient into the target); ||.||_1 = L1 norm.

(c) Optimal predictor and encoder gradient, unnumbered, Sec 3.1 (p3):
```
P^\star(E_\theta(x)) &= \text{argmin}_{P} \lVert P(E_\theta(x)) - Y \rVert_1\\
&= \text{median}(Y|E_\theta(x)).
```
```
\nabla_\theta \mathbb{E} \lVert P^\star(E_\theta(x)) - Y \rVert_1 = \nabla_\theta  \text{MAD}(Y | E_\theta(x)),
```
Paper: "in the case where the predictor is optimal, the encoder must learn to capture as much information
about the video as possible to minimize the deviation of the target." Y = random variable for
\overline{E}_theta(y); MAD = median absolute deviation.

(d) Equation (2), label eq:loss_detail, Appendix "Extended Description of V-JEPA" (p16):
```
\text{Loss} = \frac{1}{M} \sum_{k \in (i_1, \ldots, i_M)} \lVert \hat s_{k} - s_k \rVert_1,
```
Paper: "simply the average L1 distance between the output of the predictor and the y-encoder."
Gloss: M = number of masked tokens; (i_1..i_M) = their indices; \hat s_k = predictor output for mask token k;
s_k = y-encoder output at position k (full clip input). Only masked positions are graded.

(e) Attentive probe pooling, unnumbered, Appendix "Evaluation details" (p18):
```
\sum^L_{i=1}{\frac{\exp(q^\top {\bf W_k} s_i)}{\sum_j \exp(q^\top {\bf W_k} s_j)} {\bf W_v} s_i },
```
Gloss: s_i = frozen encoder token i (of L); q = learnable query; W_k, W_v in R^{d x d} = key and value maps.

## 7. Config / setup table
Source: Table 8 "pretraining hyper-parameters for V-JEPA" (p17) unless noted.

| Item | ViT-L/16_224 | ViT-H/16_224 | ViT-H/16_384 |
|---|---|---|---|
| data | VideoMix2M | VideoMix2M | VideoMix2M |
| resolution | 224 | 224 | 384 |
| num_frames / temporal_stride | 16 / 4 | 16 / 4 | 16 / 4 |
| raw frames covered | 64 | 64 | 64 |
| clip duration as stated | "roughly 3 second clips on average" (Sec 3.4, p4) vs "roughly 2 seconds ... at 30 frames per second" (Appendix p16) and "1 clip (~2 seconds)" (Table 14 caption, p22) | same | same |
| tubelet (t x h x w) | 2 x 16 x 16 | 2 x 16 x 16 | 2 x 16 x 16 |
| token grid / count | 8 x 14 x 14 = 1568 | 8 x 14 x 14 = 1568 | 8 x 24 x 24 = 4608 (our reading) |
| positional embedding | absolute 3D sin-cos | same | same |
| random_resize_scale / aspect | (0.3, 1.0) / (0.75, 1.35) | same | same |
| horizontal_flip | true | true | true |
| masking: block aspect ratio | (0.75, 1.5) | same | same |
| short-range: blocks / spatial_scale | 8 / 0.15 | same | same |
| long-range: blocks / spatial_scale | 2 / 0.7 | same | same |
| masks per clip | 2 (one short, one long) | same | same |
| temporal extent of mask | all frames ("temporal ratio of 100%", p21) | same | same |
| average masking ratio | ~90% (Sec 3.2) | same | same |
| encoder params | 200M (Table 6) | 630M (Table 6) | 630M (Table 6) |
| predictor | 12 blocks, dim 384, heads = backbone | same | same |
| [cls] token | none | none | none |
| loss | L1, mean over masked tokens (eq. 2) | same | same |
| optimizer | AdamW | AdamW | AdamW |
| batch_size | 3072 | 3072 | 2400 |
| iterations / warmup | 90000 / 12000 | 90000 / 12000 | 90000 / 12000 |
| lr start -> peak -> final | 2e-4 -> 6.25e-4 -> 1e-6 (linear warmup, cosine decay) | same | same |
| weight decay | 0.04 -> 0.4 (linear) | same | same |
| EMA momentum | 0.998 -> 1.0 (linear) | same | same |
| scheduler_scale_factor | 1.25 (schedules computed for 112,500 iters, truncated at 90,000) | same | same |
| dtype / accelerator | bfloat16 / A100 80G | same | same |
| GPU count, GPU-hours | not stated | not stated | not stated |
| samples seen | 270M (Table 5) | not stated separately | 210M (Table 16) |

Data: VideoMix2M = HowTo100M (HT) + Kinetics-400/600/700 (K710) + Something-Something-v2 (SSv2), minus overlap
with the K400/600/700 and SSv2 validation sets, "approximately 2 million videos" (Sec 3.4, p4). Table 2
(p5) gives #Samples: K710 700K, K710+SSv2 900K, K710+HT 1900K, VideoMix2M 2000K. Hours of video: not stated.

Compute statements: Fig 5 (p8) x-axis is "Pretraining Time (Hrs.)" spanning about 50-350 h (axis text from
the figure asset; per-method values not legible as text), and Sec 5.2 says V-JEPA reaches its result "with a
roughly 2x speedup compared to the large pixel prediction models". No absolute GPU-hours are given.

Frozen-eval probe (Table 9, p18): 20 epochs, AdamW, lr 1e-3 cosine to 0, wd 0.01, batch 256 (video) / 1024
(image); K400 8 clips, SSv2 1 clip during probe training (Table 9 num_clips) and 2 clips at reported eval
(Sec "Extension to multiple clips", p18); 3 spatial views at test.

README (code-only check): the Model Zoo lists ViT-L 2x16x16 224 90K 3072; ViT-H 2x16x16 224 90K 3072; ViT-H
2x16x16 384 90K 2400, and probe accuracies K400 80.8/82.0/81.9, SSv2 69.5/71.4/72.2, IN1K 74.8/75.9/77.4,
Places205 60.3/61.7/62.8, iNat21 67.8/67.9/72.6. Every number matches Table 6, so none is code-only. Code-only
items are just the config paths (configs/pretrain/vitl16.yaml, vith16.yaml, vith16_384.yaml; configs/evals/
vitl16_k400_16x8x3.yaml etc.) and the "2x16x16" patch-size notation.

## 8. Results worth remembering
Frozen attentive probe, main models (Table 6, p7; K400 16x8x3, SSv2 16x2x3; V-JEPA rows):
| Arch | Params | K400 | SSv2 | AVA | IN1K | Places205 | iNat21 |
|---|---|---|---|---|---|---|---|
| ViT-L/16 | 200M | 80.8 | 69.5 | 25.6 | 74.8 | 60.3 | 67.8 |
| ViT-H/16 | 630M | 82.0 | 71.4 | 25.8 | 75.9 | 61.7 | 67.9 |
| ViT-H/16_384 | 630M | 81.9 | 72.2 | 25.0 | 77.4 | 62.8 | 72.6 |
- Abstract headline: 81.9 (K400), 72.2 (SSv2), 77.9 (IN1K). The 77.9 uses a two-layer attentive probe
  (Sec 5.2, p9); Table 6's 77.4 is the one-layer probe.
- Reference rows in Table 6: I-JEPA ViT-H/16_512 (630M, IN22K) 79.7 / 50.0 / 19.8 / 84.4 / 66.5 / 85.7;
  DINOv2 ViT-g/14 (1100M) 83.4 / 50.6 / 24.3 / 86.2 / 68.4 / 88.8; VideoMAE ViT-H/16 79.8 / 66.2 / 20.7 /
  72.3 / 59.1 / 65.5. Text: "+21 points on SSv2" over image models; "+5 points" SSv2, "+2" K400 over video
  models (Sec 5.2, p8).

Pixels vs features (Table 1, p5; ViT-L/16, VideoMix2M, 90K iters, batch 3072, single center view 16x1x1):
| Target | K400 | SSv2 | IN1K | K400-ft (16x5x3) |
|---|---|---|---|---|
| Pixels | 68.6 | 66.0 | 73.3 | 85.4 |
| Features | 73.7 | 66.2 | 74.8 | 85.6 |

Linear/average pooling vs attentive probe:
- Table 3 (p6; ViT-L/16, single view 16x1x1): K400 Avg 56.7 vs Att 73.7; SSv2 Avg 50.1 vs Att 66.2. Caption
  says "+17.3 points on K400 and +16.1 points on SSv2"; body text says "+17 points"; 73.7 - 56.7 = 17.0.
- Table 12 (p21; multi-clip, header K400 16x5x3, SSv2 16x2x2): V-JEPA ViT-L/16 K400 Lin 56.7 / Att 80.8,
  SSv2 Lin 50.1 / Att 69.5; VideoMAE ViT-L/16 K400 52.5 / 77.8, SSv2 41.3 / 61.2.
- Table 14 (p22; K400 attentive probe): V-JEPA ViT-L/16 1 clip 73.7, 8 clips 80.9 (vs 80.8 in Tables 6 and
  12); VideoMAE 69.4 / 77.8.
- Table 13 (p21): DINOv2 ViT-g/14 K400 linear 78.4 -> attentive 83.4.

Masking ablation (Table 4, p6; ViT-L/16 pretrained on K710+SSv2, single view):
| Masking | K400 | SSv2 | IN1K |
|---|---|---|---|
| random-tube[0.9] | 51.5 | 46.4 | 55.6 |
| causal multi-block[6] | 61.3 | 49.8 | 66.9 |
| causal multi-block[12] | 71.9 | 63.6 | 72.2 |
| multi-block | 72.9 | 67.4 | 72.8 |
Appendix masking study (p21-23, Fig 8, 26 runs, ViT-B/16 on K400, linear probe): "important to maintain a high
spatial and temporal masking ratio"; several smaller blocks beat one large block; two masks per clip beat one.

Schedule comparison (Table 5, p7; ViT-L class): V-JEPA 90K iters / 270M samples seen vs VideoMAE 400K / 410M,
OmniMAE 1170K / 2400M, Hiera 1500K / 770M. Frozen K400 80.8 vs VideoMAE 77.8; IN1K 74.8 vs OmniMAE 75.1 (the
one task V-JEPA loses). Fine-tuning (Table 15, p22): V-JEPA ViT-L/16 K400 85.6, SSv2 75.1; ViT-H/16 86.6,
77.0.

Pretraining data scale (Table 2, p5; ViT-L/16 avg over K400/SSv2/IN1K): K710 70.9, K710+SSv2 71.0, K710+HT
71.1, VideoMix2M 71.5. ViT-H/16 VideoMix2M: 74.0 / 68.5 / 75.9, avg 72.8.

Low-shot (Table 7, p9; ViT-H/16_384, 5% labels): K400 68.2 vs VideoMAEv2 37.0; SSv2 54.0 vs 28.0.

Sample efficiency (Table 16, p23): V-JEPA ViT-H/16_384 210M samples seen vs DINOv2 1900M, VideoMAEv2 1600M,
OpenCLIP 39000M.

Predictor visualisation (Sec 6, Fig 6, p10): a frozen encoder + predictor; a conditional diffusion decoder is
trained on the predicted embeddings of masked regions only; samples are spatially and temporally coherent.

## 9. Limitations the paper states
- "the datasets used to train V-JEPA and other video models are too constrained and lack the visual diversity
  of the internet-scale pretraining data used by the image models" (Sec 5.2, p9). Image models still win on
  K400 (DINOv2 83.4 vs 82.0) and IN1K (86.2 vs 77.4).
- The conditioning variable is positions only; the predictor models masking, not other transformations (IWM,
  ch-iwm, adds photometric actions). Not verifiable from my sources which later paper fixes the data gap; the
  outline assigns the scale-up to ch-vjepa2.

## 10. Outline feedback (ch-vjepa)
1. Prediction prompt "why not mask random voxels?": the paper never ablates random voxels. Closest evidence is
   random-tube[0.9] (Table 4: 51.5 / 46.4 / 55.6 vs 72.9 / 67.4 / 72.8) and the temporal masking-ratio sweep
   (Fig 8c, appendix, ViT-B/16, linear probe). Reword to "why not mask random tubes?" or point the answer at
   those two results.
2. The "both" button: training samples two separate masks per clip and runs the x-encoder and predictor once
   per mask; the y-encoder runs once. "Both" should show two masks side by side (or alternate), not a union.
3. Mask config is exact: short-range = 8 blocks x 15% of the frame, long-range = 2 blocks x 70%, aspect ratio
   (0.75, 1.5), every block extended through all 8 temporal slots. The readout should display what the sampler
   computes; the paper claims "~90%" on average and does not say per mask type. With independent block
   placement, 8 blocks of 15% do not obviously union to 90% (our reading), so do not caption the readout with
   the paper's number unless the sampler reproduces it.
4. Clip duration: pick one of "roughly 3 second clips" (Sec 3.4) or "roughly 2 seconds" (appendix, Table 14
   caption) for narration and cite it; the paper gives both.
5. IN1K number: use 77.4 if the page shows Table 6, or 77.9 only with "two-layer attentive probe".
6. Delta block: the predictor output is unchanged from I-JEPA (one d-dim vector per mask token). The real
   deltas are the tubelet tokenizer, 3D sin-cos positions, masks that span time, two masks per clip, L1 loss,
   no [cls] token, and attentive-probe evaluation. Add one sentence that z is positions only (Delta_y) to set
   up ch-iwm, where z also carries photometric parameters.
7. Published date: record 2024-02-15 from abs.html; the arXiv id 2404.* is April-numbered. The timeline chapter
   should use the abs date.
