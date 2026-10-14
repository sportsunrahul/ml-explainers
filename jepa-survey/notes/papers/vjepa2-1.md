# V-JEPA 2.1 (2603.14482v3)

Lens: architecture / training method. Delta from V-JEPA 2. Sources read: `sources/vjepa2-1/src/paper.tex`,
`src/sections/methods.tex`, `src/sections/results.tex`, `paper.pdf` (38 pages, for page numbers and the two
number-bearing figures), `abs.html`. No README in the source dir, so nothing here is `code-only`.

## 1. Identity

- Title: V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning
- arXiv 2603.14482, version read: v3. abs.html: submitted 15 Mar 2026 (v1), v2 17 Mar 2026, v3 11 Jun 2026.
- Authors: Lorenzo Mur-Labadia (FAIR at Meta + Universidad de Zaragoza, work done at Meta) ... Nicolas Ballas,
  Adrien Bardes (joint last authors). 9 authors incl. Assran, Rabbat, LeCun.
- Venue: none stated (FAIR at Meta report, `fairmeta.cls`).
- Repo: https://github.com/facebookresearch/vjepa2 (paper.tex metadata line `\metadata[Code]`). meta.json has
  repo null; no README was fetched.

## 2. Core problem

V-JEPA 2 gives strong global video features (action recognition, anticipation, planning) but its patch
features are "noisy and show only fragmented local spatial structure" (Sec 2.2, p4). Linear probes on dense
tasks are poor: 22.2 mIoU ADE20K and 0.682 RMSE NYUv2 for the ViT-L baseline (Table 1, p7); 24.4 mIoU and
0.642 RMSE for V-JEPA 2 ViT-g (Table 8, p15). The paper asks how to make the same latent mask-denoising
objective produce dense, spatially and temporally consistent features without losing the global picture.

## 3. What prior methods did and why that hurt (per the paper)

1. Loss only on masked tokens. The predictor emits a token for every input position (context and mask),
   but Eq. (1) grades only the masked set M. "Therefore, the model has no incentive to encode local
   information within the context tokens and can instead devote this computation to aggregating global
   information ... similarly to register tokens" (Sec 2.2, p4, citing Darcet et al. 2023). This is the
   paper's hypothesis for the noisy PCA maps; it is verified by adding the context loss (Fig 3, p4).
2. Supervision only at the encoder output. V-JEPA 2's predictor targets only the last encoder layer; a model
   trained that way needs 4 intermediate layers at probe time to do well on dense tasks (Table 13, p37).
3. Single 3D-conv tokenizer for both modalities. V-JEPA 2 "employed a single 3D convolution as the
   patch-embedding layer. Images were duplicated temporally and treated as a 16-frame static video, which
   significantly increased their computational cost and introduced an incorrect representational bias"
   (Sec 2.3.4, p8).
4. Image data was only ImageNet (1M, weight 0.250 in VideoMix22M); V-JEPA 2.1 swaps in LVD-142M (Table 3, p8).
5. DINO-family image models (the other prior line) have good dense features but "do not directly learn
   temporal dynamics from video" (Intro, p2).

## 4. The one key idea

Grade every predictor output, not only the masked ones: add a distance-weighted L1 loss on the visible
(context) tokens, and apply both losses at four encoder depths through one multi-level predictor. Context
tokens then must stay grounded at their own location instead of becoming global aggregators.

## 5. Mechanism, step by step

Notation: x = masked view (visible patches only), y = full clean view, M = masked patch indices,
C = context (visible) patch indices, E_theta = x-encoder (trained), E_theta-bar = y-encoder (EMA),
P_phi = predictor, Delta_y = learnable mask tokens with the positions of M.

1. Tokenize by modality. Video: 3D conv 16 x 16 x 2 (patch 16, tubelet 2). Image: 2D conv 16 x 16. Add 3D
   RoPE and a learnable modality embedding (also fed to the predictor). Primary phase: 16 frames at 256^2
   -> 8 x 16 x 16 = 2048 tokens per clip; image 256^2 -> 256 tokens (our reading, computed from Table 12).
   Cooldown: 64 frames at 384^2 -> 32 x 24 x 24 = 18432 tokens; image 512^2 -> 1024 tokens (our reading).
2. Mask by dropping tokens. Same multi-block strategy as V-JEPA / V-JEPA 2 (Sec 2.1; App A). Table 12:
   spatial mask scale [0.15, 0.7], aspect ratio [0.75, 1.5], temporal mask scale [1.0, 1.0], i.e. a mask
   covers the whole clip in time. The x-encoder sees only the |C| visible tokens.
3. x-encoder forward. Take the outputs of three intermediate blocks plus the final block (ViT-G: blocks
   12, 24, 36, 48, Table 12; "4 intermediate blocks of the encoder, at equally spaced indices", App A).
   Concatenate along channels ("normalized output from intermediate encoder blocks", Fig 4 caption), then
   an MLP fuses and reduces the dimension. Output: |C| fused context tokens.
4. Predictor forward. Concatenate the fused context tokens with |M| mask tokens along the sequence axis.
   The predictor (24 blocks, width 384, vs 12 blocks in V-JEPA 2) "produces multi-level predictions for
   each token": four outputs per position, one per encoder level (Sec 2.3.2, p7). Positions graded:
   all |C| + |M| positions, at each of the four levels. V-JEPA 2 graded only M at one level.
5. Targets. The EMA y-encoder processes the full view y; its outputs at the same four blocks are the
   targets, under stop-gradient. Fig 4 shows no MLP on the y side (our reading of the diagram).
6. Loss per level: L_dense = L_predict + L_ctx (Eqs 1-3). L_predict averages an L1 over M; L_ctx averages
   lambda_i-weighted L1 over C, with lambda_i = lambda / sqrt(d_min(i, M)), d_min = distance in patches to
   the nearest masked token. lambda = 0.5 for video, 0.7 for images (App A). lambda is warmed up on a
   schedule the paper gives as "epochs 50-100" (Sec 2.3.1; training itself is counted in iterations).
   How the four levels are combined (sum or weighted) is not stated.
   Our reading: because the two terms are each normalised by their own set size, a context token's
   effective weight relative to a masked token is lambda_i * |M| / |C|, not lambda_i alone.
7. Why the context prediction is not trivial. The x-encoder's context token was computed without the
   masked content; the target is the EMA encoder's token for the same position computed from the full
   clip (our reading of Eq 2 and Fig 4). The paper's stated risk is that "the system can potentially find
   trivial solutions, such as copying the context features", which it says shows up as poor global
   performance rather than collapse (Sec 2.3.1). Mitigations: EMA + stop-gradient (as before), the
   distance weighting, the lambda warm-up, and deep self-supervision, which "allows to recover the global
   understanding capabilities" (Sec 2.3.2). Table 2 (p7) is the evidence: constant lambda = 1.0 gives
   24.6 mIoU / 51.1 SSv2; weighted lambda = 0.5 with warm-up gives 33.8 / 62.5.
8. What gets a gradient: tokenizers (2D and 3D conv), modality embeddings, x-encoder, fusion MLP,
   predictor, mask tokens. EMA y-encoder: no gradient, coefficient 0.99925. Whether the y-side tokenizers
   are EMA copies is not stated (our reading: yes, as in V-JEPA 2).
9. Training: 135k primary iterations (16 frames, 256^2) then 12k cooldown (64 frames, 384^2; images 512^2),
   per-modality worker nodes whose gradients are aggregated each step; global batch 2,304 images + 128
   clips. Ablations (Table 1) are after the primary phase; final results use the full schedule.
10. Distillation (Sec 2.3.6, App B): ViT-G -> ViT-L (300M) and ViT-B (80M). Frozen teacher replaces the
    EMA, loss only on the teacher's last layer (no deep self-supervision), new 12-block predictor, an EMA
    of the student is the released model. Two stages mirroring primary and cooldown.

Token grid for the chapter visual (video, 2048 tokens; one spatial block through all 8 tubelet slots):

```
V-JEPA 2 (Eq 1)                     V-JEPA 2.1 (Eqs 1+2, per level)
. . . . . . . .                     w w w w w w w w   w = lambda/sqrt(d_min): 1.0*lambda next to mask,
. . M M M . . .                     w W M M M W w w       0.71*lambda two patches away, 0.58 at three ...
. . M M M . . .                     w W M M M W w w   M = weight 1 (L_predict)
. . . . . . . .                     w w w w w w w w   . = no loss
 only M graded                       every token graded; weight decays away from the mask
```

Layer stack (ViT-G):

```
x-encoder blocks:  1 .. 12 .. 24 .. 36 .. 48
                        |      |      |      |      (LN'd outputs, channel-concat)
                        +------+------+------+--> MLP fuse --> [ctx | mask tokens] --> Predictor (24 x 384)
                                                                                        | 4 outputs / token
y-encoder (EMA):   1 .. 12 .. 24 .. 36 .. 48 ------ sg ------ targets at the same 4 depths --> L_dense x 4
```

## 6. Equations (TeX verbatim from methods.tex; no custom macros needed)

Eq. (1), label `eq: l1_main_loss` (methods.tex l.29-32), the V-JEPA 2 objective:

```
\mathcal{L_{\text{predict}}} =  \frac{1}{|M|} \sum_{i \in M}  \| P_{\phi}(E_{\theta}(x), \Delta_y)_i - \mathrm{sg}(E_{\overline{\theta}}(y)_{i}) \|_{1}  ,
```

Paper's wording: "M is a set containing the masked patch indexes from the x view. The loss use a
stop-gradient operator, sg, to prevent representation collapse and an exponential moving average
theta-bar of theta is used to update the weight of the encoder E_theta-bar(y) processing the video y.
L_predict is only applied on masked tokens, and not on the context tokens."

Gloss: x masked view; y clean view; E_theta(x) visible-token embeddings; Delta_y mask tokens carrying the
positions of M; P_phi(...)_i predictor output at position i; E_theta-bar(y)_i EMA-encoder embedding of
the clean view at i; sg stop-gradient; || . ||_1 L1 distance; 1/|M| mean over masked positions.

Eq. (2), label `eq:l1_ctx_loss` (methods.tex l.121-124), the context loss:

```
\mathcal{L_{\text{context}}} =  \frac{1}{|C|}  \sum_{i \in C} \lambda_i  \| P_{\phi}(E_{\theta}(x), \Delta_y)_i - \mathrm{sg}(E_{\overline{\theta}}(y)_{i}) \|_{1}  ,
```

Paper's wording: "C is the set of indexed context tokens, and lambda_i is a patch-specific weighting
parameter ... The model is trained to minimize L_predict + L_ctx." (The paper names this loss both
L_context and L_ctx, and once writes L_prediction for Eq 1.)

Gloss: same terms as Eq 1, but summed over the visible positions C, each scaled by lambda_i, mean over |C|.

Unnumbered, Sec 2.3.1 (methods.tex l.237): `\mathcal{L_{\text{dense}}} = \mathcal{L}_{\text{predict}} + \mathcal{L}_{\text{ctx}}`.

Eq. (3), label `eq:lambda_i` (methods.tex l.241):

```
\lambda_i = \frac{\lambda}{\sqrt{\text{d}_{\text{min}} (i, M)}}
```

Paper's wording: "L_ctx for a given patch i is weighted by the inverse square root of its minimum
spatio-temporal distance to any masked token in the video sequence ... d_min is the distance, in number
of blocks, between a context token and its closest mask token. This weighting emphasizes patches near
masked regions by enforcing local continuity between masked and context areas."

Gloss: lambda global scale (0.5 video, 0.7 image); d_min(i, M) patch-grid distance from context token i
to the nearest masked token; nearer tokens get a larger weight.

## 7. Config / setup table

| Item | V-JEPA 2 (as stated in this paper) | V-JEPA 2.1 | Loc |
|---|---|---|---|
| Loss positions | masked M only, 1 level | M + C, 4 levels | Sec 2.1, 2.3.1, 2.3.2 |
| Loss type | L1, EMA target, sg | same, plus lambda_i on C | Eqs 1-3 |
| lambda | 0 | 0.5 video / 0.7 image, warm-up "epochs 50-100" | Table 12; Sec 2.3.1 |
| Deep supervision blocks (ViT-G) | last only | 12, 24, 36, 48 | Table 12, p33 |
| Predictor | 12 blocks | 24 blocks, width 384 | App A, p33 |
| Tokenizer | one 3D conv; images tiled to 16 frames | 3D conv 16x16x2 video, 2D conv 16x16 image, modality token, 3D RoPE | Sec 2.3.4, p8; Fig 4 |
| Patch / tubelet | 16 / 2 | 16 / 2 | Table 12 |
| Encoder sizes | ViT-g 1B | ViT-g 1B, ViT-G 2B; distilled ViT-L 300M, ViT-B 80M | Intro p3; Sec 2.3.6 |
| Encoder depth / width / heads | not stated | ViT-G depth 48 implied by block indices; widths, heads not stated | Table 12 |
| Primary phase | -- | 135k iters, 16 frames at 4 fps, 256^2 video and image | Table 12; App A |
| Warm-up | -- | 12k iters, LR 1e-4 -> 5.25e-4, then constant | App A |
| Cooldown | -- | 12k iters, LR 6e-4 -> 1e-6 (as written), 64 frames at 384^2, images 512^2 | Table 12; App A |
| Batch (global) | -- | 2,304 images + 128 video clips, separate worker nodes per modality | Table 12; Sec 2.3.3 |
| Weight decay / EMA | -- | 0.04 / 0.99925 | Table 12 |
| Masks | V-JEPA multi-block | spatial scale [0.15, 0.7], temporal [1.0, 1.0], aspect [0.75, 1.5] | Table 12 |
| Random resize | -- | aspect [0.75, 1.35], scale [0.3, 1.0] | Table 12 |
| Data | VideoMix22M: SSv2 168K, Kinetics 733K, HowTo100M 1.1M, ImageNet 1M, YT-1B 19M (1.6M hours) | VisionMix163M: same videos, ImageNet replaced by LVD-142M (142M curated images) | Table 3, p8 |
| Sampling weights (V-JEPA 2 -> 2.1) | SSv2 0.056, Kinetics 0.188, HowTo100M 0.318, ImageNet 0.250, YT-1B 0.188 | SSv2 0.170, Kinetics 0.010, HowTo100M 0.100, ImageNet 0, YT-1B 0.720; LVD-142M cells blank | Table 3 |
| Compute | -- | GPU type and hours not stated; robot CEM runs on a single A100 | Table 6 caption |
| Hyperparameters across sizes | -- | identical for ViT-L, -g, -G | Table 12 caption |

Evaluation protocols (App C, p34-36): dense linear probes follow DINOv3 (Simeoni et al. 2025) on frozen
last-layer patch features, no intermediate layers; ADE20K and VOC at 512 px (1024 patch tokens), Cityscapes
1024 px height; VOS is non-parametric label propagation at 480 px short side; global tasks use a 4-block
attentive probe (16 heads) at 384^2. Table 9 caption says SSv2 uses 16 x 2 x 3 inputs; App C.5 writes
"64 x 2 x 3 input for SSv2 (16 frames clip, ...)" -- inconsistent, quote with care.

## 8. Results worth remembering

Ablation, ViT-L baseline, after primary phase except last row (Table 1, p7; Fig 5, p6):

| Step | IN1K | SSv2 | NYU RMSE | ADE20K mIoU |
|---|---|---|---|---|
| V-JEPA 2 | 82.2 | 72.8 | 0.682 | 22.2 |
| + Context loss | 72.6 | 62.5 | 0.474 | 33.8 |
| + Multi-level pred. | 80.8 | 72.1 | 0.463 | 38.6 |
| + Vision Mix | 81.6 | 72.6 | 0.418 | 40.8 |
| + Multi-modal tok. | 81.6 | 72.6 | 0.415 | 41.4 |
| + Model scaling (ViT-G) | 84.8 | 76.1 | 0.365 | 47.1 |
| + Cool-down | 85.5 | 77.7 | 0.307 | 47.9 |

Text variants that differ from Table 1 (do not reconcile): Sec 2.2 and 2.3.1 prose give 33.9 mIoU and
0.473 RMSE for the context loss; Sec 2.3.2 prose gives 72.0 SSv2 for deep self-supervision.

Weighting ablation (Table 2, p7; ADE20K mIoU / SSv2): lambda = 0: 22.2 / 72.8; const 0.05: 26.4 / 71.0;
const 0.2: 29.6 / 62.5; const 0.5: 27.5 / 53.8; const 1.0: 24.6 / 51.1; 0.2 + warm-up: 30.5 / 60.5;
0.5 + warm-up: 32.2 / 61.5; 0.5 weighted + warm-up: 33.8 / 62.5.

Dense tasks (Table 8, p15; NYUv2 RMSE, KITTI RMSE, ADE20K, Cityscapes, VOC mIoU, DAVIS-S, YT-VOS-S J&F):

| Model | NYUv2 | KITTI | ADE20K | Citysc. | VOC | DAVIS-S | YT-VOS-S |
|---|---|---|---|---|---|---|---|
| DINOv3 7B/16 | 0.309 | 2.346 | 55.9 | 81.1 | 86.6 | 71.1 | 74.1 |
| DINOv3 ViT-H+ 0.8B/16 | 0.352 | 2.635 | 54.8 | 79.5 | 85.8 | 71.1 | 74.0 |
| DINOv2 1B/14 | 0.372 | 2.624 | 49.5 | 75.6 | 83.1 | 63.9 | 65.6 |
| V-JEPA 2 ViT-g 1B/16 | 0.642 | 4.650 | 24.4 | 45.9 | 63.9 | 52.5 | 53.7 |
| V-JEPA 2.1 ViT-g 1B/16 | 0.350 | 2.601 | 47.8 | 71.8 | 84.7 | 68.1 | 72.3 |
| V-JEPA 2.1 ViT-G 2B/16 | 0.307 | 2.461 | 47.9 | 73.5 | 85.0 | 69.0 | 72.7 |

Gains vs V-JEPA 2 ViT-g, as the paper states them: +23.4 ADE20K, +27.6 Cityscapes, +20.7 VOC (Sec 3.5,
p15). Fig 2 (p2) relative gains: NYUv2 +52%, Ego4D STA +35%, DAVIS +31%, ADE20K +96%. "State of the art"
in linear-probe depth is NYUv2 only (0.307 vs DINOv3 7B 0.309); on KITTI DINOv3 7B is better (2.346 vs
2.461) and the paper says "best across models that have less than 2 billion parameter". VOS: "second best
... surpassing all prior encoders except DINOv3" (Sec 3.6). Do not use Fig 2 for third-party numbers: it
draws the SSv2 prior SOTA as 71.1 and IN1K as 88.1, while Table 9 has 70.1 and 88.4.

Global tasks (Table 9, p18; top-1, attentive probe; V-JEPA 2 and 2.1 at 384^2):

| Model | SSv2 | Diving-48 | K400 | IN1K |
|---|---|---|---|---|
| V-JEPA 2 ViT-g 1B | 77.3 | 90.2 | 87.3 | 85.1 |
| V-JEPA 2.1 ViT-g 1B | 76.9 | 89.0 | 87.0 | 84.8 |
| V-JEPA 2.1 ViT-G 2B | 77.7 | 89.2 | 87.7 | 85.5 |
| DINOv3 7B (literature) | 70.1 | -- | 87.8 | 88.4 |
| InternVideo2 2s-1B (same protocol) | 69.7 | 86.4 | 89.4 | 85.8 |

Text (Sec 3.7): 77.7 SSv2 is "a new absolute state-of-the-art ... compared to 77.5 for InternVideo2 full
fine-tuning, 77.3 for V-JEPA 2". Caveat for the page: parity is ViT-G (2B) vs V-JEPA 2 ViT-g (1B); at
matched 1B, 2.1 ViT-g is below V-JEPA 2 ViT-g on all four columns, and Diving-48 is below even at 2B.

Action anticipation EK100 (Table 5, p12; recall@5 verb / noun / action): V-JEPA 2 ViT-g 63.6 / 57.1 /
39.7; V-JEPA 2.1 ViT-g 63.6 / 56.2 / 38.4; V-JEPA 2.1 ViT-G 64.3 / 59.9 / 40.8; PlausiVL 8B 27.6 action.
Paper calls 40.8 "a +2.8% improvement" (relative to 39.7, our reading).

Ego4D short-term anticipation (Table 4, p11; Top-5 mAP All): StillFast 3.96; STAformer 5.67; DINOv2 ViT-g
5.44; DINOv3 ViT-7B 5.68; V-JEPA 2 ViT-g 6.02; V-JEPA 2.1 ViT-g 6.75; V-JEPA 2.1 ViT-G 7.71 ("approximately
35%" relative over STAformer). Box AP: ViT-G 50.7 vs DINOv3 7B 50.2. AP b+V 25.8, AP b+delta 20.2.

Robot manipulation (Table 6, p13; Sec 3.3, p12-13). Protocol: V-JEPA 2-AC recipe from the public
V-JEPA 2 code with the 2.1 encoder swapped in; frame-causal action-conditioned predictor, 24 layers,
~300M params, trained on Droid raw with teacher-forcing + two-step rollout loss; zero-shot on a table-top
Franka Panda with parallel-jaw gripper; reach, grasp, pick-and-place of a cup with image goals via MPC;
CEM on a single A100; same task config files as V-JEPA 2; 10 tasks per skill.

| Method | CEM samples | Iter. | Horizon | Time | Reach | Grasp | Pick-and-place |
|---|---|---|---|---|---|---|---|
| V-JEPA 2 | 800 | 10 | 1 | 3 sec | 100% | 60% | 80% |
| V-JEPA 2.1 | 800 | 10 | 1 | 3 sec | 100% | 70% | 80% |
| V-JEPA 2.1 | 300 | 15 | 8 | 14 sec | 100% | 80% | 80% |

The "20%" grasp headline (abstract, intro, Table 6 caption) is 60 -> 80 and needs the second planning
setting (300 samples, 15 iterations, horizon 8, 14 s per step). At matched planning settings the gain is
10% (60 -> 70), which is what the body text says (Sec 3.3: "leading to a 10% improvement"). The paper
reports that V-JEPA 2 degrades with longer horizons (no number). Remaining failures are attributed to
gripper-action planning, not spatial understanding.

Navigation (Table 7, p14; Sec 3.4): CDiT latent world model on 2.1 features vs NWM on SD-VAE. Planning
time 10.6 s vs 103.2 s ("10x"); 8 denoising steps vs 128; CEM N = 480 trajectories of 2 seconds at 4 FPS,
3-DoF actions. Tartan Drive ATE: NWM 5.831, ViT-g 5.758, ViT-G 5.687. Average ATE: 3.032 / 2.975 / 2.990.

VideoQA (Table 10, p19): same filtered PerceptionLM data (18% removed), Llama 3.1 8B. V-JEPA 2 ViT-g vs
2.1 ViT-G: avg 57.8 vs 57.9; PerceptionTest 80.1 vs 83.1; worse on TemporalBench (32.3 vs 28.5) and TOMATO
(41.4 vs 38). 2.1 alignment uses 64 frames vs 32.

Distillation (Table 11, p23): ViT-L scratch vs distilled: SSv2 74.2 -> 76.5, ADE20K 42.0 -> 46.7, NYU
0.381 -> 0.336, KITTI 2.914 -> 2.490, DAVIS 64.5 -> 68.7. ViT-B distilled: SSv2 74.5, ADE20K 42.0, DAVIS
67.0. Text says distilled ViT-L DAVIS 68.7 is "close to 67.0" from ViT-G, but Table 11 lists ViT-G at 69.0
(67.0 is the ViT-B row).

Deep self-supervision vs multi-scale probing (Table 13, p37; ViT-L from scratch; last layer / 4 layers):
without: ADE20K 34.9 / 39.1, NYU 0.513 / 0.463, Diving-48 85.8 / 86.9; with: 42 / 43.9, 0.381 / 0.370,
87.2 / 88.1. Cooldown resolution (Table 14, p37; ViT-G 256 vs 384): SSv2 77.1 / 77.7, NYU 0.313 / 0.307,
EK100 38.8 / 40.8.

## 9. Limitations the paper states

- ADE20K and Cityscapes "remains slightly behind the best image encoders"; hypothesis: VisionMix has
  fewer cluttered scenes (Sec 3.5). DINOv3 7B leads on every dense column except NYUv2.
- VOS second to DINOv3 (Sec 3.6). VideoQA worse than V-JEPA 2 on TemporalBench and TOMATO (Sec 3.8).
- Robot failures from gripper-action planning (Sec 3.3). Future work: scale past 2B (DINOv3 went to 7B),
  larger video data, dense-prediction world models (Sec 5). Not stated but visible in Tables 5 and 9: at
  matched 1B size the global numbers drop slightly.
- No later paper in this source set fixes these.

## 10. Outline feedback (ch-vjepa21)

- "let EVERY token carry loss" is right but under-specified: context tokens carry a weight
  lambda / sqrt(d_min) that fades with distance from the mask, lambda is warmed up, and a flat weight
  hurts (Table 2: const lambda = 1.0 drops SSv2 72.8 -> 51.1). The before/after token grid should show a
  weight gradient around the mask, not uniform shading. Masks span the whole clip in time, so the grid is
  one spatial block repeated over the 8 tubelet slots.
- The layer-stack visual should show ONE predictor with four outputs per token, fed by a channel-concat
  + MLP of blocks 12/24/36/48 (ViT-G), with EMA targets at the same four depths; not four predictors.
- Add the ablation beat: the context loss alone costs ~10 points SSv2 and ~10 IN1K; deep self-supervision
  recovers them (Table 1 rows 2-3). That is the argument for needing both ingredients.
- "robotics (grasp success vs 2-AC)": the 20% is 60 -> 80 with a changed planner (horizon 8, 300 samples,
  15 iters, 14 s); matched settings give 60 -> 70. The page must say which it quotes.
- "nothing was lost" holds for ViT-G 2B vs V-JEPA 2 ViT-g 1B (SSv2 77.7 vs 77.3; K400 87.7 vs 87.3; IN1K
  85.5 vs 85.1; Diving-48 89.2 vs 90.2 is a small loss). At matched 1B the 2.1 model is slightly below on
  all four and on EK100. Phrase as "at 2B" or show both rows.
- The tokenizer alone is a small step (ADE20K 40.8 -> 41.4); the image-data swap to LVD-142M is the larger
  data effect. Do not overweight the tokenizer in the chapter.
- Depth SOTA claim is NYUv2 only; DINOv3 7B still wins KITTI and all segmentation columns.
- Dates: v1 2026-03-15, v3 2026-06-11. Sizes as in outline (ViT-g/G, 1B/2B) plus distilled ViT-L/B
  (300M/80M) if wanted.
