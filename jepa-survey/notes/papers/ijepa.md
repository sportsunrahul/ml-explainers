# I-JEPA (2301.08243v3)

Sources read: sources/ijepa/src/main.tex (+ math_commands.tex), paper.pdf (17 pp, for table/figure/page
numbers), abs.html, README.md. Labels: `code-only` = README; `our reading` = inferred, not stated.

## 1. Identity
- Title: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
- arXiv 2301.08243, version read: v3. v1 2023-01-19, v2 2023-03-30, v3 2023-04-13 (abs.html submission history).
- Authors: Mahmoud Assran (first) ... Nicolas Ballas (last); 8 authors, Meta AI (FAIR) + McGill/Mila/NYU.
- Venue: CVPR 2023 (main.tex `\confName{CVPR}` `\confYear{2023}`; README: "published @ CVPR-23").
  abs.html "Comments" field reads "2023 IEEE/CVF International Conference on Computer Vision" (ICCV wording);
  treat as loose wording, cite CVPR 2023.
- Repo: github.com/facebookresearch/ijepa (meta.json). README requires PyTorch 2.0, Python 3.8+ (code-only).

## 2. Core problem
Learn semantic image representations by self-supervision without hand-crafted view augmentations and
without reconstructing pixels. Invariance methods need augmentations that bake in biases; pixel/token
reconstruction methods learn lower-level features that need fine-tuning to be useful (Sec. 1).

## 3. What prior methods did and why that hurt (paper's account, Sec. 1-2)
- Invariance-based (SimCLR, BYOL, DINO, MSN, iBOT): two or more augmented views must embed the same.
  Augmentations encode priors that can hurt other tasks (classification vs segmentation need different
  invariances) and do not transfer to other modalities (audio). Multiple views per image cost compute.
- Generative / mask-denoising (MAE, BEiT, SimMIM, data2vec): fill in masked pixels or tokens. Less prior
  knowledge, but representations are lower semantic level; linear probing and low-shot are weak; need
  end-to-end fine-tuning. Pixel targets force the model to spend capacity on irrelevant detail.
- Closest predecessors: data2vec (predicts online-teacher representations of masked patches) and CAE.
  I-JEPA claims better efficiency and more semantic off-the-shelf features than both (Sec. 4).

## 4. The one key idea
From one context block, predict the representations (not pixels) of several large target blocks of the
same image, where targets come from an EMA target encoder applied to the full image. Two design choices
carry the result: predict in representation space, and multi-block masking with large (0.15-0.2 scale)
targets and a large (0.85-1.0 scale) spatially distributed context (abstract, Sec. 3).

## 5. Mechanism, step by step (Sec. 3, Appendix A)
1. Patchify image y into N non-overlapping patches (224 px, patch 16 -> 14x14 = 196; patch 14 -> 16x16 = 256).
2. Target encoder f_{theta-bar} (ViT, EMA copy, no gradient) processes ALL N patches ->
   s_y = {s_{y_1}, ..., s_{y_N}}, one D-dim vector per patch. Targets are the LAST-LAYER outputs; no
   normalisation is stated in paper or README.
3. Sample M = 4 target blocks B_1..B_4 on the patch grid: scale in (0.15, 0.2), aspect ratio in (0.75, 1.5).
   They may overlap each other. s_y(i) = {s_{y_j}}_{j in B_i}. The mask is applied to the OUTPUT of the
   target encoder, not its input (Table 11 shows this matters: 67.3 vs 56.1).
4. Sample one context block B_x: scale in (0.85, 1.0), UNIT aspect ratio (square). Remove every patch that
   lies in any target block. Average remaining context = 0.25 of the image's patches (Table 6, measured on
   ViT-B/16).
5. Context encoder f_theta (ViT, trainable) sees only the context patches -> s_x = {s_{x_j}}_{j in B_x}.
6. Predictor g_phi (narrow ViT, width 384, trainable) is run once per target block (M = 4 passes):
   input = s_x plus one mask token per target patch; mask token = shared learnable vector + positional
   embedding of that patch's location; output = hat-s_y(i), one vector per target patch.
7. Loss = mean over the 4 blocks of the summed squared L2 distance between predicted and target patch
   vectors (equation below). Gradient reaches phi (predictor) and theta (context encoder) only.
8. Target encoder update: EMA of context-encoder weights, momentum 0.996 -> 1.0 linearly over training;
   initialised identical to the context encoder. No equation given (see Sec. 6).
9. Optimiser AdamW; batch 2048; LR linear 1e-4 -> 1e-3 over the first 15 epochs then cosine to 1e-6;
   weight decay linear 0.04 -> 0.4; no [cls] token. Evaluation uses the TARGET encoder, average-pooled.
10. Batching detail: all context masks on one GPU share a size, all target masks on one GPU share a size,
   so tensors stay rectangular; the mask sampler runs in the data-loader collator (Appendix A, "Masking").

## 6. Equations
### Loss (Sec. 3, paragraph "Loss"; UNNUMBERED display equation, the only one in the paper)
```tex
\[
    \frac{1}{M} \sum^M_{i=1} {D}\left(\hat{\vs}_y(i), \vs_y(i)\right) = \frac{1}{M}  \sum^M_{i=1}\sum_{j \in B_i} \lVert\hat{\vs}_{y_j} - \vs_{y_j}\rVert^2_2.
\]
```
Macros (math_commands.tex): `\def\vs{{\bm{s}}}`, `\def\vm{{\bm{m}}}`, `\def\vx{{\bm{x}}}`, `\def\vy{{\bm{y}}}`.
Paper's wording: "The loss is simply the average L_2 distance between the predicted patch-level
representations hat-s_y(i) and the target patch-level representation s_y(i)".
Gloss: M = number of target blocks (4); B_i = set of patch indices in target block i; s_{y_j} = target
encoder output for patch j (D-dim); hat-s_{y_j} = predictor output for patch j; ||.||_2^2 = squared
Euclidean distance. NOTE (our reading): the prose says "average" but the formula SUMS over patches inside
a block and over the D dims, and averages only over the M blocks. The stepper should show the formula.
Predictor definition (same section): `\hat{\vs}_y(i)=\{\hat{\vs}_{y_j}\}_{j \in B_i} = g_\phi(\vs_x, \{\vm_j\}_{j \in B_i})`.

### EMA update: NOT an equation in the paper
Text only (Appendix A, "Optimization"): "The target-encoder weights are identical to the context-encoder
weights at initialization, and updated via an exponential moving average thereafter ... We use a momentum
value of 0.996, and linearly increase this value to 1.0 throughout pretraining, following [DINO, MSN]."
Our reading of the standard form: theta-bar <- m * theta-bar + (1 - m) * theta, m: 0.996 -> 1.0.
Attribute the written form to ch-ssl's BYOL extraction, not to this paper.

## 7. Config / setup (architecture lens)
### Stated in the paper
| item | value | loc |
|---|---|---|
| resolution | 224 x 224 (all models unless stated); ViT-H/16_448 at 448 x 448 | Sec. 5 p.4; Table 1 caption |
| patch size | 16 (B/16, L/16, H/16, G/16); 14 (H/14) | model names |
| encoders | "standard ViT architectures"; no [cls] token | App. A "Architectures" |
| ViT-L width | 1024 channels | App. "Predictor width" paragraph |
| predictor width | 384 (prose in App. "Predictor width" says "386 channels": typo) | App. A; Table 14 |
| predictor heads | equal to the backbone context encoder's heads | App. A |
| predictor depth | 6 (ViT-B/16); 12 (ViT-L/16, ViT-H/16, ViT-H/14); 16 (ViT-G/16) | App. A |
| mask tokens | shared learnable vector + positional embedding, one per target patch | Sec. 3 "Prediction" |
| targets | M = 4 blocks, scale (0.15, 0.2), aspect (0.75, 1.5), may overlap; mask applied to target-encoder OUTPUT | Sec. 3; Fig. 4 caption; App. A (writes "(015, 0.2)": typo) |
| context | 1 block, scale (0.85, 1.0), unit aspect ratio, minus all target patches; avg 0.25 of patches | Sec. 3; Table 6 |
| loss | mean over blocks of summed squared L2 (above) | Sec. 3 |
| EMA momentum | 0.996 -> 1.0, linear | App. A |
| optimizer | AdamW | App. A |
| batch | 2048 | App. A |
| LR | 1e-4 -> 1e-3 linear over first 15 epochs, then cosine to 1e-6 (paper writes 10^{-4}, 10^{-3}, 10^{-6}) | App. A |
| weight decay | 0.04 -> 0.4 linear (ablation: fixed 0.05, Table 13) | App. A; Table 13 |
| epochs | B/16 600; L/16 600; H/14 300; H/16_448 300 (IN-1k) | Table 1 |
| compute | ViT-H/14: 16 A100 GPUs, under 72 hours (abstract); "less than 1200 GPU hours" (Sec. 1) | abstract; Sec. 1 |
| augmentations | none: "without using any view data augmentations during pretraining" (Fig. 1 caption); "only requires processing a single view of each image" (Sec. 4) | Fig. 1; Sec. 4 |
| eval features | target encoder, average-pooled last layer, or concat of last 4 layers (best of) | App. A "Downstream" |
| linear probe IN-1k | LARS, batch 16384, 50 epochs, step decay /10 every 15 epochs | App. A |
| 1% low-shot | fine-tune 50 epochs, AdamW, batch 512, layer decay 0.75, label smoothing 0.1, lr 3e-5; WD 5e-2 (L/16), 4e-1 (H/14, H/16_448); ~12-13 images per class | App. A; Sec. 5 |

### code-only (README.md)
| item | value |
|---|---|
| ViT-H/14 run | "should be run on 16 A100 80G GPUs for an effective batch-size of 2048" (so 128 per GPU: our reading) |
| config files | configs/in1k_vith14_ep300.yaml, in1k_vith16-448_ep300.yaml, in22k_vith14_ep66.yaml, in22k_vitg16_ep44.yaml |
| released models | ViT-H/14 224 300 ep IN-1k; ViT-H/16 448 300 ep IN-1k; ViT-H/14 224 66 ep IN-22k; ViT-g/16 224 44 ep IN-22k |
| transforms | `src/transforms.py # pre-train data transforms` exists (contents not in sources) |
| predictor depth/width | NOT stated in README (only in paper) |

### Paper vs README differences
- IN-22k epochs: paper counts IN-1k-equivalent epochs (ViT-H/14 "equivalent of 900 IN1K epochs", ViT-G/16
  "600 IN1k epochs", Table 5 caption); README counts IN-22k epochs (66, 44). Ratio 13.6 both times;
  checkpoint file names carry "900e"/"600e". Same runs, different units.
- Naming: paper "ViT-G/16" (Table 5), README "ViT-g" (config `in22k_vitg16`). Width/depth of this model are
  in neither source.
- Table 5 caption says "The ViT-H/16 is trained for the equivalent of 600 IN1k epochs"; the row is ViT-G/16
  (typo).

### Not in the sources (do NOT fill from memory; fetch if the page needs them)
Encoder width/depth/heads for B/L/H (only "standard ViT" + ViT-L = 1024 are stated); predictor input/output
projections; positional-embedding type (learned vs sin-cos); any target normalisation; the loss function
actually used in code; per-GPU batch; yaml warmup/start LR; crop/flip transforms. Fetch
`src/train.py`, `src/models/vision_transformer.py`, `configs/in1k_vith14_ep300.yaml` from the repo.

### Computed shapes (our reading; encoder dims are the standard ViT values from Dosovitskiy et al. 2020,
### not stated in this paper; head dims follow from width / heads)
| | ViT-B/16 @224 | ViT-L/16 @224 | ViT-H/14 @224 | ViT-H/16 @448 |
|---|---|---|---|---|
| grid, N patches | 14x14 = 196 | 14x14 = 196 | 16x16 = 256 | 28x28 = 784 |
| patch vector | 3*16*16 = 768 | 768 | 3*14*14 = 588 | 768 |
| encoder width D / depth / heads | 768 / 12 / 12 (uncertain) | 1024 (paper) / 24 / 16 (uncertain) | 1280 / 32 / 16 (uncertain) | 1280 / 32 / 16 (uncertain) |
| encoder head dim | 64 | 64 | 80 | 80 |
| predictor width / depth / heads | 384 / 6 / 12 | 384 / 12 / 16 | 384 / 12 / 16 | 384 / 12 / 16 |
| predictor head dim | 32 | 24 | 24 | 24 |
| target block, patches (0.15-0.2 N) | 29-39 | 29-39 | 38-51 | 118-157 |
| context block before removal (0.85-1.0 N) | 167-196 | 167-196 | 218-256 | 666-784 |
| context after removal (0.25 N; measured on B/16 only) | ~49 | ~49 | ~64 (extrapolated) | ~196 (extrapolated) |

Tensor trace for one image, ViT-H/14 (our reading; B = per-GPU batch):
```
image [3,224,224] -> patchify [256,588] -> embed + pos [256,1280]
target enc (EMA, no grad): [256,1280] -> 32 layers -> s_y [256,1280] -> select block i: [n_i,1280], n_i in 38..51
context enc (train):       keep context patches [n_x,1280], n_x ~ 64 -> 32 layers -> s_x [n_x,1280]
predictor (train), x4:     s_x -> 384-d (projection implied by 384 != 1280) [n_x,384]
                           + mask tokens [n_i,384] = shared vector + pos-embed(j), j in B_i
                           -> 12 layers, 16 heads -> take the n_i mask positions -> back to 1280-d [n_i,1280]
loss:                      (1/4) * sum_i sum_{j in B_i} || hat_s[j] - s_y[j] ||^2   (red)
backward:                  d loss -> predictor phi, context encoder theta; target encoder gets none
update:                    AdamW(theta, phi); theta_bar <- m theta_bar + (1-m) theta; step LR, WD, m schedules
```
Iterations (our reading; IN-1k size 1.28M is not in the sources): 1,281,167 / 2048 = ~626 iterations per
epoch; 300 epochs = ~188k iterations.

## 8. Results worth remembering
- Table 1 (p.5), IN-1k linear probe top-1: I-JEPA ViT-B/16 600 ep 72.9; ViT-L/16 600 ep 77.5; ViT-H/14
  300 ep 79.3; ViT-H/16_448 300 ep 81.1. Baselines: MAE B/16 68.0, L/16 76.0, H/14 77.2 (1600 ep);
  data2vec L/16 77.3 (1600); CAE B/16 70.4, L/16 78.1 (1600); DINO B/8 80.1; iBOT L/16 81.0 (250 ep);
  SimCLR v2 RN152 (2x) 79.1.
- Table 2 (p.5), IN-1k 1% labels: I-JEPA ViT-L/16 69.4; ViT-H/14 73.3; ViT-H/16_448 77.3. MAE L/16 67.1,
  H/14 71.5; data2vec L/16 73.3; iBOT B/16 69.7; DINO B/8 70.0; SimCLR v2 70.2; BYOL 71.2; MSN B/4 75.7.
  Adapted by fine-tuning or linear probe, "whichever works best".
- Table 3 (p.6), linear transfer, ViT-H/14: CIFAR100 87.5, Places205 58.4, iNat18 47.6 (MAE H/14: 77.3,
  55.0, 32.9; data2vec L/16: 81.6, 54.6, 28.1; DINO B/8: 84.9, 57.9, 55.9; iBOT L/16: 88.3, 60.4, 57.3).
- Table 4 (p.6), low-level linear transfer, ViT-H/14: Clevr/Count 86.7, Clevr/Dist 72.4 (MAE H/14 90.5,
  72.4; data2vec 85.3, 71.3; DINO 86.6, 53.4; iBOT 85.7, 62.8). Beats view-invariance methods, not MAE on
  counting.
- Table 5 (p.7), IN-22k pretraining: H/14 89.5 / 57.8 / 50.5 / 88.6 / 75.0; G/16 89.5 / 59.1 / 55.3 / 86.7 /
  73.0 (CIFAR100 / Places205 / iNat18 / Count / Dist).
- Compute (text): "16 A100 GPUs in under 72 hours" (abstract); "less than 1200 GPU hours, which is over
  2.5x faster than a ViT-S/16 pretrained with iBOT and over 10x more efficient than a ViT-H/14 pretrained
  with MAE" (Sec. 1); "about 7% slower time per iteration" than MAE but "converges in roughly 5x fewer
  iterations" (Sec. 6). "ViT-H/14 requires less compute than a small iBOT model (ViT-S/16)".
- Figures 1 and 5 (GPU-hours axes, log scale): values are NOT printed; read from the figure PDFs' vector
  geometry (our reading, +-5%). Fig. 1 (p.1, linear eval): I-JEPA H/14 ~1170 h / 79.3; H/16_448 ~5600 h /
  81.1; iBOT S/16 ~3100 h / 77.9; CAE L/16 ~6900 h / 78.1; data2vec L/16 ~9900 h / 77.3; MAE H/14
  ~12300 h / 77.2. Fig. 5 (p.7, 1%): I-JEPA B/16 600 ep ~770 h / 62.4; H/14 ~2500 h / 73.3; H/16_448
  ~5600 h / 77.3; iBOT S/16 800 ep ~3100 h / 62.0, B/16 400 ep ~5000 h / 69.7; MAE L/16 ~6000 h / 67.1,
  H/14 ~12300 h / 71.5; data2vec L/16 ~9900 h / 73.3. DISCREPANCY: the I-JEPA ViT-H/14 point sits at
  ~1170 GPU hours in Fig. 1 but ~2500 in Fig. 5; all other methods agree across the two figures, and the
  text's 2.5x / 10x ratios match Fig. 1's placement. Cite the text values on the page, not the figures.
- Table 6 (p.8), masking strategy, ViT-B/16 300 ep, 1% linear probe: multi-block 54.2 (ctx ratio 0.25);
  rasterized 15.5 (0.25); block 20.2 (0.4); random 17.6 (0.4). Rasterized = one quadrant predicts the other
  three; block = single Block(0.6) target, context = complement; random = Random(0.6) patches, complement.
- Table 7 (p.8), targets, ViT-L/16, 1% linear probe: target-encoder output 66.9 (500 ep) vs pixels 40.7
  (800 ep). Pixel run trained LONGER and still loses.
- Table 8 (p.14), target scale, ViT-B/16 300 ep, 1% linear probe, 4 targets, ctx (0.85,1.0):
  (0.075,0.2) 19.2; (0.1,0.2) 39.2; (0.125,0.2) 42.4; (0.15,0.2) 54.2; (0.2,0.25) 38.9; (0.2,0.3) 33.6.
- Table 9 (p.14), context scale, same protocol: (0.40,1.0) 31.2; (0.65,1.0) 47.1; (0.75,1.0) 49.3;
  (0.85,1.0) 54.2.
- Table 10 (p.14), number of targets, same protocol: 1 -> 9.0; 2 -> 22.0; 3 -> 48.5; 4 -> 54.2.
  (Answers the outline's "tiny single patches" prompt only indirectly; Table 8's small-scale rows do it.)
- Table 11 (p.15), mask target-encoder output vs input, ViT-H/16 300 ep, 1% linear probe: 67.3 vs 56.1.
- Table 12 (p.15), predictor depth, ViT-L/16 500 ep, 1% linear probe: 6 layers 64.0; 12 layers 66.9.
- Table 13 (p.15), weight decay, ViT-L/16 600 ep: 0.04->0.4 gives 1% (fine-tune) 69.4, linear 77.8;
  fixed 0.05 gives 70.7, 76.4. NOTE: Table 1 lists the same ViT-L/16 600 ep linear probe as 77.5; Table 13
  says 77.8. Cite each with its table; do not reconcile.
- Table 14 (p.15), predictor width, ViT-L/16 600 ep, 1% FINE-TUNING (not linear): 384 -> 70.7; 1024 -> 68.4.
- Table 15 (p.16), full IN-1k fine-tune: I-JEPA ViT-H/16_448 300 ep 87.1 vs MAE ViT-H/14_448 1600 ep 87.8;
  "5.3 times less epochs".
- Protocol warning: Tables 7, 8-10, 11, 12 and Table 13's linear column use a frozen linear probe on 1%;
  Table 14 and Table 13's 1% column use fine-tuning. Do not put 70.7 next to 66.9 on one axis.
- Fig. 6 (p.8) / Sec. 7: RCDM decoder on average-pooled predictor output shows the predictor captures
  positional uncertainty and object parts in correct pose; discards low-level detail and background.

## 9. Limitations stated, and who fixed them
- The paper states no limitations section. Implicit ones it reports: ViT-G/16 does not improve low-level
  tasks (larger patches hurt local prediction, Sec. 6); counting still trails MAE (Table 4); EMA target
  encoder is "essential" (Sec. 3), i.e. collapse avoidance stays heuristic.
- Later sources in this survey that address them: V-JEPA (2404.08471) extends the recipe to video;
  LeJEPA (2511.08544) replaces the EMA/stop-gradient heuristics with SIGReg; V-JEPA 2.1 (2603.14482)
  adds dense per-token loss for spatial precision. (Mapping is the outline's; not claimed by this paper.)

## 10. Outline feedback (ch-ijepa and interlude)
1. "Equations: the loss (paper's Eq.)": the loss is an UNNUMBERED display equation; there is no equation
   number to cite. Say "Sec. 3, 'Loss'". "the EMA update": the paper has no EMA equation, only momentum
   values; attribute the written update to BYOL (ch-ssl) and cite I-JEPA for 0.996 -> 1.0.
2. Masking line should add: context block has UNIT aspect ratio (always square); target blocks may overlap
   each other; the context loses every patch inside any target. The playground's 14x14 grid is ViT-B/16;
   ViT-H/14 is 16x16.
3. "Results: IN-1k linear probe for ViT-H/14": give all four sizes (72.9 / 77.5 / 79.3 / 81.1), since the
   81.1 headline is ViT-H/16 at 448, not H/14. The brief's example claim (77.3, "Table 1, row ViT-H/14 448")
   is wrong on both counts: 77.3 is Table 2 (1% low-shot) for ViT-H/16_448.
4. "compute vs MAE and iBOT (GPU hours)": the paper prints no GPU-hour numbers for baselines; only ratios
   (2.5x iBOT ViT-S/16, 10x MAE ViT-H/14) and "<1200 GPU hours". Figures 1 and 5 disagree on I-JEPA's own
   H/14 point (~1170 vs ~2500 h, our geometry reading). Use the text ratios; if a chart is drawn, mark it
   "read from Fig. 1 geometry" and show the ratios, not absolute hours.
5. "ablation table rows: predictor depth": Table 12 is ViT-L/16 at 500 ep (6 vs 12 layers: 64.0 vs 66.9).
   Add the target-count ablation (Table 10: 1 target 9.0 -> 4 targets 54.2) and target-scale ablation
   (Table 8) since the chapter's prediction prompt ("tiny single patches") is answered by those, not by
   the masking-strategy table. Table 11 (output vs input masking, 67.3 vs 56.1) backs the stepper's
   "target encoder sees the whole image" step; worth one line.
6. ArchFlow: encoder widths/depths/heads are not in the paper or README (only ViT-L = 1024). Label them as
   standard ViT values, or fetch `src/models/vision_transformer.py` to make them code-only. Predictor
   in/out projections are implied, not stated.
7. Interlude: per-GPU batch (128 = 2048/16) and iterations per epoch (~626) are our reading; IN-1k size is
   not in the sources. Table 6's 0.25 context ratio was measured on ViT-B/16; for H/14 it is extrapolated.
8. Paper typos to preserve as-is when quoting: "(015, 0.2)" (App. A masking), "386 channels" (App.
   predictor width prose; table says 384), Table 5 caption "ViT-H/16" for the ViT-G/16 row.
