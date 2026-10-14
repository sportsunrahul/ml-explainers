# DN-DETR and DINO (one chapter)

## Sources and caveats

- **DN-DETR**: Li, Zhang, Liu, Guo, Ni, Zhang. "DN-DETR: Accelerate DETR Training by Introducing Query DeNoising". arXiv 2203.01305v3 (last revised 8 Dec 2022). The v3 source is the **journal extension** (IEEEtran compsoc) of the CVPR 2022 oral paper, so it has extra models (DN-Deformable-DETR++, DN-Vanilla-DETR, DN-Anchor-DETR, DN-Faster R-CNN, DN-Mask2Former). Read: `sources/dn-detr/tex/main.tex` plus every `\input`; numbering checked against `paper.pdf`.
- **DINO**: Zhang, Li, Liu, Zhang, Su, Zhu, Ni, Shum. "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection". arXiv 2203.03605v4 (last revised 11 Jul 2022). This is the detection DINO, not the self-supervised DINO of Caron et al. Read: `sources/dino/tex/main.tex` plus every `\input`; numbering checked against `paper.pdf`.
- **Code links as printed in the papers**: DN-DETR abstract gives `https://github.com/IDEA-Research/DN-DETR` (not "IDEA-opensource"). DINO abstract gives `https://github.com/IDEACVR/DINO`, which now 301-redirects to `https://github.com/IDEA-Research/DINO`. Anything tagged **[repo]** below comes from the current main-branch code or README, not from the paper, and may not be the exact code behind the paper's numbers.
- **DN-DETR PDF numbering traps** (cite carefully):
  - The tex does `\setcounter{table}{0}` before the speedup table, so the PDF has **two table series**: Tables 1-6 on pp. 7-10 (main results, 1x setting, extensions, SOTA, two ablations), then a second "Table 1" (25 vs 50 epoch speedup, p. 11), second "Table 2" (training wall-clock, p. 11), second "Table 3" (known-object detection, p. 11), second "Table 4" (known-label detection, p. 12). Claim locs below say "Table 1b" etc. for the second series.
  - Label `fig:vib` is defined twice. The IS plot is **Fig. 2** by its PDF caption, but the body text says "Fig. 3 shows a comparison of IS" (Fig. 3 is actually the anchor-target L1 distance plot).
  - Eq. (2) sums over index `j` but the summand uses `n` (typo in the paper).
- **DINO notation trap**: DINO reuses the names lambda_1, lambda_2 with a **different meaning** from DN-DETR. In DN-DETR they are the center-shift and box-scale noise scales. In DINO they are the inner (positive) and outer (negative) thresholds of contrastive denoising. See "DINO noise scales" below.

---

# Part 1: DN-DETR

## Problem: bipartite matching is unstable early in training

**Diagnosis (Sec. 1, Sec. 3.1).** DETR needs 500 epochs on COCO versus 12 for the original Faster R-CNN [dn-500ep]. Prior work blamed the architecture (cross-attention, query design). DN-DETR argues the slow convergence **also** comes from the discrete Hungarian (bipartite) matching step, which is unstable "especially in the early stages of training due to the nature of stochastic optimization": "for the same image, a query is often matched with different objects in different epochs, which makes optimization ambiguous and inconstant." A small change in the cost matrix can cause "an enormous change in the matching result" (citing work showing Hungarian matching is not a stable matching because blocking pairs exist).

**Two-stage view (Sec. 3.1).** Training a DETR-like model = learning "good anchors" plus learning relative offsets. Queries learn the anchors; if the anchor a query is responsible for keeps changing, the offset is hard to learn. A noised GT box is a "good anchor" with a GT box nearby, so its target is unambiguous.

**Disagreement with TSP (Sun et al.).** Sun et al. fed a pretrained DETR's assignments to a student and found it only helped early convergence, concluding the Hungarian loss is not the main reason for slow convergence. DN-DETR says it reaches "a different conclusion" (Sec. 2.2).

**How they measure instability: the IS metric (Sec. 3.1).** For one training image, epoch i:
- predictions O^i = {O_0^i, ..., O_{N-1}^i} (N = number of predicted objects), GT T = {T_0, ..., T_{M-1}}.
- after matching, index vector V^i: V_n^i = m if prediction n matched GT m, else -1 (Eq. 1).
- IS^i = number of query slots whose assignment changed between epoch i-1 and epoch i (Eq. 2); averaged over all images.
- Evaluated on COCO 2017 val, which has 7.36 objects per image on average, so "the largest possible IS is 7.36 x 2 = 14.72" [dn-is-objects-per-image, dn-is-max]. (Our reading of the factor 2, not stated in the paper: each GT that switches owner changes two slots, the old owner and the new owner.)
- Fig. 2 (PDF caption): IS of DAB-DETR vs DN-DETR, both trained 12 epochs on the same setting, measured between each pair of consecutive epochs on the validation set. The text only says DN-DETR "effectively alleviates the instability"; per-epoch values are not given in the text [dn-is-fig].

**Second symptom: queries search too far (Sec. 3.2).** Mean l1 distance between initial anchors (positional queries) and the matched GT boxes in the last decoder layer is smaller for DN-DETR than DAB-DETR (Fig. 3; y-axis 0.08-0.15, exact values not in text; read off the plot, DAB-DETR falls from ~0.146 at epoch 1 to ~0.096 at epoch 12 and DN-DETR from ~0.110 to ~0.082) [dn-dist-fig]. Fig. 4 shows arrows from anchor to matched target; DN-DETR arrows are shorter. Interpretation: denoising teaches each query to search locally, which "prevents potential prediction conflicts between queries".

## Key idea

Besides the normal matching queries, feed **noised copies of the ground-truth boxes and labels** into the decoder as extra queries, and train them to reconstruct the original GT. These denoising queries skip Hungarian matching entirely: query k's target is simply the GT it was made from. This is an easier auxiliary task with a fixed target, and it teaches box refinement from nearby anchors. The denoising part is removed at inference, so test-time cost is unchanged [dn-no-test-cost].

## Mechanism

**Base model.** DAB-DETR: queries are 4D anchor boxes (x, y, w, h), refined layer by layer by predicted (dx, dy, dw, dh). The only architectural change: the decoder embedding (content query) is specified as a **class label embedding**, so labels can be noised too (Sec. 4.1, 4.2, Fig. 5).

**Two query parts (Sec. 4.1).**
- Matching part Q = {Q_0, ..., Q_{L-1}}: learnable anchors, Hungarian matching, same loss as DETR.
- Denoising part q = {q_0, ..., q_{K-1}}: noised GT box-label pairs ("GT objects"); outputs must reconstruct the GT objects.

**Box noise (Sec. 4.3).** Two kinds, each with its own scale:
1. **Center shifting**: add (dx, dy) with |dx| < lambda_1 w / 2 and |dy| < lambda_1 h / 2, lambda_1 in (0, 1), so the noised center stays inside the original box.
2. **Box scaling**: new width sampled in [(1 - lambda_2) w, (1 + lambda_2) w], height in [(1 - lambda_2) h, (1 + lambda_2) h], lambda_2 in (0, 1).
- Noise is **uniform** [dn-uniform-noise]. Defaults: lambda_1 = 0.4, lambda_2 = 0.4 [dn-lambda1, dn-lambda2].
- [repo] `IDEA-Research/DN-DETR` `models/dn_dab_deformable_detr/dn_components.py`: box in (cx, cy, w, h); offset = uniform[-1, 1) * (w/2, h/2, w, h) * box_noise_scale, then clamp to [0, 1]. Default `--box_noise_scale 0.4` in `main.py`. This matches the paper with lambda_1 = lambda_2 = box_noise_scale [dn-repo-box-noise].

**Label noise (Sec. 4.3).** **Label flipping**: randomly flip some GT labels to other classes, ratio gamma = 0.2 [dn-gamma]. Forces the model to use the (noised) box to recover the correct label, i.e. learn the label-box relationship. [repo] default `--label_noise_scale 0.2`.

**Losses (Sec. 4.3).** Reconstruction: l1 + GIoU for boxes, focal loss for labels, as in DAB-DETR. Plus the usual Hungarian loss on the matching part.

**Label embedding details (Sec. 4.5).** 80 COCO class embeddings plus an **unknown-class embedding** used as the content query of the matching part (to be "semantically consistent" with the denoising part). An **indicator** bit is appended: 1 for denoising queries, 0 for matching queries.

**Denoising groups (Sec. 4.4).** Multiple noised versions of all GT objects: q = {g_0, ..., g_{P-1}}, each group g_p holds M queries, one per GT object (M = number of GT objects in the image). Default 5 groups [dn-groups-default]; DN-Deformable-DETR uses 10, DN-Deformable-DETR++ 5, Faster R-CNN 10, Anchor DETR 5 [dn-groups-variants].

**Attention mask (Sec. 4.4).** Two leakage paths must be blocked in decoder self-attention:
1. the matching part must not see the denoising part (otherwise it would just copy nearly-GT boxes);
2. one denoising group must not see another group (another noised copy of the same GT would leak the answer).
- Denoising queries seeing matching queries is harmless, since matching queries are learned and contain no GT information.
- Without the mask, denoising **hurts**: AP 24.0 vs 43.4 with it (Table 5) [dn-abl-full, dn-abl-nomask].
- Cost: with 5 groups, training GFLOPs rise from 94.4 to 94.6 for DAB-DETR R50 (Sec. 4.4 text) [dn-gflops-text]; the wall-clock table reports 94.5 [dn-train-gflops-table]. Zero test overhead.

**Extensions (Sec. 4.6-4.9).** Deformable DETR (queries specified as 4D boxes; "++" adds deformable attention in the decoder too), Anchor DETR (2D points, center shifting only), vanilla DETR (linear box embedding of the noised box plus label embedding), Faster R-CNN (noised boxes fed to the detection head alongside RPN boxes; box denoising only), Mask2Former (shift GT masks, feed as attention mask).

## Equations (verbatim from main.tex, numbers from the PDF)

Eq. (1), matching index vector:
```latex
V^i_n=\left\{\begin{array}{ll}
 m, &\text{if } O^i_n \text{ matches } T_m\\
 -1, &\text{if } O^i_n \text{ matches nothing}
\end{array}\right.
```
Words: for epoch i, slot n records which GT the n-th prediction was matched to, or -1.

Eq. (2), instability:
```latex
IS^i=\sum_{j=0}^{N}\mathbbm{1}(V^i_n\neq V^{i-1}_n)
```
Words: count the slots whose assignment differs from the previous epoch. (Paper typo: sum index j, summand index n; read both as n.) Averaged over images for the dataset value.

Eq. (3), decoder with mask:
```latex
\mathbf{o}=D(\mathbf{q},F|A)
```
Words: decoder D maps queries q and encoder features F to outputs o, under attention mask A.

Eq. (4), with the two parts separated:
```latex
\mathbf{o}=D(\mathbf{q},\mathbf{Q},F|A)
```
Words: q = denoising part, Q = matching part, decoded jointly in one pass.

Eq. (5) and (6), groups:
```latex
\mathbf{q}=\left\{\mathbf{g_0}, \mathbf{g_1}, ..., \mathbf{g_{P-1}}\right\}
\mathbf{g_p}=\left\{q^p_0, q^p_1, ..., q^p_{M-1} \right\}
```
where q^p_m = delta(t_m), delta = the noising function, t_m = m-th GT object.

Eq. (7), attention mask, A = [a_ij] of size W x W with W = P x M + N, first P x M rows/cols = denoising part, a_ij = 1 means query i cannot see query j:
```latex
a_{ij}=\left\{\begin{array}{ll}
1, & \text { if } j<P\times M \text{ and } \lfloor\frac{i}{M}\rfloor \neq \lfloor\frac{j}{M}\rfloor;\\
1, & \text { if } j<P\times M \text{ and } i\geq P\times M;\\
0, & \text{otherwise.}
\end{array}\right.
```
Words: block row i from seeing a denoising column j if j is in a different group from i (line 1) or if i is a matching query (line 2). Everything else is visible. Note N here is the number of matching queries (the text reuses N/L).

## Results (COCO val2017, trained on train2017)

Setup: 6 enc + 6 dec layers, hidden 256, lr 1e-4, AdamW wd 1e-4, 8 A100, batch 16, 50-epoch schedule drops lr at epoch 40, 12-epoch schedule at epoch 11 [dn-setup]. All DETR-like models use 300 queries except DETR (100) [dn-queries].

**Same-setting gains over DAB-DETR, single scale, 50 epochs (Table 1, p. 7).**
- R50: 42.2 -> 44.1 (+1.9) [dn-t1-r50]
- R101: 43.5 -> 45.2 (+1.7) [dn-t1-r101]
- DC5-R50: 44.5 -> 46.3 (+1.8) [dn-t1-dc5r50]
- DC5-R101: 45.8 -> 47.3 (+1.5) [dn-t1-dc5r101]
- Same GFLOPs and params as DAB-DETR (94 GFLOPs, 44M for R50) [dn-t1-r50].

**12-epoch (1x) setting (Table 2, p. 8).** Sec. 5.1: the Table 2 and Table 4 comparisons use 3 pattern embeddings (as in DAB-DETR / Anchor DETR), although the rows are not starred; e.g. DAB-DETR-DC5-R50 here has 216 GFLOPs, the DAB-DETR paper's figure for its 3-pattern DC5-R50* model.
- DN-DETR-DC5-R50 41.7 (+3.7 over DAB-DETR-DC5-R50 38.0) [dn-t2-dc5r50]
- DN-DETR-DC5-R101 42.8 (+2.5 over 40.3) [dn-t2-dc5r101]
- DN-Deformable-DETR-R50 43.4; DN-Deformable-DETR-R50++ 46.0 [dn-t2-deform, dn-t2-deformpp]
- DN-Deformable-DETR-R101 44.1 at 12 epochs vs Faster R-CNN R101 FPN 44.0 at 108 epochs ("9x faster") [dn-t2-r101-vs-frcnn]
- The abstract's "46.0 AP ... 12 epochs" and "49.5 AP ... 50 epochs" with R50 are **DN-Deformable-DETR-R50++**, not plain DN-DETR [dn-abs-46-49.5].

**Multi-scale, 50 epochs, R50 (Table 4, p. 9).**
- DN-Deformable-DETR-R50 48.6 vs DAB-Deformable-DETR-R50 46.9 (+1.7) [dn-t4-deform]
- DN-Deformable-DETR-R50++ 49.5 [dn-t4-deformpp]
- vs Dynamic DETR-R50 47.2 (test-dev, 5 scales + FPN): +1.4 [dn-t4-dynamic]

**2x speedup (Table 1b, p. 11).** DN at 25 epochs matches baseline at 50: DC5-R50 44.4 vs 44.5; Deformable-R50 46.8 vs 46.9; Deformable-R50++ 48.4 vs 48.7; DN-Vanilla-DETR-R50 at 250 epochs 42.2 vs DETR-R50 at 500 epochs 42.0 [dn-t1b-*]. Fig. 7a: before lr drop, DN-DETR (DC5-R50) reaches 40 AP in 20 epochs, DAB-DETR needs 40 [dn-fig7a]. Fig. 7b: DN-Deformable-DETR reaches 47.8 AP in 30 epochs, 0.9 above converged DAB-Deformable-DETR [dn-fig7b].

**Wall clock (Table 2b, p. 11).** DAB-DETR-R50 2555 min for 50 epochs vs DN-DAB-DETR-R50 1443 min for 25 epochs; 51.1 vs 57.7 min per epoch [dn-t2b-time, dn-epoch-time].

**Ablations, R50, 50 epochs.**
- Table 5 (1 denoising group): box+label+mask 43.4; box+mask 43.0; mask only (no denoising) 42.2; box+label without mask 24.0 [dn-abl-*].
- Table 6 (groups: none / 1 / 5): R50 42.2 / 43.4 / 44.1; R50-DC5 44.5 / 45.6 / 46.3; R101 43.5 / 45.0 / 45.2; R101-DC5 45.8 / 46.5 / 47.3. Gains diminish with more groups [dn-t6-*].
- Fig. 8 noise-scale sweep (20 epochs, batch 64, R50, no lr drop; the other scale fixed at 0.4): both noises help; too much noise hurts. Figure annotations: best 35.5 AP at center-shift lambda_1 = 0.2 and best 35.7 AP at box-scale lambda_2 = 0.4; the sweep runs 0.0 to 0.8 [dn-fig8].

**Generality (Table 3, p. 8), 12 epochs unless noted.** Anchor-DETR-DC5-R50 38.2 -> 39.4 (+1.2); Group-DAB-DETR 41.9 -> 44.5 (+2.6); Faster R-CNN-FPN-R50 37.9 -> 38.4 (+0.5); SAM-DETR++ 43.2 -> 44.8 (+1.6); **DINO-R50 w/o DN 46.0 -> w/ DN 47.4 (+1.4)** (note: DINO's own Table 7 reports 45.1 for 'No DN'; 46.0 there is its '10 DN' row); vanilla DETR-R50 300 epochs 40.6 -> 42.6 (+2.0); Mask DINO 40.7 -> 41.4 (+0.7); Mask2Former 38.7 -> 39.7 (+1.0) [dn-t3-*]. Text: "1.2-2.6 AP" for DETR-like detectors [dn-t3-range].

## Limitations (paper-stated)

Noise is only sampled from a uniform distribution; more complex noising schemes are left to future work (Sec. 6). Known-object / known-label experiments are "preliminary" and not on the best-tuned model (Sec. 5.7.1).

---

# Part 2: DINO

## Delta from DN-DETR, DAB-DETR, Deformable DETR

DINO = DAB-DETR's dynamic 4D anchor queries + DN-DETR's denoising + Deformable DETR's deformable attention, multi-scale features, query selection ("two-stage") and iterative box refinement ("look forward once"), plus three new pieces (Sec. 3.1-3.2):
1. Contrastive DeNoising (CDN)
2. Mixed query selection
3. Look forward twice

Other engineering (Appendix D.3): 900 queries (= DN-DETR's 300 x 3 patterns, "same computation cost") [dino-900q]; shared prediction head across decoder layers (about one million fewer params) [dino-shared-head]; DAB-DETR's conditional queries dropped; **dynamic DN groups**: fix the total number of DN queries and set the number of groups per image from its object count, since COCO images have 1 to 80 objects and padding to the batch max wastes memory (App. D.1) [dino-dyn-groups]. DINO's summary of DN-DETR's noise (Sec. 3.1) writes a single lambda with |dx| < lambda w/2, |dy| < lambda h/2, |dw| < lambda w, and (typo) |dy| < lambda h (should be dh); footnote says DN-DETR set lambda_1 = lambda_2.

## Idea 1: Contrastive denoising (Sec. 3.3)

**Why.** DN teaches the model to refine anchors that have a GT nearby, but "it lacks a capability of predicting 'no object' for anchors with no object nearby." CDN adds hard negatives so the model learns to **reject** anchors.

**Mechanism.**
- Two hyperparameters lambda_1 < lambda_2 (Fig. 3, concentric squares around a GT box).
- **Positive** query: noise scale < lambda_1 (inside the inner square). Target: reconstruct its GT box and label.
- **Negative** query: noise scale in (lambda_1, lambda_2) (between squares). Target: "no object" (background), via focal loss.
- "We usually adopt small lambda_2 because hard negative samples closer to GT boxes are more helpful."
- Each CDN group has 2 x n queries for n GT boxes (one positive and one negative per GT). Multiple CDN groups, as in DN-DETR. Losses: l1 + GIoU for boxes, focal for classes, focal for negatives-as-background.
- The squares are a 2D illustration of a 4D region (Fig. 3 caption).
- Setting (App. D.3): lambda_1 = 1.0, lambda_2 = 2.0, 100 CDN pairs (100 positive + 100 negative queries) [dino-cdn-lambdas, dino-cdn-pairs]. Large (SwinL) model: 1000 DN queries [dino-swinl-dn].

**DINO noise scales: how lambda_1 = 1.0, lambda_2 = 2.0 fit with dn_box_noise_scale = 0.4 (Table 8).** The paper does not define the norm behind "noise scale", and lists both lambda_1 = 1.0 / lambda_2 = 2.0 (text) and dn_box_noise_scale = 0.4 (Table 8). The official code reconciles them [dino-repo-cdn-noise, uncertain]:
- [repo] `IDEA-Research/DINO` `models/dino/dn_components.py` `prepare_for_cdn`: convert GT (cx, cy, w, h) to corners (x1, y1, x2, y2). For each of the 4 corner coords draw u ~ Uniform[0, 1); for negatives add 1.0 (so u in [1, 2)); multiply by a random sign; offset = u * (w/2 or h/2) * box_noise_scale (0.4 in `config/DINO/DINO_4scale.py`). Clamp to [0, 1], convert back to (cx, cy, w, h).
- So in practice, per corner coordinate: **positive** = shift of less than 0.4 x w/2 = 0.2 w (0.2 h vertically); **negative** = shift between 0.2 w and 0.4 w (every one of the 4 coordinates is pushed into the outer shell). lambda_1 = 1.0 and lambda_2 = 2.0 act as multipliers on the base scale 0.4.
- Positive/negative is decided by construction (which sample it is), not by measuring IoU afterwards. A negative is always further out than any positive on every coordinate, before clamping.
- This differs from DN-DETR, which perturbs center and size (not corners).
- Label noise [repo]: `dn_label_noise_ratio = 0.5` (Table 8 also says 0.5), but code flips a label when p < 0.5 x 0.5, i.e. effective flip rate 0.25 [dino-label-noise, uncertain].
- Also [repo] `dn_number = 100` in the 4-scale config.

**Why it helps (Sec. 3.3 Analysis).** When several anchors are near one object the model is "confused": (1) duplicate predictions (Fig. 8: with DN queries instead of CDN the boy has 3 duplicate boxes) [dino-fig8-dup]; (2) a farther anchor may be selected. CDN teaches the model to prefer the closer anchor and reject the others.

**Metric: Average Top-K Distance, Eq. (1).** For each GT box b_i take the initial decoder anchor a_i whose final refined box was matched to it; ATD(k) = mean of the k largest l1 distances ||b_i - a_i||_1. Top-K because confusion is likelier when a GT is matched with a far anchor. Fig. 4 (ATD(100)): DN is "good enough" overall, CDN finds better anchors for small objects; Fig. 4c: CDN gives +1.3 AP_S over DN at 12 epochs, R50, multi-scale [dino-cdn-aps].

## Idea 2: Mixed query selection (Sec. 3.4)

- Static queries (DETR, DN-DETR, DAB-DETR): positional queries/anchors learned from data, same for every image; content queries all 0 vectors (DN-DETR/DAB-DETR) (Fig. 5a). Deformable DETR learns both positional and content queries statically.
- Pure query selection (Deformable DETR "two-stage", Efficient DETR): pick top-K encoder features (by class/objectness score) from the last encoder layer; **both** positional and content queries come from a linear transform of those features; an auxiliary head's boxes initialize reference boxes (Fig. 5b).
- **Mixed** (DINO, Fig. 5c): use the top-K selected features' **boxes only** to initialize the anchor (positional) queries; keep the **content queries learnable (static)**. Reason: selected encoder features are "preliminary content features without further refinement" and may cover multiple objects or only part of one, so using them as content "could be ambiguous and misleading". Better positions let the decoder pool more comprehensive content from the encoder.
- Extra intermediate losses after the query selection module, same components as decoder layers (App. D.3).

## Idea 3: Look forward twice (Sec. 3.5)

- Deformable DETR's iterative box refinement detaches the gradient between layers to stabilize training: layer i's parameters are updated only from the loss on its own box b_i ("look forward once", Fig. 6a).
- DINO conjecture: the better box from layer i+1 can help correct layer i. So layer i's parameters get gradients from the losses of layer i **and** layer i+1 (Fig. 6b).
- A layer's final box quality depends on the input box b_{i-1} and the predicted offset delta b_i. Look forward once optimizes only the offset. Look forward twice also improves the input box: layer i+1's prediction is computed from the **undetached** b_i', so its loss flows back into layer i.
- Each offset delta b_i is used twice: once for b_i' and once for b_{i+1}^(pred), hence the name.

Eq. (2), verbatim:
```latex
\begin{aligned}
     \Delta b_{i} &= \mathrm{Layer_{i}}(b_{i-1}), \quad\quad
     &b_{i}' &= \mathrm{Update}(b_{i-1}, \Delta b_{i}), \\
     b_{i} &= \mathrm{Detach}(b_{i}'), \quad\quad
     &b_{i}^{(\mathrm{pred})} &= \mathrm{Update}(b_{i-1}', \Delta b_{i}),
\end{aligned}
```
Words: layer i reads the detached box b_{i-1} and predicts an offset. b_i' (with gradient) and b_i (detached copy) are the updated box; b_i is what the next layer takes as input. The box that is **supervised** at layer i, b_i^(pred), is the offset applied to the **undetached** previous box b_{i-1}', so layer i's loss reaches layer i-1's offset. Update(b, d) = sigmoid(inverse_sigmoid(b) + d) on normalized boxes, as in Deformable DETR: the head's raw output d is already an offset in logit space and is not passed through inverse_sigmoid (footnote to Sec. 3.5 loosely says "we sum them after inverse sigmoid"; [repo] DINO `deformable_transformer.py`: `delta_unsig + inverse_sigmoid(reference_points)` then `.sigmoid()`).

## Other equations

Eq. (1), ATD:
```latex
ATD(k)=\frac{1}{k}\sum\left\{\mathop{topK}\left(\left\{\lVert b_0-a_0 \rVert_{1}, \lVert b_1-a_1 \rVert_{1}, ..., \lVert b_{N-1}-a_{N-1} \rVert_{1}\right\},k\right)\right\}
```
Eq. (3), one-to-one TTA ensembling (App. A): for each prediction of the main augmentation, take the highest-IoU box from each other augmentation above a threshold, then
```latex
b=\frac{1}{\sum I^i}\sum_{i=o}^{n-1} I^iw^is^i_{idx(i)}b^i_{idx(i)}
```
(weighted average; w^i = augmentation weight, s = score, I^i = 1 if augmentation i had a box over the IoU threshold). Used only for the SwinL model.

## Results (COCO val2017 unless noted)

Setup: R50 on ImageNet-1k, trained on train2017 only (no extra data). lr 1e-4, drops at epoch 11 / 20 / 30 for the 12 / 24 / 36 epoch settings; AdamW wd 1e-4; batch 16 [dino-lr-sched]. 4-scale = backbone stages 2, 3, 4 + a downsampled extra map; 5-scale adds stage 1 [dino-scales]. Note v4's R50 numbers are higher than v1's "due to engineering techniques" (Sec. 4.2).

**ResNet-50, 12 epochs (Table 1), no extra data.**
- DINO-4scale 49.0 (+5.6), AP_S 32.0 (+7.2), 279 GFLOPS, 47M, 24 FPS [dino-t1-4s]
- DINO-5scale 49.4 (+6.0), AP_S 32.3 (+7.5), 860 GFLOPS, 47M, 10 FPS [dino-t1-5s]
- Deltas are vs DN-Deformable-DETR(4scale) 43.4 at 12 epochs [dino-t1-dn]. Other rows: Deformable DETR(4scale) 41.1, Dynamic DETR(5scale) 42.9, DyHead 43.0, HTC 42.3, Faster-RCNN 37.9 [dino-t1-others].
- GFLOPS/FPS measured on the same A100 for all rows. (DN-Deformable-DETR is listed at 265 GFLOPS here vs 195 in the DN-DETR paper; not explained, but DINO's row is daggered as a 900-query / 300 x 3 pattern model.) [dino-t1-dn-gflops]

**ResNet-50, 24 and 36 epochs (Table 2), no extra data.**
- 24 epochs: 4scale 50.4 (+1.8), 5scale 51.3 (+2.7) [dino-t2-24]
- 36 epochs: 4scale 50.9 (+2.3), 5scale 51.2 (+2.6) [dino-t2-36]
- Deltas are vs **DN-Deformable-DETR at 50 epochs (48.6)**, so the 24-epoch comparison is DINO at under half the epochs [dino-t2-baseline]. The abstract's "+2.7 AP in 24 epochs compared to DN-DETR" is this comparison.
- 5scale at 36 epochs (51.2) is slightly below 24 epochs (51.3).
- The paper skips 50 epochs because DINO "converges faster and yields only a smaller additional gain with 50-epoch training".

**Swin-L + Objects365 pretraining (Table 3): EXTRA DATA.**
- DINO-SwinL, 218M params [dino-t3-params], backbone pre-trained on IN-22K-14M, detector pre-trained on O365, then fine-tuned on COCO: val2017 63.1 w/o TTA, 63.2 w/ TTA; test-dev 63.2 w/o TTA, 63.3 w/ TTA [dino-t3-val, dino-t3-testdev]. The headline 63.2 / 63.3 are **with TTA**.
- Comparisons in Table 3: SwinV2-G 3.0B params, test-dev 63.1 w/ TTA; Florence-CoSwin-H test-dev 62.4; GLIP 61.5; Soft Teacher+SwinL 61.3; DyHead 60.6; SwinL (HTC++) 58.7 [dino-t3-others]. All prior SOTA use non-Transformer-decoder heads (HTC++, DyHead); DINO is the only "end-to-end" row.
- Paper claims 1/15 the params of SwinV2-G, 1/60 the backbone pre-training images and 1/5 the detection pre-training images of Florence [dino-ratios]. Objects365: 1.7M annotated images [dino-o365-size].
- Recipe (App. D.2): O365 pre-training 26 epochs on 64 A100; COCO fine-tuning 18 epochs on 16 A100; batch 1 per GPU; fine-tune image size 1.5x (max 1200 x 2000, adds about 0.5 AP); checkpointing + mixed precision; 1000 DN queries; first 5,000 of 80,000 O365 val images kept for validation, rest added to training [dino-swinl-recipe].
- [repo] README (2022/8/6 entry): a SwinL 5-scale model "without any tricks" (no O365, no large image size, no multi-scale test) gets 58.5 AP on COCO val [dino-readme-swinl-noextra]. Not in the paper.

**Ablation (Table 4), R50.** Setting not stated in the caption; 47.9 for the full model matches Table 5 (12 epochs) and Table 7 (100 CDN), so almost certainly 12 epochs, 4-scale, pre-engineering-update numbers [dino-abl-setting, uncertain].
1. DN-DETR 43.4
2. Optimized DN-DETR 44.9 (+1.5)
3. Strong baseline = Row 2 + pure query selection 46.5 (+1.6)
4. + mixed query selection 47.0 (+0.5)
5. + look forward twice 47.4 (+0.4)
6. + contrastive DN = DINO 47.9 (+0.5) [dino-abl-*]
- Oddity: AP_S drops from 31.1 (Row 4) to 29.9 (Row 5), then 31.2 with CDN (Row 6).

**Denoising query count (Table 7).** 100 CDN pairs 47.9; 1000 DN 47.6; 200 DN 47.4; 100 DN 47.4; 50 DN 46.7; 10 DN 46.0; no DN 45.1. Gains saturate after 100 [dino-t7].

**Encoder/decoder depth (Table 6), 12 epochs, 100 DN without negatives.** 6/6 47.4; 4/6 46.2; 3/6 45.8; 2/6 45.4; 6/4 46.0; 6/2 44.4; 2/4 44.1; 2/2 41.2. Decoder 6 -> 2 costs 3.0 AP, vs 13.8 AP (42.9 -> 29.1) for Dynamic DETR; attributed to mixed query selection giving a good initialization [dino-t6, dino-t6-drop].

**Training cost (Table 5), R50, 8 A100.** DINO: batch 2/GPU, about 55 min/epoch, 16GB, 12 epochs, 47.9 AP. Deformable DETR (vanilla, without two-stage or box refinement): same batch/time/memory, 50 epochs, 45.4. DETR: about 16 min/epoch, 26GB, 300 epochs, 41.2. Faster RCNN: about 60 min/epoch, 13GB, 108 epochs, 42.0 [dino-t5].

**Loss/matching hyperparameters (App. D.3, Table 8).** Focal alpha 0.25, gamma 2; loss coefs class 1.0, L1 5.0, GIoU 2.0; matching costs class 2.0, bbox 5.0, GIoU 2.0; 8 heads, FFN 2048, dropout 0.0, 4 sampling points in enc and dec [dino-t8].

## Limitations

DINO v4 has **no Limitations section**. Paper-stated caveats only:
- TTA "does not obtain an inspiring gain" (App. A); the gain is 0.1 AP on both val and test-dev (Table 3) [dino-tta-gain].
- 5-scale costs far more compute than 4-scale for +0.4 AP at 12 epochs: 860 vs 279 GFLOPS, 10 vs 24 FPS (Table 1) [dino-4v5-gap, dino-t1-4s, dino-t1-5s].
- CDN mainly helps small objects (Sec. 3.3); DN is already "good enough" for anchor selection overall.
- Reproducibility note (mine, not the paper's): the noise-scale description in the paper (lambda_1 = 1.0, lambda_2 = 2.0 vs dn_box_noise_scale = 0.4, label noise 0.5) is only interpretable via the code.

---

# Facts useful for visuals

- **DN-DETR noise box widget**: GT (x, y, w, h). Center shift uniform in |dx| < lambda_1 w/2, |dy| < lambda_1 h/2; width uniform in [(1 - lambda_2) w, (1 + lambda_2) w], height likewise; defaults lambda_1 = lambda_2 = 0.4 [dn-lambda1, dn-lambda2]. All noised boxes are "positive" (reconstruct GT). Label flip rate gamma = 0.2 [dn-gamma].
- **DINO positive vs negative widget**: corners (x1, y1, x2, y2), each shifted by sign * u * (w/2 or h/2) * 0.4; positive u in [0, 1) so |shift| < 0.2 w; negative u in [1, 2) so 0.2 w <= |shift| < 0.4 w; clamp to image [dino-repo-cdn-noise]. Paper wording: positive "noise scale smaller than lambda_1", negative "larger than lambda_1 and smaller than lambda_2", lambda_1 = 1.0, lambda_2 = 2.0 [dino-cdn-lambdas]. Draw as concentric squares (Fig. 3). Negative target = "no object".
- **IS counter**: COCO val averages 7.36 objects per image, max IS 14.72 [dn-is-max]. A visual can show a query's assigned GT flipping between epochs and IS incrementing by 2 per swap.
- **Attention mask grid**: W = P x M + N; denoising block first. Grey (blocked): matching rows x denoising columns, and off-diagonal group blocks within the denoising columns. Everything else open. DINO CDN groups have 2n queries each (n positives + n negatives).
- **Mask ablation shock number**: denoising without the mask 24.0 AP vs 43.4 with it [dn-abl-nomask].
- **Speedup**: 25 DN epochs ~= 50 baseline epochs (44.4 vs 44.5) [dn-t1b-dc5r50]; 1443 vs 2555 min [dn-t2b-time].
- **DINO ladder** (Table 4): 43.4 -> 44.9 -> 46.5 -> 47.0 -> 47.4 -> 47.9 [dino-abl-*].
- **Headline split for a legend**: R50 no extra data: 49.4 @ 12 ep, 51.3 @ 24 ep, 51.2 @ 36 ep (5-scale) [dino-t1-5s, dino-t2-24, dino-t2-36]; Swin-L + Objects365: 63.2 val / 63.3 test-dev with TTA [dino-t3-val, dino-t3-testdev].
- **Look forward twice diagram**: arrows b_{i-1} -> Layer_i -> delta b_i; b_i' = Update(b_{i-1}, delta b_i) keeps gradient; Detach(b_i') feeds layer i+1; supervised box b_{i+1}^(pred) = Update(b_i', delta b_{i+1}) sends gradient back into layer i.
- **Mixed query selection diagram**: top-K encoder tokens -> their boxes become anchors; content queries remain learned embeddings (not taken from the tokens).
- **Duplicate example**: DN-trained model gives 3 boxes on one boy; CDN gives 1 (Fig. 8) [dino-fig8-dup].
