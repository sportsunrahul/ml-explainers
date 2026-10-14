# SAM 3: Segment Anything with Concepts -- extraction notes (chapter ch-sam3)

Sources read: `sources/sam3/latex/main.tex` (all body + appendices A-C, D, E, F.2, F.5, model card),
`latex/multiplex.tex` (App. H, v2 only), `abs.html`, `paper.pdf` (page/table numbers, Fig. 10),
`sources/repos/sam3.README.md` (code_only facts).
Macro note: `\model`, `\task`, `\dsname`, `\cgf`, `\ilmcc`, `\pmf` and all dataset-size macros
(`\hqvocab`=4M, `\hqimgs`=5.2M, `\hqmasks`=52M, `\synvocab`=38M, `\synmasks`=1.4B,
`\imgplusvideotestvocab`=207K, ...) are defined in `main.tex` lines 173-237, NOT in `text_boxes/inc_macros.tex`.
Lens tags: `[arch]` = architecture/training lens leads; `[data]` = dataset/benchmark (claim-card) lens leads.

## 1. Identity
- Title: SAM 3: Segment Anything with Concepts.
- arXiv 2511.16719, version read: v2. v1 2025-11-20 (Thu, 20 Nov 2025 18:59:56 UTC); v2 2026-03-28
  (Sat, 28 Mar 2026 16:54:56 UTC) (abs.html Submission history). v2 adds App. H "Object Multiplex".
- Authors: Nicolas Carion (first) ... Christoph Feichtenhofer (last); Meta Superintelligence Labs (README).
  README: "* core contributor, ... order is random within groups".
- Venue: not stated (arXiv; model card "Version 1.0, Date 2025, Organization Meta SAM Team").
- Repo: https://github.com/facebookresearch/sam3. README (code_only) describes the SAM 3.1 release of
  2026-03-27 (Object Multiplex checkpoints on HF `facebook/sam3.1`).

## 2. Core problem [arch]
SAM 1/2 do Promptable Visual Segmentation (PVS): one prompt (point/box/mask) -> one object. They do not
"find and segment all instances of a concept appearing anywhere in the input (e.g. all 'cats' in a video)"
(Sec. 1). SAM 3 defines Promptable Concept Segmentation (PCS): given an image or short video (<=30 s) and a
short noun phrase (NP), image exemplars, or both, return masks + unique identities for every matching
instance (Sec. 2).

## 3. What prior methods did and why that hurt (per the paper)
- SAM 1/2: geometric prompts, single instance per prompt; no notion of category (Sec. 1, Sec. 3 "Image Exemplars").
- Open-vocabulary detectors (OWLv2, GroundingDINO, LLMDet) + SAM boxes: on SA-Co/Gold best baseline
  OWLv2* reaches 24.6 cgF1 vs SAM 3 54.1 (Table 1).
- Plain DETR classification asks each query to both recognise (global) and localise (local); "forcing
  proposal queries to understand the global context can be counterproductive, as it conflicts with the
  inherently local nature of the localization objective" (Sec. 3, Presence Token).
- SAM 2's Hiera (MAE) encoder suits geometric PVS but lacks semantic/linguistic grounding and an aligned
  text encoder (App. A.1, Vision and Text Encoder; Table 11).
- Prior data engines used homogeneous web sources and few labels; existing benchmarks have ~50x fewer
  concepts (Sec. 1; App. E.2).
- AP ignores calibration and is noisy with tens of thousands of rare labels (App. E.3).

## 4. The one key idea
Split "is this concept in the image at all?" (one global presence token) from "which query is an
instance?" (DETR queries), conditioned on an aligned vision-language backbone (PE), and feed it with a
data engine whose MLLM "AI verifiers" and adversarial hard negatives scale exhaustive concept labels.
For video, keep detection (identity-agnostic) and tracking (identity-preserving) as separate heads on
one shared backbone (Sec. 1: "The decoupled design for detection and tracking avoids task conflict").

## 5. Mechanism, step by step [arch]
Detector (image or each video frame):
1. Image 1008x1008 (fixed square; eval resizes without keeping aspect ratio; train pads randomly on all
   sides) (App. C.4 Data augmentation) -> PE vision encoder ("L+" size, ~450M params, patch 14, 32 layers,
   global attention in 4 of 32 layers, else windowed attention over 3x3 non-overlapping windows of 336 px;
   RoPE in every layer + windowed absolute pos. emb.) (App. C.2 Image and Text Encoders; Table 11 patch size).
   Run once per frame for the whole interaction (App. C.2).
   Token grid (our reading): 1008/14 = 72 -> 72x72 = 5184 tokens; each 336 px window = 24x24 tokens
   (App. A.1 Implementation Details states windowed "24x24 tokens" attention).
   Token count: ablations run a stride-28 setting. App. A.1 says encoder tokens are "downsampled by 2x2 to
   1296 tokens before being passed to the fusion encoder and detector" (1296 = (1008/28)^2 = 36x36), and App. A.2
   says ablations use "a stride-28 (instead of 14) variant of SAM 3 using 4x fewer tokens in the detector". The
   phrase "instead of 14" implies the released detector runs at stride 14 = 72x72 = 5184 tokens, but the paper
   never states this outright. Stepper: show 1008 -> 72x72 (our reading); show 36x36 only as the ablation config.
   Multi-scale features for the mask head come from SimpleFPN (single-scale ViT) (App. C.2 Segmentation Head).
2. Text NP -> PE text encoder (~300M params, causal, max context length 32) (App. C.2; App. C.4 Stage 1).
3. Image exemplars (box + positive/negative label), if any -> exemplar encoder: per exemplar, position
   embedding + label embedding + ROI-pooled visual features, concatenated, then a small transformer;
   tokens self-attend and cross-attend to the unconditioned frame embeddings (Sec. 3; App. C.2 Geometry and
   Exemplar Encoder). Same encoder makes "geometry tokens" from points/boxes for PVS (auxiliary, Stages 2-3).
   Prompt tokens = text tokens ++ exemplar/geometry tokens.
4. Fusion encoder: 6 transformer blocks, each self-attention + cross-attention (image tokens -> prompt
   tokens) + MLP, vanilla attention; output = "conditioned" frame embeddings (App. C.2 Fusion Encoder).
5. Decoder: 6 transformer blocks, Q = 200 learned object queries (default) + 1 presence token; queries
   self-attend, cross-attend to prompt tokens and to conditioned frame embeddings, then MLP. Box-to-pixel
   relative position bias (PlainDETR) in the image cross-attention; "unlike recent DETR models, we stick to
   vanilla attention" (our reading: no deformable attention). Named DETR tricks: iterative box refinement (each layer
   predicts a delta from the previous layer's box, Deformable DETR), look-forward-twice (DINO), hybrid
   matching (H-DETR), Divide-And-Conquer DETR (DAC) dual supervision, Align loss (Align-DETR). Box and score
   MLPs on queries. Presence token is concatenated with queries in all operations but excluded from DAC
   (Sec. 3 Detector Architecture; App. C.2 Decoder, Presence Head).
   Also 4 learned "geometric queries" (= SAM output tokens) used for PVS during Stages 2-3; presence set to 1
   for PVS (App. C.2).
6. Scores: final per-query score = per-query (conditional) score x shared presence score (Eq. in Sec. 6).
   For counting with a known-present concept one can set presence = 1 (App. C.2).
7. Mask head: MaskFormer-style; instance masks from conditioned fusion features + decoder output queries;
   semantic segmentation head uses the conditioned fusion features only, predicts a binary per-pixel mask
   for the prompt, reuses the presence score, trained on positives only (Sec. 3; App. C.2).
8. Ambiguity head (App. C.2 Handling Ambiguity): K experts trained winner-takes-all (only the lowest-loss
   expert gets gradient); K = 2 best, K > 3 hard due to mode collapse; a classifier picks the expert (CE on
   which expert had min loss); it "adjusts only the classification logits"; trained on top of a frozen
   SAM 3. Overlap detection uses Intersection-over-Minimum (IoM). Result: "15% reduction in overlapping
   instances".
9. Interactivity: exemplar boxes added on errors refine all instances (concept mode); clicks refine one
   instance through the SAM 2 prompt encoder + mask decoder (instance mode); a hard "mode-switch" separates
   them (Sec. 3 Instance Refinement; App. B).

Tracker (video):
10. Per frame: `M_hat_t = propagate(M_{t-1})`, `O_t = detect(I_t, P)`, `M_t = match_and_update(M_hat_t, O_t)`
    (Sec. 3). Tracker = SAM 2 design (prompt encoder, mask decoder = two-way transformer, memory encoder,
    memory attention, memory bank of past frames + conditioning frames, object pointers, occlusion head),
    shared PE backbone via a separate neck; 3 output masks per object per frame, most confident one kept;
    only frames where the object is confidently present are retained in memory (Sec. 3; App. C.3 Tracker).
    Memory-bank size: not stated.
11. Matching: IoU match of tracked masks to detections; unmatched detections spawn new masklets (Sec. 3).
    Delta_i(tau) = +1 if some detection has IoU > iou_threshold with masklet i, else -1; Masklet Detection
    Score S_i(t,t') = sum of Delta (App. C.3).
12. Track confirmation delay T = 15 frames (~0.5 s at 30 fps): output at frame tau shown after seeing tau+T.
    Remove unconfirmed masklets: S_i(t,t+T) < V with V = 0 and first seen inside the window. Remove
    duplicates: two masklets both IoU >= iou_threshold with the same detection for >= ceil(T/2) frames ->
    drop the later-born one (if born in window); iou_threshold = 0.1 (App. C.3).
13. Masklet suppression: confirmed masklet with lifetime S_i(t_first, tau) < 0 -> output mask zeroed but kept
    in tracker state (App. C.3).
14. Periodic re-prompting every N = 16 frames: if detection IoU >= 0.8 with the tracked mask and both
    detection score and masklet score > 0.8, re-initialise the track from the detection mask.
    Detection-guided re-prompting: if best-matching detection has box IoU < 0.85 with the track, recondition
    from the detection (App. C.3).

Object Multiplex (v2, App. H): group objects into buckets of M = 16 slots; memory encoding, memory attention
and mask decoding run once per bucket ("O(N) to O(ceil(N/M))") with a shared spatial memory plus per-object
embeddings; mux/demux map between object-indexed "data space" and bucket/slot "multiplex space"; decoder emits
5 tokens per slot (3 mask, 1 IoU, 1 object-score) = 5M = 80 tokens per bucket; slot indices randomised in
training. Uses the same detector as SAM 3.

## 6. Equations
E1. Presence decomposition (App. C.2 Presence Head, p. 20; unnumbered display):
```
p(\text{query}_i \text{ matches NP}) =
p(\text{query}_i \text{ matches NP}\mid \text{NP appears in image}) \cdot p(\text{NP appears in image}).
```
Paper: "instead of predicting p(query_i matches NP) directly, we break it down"; Sec. 3: "The final score for
each proposal query is the product of its own score and the presence score." App. A.1: "During inference, we
use the product of the global presence score and the object score as the total object score."
Gloss: left = total score of query i; first factor = the query's own score (localisation, conditional on the
concept being present); second factor = output of the presence token's MLP head, shared by all 200 queries.
Training: presence BCE always; per-query classification loss "only ... when the NP is present in the image"
(Setting (a), Table 10).

E2. Video loop (Sec. 3, Tracker and Video Architecture, unnumbered):
```
\mathcal{\hat{M}}_{t} = \mathrm{propagate}\left(\mathcal{M}_{t-1}\right),~~
\mathcal{O}_{t} = \mathrm{detect}\left(I_{t}, P\right),~~
\mathcal{M}_{t} = \mathrm{match\_and\_update}\left(\mathcal{\hat{M}}_{t}, \mathcal{O}_{t}\right).
```
M_{t-1}: masklets up to the previous frame; M_hat_t: their propagated masks on frame t; I_t: frame; P: prompt;
O_t: new detections; M_t: updated masklets.

E3. Frame match and MDS (App. C.3 Disambiguation Strategy, unnumbered):
```
\Delta_i(\tau) = \begin{cases} +1, & \text{if } \exists~d \in \mathcal{D}_\tau \text{ s.t. } \mathrm{IoU}(d, \mathcal{\hat{M}}^i_\tau) > \mathrm{iou\_threshold} \\ -1, & \text{otherwise}, \end{cases}
\qquad S_i(t, t') = \sum_{\tau=t}^{t'} \Delta_i(\tau).
```
D_tau: detector outputs on frame tau; M_hat^i_tau: tracker mask for object i. S sums +1/-1 over a window:
"measures how a masklet is consistently matched to a detection within a temporal window".

E4. Ambiguity head losses (App. C.2 Handling Ambiguity, unnumbered):
```
\mathcal{L}_{\text{MoE}} = \sum_{k=1}^K p_k \, \mathcal{L}_k ;\quad
k^\star = \arg\min_{k} \, \mathcal{L}_k, \qquad \mathcal{L}_{\text{WTA}} = \mathcal{L}_{k^\star}
```
Gradient of L_WTA reaches only expert k*. "allows each expert to specialize to one kind of interpretation".

E5. pmF1, Eq. (1) (`eqn:pmf`, App. E.3, p. 36-37):
```
\mathrm{pmF}_1^\tau = \frac{2\mathrm{TP}^\tau_{total}}{2\mathrm{TP}^\tau_{total} + \mathrm{FP}^\tau_{total} + \mathrm{FN}^\tau_{total}}, \quad \pmf = \frac{1}{10}\sum_{\tau \in \textrm{np.linspace}(0.5, 0.95, 10)} \mathrm{pmF}_1^\tau.
```
Only positive datapoints (>= 1 GT mask). Per datapoint: IoU matrix, optimal bipartite matching; matched
prediction with IoU >= tau = TP, else FP; unmatched GT = FN. Counts are summed over all positive datapoints
(micro), then averaged over 10 IoU thresholds 0.5..0.95. Only predictions with confidence > 0.5 count (Sec. 5).

E6. IL_MCC, Eq. (2) (`eqn:ilmcc`):
```
\mathrm{IL\_MCC} = \frac{\mathrm{IL\_TP}\cdot\mathrm{IL\_TN} - \mathrm{IL\_FP}\cdot\mathrm{IL\_FN}}{\sqrt{(\mathrm{IL\_TP} + \mathrm{IL\_FP})\cdot(\mathrm{IL\_TP} + \mathrm{IL\_FN})\cdot(\mathrm{IL\_TN} + \mathrm{IL\_FP})\cdot(\mathrm{IL\_TN} + \mathrm{IL\_FN})}}.
```
Image-level: positive pair + any mask predicted (conf > 0.5) = IL_TP, else IL_FN; negative pair + any mask =
IL_FP, else IL_TN. Range [-1, 1]; ignores mask quality.

E7. cgF1, Eq. (3) (`eqn:cgf`):
```
\mathrm{cgF}_1 = 100 \cdot \mathrm{pmF}_1 \cdot \mathrm{IL\_MCC}.
```
"classification-gated F1": localisation quality gated by presence classification. Why new: AP ignores
calibration, is noise-dominated with tens of thousands of rare labels, full PR curves are infeasible; plain F1
"is sensitive to high ratios of negative annotations: no extra credit is given for correctly predicting
nothing for a negative" (App. E.3). Oracle on SA-Co/Gold: compare predictions to each of 3 GTs, keep the
best local F1 (App. E.3).
Video cgF1 differs: VL_MCC (video-level) x positive MACRO F1 with volume IoU (App. F.5), vs image IL_MCC x
positive MICRO F1.

## 7. Config / setup table [arch]
| Item | Value | Loc |
|---|---|---|
| Total params | ~850M (README: 848M, code_only) | App. C.1 |
| Vision / text encoder params | ~450M / ~300M ("L+" size) | App. C.1, C.4 Stage 1 |
| Detector + tracker params | ~100M | App. C.1 |
| Input resolution | 1008x1008 | App. C.2, C.4 Data aug. |
| Patch size | 14 (PE-L+) | Table 11 |
| Vision layers / global layers | 32 / 4 | App. C.2 |
| Window | 3x3 windows of 336 px (24x24 tokens) | App. C.2, App. A.1 |
| Token grid | 72x72 = 5184 at stride 14 (our reading, implied by A.2 "instead of 14"); ablations: 1296 = 36x36 (stride-28 equivalent) | App. A.1, A.2 |
| Text context | 32 tokens, causal | App. C.2 |
| Fusion encoder | 6 blocks (self-attn, cross-attn to prompt, MLP) | App. C.2 |
| Decoder | 6 blocks, Q = 200 object queries + 1 presence token (+4 geometric queries for PVS) | App. C.2 |
| Width / heads / exemplar-encoder depth / memory frames | not stated | -- |
| Ambiguity experts | K = 2 | App. C.2 |
| PE pretraining data | 5.4 billion image-text pairs | App. C.2, C.4 |
| Compute (released model) | 172k A100 GPU hours + 86k H200 GPU hours | App. I Model card, "Cost and impact of compute" |
| Inference | 30 ms per image with 100+ objects on one H200 | Sec. 1 |
| Video real-time (30 FPS) | up to 10 obj on 2 H200s, 28 on 4, 64 on 8 | App. B |

Training stages (Sec. 3 Training Stages; App. C.4; Table 18):
| Stage | Trains | Frozen | Data | Recipe |
|---|---|---|---|---|
| 1 PE pretraining | image + text encoders (contrastive) | -- | 5.4B image-text pairs; no distillation, no video fine-tuning | per PE paper |
| 2 Detector pretraining | detector + vision + text encoders | -- | SA-Co/SYN, HQ, EXT, VIDEO frames (PCS); SA-1B, SA-Co/VIDEO(-EXT) frames (PVS) | ~95k it, batch 896, AdamW, 5k warmup/cooldown, rsqrt schedule, wd 0.1, layer-wise LR decay 0.9 (vision), LR 5e-4 vision / 1e-4 text / 1e-3 rest; dropout 0.1; p=0.2 drop NP (visual query), p=0.2 add input boxes; PVS with 4 interactivity steps; Align loss replaces IoU-prediction loss |
| 3 Detector fine-tuning + presence post-training | same as stage 2 | -- | highest-quality exhaustive data only (SYN dropped): HQ, EXT, VIDEO frames | ~5k it, batch 512, all LRs x0.025; PVS 7 interactivity steps; PCS exemplar interactivity 5 iterations (positives from FNs, negatives from high-conf FPs); presence token + presence BCE (weight 20) introduced here |
| 4 Tracker | tracker | PE backbone (and detector, our reading: "tracker decoder is trained on top of the frozen backbone") | SA-1B, SA-Co/VIDEO, VIDEO-EXT (frames and videos) | batch 512, ~190k it, cosine, peak LR 5.0e-4, 1k warmup; then 16- and 32-frame clips for 60k it at LR x0.1 |
Stage 4 freeze reason (App. C.4): "Freezing the backbone at this stage is made possible by pre-training on VOS
data in previous stages at the video frame level. This stage retains the strong spatial grounding of the
previous stage and focuses on spatial-temporal tracking without degrading other capabilities."
Losses: Stage 2 (App. C.4): "For boxes, we use L_1 and gIoU losses with weights of 5 and 2. Classification loss
uses a weight of 100 and focal and dice losses use weights of 200 and 10 respectively." Presence BCE weight 20
(Stage 3). Stage 4 verbatim: "a linear combination of focal and dice losses for mask prediction, mean absolute
error (MAE) loss for IoU prediction, and cross-entropy loss for object occlusion prediction, with respective
weights of 20:1:1:1. For multi-mask predictions, we only apply supervision to the mask with the lowest
segmentation loss." Occlusion head always supervised, masks not supervised when the frame has no GT mask.

## 8. Results worth remembering
Architecture-side:
- Presence head ablation (Table 9a, shorter runs): without 50.7 cgF1 / 0.77 IL_MCC / 65.4 pmF1; with 52.2 /
  0.82 / 63.4. "+1.5" cgF1, "+0.05" IL_MCC. Note pmF1 drops by 2.0.
- Presence supervision (Table 10): (a) mask scores only when present: 54.0/0.82/65.5; (b) always (DETR-style):
  52.2/0.81/64.2; (c) supervise total score: 54.9/0.83/66.0; (d) total, detach presence: 53.6/0.83/64.9.
  Paper picks (a) for flexibility although (c) is slightly higher.
- Encoder ablation (Table 11): PE-L+ 43.2 Gold cgF1 / 42.5 COCO-O AP; PE-L+ w/ DistilRoBERTa 38.1/39.6;
  DINOv2-L 35.3/31.9; Hiera-L 32.8/22.0.
- Temporal disambiguation (Table 39a, SA-V test): with 30.3 cgF1 / 58.0 pHOTA; without 27.1 / 55.9.
- Multiplex (App. H.2): ~7% latency overhead at 1 object; faster from >= 2 objects; 5.2x speedup at 128 objects
  on one H100 (Fig. 30). VEval YT-Temporal-1B +2.7 cgF1 (50.8 -> 53.5), +0.9 pHOTA (Table 43); OVIS +1.8 mAP;
  slight regressions LVVIS (36.3 -> 34.2), BURST (44.5 -> 43.1), YTVIS21 (57.4 -> 56.3). VOS MOSEv1 78.4 -> 79.6,
  MOSEv2 60.3 -> 62.3 (Table 44).

Benchmark-side:
- Table 1 (SA-Co/Gold instance cgF1): Human 72.8; SAM 3 54.1; OWLv2* 24.6; OWLv2 17.3; DINO-X 21.3; APE-D* 16.4;
  Gemini 2.5 13.0; LLMDet-L 6.5; gDino-T 3.3. Silver 49.6, Bronze 42.6, Bio pmF1 55.4. Box: Gold 55.7 (Human 74.0).
  "more than double ... OWLv2*, and 74% of the estimated human performance" (Sec. 6).
- LVIS zero-shot mask AP 48.5 (Table 1) vs intro text "48.8 on LVIS vs. the current best of 38.5" (Sec. 1)
  -- the paper is inconsistent; 38.5 = DINO-X (Table 1). LVIS box AP 53.6; COCO box AP 56.4; COCO-O 55.7.
  Semantic mIoU: ADE-847 13.8, PC-59 60.8, Cityscapes 65.2 (Table 1).
- Few-shot (Table 2): ODinW13 AP0 61.0, AP10 71.8; RF-100VL AP0 15.2 (gDino-T 15.7), AP10 36.5.
- 1 exemplar (Table 3) AP+ image-only: COCO 76.8, LVIS 76.0, ODinW13 82.2 vs T-Rex2 58.5 / 65.8 / 61.8
  (+18.3, +10.3, +20.5 per text). T+I: 78.1 / 78.4 / 81.8.
- K exemplars (Fig. 7, Sec. 6 text): after 3 clicks interactive PCS beats text-only by +21.6 cgF1 and PVS
  refinement by +2.0; plateau after 4.
- Counting (Table 4): CountBench MAE 0.12, Acc 93.8; PixMo-Count MAE 0.21, Acc 86.2.
- Video PCS (Table 5): VEval SA-V 30.3 cgF1 / 58.0 pHOTA (human 53.1/70.5); YT-Temporal-1B 50.8/69.9 (human
  71.2/78.4); SmartGlasses 36.4/63.6 (human 58.5/72.3); LVVIS 36.3 mAP, BURST 44.5 HOTA, YTVIS21 57.4 (GLEE
  62.2), OVIS 60.5. "over 80% of human pHOTA" (Sec. 6).
- VOS (Table 6): MOSEv2 60.3 vs SAM 2.1 L 47.9 and SeC 53.8 ("outperforms prior work by 6.5"); SA-V test 84.4
  vs 78.4; DAVIS17 92.2 vs 90.7.
- Interactive image SA-37 (Table 7): 1/3/5 clicks mIoU SAM 3 66.1/81.3/85.1 vs SAM 2.1 L 66.4/80.3/84.3 vs SAM 1
  H 58.5/77.0/82.1; FPS SAM 3 43.5 vs SAM 2.1 L 93.0. SAM 3 is slightly behind SAM 2.1 at 1 click and ~2x slower.
- Hard negatives (Table 9b, per image in HQ): 0 -> 28.3 / 0.44 / 62.4; 5 -> 39.4 / 0.62 / 62.9; 15 -> 41.8 /
  0.67 / 62.4; 30 -> 43.0 / 0.68 / 62.8. IL_MCC "from 0.44 to 0.68"; pmF1 flat. (App. A.2 ablation setting.)
- Training data (Table 9c): EXT 23.7; +SYN 32.8; +HQ 45.5; all 47.4 ("+8.8", "+14.6").
- AI verifiers on top of SAM 3 (Table 9d): SAM 3 54.0; +EV 61.2 (+7.2); +MV 62.3 (+1.1); human 72.8.
- Human oracle vs random pair (Table 29, Gold 3-reviewed subset): Oracle 76.2 cgF1 / 0.99 / 77.0; Random Pair
  55.5 / 0.87 / 63.4. Ablation-version internal Gold human = 70.8 (App. A.2).

Data engine [data] (Sec. 4; App. D; Table 19):
- Phase 1 (human verification): BLIP-2 captioner + spaCy parser propose NPs; FIBER or OWLv2 boxes prompt SAM 2
  for masks; humans do MV, EV, correction. 1.2M images, 4.3M image-NP pairs, 1 domain.
- Phase 2 (human + AI verification): Llama 3.2 fine-tuned on Phase 1 MV/EV labels as AI verifiers (MV: 5-way
  choice, EV: 6-way choice; pretrain on >200M image-text pairs, fine-tune on ~10M); Llama-based NP proposer +
  hard-negative proposer (ontology siblings/cousins/uncles + Llama 4), kept only if adversarial to SAM 3. SAM 3
  retrained 6 times. 2.4M images, 122.2M image-NP pairs, 5 domains. Gold cgF1 21.4 at end of phase.
  "roughly doubles the data engine's throughput vs. human annotators".
- Phase 3 (scaling, domain expansion): 15 domains; 22.4M-node Wikidata ontology (17 top-level, 72 sub-cats);
  alt-text NPs; mining crowded / small-object images; SAM 3 iterated 7 times, AI verifiers 3 times. 1.6M images,
  19.5M image-NP pairs; SYN 39.4M images / 1.7B image-NP; EXT 9.3M / 136.6M. Gold 54.4.
- Phase 4 (video): O(1M) hours pool; 5-30 s single-shot clips; PLM as challenge judge; keep NPs with >= 3
  objects and videos with >= 2 such NPs; humans verify and correct with SAM 2. 52.5K videos, 134.3K video-NPs.
  VEval test pHOTA 63.9.
- AI verifier accuracy (Table 23, average): MV human 75 vs AI 77.7; EV human 81.8 vs AI 81.1.
- Throughput (Table 15, Wiki-Food&Drink): time per datamix item 90 s (human from NP) -> 23 s (engine full);
  positive NP 236 -> 152 s; negative NP 71 -> 6 s. "5x speed up on negative phrases and a 36% speed up for
  positive phrases". Video (Table 16): engine-all-human is 40% slower but yields 9% more masklets.

SA-Co data [data] (Sec. 5; Tables 24, 25, 28):
- HQ: 4.0M NPs, 5.2M images, 146.1M image-NPs, 88.5% negatives, 52.3M masks.
- SYN: 38.0M NPs, 39.4M images, 1.7B image-NPs, 74.0% negatives, 1.4B masks.
- EXT: 497.4K NPs, 9.3M images, 136.6M image-NPs, 71.8% negatives, 70.5M masks.
- VIDEO: 24.8K NPs, 52.5K videos, 134.3K video-NPs, 26.7% negatives, 467.1K masklets; avg 84.1 frames at 6 fps.
- Benchmark total: 207K unique phrases, 121K images and videos, over 3M media-phrase pairs (Sec. 5);
  "120K images and 1.7K videos" (Sec. 1). Gold 51.8K NPs / 15.8K images / 168.9K image-NPs / 84.4% neg / 7
  domains / 3 annotators; Silver 54.6K / 66.1K / 1.8M / 94.0% / 10 domains / 1 annotator; Bronze 105.3K / 32.5K /
  1.0M; Bio 166 / 5.4K / 35.9K; VEval 5.2K NPs / 1.7K videos / 10.3K video-NPs / 75.4% neg / 3 domains.
- Gold has "~50x more unique phrases" than LVIS-test (App. E.2).

## 9. Limitations the paper states (App. B)
- "SAM 3 struggles to generalize to fine-grained out-of-domain concepts (e.g., aircraft types, medical terms) in
  a zero-shot manner, especially in niche visual domains (e.g., thermal imagery)." Mitigation shown: few-shot
  fine-tuning; synthetic in-domain data with no humans (Fig. 8/9).
- "constrained to simple noun phrase prompts and does not support multi-attribute queries beyond one or two
  attributes or longer phrases including referring expressions" -> SAM 3 Agent with an MLLM.
- "the cost of SAM 3 inference scales linearly with the number of objects being tracked"; "no shared
  object-level contextual information" across objects. -> Fixed in v2 by Object Multiplex (App. H), shared
  memory per 16-object bucket, released as SAM 3.1 (README).
- Concept vs instance interactivity needs a hard "mode-switch".
- Industrial relevance (our reading): the thermal-imagery and fine-grained OOD limitation maps directly to
  inspection domains; the paper's remedy is few-shot fine-tuning (RF-100VL AP10 36.5).

## 10. Outline feedback (ch-sam3)
- Add the ambiguity head (K = 2 experts, winner-takes-all, picker head, frozen SAM 3, IoM, 15% fewer overlaps):
  it is the concept-level analogue of SAM 1's three-mask output and links to ch-promptable.
- The sam3 scene's "per-query score x presence" is correct (App. C.2), but the training rule matters: query
  scores get no gradient on image-level negatives; the presence head alone learns absence (Table 10 (a)).
- Show the honest trade-off: presence raises IL_MCC (0.77 -> 0.82) but lowers pmF1 (65.4 -> 63.4) (Table 9a).
  Hard negatives move IL_MCC 0.44 -> 0.68 with pmF1 flat (Table 9b): the hard-negative toggle in the scene
  should change the "is it here" gate, not mask quality.
- ArchFlow width and heads are not in the paper; only resolution 1008, patch 14, 72x72 grid (our reading),
  32 layers, 6+6 blocks, Q = 200, text context 32 are sourced. Released detector tokens 72x72 = 5184 are implied
  (A.2 "instead of 14") but not stated; 1296 is the ablation setting. Label it.
- Tracker heuristics are worth a mini-stepper: T = 15 delay, V = 0, iou_threshold 0.1, re-prompt every 16 frames
  at IoU >= 0.8 and score >= 0.8, detection-guided re-prompt at box IoU < 0.85.
- Multiplex is a v2 appendix result (M = 16, 80 tokens per bucket, 5.2x at 128 objects on one H100, ~7% overhead
  at 1 object). Put it as a footnote card, not the main architecture.
- Inconsistencies to label, not reconcile: LVIS mask AP 48.8 (Sec. 1) vs 48.5 (Table 1); SA-Co/EXT "15 external
  datasets" (Sec. 5) vs "eighteen" (App. E.1, Table 26 lists 18); SA-Co/HQ "phases 1-4" (Sec. 5) vs "Phases 1-3"
  (App. E.1, Table 19); benchmark 207K concepts (paper) vs 270K (README); 850M (paper) vs 848M (README); human
  Gold cgF1 72.8 (Table 1) vs 76.2 (Table 29 oracle, 3-reviewed subset) vs 70.8 (internal, App. A.2).
- SAM 3 is not better on everything: SA-37 1-click 66.1 vs SAM 2.1 L 66.4, and 43.5 vs 93.0 FPS (Table 7).
- ch-inspection can reuse: fine-grained OOD / thermal limitation; few-shot RF-100VL; domain adaptation with
  synthetic data (Fig. 8: SYN-Food catches up with HQ-Food without human labels).
