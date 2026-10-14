# Outline: when a detector gets a text encoder

Survey of open-vocabulary and grounded object detection, 2021-2025. Long page (no narration; cue ids and
player hooks kept) plus an atlas page. Reader: basic deep learning; has the vlm-survey (CLIP, SigLIP, ViT)
and detr-family (DETR to DINO, RT-DETR) explainers, so CLIP and DETR get one recap paragraph each with a link;
two-stage detectors and YOLO necks are taught from scratch.

The frame for the whole page: a closed-set detector scores each region with a fixed matrix `W` (one row per
class). Every paper here swaps `W` for text embeddings, and they differ in WHERE text enters the network:

```
 image -> backbone -> [neck] -> [proposals / queries] -> region features -> score = region . text
 ViLD            text at the classifier only; CLIP distilled into region features
 OWL-ViT/v2      every ViT token is a box; text at the classifier only (late fusion)
 GLIP            text fused into the last encoder layers (deep fusion)
 Grounding DINO  fused in the encoder + text picks the queries + text cross-attention in the decoder
 YOLO-World      text fused into the neck, then folded into conv weights for deployment
```

Color key (one meaning per color, whole page): blue = image / region features; yellow = text / prompts;
green = a match (region-text score that wins, positive target); red = loss / gradient; brown = frozen or
background / no object.

Hero: a closed-set detector draws boxes; its classifier block is a stack of K labelled rows `W`; the rows
slide out and a text encoder slides in, emitting one yellow vector per typed class name; then five colored
taps show where text enters for each model, ending on the static "where text enters" diagram. Replay button.

## Act I: a detector that reads

### 1. `ch-text-classifier` The class list becomes a text encoder
- Intuition: a detector's class score is a dot product between a region feature and one row of `W`. Replace
  the rows with text embeddings and a new class is a new sentence, not a new training run.
- Also defines the vocabulary the page uses: base vs novel classes, LVIS frequent/common/rare, AP and APr
  (rare-class AP), zero-shot transfer. CLIP recap (one paragraph, link to vlm-survey ch-shared-space).
- Visual `viz-classifier`: 2D toy plane of region features (blue dots) and class directions (arrows). Buttons:
  `btn-classifier-closed` (fixed rows, a "forklift" region has no row and is forced into the nearest wrong
  class), `btn-classifier-text` (rows become text vectors), `btn-classifier-add` (add "forklift" as text: a
  new arrow appears, the region flips to green). Toy.
- Equation: GLIP Eq. 2 and Eq. 3 (S_cls = O W^T -> S_ground = O P^T), built term by term.
- LVIS split as a small sourced bar (counts of rare/common/frequent classes; ViLD sec. 4.1).

### 2. `ch-map` The field in one picture
- Intuition: the area went from a handful of papers in 2021 to over a hundred a year, and almost none of it
  studies factories.
- Visual `viz-map`: counts by year from the arXiv scan (522 papers, three queries, 2026-10-09) with a
  domain-split bar (title-keyword heuristic: 3D 63, remote sensing 27, medical 8, industrial 3...) and the
  family tree preview. Every count is a `data-claim` whose loc is the query. Computed from the scan.

## Act II: borrow CLIP's eyes (2021)

### 3. `ch-two-stage` How a two-stage detector finds boxes (from scratch)
- Intuition: proposals do not care what the object is. The RPN learns "objectness", so it can box objects
  it was never told the name of; only the second stage names them.
- Visual `viz-rpn`: toy image with anchors at one feature-map cell; buttons step anchors -> objectness scores
  -> box deltas -> top-k proposals -> RoIAlign 7x7 crop -> head. Slider: number of proposals kept. Toy.
- Sourced: ViLD Table 1 (RPN trained on base only vs base+novel gives comparable recall on novel classes).
- Archflow walkthrough of Mask R-CNN is folded into chapter 4 (one figure, not two).

### 4. `ch-vild` ViLD: teach the region head to see like CLIP
- Intuition: CLIP already knows rare things but is slow on crops; train the detector's region embedding to
  (a) be classified by CLIP text vectors (ViLD-text) and (b) copy CLIP's image embedding of each crop
  (ViLD-image). The second loss reaches proposals of novel objects, which no label ever touches.
- Visual `viz-vild3d` (the page's 3D figure, pe3): unit sphere; yellow text points fixed; blue region points
  and their CLIP-crop targets; buttons `btn-vild3d-text` (only base-class regions move to their text points),
  `btn-vild3d-image` (all regions, including novel ones, move toward their CLIP points), `btn-vild3d-both`.
  A novel "unseen" region ends near its own text point only with the image loss. Toy, labelled.
- Equations: ViLD Eq. 1 (crop ensemble), Eq. 2 (ViLD-text with learned background), Eq. 3 (L1 distillation),
  Eq. 4 (total). Sourced table: APr of CLIP-on-crops, ViLD-text, ViLD-image, ViLD, supervised RFS, GloVe.
- Architecture walkthrough `viz-vildflow`: image -> ResNet-50 FPN -> RPN -> RoIAlign -> region head ->
  projection -> embedding -> cosine with T(C_B) + e_bg; distill branch crop -> frozen CLIP image encoder
  (precomputed offline). Backward: backbone, RPN, head, projection, e_bg train; CLIP image + text encoders
  frozen, the gradient stops there.
- Branch deltas (one paragraph each, on the family tree): OVR-CNN (caption pretraining of a vision-to-language
  projection), RegionCLIP (region-text pretraining on caption-derived pseudo pairs), Detic (image-level
  labels train the classifier on the largest proposal, ImageNet-21k classes).

## Act III: make CLIP itself the detector (2022-2023)

### 5. `ch-owl` OWL-ViT: every patch is a candidate object
- Intuition: drop the pooling at the top of a CLIP ViT; every output token now predicts one box and one
  embedding; the text encoder runs once per query, independent of the image, so thousands of queries are
  cheap. The same head accepts an image embedding as the query: one-shot detection by example.
- Visual `viz-owl`: toy image with the ViT token grid; each token draws its box faintly; text query buttons
  re-score tokens; `btn-owl-image` switches to an exemplar query. Threshold slider. Toy.
- Archflow `viz-owlflow` with computed shapes for B/32 at 768 (576 tokens), B/16 at 768, L/14 at 840; text
  tower; box MLP; class head; DETR matching with focal sigmoid loss; trainable vs frozen per the paper.
- Sourced: LVIS APr (open-vocab) from Table 1; one-shot COCO AP50 from Table 2.

### 6. `ch-owlv2` OWLv2: label the web with yourself
- Intuition: detection data is small; run OWL-ViT over 10B web images with N-gram queries from their alt
  text, keep confident boxes, train a new detector on that, and keep scaling. Cheaper training through token
  dropping, an objectness head and mosaics.
- Visual `viz-owlst`: the self-training loop as a stepper (web image + alt-text -> N-grams -> teacher boxes ->
  threshold filter -> student). Second panel: token dropping toy (drop lowest-variance patches; slider for
  drop rate). Toy for the panels.
- Sourced: main table LVIS APr for OWL-ST vs OWL-ST+FT; the fine-tuning trade-off (LVIS up, ODinW13 down,
  weight ensemble recovers); the 13-vs-35 ODinW mean/median gap. These numbers feed chapter 12.

## Act IV: detection as grounding (2021-2023)

### 7. `ch-glip` GLIP: a detector is a grounding model
- Intuition: write the classes as one sentence, "person. bicycle. car.", score every region against every
  token, and detection becomes phrase grounding. Then any captioned image with boxes is training data, and
  the vocabulary grows from at most about 2,000 detection categories to 44,518 phrases (Flickr30K), 110,689
  (VG Caption) and 58.4M pseudo-labelled noun phrases.
- Visual `viz-ground`: alignment matrix regions x tokens; phrase chips; subword split shown ("tooth #brush");
  click a phrase -> its tokens become the positive targets for the matched region. Toy.
- Equations: GLIP Eq. 2-3 recalled, the token-level target expansion. Sourced: phrase counts (Flickr30K,
  VG captions), "0.8M gold grounding data beats 2M extra detection data on rare categories", pretraining
  data (gold + pseudo-labeled image-text pairs), zero-shot COCO and LVIS.
- Branch: MDETR (the DETR-based ancestor of grounding pretraining), GLIPv2.

### 8. `ch-fusion` Deep fusion: let the words change what the image encoder sees
- Intuition: in late fusion the image features are the same whatever you ask; with cross-attention in every
  upper layer the image features become prompt-dependent. Price: the fusion layers rerun for every prompt
  (GLIP-T 4.84 -> 2.52 FPS, Table 7), which is the OWL-ViT argument against fusion. Benefit: zero-shot ODinW
  35.5 (late) vs 44.4 (deep), Table 8; prompt tuning only works with deep fusion.
- Visual `viz-fusion`: two columns of layers (image, text) with X-MHA arrows; toggle late vs deep; a toy
  feature map whose "stingray" response sharpens layer by layer only in deep mode. Toy.
- Equations: GLIP Eq. 4-6 (X-MHA, DyHead update, BERT update), attention lines.
- Archflow `viz-glipflow`: Swin-T -> DyHead x6 with X-MHA; BERT-base -> added BERT layers; shapes.
- Sourced: manual prompt table (Aquarium, Rabbits, EgoHands, Mushrooms, Packages, Pothole: original vs
  descriptive prompt AP). Sets up chapter 13's "the name is the failure" point.

### 9. `ch-gdino` Grounding DINO: DINO-DETR that reads
- Intuition: take DINO (recap + link to detr-family) and add text at three places: fuse in the encoder
  (feature enhancer), let the text choose which encoder tokens become queries (language-guided query
  selection), and let each decoder query attend to the text.
- Visual `viz-qsel`: toy grid of image tokens; prompt chips; heat = max over text tokens of the dot product;
  top-K tokens light green and become queries. Switching the prompt moves the queries. Toy.
- Second visual `viz-subsentence`: attention mask between category names (word level vs sub-sentence). Toy.
- Equation: Grounding DINO Eq. 1 (Top_Nq(Max(X_I X_T^T))). Archflow `viz-gdinoflow` with shapes (Swin-T,
  multi-scale tokens > 10,000, text <= 256, d = 256, 900 queries, 6+6 layers), loss (L1 + GIoU + focal
  contrastive), what is frozen.
- Sourced: COCO zero-shot AP, LVIS, ODinW.

### Interlude `ch-training` One Grounding DINO training step, end to end (not narrated)
- Stepper: batch + prompt construction (category names concatenated, negatives) -> Swin -> BERT -> feature
  enhancer -> query selection -> decoder -> matching -> losses. Shapes in a readout; links back to ch 9.

## Act V: real time (2024)

### 10. `ch-yoloworld` YOLO-World: fold the vocabulary into the weights
- From scratch: what YOLO is (dense predictions at every cell of three feature maps, a PAN neck mixing top-
  down and bottom-up, a decoupled head, task-aligned assignment instead of Hungarian matching).
- Intuition: inject text into the neck cheaply (max-sigmoid gate per pixel, image-pooled attention back into
  text), score with a contrastive head; at deployment the user's vocabulary is encoded once and becomes
  ordinary conv/linear weights: prompt-then-detect.
- Visual `viz-repvl`: toy feature map gated by the max over text sims (T-CSPLayer); button
  `btn-repvl-reparam` collapses the text encoder into a 1x1 conv block. Toy.
- Sourced chart `viz-speed`: LVIS minival zero-shot AP vs FPS (YOLO-World S/M/L vs GLIP, Grounding DINO,
  DetCLIP) from YOLO-World Table 2.
- Equations: YOLO-World Eq. 1-3. Archflow `viz-yolowflow` with shapes (640 input -> 80x80, 40x40, 20x20 ->
  8400 predictions, vocab of at most 80 nouns per mosaic during training).
- Branch: YOLOE (one line).

## Act VI: when words run out, and the practical question

### 11. `ch-visual-prompts` Show, do not tell: visual prompts
- Intuition: if the thing has no good name, draw a box around an example. OWL-ViT already accepts an image
  embedding as the query (one-shot COCO AP50 means 24.3 -> 41.8 vs the prior AIT). T-Rex2 trains text and box
  prompts into one space and reports that they complement each other.
- Honest framing (from extraction): T-Rex2's "visual prompt" numbers average about 16 labelled training-set
  examples per class, so they are not zero-shot like a text prompt; "text for common, visual for rare" flips
  with the split (LVIS-val rare: visual 32.4 vs text 29.0; LVIS-minival rare: text 37.4 vs visual 29.9).
  Independent tests disagree with the optimistic reading: image-only prompting loses to class names in
  RF100-VL (All 6.4 vs 12.2, MQ-GLIP) and in FSOD-VLM's rarest group (3.34 vs 8.41); YOLOE adds visual and
  prompt-free modes.
- Visual `viz-vprompt`: toy scene with an odd part; text prompt vs exemplar box vs both; scores update; a
  second panel plots the sourced T-Rex2 and RF100-VL numbers side by side with their settings. Toy + sourced.

### 12. `ch-when` When can a text prompt replace a custom detector?
- Claim cards, each with how it was measured and what would falsify it:
  1. RF100-VL label-budget ladder by domain (Table 2 zero-shot and 10-shot; Table 7 10% labels and full).
     Industrial: zero-shot best 10.3 (MM-GroundingDINO-L*, all classes in one prompt), 10-shot fine-tuned
     Grounding DINO 37.8, 10% labels 50.7-52.9, full labels 64.3-68.2. "Industrial" is a 22-dataset CLIP
     cluster, about 7 of them defect or inspection sets: say so on the chart.
  2. ODinW per-dataset zero-shot AP strip plot (OWL-ST L/14 and Grounding DINO, 35 datasets each, from their
     own appendix tables): mean vs median; count of datasets below 1 / 5 / 10 AP.
  3. GLIP data efficiency (zero-shot vs 1/3/5/10-shot vs full, ODinW-13 average; one table only) and
     prompt tuning vs full tuning (works only with deep fusion).
  4. Fine-tuning forgets (OWLv2): median over 35 ODinW datasets falls at every size (L/14 16.2 -> 6.3) while
     LVIS rises; a weight-averaged checkpoint recovers (ODinW-13 56.3, curated-vocabulary model).
- Visual `viz-ladder`: sourced chart, domain buttons, label budget on x, AP on y.
- Decision guide (labelled "our reading"): a prompt alone is enough when the concept is common on the web,
  its everyday name is unambiguous, it looks like an object, and your label policy matches the web meaning;
  otherwise few-shot fine-tune an open-vocabulary detector (the best 10-shot learner in RF100-VL), and keep a
  weight-averaged copy if you also need open-vocabulary behaviour.

### 13. `ch-failures` How they fail on rare industrial objects
- Taxonomy, each item tied to a source:
  1. The name is jargon, an acronym or context-dependent: RF100-VL hard-example types (its wire-rope defect
     "Thunderbolt"), GLIP's Mushrooms prompt (13.6 -> 73.8 by describing the look), MM-G Brain Tumor with
     numeric labels (0.4 zero-shot -> 47.5 fine-tuned), Grounding DINO PlantDoc 0.36 AP, GD 1.5 renaming ODinW
     categories where it scored poorly.
  2. The label policy differs from the web meaning of the name (FSOD-VLM: nuImages "truck" boxes the cab,
     not the whole vehicle). A zero-shot model cannot know your annotation guide.
  3. The target is a texture or region, not a bounded object (our reading; weak support in the sources, so
     labelled as such).
  4. Tiny objects at fixed input resolution (our reading + input sizes from the configs).
  5. Scores are not calibrated across prompts; thresholds do not transfer (OWLv2 limitations; ViLD App. B:
     CLIP scores do not reflect box quality).
  6. Attributes and negation are ignored (OVDEval; D3, phrased as "negation is ignored", since D3's Table 2
     contradicts its own stronger sentence for some models).
  7. Fine-tuning trades away open-vocabulary robustness (chapter 12).
- Visual: the 3D cloud from chapter 4 again (`viz-gap3d`): add a jargon name and watch its text point land
  far from the visual cluster; add a descriptive prompt and watch it land closer. Toy, labelled.

## Closing (unnarrated)
- `ch-tree` family tree and timeline: spine + branches, colored by where text enters.
- `ch-summary` one paragraph.
- `ch-deeper` per paper: section, equation and table to read next.

## Atlas page (one stage, step tabs, glossary dialogs)
Nine chapters: text-classifier, two-stage + ViLD, OWL-ViT/v2, GLIP fusion, Grounding DINO, YOLO-World,
visual prompts, when prompts suffice, failure modes. Scenes reuse the long page's figure logic.

## Open items for extraction
- LVIS rare/common/frequent class counts and the base/novel split as used by ViLD.
- OWL-ViT and OWLv2 main-table APr values, ODinW13 numbers, fine-tune trade-off values (table, not figure).
- GLIP pretraining data sizes, zero-shot COCO/LVIS, data-efficiency table values.
- Grounding DINO COCO zero-shot, LVIS, ODinW; hyperparameters for shapes.
- YOLO-World Table 2 speed and APr values; input size; strides.
- T-Rex2 text vs visual prompt numbers; FSOD-VLM (2312.14494) zero-shot vs few-shot; OVDEval and D3 failure
  findings.

## Corrections from extraction (2026-10-09), supersede the chapter specs above
ViLD group (notes/papers/vild.md, claims/vild.json):
- Ch 1: sources give only 1,203 total, 866 base (frequent + common), 337 rare. No separate frequent/common
  counts and no thresholds; show base vs novel only (done).
- Ch 4 viz-vildflow: trained from scratch, 180,000 iterations, no ImageNet init; backbone trainable from random
  weights. CLIP text and image embeddings are precomputed offline, so the frozen encoders are not in the
  training graph at all. N, M, the region-embedding width and the projection size are not stated; 512 is
  derived and uncertain. "1000 proposals" (App. D) belongs to the R-CNN-style pipeline only. No hardware.
- Metric labels: RegionCLIP LVIS numbers read as box AP; ViLD Table 3 is mask AP; Detic Table 2 is mask AP
  under ViLD's settings (17.8 vs 16.1, like for like); OVR-CNN 22.8 is the generalized setting.
- Family tree: OVR-CNN v1 is 2020-11-20, so Act II is "2020-2021".
- Ch 3: Detic App. A Table 8(a) is a second RPN-generalization source; do not mix with ViLD Table 1.
- Ch 13: ViLD supports item 4 (App. B Fig. 12: CLIP scores ignore box quality). Item 2 (texture targets)
  has weak ViLD support: its failures are missed objects and visual-similarity confusions.
- Ch 4 delta: OVR-CNN and RegionCLIP use a fixed all-zero background embedding; ViLD learns one.

YOLO-World group (notes/papers/yoloworld.md, claims/yoloworld.json):
- Frame line: "folded into weights" holds for the T-CSPLayer gate (App. Eq. 4, a 1x1 conv) and the classifier;
  I-Pooling Attention still reads 27 pooled tokens per image after re-parameterization. Say so.
- Shapes: 640 input, 80x80/40x40/20x20 and 8400 are not in the paper; label "our reading (YOLOv8
  convention)". "At most 80 nouns per mosaic" is sourced.
- YOLO primer: the paper only names Darknet, C2f, task-aligned assignment and DFL; teach as background,
  labelled, without numbers we cannot source.
- viz-speed: Fixed AP on LVIS minival, one V100, no TensorRT; plot both FPS values (with/without the text
  encoder) on a log axis; do not mix in Table 6 FPS (TensorRT).
- Text encoder: "CLIP-base (ViT-base)", frozen in pretraining; B/32 is not confirmed.
- Tables 3-5 ablations are not comparable with Table 2 (32.5 vs 35.0 AP for the same data); do not chart
  together. Add App. Eq. 4 for btn-repvl-reparam; quote Eq. 6 with its source typo noted.
- YOLOE removes the neck fusion (its Table 5: -1.9 AP, 1.28x faster): family tree "classifier only", and
  mention in ch 11 (it has visual and prompt-free modes).

OWL group (notes/papers/owlvit.md, owlv2.md, claims/owl.json, data/odinw35-owl.json):
- Neither paper has displayed equations; the loss is described in words. Ch 5/6 show no paper equation; use
  the GLIP-style score equation from ch 1 and say OWL computes per-token logits against each query.
- Ch 5: nothing is frozen in either paper; the text encoder trains at lr 2e-6; a frozen text encoder was
  tested and did worse (Table A5). Archflow: both towers trainable, text tower at a tiny lr.
- Ch 5 one-shot: Table 2 is an R50+H/32 model at 960; use the means 24.3 -> 41.8 (AIT vs OWL-ViT). The
  intro's "26.0" is AIT split 1, not the mean. OWL-ViT Table 1 copies ViLD rows from ViLD Table 3 (mask AP).
- Ch 6 token-drop panel can use real data: Table A1 LVIS AP/APr at drop rates 0, .25, .33, .5, .7.
- Ch 12 card 4: ODinW13 does not fall for G/14 (49.9 -> 50.1); median35 falls at every size (L/14 16.2 ->
  6.3). Use median35 as the evidence. The weight-ensemble row (ODinW13 56.3) averages the curated-vocabulary
  model (row 21); no table gives its LVIS AP.
- Ch 13 item 2: token dropping happens only in training; at inference OWLv2 is identical to OWL-ViT and
  drops nothing. Do NOT cite it as an inference-time failure cause.
- Unresolved conflicts (do not show these numbers): OWL-ViT mosaic probabilities (text vs Table A3); O365 AP
  15.8 (text) vs 15.5 (Table A4); OWLv2 data size "2.3B pairs" (intro) vs 2.3 x 10^8 examples (Table A2);
  score threshold 0.1/0.3 vs 0.3; OWLv2 venue uncertain.
- Token counts: 576 (B/32 at 768) and 5184 (L/14 at 1008) printed; 2304 (B/16 at 768), 3600 (L/14 at 840)
  and 3600 (OWLv2 B/16 at 960) computed (our reading).

GLIP group (notes/papers/glip.md, mdetr.md, glipv2.md, claims/glip.json; code values in claims/glip-code.json):
- Shapes for viz-glipflow: the paper gives no resolution, DyHead depth, added BERT layers or d. Code-only
  (official config): GLIP-T DyHead 6 layers (Swin-L 8), 256 channels, bert-base-uncased (not frozen), 800 x
  1333 input, batch 64, lr 1e-4 / text 1e-5, 85 sampled negative names, ". " separator. Label code-only.
  Number of added BERT layers: not in paper or config; keep symbolic.
- Metric traps: GLIP zero-shot COCO is domain transfer (Objects365 contains all 80 COCO classes), not novel
  classes. GLIP LVIS numbers are box AP with no held-out classes, chosen on minival: never beside ViLD mask APr
  as equal. MDETR gray row in GLIP Table 3 = MDETR fine-tuned on 10% LVIS, fixed AP.
- Ch 7: replace "hundreds of thousands of phrases" with 44,518 (Flickr30K), 110,689 (VG Caption) and 58.4M
  pseudo-labelled noun phrases. "0.8M gold grounding beats 2M detection data" holds only for APr: 17.7 vs
  15.0 (Table 5, rows 6 vs 8); row 8 is better on APc, APf and AP. Say exactly that.
- Ch 8: "image encoder reruns per prompt" overstates it: only the fusion layers rerun; GLIP-T 4.84 -> 2.52
  FPS (Table 7). New evidence: late vs deep fusion zero-shot ODinW 35.5 vs 44.4 (Table 8); prompt tuning
  works only with deep fusion: GLIP-T (A) prompt tuning 43.3 vs full tuning 63.6. MDETR cannot be tested
  zero-shot (always predicts a box for any text).
- Cited-as-printed discrepancies: GLIP-T (A) ODinW 28.7 (Tables 8, 11) vs 28.8 (Table 14); Table 10 caption
  "CLIP-T" typo; MDETR 100% APr 7.4 vs 7.5; GLIPv2 says GLIP-T 1-shot 51.3 vs GLIP's 51.1; Fig. 5 VOC AP50 vs
  Table 14 AP. Use one table per chart and cite it.

Grounding DINO group (notes/papers/gdino.md + 5 branch notes, claims/gdino.json, data/odinw35-gdino.json):
- Ch 9: nothing is frozen in default training (Table 8: backbone and BERT at lr 1e-5); freezing only in the
  DINO-transfer experiment (Table 9). Class-cost/loss weights conflict (Sec. 4.1: cost 2.0, loss 1.0; Table 8:
  1.0 and 2.0); L1 5.0 and GIoU 2.0 agree. Show L1 and GIoU weights only, or cite both class weights.
- viz-qsel: the selected tokens initialize only the positional part (anchor boxes) of each query; content
  queries are learned. Say so in the figure.
- "Grounding DINO L": COCO 52.5 is a different L checkpoint than LVIS 33.9 / ODinW 26.1 (more data:
  Cap4M, COCO, RefC). Never present them as one model's scores.
- Interlude: enhancer layer order is ambiguous (Fig. 3 self-attention first; MM-G Algorithm 1 from the code
  cross-attention first). No epochs, image size or negative count in the paper; MM-G / OV-DINO state
  negatives (OV-DINO caps prompts at 150 per image), labelled with their own source.
- Ch 12: Grounding DINO's printed ODinW means are ODinW-35 (recomputed from Tables 12-14). ODinW-13 is only
  printed by later papers, which disagree (49.8 T / 56.9 L in GD 1.5; 51.4 T in MM-G): do not show a GD ODinW-13.
  Never take GD numbers from other papers' rows (OV-DINO copies GLIP-T's APr/c/f into GD's row; MM-G re-evaluates
  LVIS 28.8, ODinW-35 22.7). GD 1.5, OV-DINO, LLMDet report LVIS fixed AP; GD never states its protocol.
- Ch 13 evidence: MM-G Brain Tumor (numeric labels) 0.4 AP zero-shot -> 47.5 fine-tuned; GD 1.5 admits
  rewriting ODinW category names where it scored poorly; GD's own PlantDoc 0.36 AP.

Evidence group (notes/papers/rf100vl.md, fsod-vlm.md, ovdeval.md, d3.md, trex2.md, claims/evidence.json,
data/rf100vl.json): integrated into chapters 11-13 above. Also: take fully-supervised RF100-VL values from
Table 7, not Table 1 (stale column); OWLv2 in RF100-VL gets one class per prompt, GroundingDINO all classes in
one prompt; OWL-ViT is not evaluated in OVDEval; text-vs-table mismatches recorded as printed (RF100-VL "<2%"
medical vs 2.1; FSOD-VLM Detic 10-shot 16.09 / 15.53 / 15.28; OVDEval "3K labels").

## Fact-check corrections (2026-10-10; notes/factcheck/res-*.txt). These win over everything above.
- Span is 2020-2025 (OVR-CNN v1 2020-11-20); Act II heading "2020-2021".
- LVIS split: T-Rex2 gives val 405 frequent : 461 common : 337 rare; minival 389 : 345 : 70. Use T-Rex2 as the
  source if the page shows frequent/common counts; note minival has only 70 rare classes (APr on minival vs val
  is not comparable).
- Detic like-for-like result (17.8 mask APr) uses IN-L (997 ImageNet-21K classes overlapping LVIS); full 21K is
  Sec. 5.5 only.
- RoIAlign 7x7 is not in any local source: toy-figure only.
- GLIP: batch 64, lr 1e-4, text lr 1e-5 are in the paper (App. C.1); only AdamW is code-only. "85" is the cap on
  total category names in the prompt (positives always kept), not "85 negatives". 800 x 1333 is the test size;
  training is multi-scale (short side 480-800). Label the fusion comparison "ODinW-13" (GLIP's 13-dataset
  average, O365+GoldG pretraining, zero-shot: 35.5 late vs 44.4 deep, Table 8).
- OWLv2 card: ODinW-13 does not fall for G/14; use median35 (B/16 11.6 -> 6.0, L/14 16.2 -> 6.3, G/14 17.5 ->
  9.5; LVIS APr val rises 29.6 -> 36.2, 34.9 -> 44.6, 37.5 -> 47.2). The weight ensemble recovers on ODinW-13
  mean (56.3) but its median35 is 10.6, below OWL-ST L/14's 16.2: say both, do not say "recovers" unqualified.
- Ch 13 calibration item: cite ViLD App. B for "the class score ignores box quality" and OWLv2 Sec. 5 for
  "fine-tuned models' scores on out-of-distribution queries may be poorly calibrated and depend on the precise
  wording". Do not claim the sources show thresholds failing to transfer; that part is "our reading".
- YOLO-World paper names PAN (top-down + bottom-up over P3-P5), a decoupled head with two 3x3 convs, IoU loss,
  DFL, task-aligned assignment and 4-image mosaic: these can be taught as sourced. Its Table 7 mixes Detic mask AP
  and ViLD box AP under a "bbox AP" caption: do not chart Table 7.
- Grounding DINO: PlantDoc 0.36 is the T model with O365+GoldG+Cap4M (Tables 13, 18); O365+GoldG T scores 0.34;
  L scores 2.02. "MM-G re-evaluates" is our inference. Pretraining epochs/size are absent; side experiments state
  12 and 9 epochs and a 2000 max size for the 1.5x COCO runs.
