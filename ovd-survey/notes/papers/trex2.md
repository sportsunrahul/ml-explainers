# T-Rex2

## 1. Identity
- Title: T-Rex2: Towards Generic Object Detection via Text-Visual Prompt Synergy
- arXiv 2403.14610, read v1 (only version). v1 2024-03-21 (abs.html).
- Authors: Qing Jiang (South China University of Technology, IDEA) ... Lei Zhang (SCUT, IDEA, corresponding). Six authors: Jiang, Li, Zeng, Ren, Liu, Zhang.
- Venue: none stated; abs.html comment "Technical Report". The LaTeX uses the ECCV 2024 template (src/README.md), which is not a venue claim.
- Repo: github.com/IDEA-Research/T-Rex (model API only; abstract). Macro note: `\ourmodel` = `\textit{T-Rex2}`.

## 2. Core problem
Text prompts describe common concepts well but fail on rare objects (little training data for the name) and on things that are hard to name ("the microorganisms in the microscope image"). Visual prompts (a box or point on an example) work for things you cannot name but generalize poorly from few examples ("dog" spans many breeds). The paper wants one detector that accepts either, and claims the two help each other when trained into one embedding space.

## 3. What prior methods did and why that hurt (per the paper)
- Text-prompt OVD (GLIP, Grounding DINO, OWL-ViT, Detic ...): "Long-tailed data shortage" and "Descriptive limitations" (Sec. 1, Fig. 2).
- Visual-prompt methods (SEEM, DINOv, T-Rex): less effective at the general concept of a class (Sec. 1). DINOv Visual-G ODinW AP_avg 14.9 / 15.7 (Table 1).
- Naive joint training of both without alignment: text and visual embeddings of the same class sit apart (t-SNE, Fig. 5a) and text-prompt AP drops (Table 4).

## 4. The one key idea
Encode a visual prompt into the same space as the CLIP text embedding of its class name, and pull them together with a region-level InfoNCE loss, so the detector's classifier can be built from text, from examples, or from their average.

## 5. Mechanism, step by step (where text and examples enter)
1. Image encoder (DINO / Deformable DETR style): Swin-T or Swin-L backbone -> multi-scale maps -> 6 deformable self-attention encoder layers -> features f_i in R^{C_i x H_i x W_i}, i = 1..L (Sec. 3.1, Sec. 4.2). The image encoder never sees the prompt: the paper calls T-Rex2 "a late fusion model" (Sec. 3.4).
2. Text prompt encoder: CLIP-B text encoder, fine-tuned; the [CLS] output of a category name or short phrase is the text prompt embedding T (Sec. 3.1, Sec. 4.2).
3. Visual prompt encoder (Sec. 3.1, Eq. 1-5): K user boxes b_j (or points p_j), normalized, on a reference image -> fixed sine-cosine position embedding -> linear to K x D (Eq. 1, 2). A learnable content embedding C (K x D) plus one global content token C' (1 x D) with global position [0.5, 0.5, 1, 1] (box) or [0.5, 0.5] (point) -> concatenate content and position on channels -> linear -> queries Q, (K+1) x D (Eq. 3). Three multi-scale deformable cross-attention layers, each query sampling the image features around its own box or point (Eq. 4; layer count from Sec. 4.2). Self-attention among the K+1 queries, FFN (hidden 1024); the global token's output is the visual prompt embedding V (Eq. 5). One V per class per image.
4. Generic visual prompt: average V over n example images (Eq. 11). Evaluation default n = N = 16 images per category drawn from the benchmark's training split, one GT box each (Sec. 4.3).
5. Box decoder (DINO style): select the top 900 encoder tokens by similarity to the prompt embeddings to initialize anchors (Grounding DINO's query selection); decoder queries use deformable cross-attention to image features and predict anchor offsets (Eq. 6, 7) (Sec. 3.1). Prompts enter here only through query selection and the classifier.
6. Classifier: prompt embeddings are the classifier weights, Label = V . Q_dec^T, C x N (Eq. 8); the same with T for text, or P_mixed = (T + V)/2 for mixed prompts (Eq. 12).
7. Alignment: for K prompt regions in an image with visual embeddings v_i and text embeddings t_i of the same regions, InfoNCE pulls v_i to t_i and away from other t_j (Eq. 9).
8. Training: alternate text-prompt and visual-prompt iterations ("cyclical", Sec. 3.3). Visual prompts are drawn from the same image ("current image prompt, current image detect"): for each category pick 1 to all GT boxes; 50% chance to convert to center points. Text: positive category names plus sampled negatives; grounding phrases plus negatives from a global dictionary of names that occur > 100 times.
9. Loss: L1 + GIoU box loss, sigmoid focal contrastive classification loss (dot product of query and prompt embedding), DINO denoising loss, alignment loss (Eq. 10). Matching costs: cls 2.0, L1 5.0, GIoU 2.0; final weights: cls 1.0, L1 5.0, GIoU 2.0, contrastive 1.0 (App. A).

## 6. Equations
TeX verbatim from sec/3_method.tex (PDF numbers). Macro: none needed beyond standard `\operatorname`.
- Eq. (1)-(2) (one `gather`):
  `B = \operatorname{Linear}(\operatorname{PE}(b_1,...b_K); \theta_B) : \mathbb{R}^{K \times 4D} \to \mathbb{R}^{K \times D}`
  `P = \operatorname{Linear}(\operatorname{PE}(p_1,...p_K); \theta_P) : \mathbb{R}^{K \times 2D} \to \mathbb{R}^{K \times D}`
  Paper: encode "coordinate inputs into position embeddings through a fixed sine-cosine embedding layer", then "two distinct linear layers" project to "a uniform dimension". Gloss: b_j = (x, y, w, h) normalized box; p_j = (x, y) point; PE gives D dims per coordinate (4D for a box, 2D for a point); theta = linear weights; output one D-vector per prompt. Box and point are distinct prompt types (not a point as a tiny box).
- Eq. (3):
  `Q = \begin{cases} \operatorname{Linear}\left(\operatorname{CAT}\left(\left[C; C'\right], \left[B; B'\right]\right); \varphi_B\right) , \operatorname{box} \\ \operatorname{Linear}\left(\operatorname{CAT}\left(\left[C; C'\right], \left[P; P'\right]\right); \varphi_P\right), \operatorname{ point} \end{cases}`
  Paper: content embeddings "are concatenated with position embeddings along the channel dimension, and a linear layer is applied for projection, thereby constructing the input query embedding Q". Gloss: C = learnable content vector repeated K times; C' = one global class token; B', P' = position embedding of the whole image; CAT = channel concatenation; varphi = linear weights. Shape (our reading): (K+1) x 2D -> (K+1) x D.
- Eq. (4):
  `{Q'_j} = \begin{cases} \operatorname{MSDeformAttn}(Q_j, b_j, \left\{\boldsymbol{f}_i\right\}_{i=1}^L) , \operatorname{box} \\ \operatorname{MSDeformAttn}(Q_j, p_j, \left\{\boldsymbol{f}_i\right\}_{i=1}^L) , \operatorname{point} \end{cases}`
  Paper: "each query will selectively attend to a limited set of multi-scale image features encompassing the regions surrounding the visual prompts". Gloss: Q_j = j-th prompt query; b_j or p_j = its reference location; f_i = encoder feature map at level i; MSDeformAttn = multi-scale deformable attention (samples a few points near the reference).
- Eq. (5): `V=\operatorname{FFN}(\operatorname{SelfAttn}(Q'))[-1]`
  Paper: "The output of the global content query will be used as the final visual prompt embedding V." Gloss: [-1] = last token = the global token C'.
- Eq. (6)-(7): `(\Delta x, \Delta y, \Delta w, \Delta h) = \operatorname{MLP}(Q_{dec})` and `\operatorname{Box} = (x+\Delta x, y+\Delta y, w+\Delta w, h+\Delta h)`. Gloss: Q_dec = decoder query features; (x, y, w, h) = anchor from query selection.
- Eq. (8): `\operatorname{Label} = V \cdot Q_{dec}^T : \mathbb{R}^{C \times D} \times \mathbb{R}^{D \times N}  \rightarrow \mathbb{R}^{C \times N}`
  Paper: "we utilize the prompt embeddings as the weights for the classification layer". Gloss: C = number of prompt classes; N = number of detection queries; D = embedding width. This is the page's frame: W is replaced by prompt embeddings.
- Eq. (9): `\mathcal{L}_{align} = -\frac{1}{K} \sum_{i=1}^K \log \frac{\exp(v_i \cdot t_i)}{\sum_{j=1}^K \exp(v_i \cdot t_j)}`
  Paper: "region-level contrastive learning to align these two modalities"; "can be regarded as a mutual distillation process". Gloss: v_i = visual prompt embedding of region i; t_i = text embedding of its class; K = prompt regions in the image; one direction (visual-to-text), no temperature written.
- Eq. (10): `\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{L1}} + \mathcal{L}_{\text{GIoU}} + \mathcal{L}_{\text{DN}} + \mathcal{L}_{\text{align}}` (weights in App. A).
- Eq. (11): `V = \frac{1}{n} \sum_{i=1}^{n} V_i` (generic visual embedding over n images).
- Eq. (12): `P_{\text{mixed}} = \frac{T + V}{2}` (mixed prompt).
- Supplement Eq. (13) (region classification) and Eq. (14) (counting MAE) are not needed for the page.

## 7. Config and measurement setup
| Item | Value | Loc |
|---|---|---|
| Base detector | DINO | Sec. 4.2 |
| Backbone | Swin-T or Swin-L, ImageNet pre-trained | Sec. 4.2, App. A |
| Encoder | 6 transformer encoder layers (deformable) | Sec. 4.2 |
| Text encoder | CLIP-B text tower, fine-tuned, lr 1e-5 | Sec. 4.2 |
| Visual prompt encoder | 3 deformable cross-attention layers, FFN hidden 1024 | Sec. 4.2 |
| Queries | top 900 by prompt similarity | Sec. 3.1 |
| Embedding width D | not stated (our reading: DINO default 256; uncertain) | -- |
| Decoder layers | not stated | -- |
| Optimizer | AdamW; lr 1e-5 backbone and text encoder, 1e-4 others | Sec. 4.2 |
| Compute | Swin-T: 16 A100, batch 128; Swin-L: 32 A100, batch 64; AMP | App. A |
| Text data | 3.15M labeled + 3.39M pseudo-labeled images (O365, OpenImages, GoldG; pseudo-boxes from Grounding DINO on CC3M, LAION-400M, Bamboo) | Sec. 4.1 |
| Visual data | 2.4M labeled + 0.65M pseudo-labeled images (O365, OpenImages, HierText, CrowdHuman; SA-1B boxes named by TAP with a 2560-class dictionary) | Sec. 4.1 |
| Speed (RTX 3090, batch 1) | Swin-T 10.41 FPS, interactive 33.33 FPS; Swin-L 3.62 FPS, interactive 19.96 FPS | Table 7 |

Evaluation protocols (Sec. 4.3), all called "zero-shot" (benchmark not used for training weights):
- Text: all benchmark class names as prompts.
- Visual-G (generic): per category, average the embeddings of N = 16 GT boxes from 16 random training-split images. In-context exemplars from the target dataset's labels; zero-shot only in the sense that weights are not updated.
- Visual-I (interactive): for each category present in the test image, one GT box (or its center) from that same test image. The paper says this comparison "may not be entirely equitable" (Sec. 4.5).
- LVIS splits: val 1203 classes, frequent:common:rare = 405:461:337; minival 804 classes, 389:345:70 (Sec. 4.3).
- "Roboflow100" in Table 1 is the original RF100 (Ciaglia et al.), not RF100-VL.

Analysis lens for the synergy claim: claim = text wins on common classes, visual wins on rare / long-tail; measured = Table 1 (Text vs Visual-G, same weights), Fig. 4 (per-class AP difference on LVIS val, classes sorted by frequency in the text training data), Table 4 (ablation). Would be falsified if Visual-G <= Text on rare classes. LVIS-minival rare already does this (see Section 10).

## 8. Results worth remembering
Table 1 (same weights; AP):
| Setting | COCO val | LVIS minival AP / APr | LVIS val AP / APr | ODinW-35 AP_avg / AP_med | RF100 AP_avg |
|---|---|---|---|---|---|
| T-Rex2 Swin-T Text | 45.8 | 42.8 / 37.4 | 34.8 / 29.0 | 18.0 / 4.7 | 8.2 |
| T-Rex2 Swin-T Visual-G | 38.8 | 37.4 / 29.9 | 34.9 / 32.4 | 23.6 / 17.5 | 17.4 |
| T-Rex2 Swin-L Text | 52.2 | 54.9 / 49.2 | 45.8 / 42.7 | 22.0 / 7.3 | 10.5 |
| T-Rex2 Swin-L Visual-G | 46.5 | 47.6 / 45.4 | 45.3 / 43.8 | 27.8 / 20.5 | 18.5 |
| GLIP-L Text (ref.) | 49.8 | 37.3 / 28.2 | 26.9 / 17.1 | 23.4 / 11.0 | 8.6 |
| Grounding DINO Swin-L Text (ref.) | 52.5 | 33.9 / 22.2 | - | 26.1 / 18.4 | - |

- Paper's own deltas (Sec. 4.4, Swin-T): text beats Visual-G by 7 AP on COCO and 5.4 AP on LVIS-minival; Visual-G beats text by 3.4 AP on LVIS-val rare, 5.6 on ODinW, 9.2 on RF100.
- Fig. 4 (LVIS val, per-class AP_text - AP_visual; backbone and prompt mode not stated in the caption): Text:Visual wins = 254:151 (frequent), 210:251 (common), 84:253 (rare). Counts sum to 405 / 461 / 337, the val split.
- ODinW median is where the gap is largest: Swin-T AP_med 4.7 (text) vs 17.5 (Visual-G); Swin-L 7.3 vs 20.5. Our reading: text fails badly on many ODinW datasets while 16 exemplars do not.
- Ablation (Table 4, Swin-T; COCO AP, LVIS-val AP / APr): text-only training 46.4, 32.8 / 32.1; visual-only training Visual-G 14.0, 15.3 / 8.6; joint without alignment: text 44.4, 32.2 / 28.2, Visual-G 38.7, 30.2 / 29.4; joint with alignment: text 45.8, 34.8 / 29.0, Visual-G 38.8, 34.9 / 32.4.
- Number of exemplars (Table 5, Visual-G Swin-T; COCO AP, LVIS-val AP / APr): 1 -> 29.2, 26.2 / 27.6; 4 -> 32.9, 32.9 / 32.0; 16 -> 38.8, 34.9 / 32.4; 32 -> 41.3, 35.1 / 32.2; 64 -> 41.4, 35.2 / 32.4.
- Mixed prompt (Table 8, Swin-T): COCO Text 45.8, Visual-G 38.8, Text + Visual-G 42.5; LVIS-val AP 34.8 / 34.9 / 37.0, APr 29.0 / 32.4 / 34.3.
- Interactive (Table 2, Visual-I box, Swin-T): COCO 56.6; LVIS-val AP 62.6, APr 71.9; ODinW AP_avg 37.7; RF100 30.6.
- Data engine (Table 6): same O365 + GoldG data (1.4M): Grounding DINO-T LVIS-minival AP 25.6 / APr 14.14 vs T-Rex2-T text 34.9 / 32.7. SA-1B pseudo-data raises Visual-I sharply (LVIS-minival AP 40.6 -> 59.3) but lowers Visual-G COCO (41.1 -> 38.8).
- Counting (Table 3): FSC147 MAE 10.94 (T-Rex 8.72); FSCD-LVIS AP 43.35 (T-Rex 40.32).
- Region classification (supp. Table 2): GT box as visual prompt, nearest class-name text embedding: COCO top-1 72.6 (Swin-T), 82.2 (Swin-L); LVIS 40.8 / 49.8; CLIP ViT-B on crops 37.6 / 9.0.

## 9. Limitations the paper states, and failure cases (where visual prompts fail)
- "Visual prompts may sometimes interfere with text prompts, especially in scenarios involving common objects": mixed prompt COCO 42.5 < text 45.8 (Sec. 5, Table 8).
- "The requirement for up to 16 visual examples to ensure reliable detection due to visual diversity" (Sec. 5). With 1 exemplar, Visual-G COCO is 29.2 vs text 45.8 (Tables 5, 1).
- Visual prompts alone cannot form a general concept: visual-only training gives 14.0 COCO AP; "when every example the model sees is drastically different from the last" (Sec. 4.6, Table 4, "chair" example).
- Intro: one visual example of "dog" does not convey all breeds, sizes and colors (Sec. 1).
- SA-1B data with TAP names "slightly weakens" generic capability, likely from wrong TAP labels (Sec. 4.6).
- Joint training costs text-prompt rare-class AP: text-only training APr 32.1 vs joint-with-alignment text APr 29.0 (Table 4). The paper reports the +0.8 relative to the no-alignment joint model (28.2), not relative to text-only.

## 10. Outline feedback
- Supports (ch 11): one detector can take text, exemplars, or both; exemplars live in the same space as class-name embeddings (Eq. 8, Eq. 9); mixed prompts help on LVIS (APr 29.0 / 32.4 -> 34.3) and hurt on COCO (45.8 -> 42.5).
- Supports, with a large effect, "visual exemplars for datasets the names do not cover": ODinW AP_med 4.7 -> 17.5 and RF100 8.2 -> 17.4 (Swin-T). This is the closest T-Rex2 evidence to the industrial question.
- Does NOT cleanly support "text for common, visual for rare" as stated in the outline:
  - On LVIS-minival (70 rare classes) text beats Visual-G on rare: 37.4 vs 29.9 (Swin-T), 49.2 vs 45.4 (Swin-L). Only on LVIS-val (337 rare) does Visual-G win: 32.4 vs 29.0 (Swin-T), 43.8 vs 42.7 (Swin-L). Table 1's red/green coloring hides the minival case.
  - The joint model's text APr (29.0) is below a text-only-trained model's APr (32.1, Table 4), so part of the val "visual wins rare" gap comes from joint training lowering text, not from exemplars being better.
  - Visual-G uses 16 labeled exemplars per class from each benchmark's train split; text uses none. The comparison is "16-shot in-context exemplars vs zero-shot names", not two zero-shot modes. Say this on the page.
- Conflicts with RF100-VL (same group): there, in-context image prompts (MQ-GLIP-Image) were worse than names (All 6.4 vs 12.2) and gradient fine-tuning beat every in-context method. Ch 11 should present T-Rex2 as one model's result and RF100-VL as the independent test that went the other way. Our reading: the visual-prompt encoder design matters (T-Rex2 trains it on 3.1M images; MQ-Det adds a lightweight module to a frozen GLIP), so "visual prompts help" is a property of a model, not of the modality.
- RF100 in T-Rex2 is the original Roboflow100 (805 classes per RF100-VL Table 13), not RF100-VL. Do not put the 17.4 next to RF100-VL numbers.
- "Where text enters" for the frame (ch 11 or the tree): T-Rex2 is late fusion. Prompts (text or visual) enter at query selection (top 900) and as classifier weights (Eq. 8); the image encoder does not see the prompt; the CLIP text tower is fine-tuned (lr 1e-5), unlike ViLD's frozen CLIP text encoder.
- Visual-I numbers (COCO 56.6, LVIS-val APr 71.9) use a GT box from the test image; do not compare them with zero-shot text numbers.
