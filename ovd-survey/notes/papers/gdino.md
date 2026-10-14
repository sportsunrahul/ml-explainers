# Grounding DINO -- Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

## 1. Identity
- Title: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"
- arXiv 2303.05499, read v5. v1 2023-03-09 (Thu, 9 Mar 2023 18:52:16 UTC). v2 2023-03-10, v3 2023-03-16, v4 2023-03-20, v5 2024-07-19 (Fri, 19 Jul 2024 06:00:41 UTC) (abs.html submission history).
- Authors: Shilong Liu (first; Tsinghua + IDEA), ..., Lei Zhang (last; IDEA, corresponding with Jun Zhu). 12 authors; Chunyuan Li and Jianwei Yang at Microsoft Research.
- Venue: ECCV 2024 is implied only by the LaTeX template (`\usepackage{eccv}` camera-ready line, `README.md` "Official ECCV 2024 Template"); the paper text and abs.html do not state a venue.
- Repo: https://github.com/IDEA-Research/GroundingDINO (abs.html Comments; abstract says "We release some checkpoints and inference codes").
- Source read: `src/sec/*.tex`, `src/tables/*.tex`, `src/image_tex/*.tex` (equations, tables), `paper.pdf` (33 pages; page, table and figure numbers). Table numbers below are the PDF numbers.

## 2. Core problem
Detect arbitrary objects named by human language (category names or referring expressions) -- "open-set object detection"; the paper treats open-set, open-world and open-vocabulary detection as the same task (Sec. 1 footnote). Evaluate fully zero-shot ("the training split of the test dataset is not utilized", abstract footnote) on COCO, LVIS, ODinW, plus referring expression comprehension (REC) on RefCOCO/+/g.

## 3. What prior methods did and why that hurt (per the paper)
- A closed-set detector has a backbone, a neck and a head (Fig. 2). Open-set detectors learn language-aware region embeddings with a contrastive loss at the neck and/or head outputs (Sec. 1).
- Fusion can happen in three phases: neck (phase A), query initialization (phase B), head (phase C) (Sec. 1, Fig. 2). Prior work fuses in only part of them: GLIP = A only; OV-DETR = B only (language-aware queries); MDETR and DQ-DETR = A,C; OmDet = C; FindIt = A; ViLD, RegionCLIP, OWL-ViT, DetCLIP = no fusion ("-") (Table 1, col "Fusion"). "previous works only fuse multi-modal information in partial phases, which may lead to sub-optimal language generalization ability" (Sec. 2).
- Classical detectors (Faster R-CNN) make fusion hard "in most blocks"; DINO's layer-by-layer Transformer design matches language blocks (Sec. 1).
- CLIP-based detectors: CLIP trained on image-text pairs is limited for region-text tasks (citing RegionCLIP) (Sec. 1).
- GLIP concatenates all category names into one sentence in random order; word-level text features then let unrelated category names attend to each other (Sec. 1, Sec. 3.4, Fig. 4b). Sentence-level encoding (DetCLIP, OWL-ViT) removes the interaction but "losing fine-grained information" (Sec. 3.4).
- Prior open-set evaluation mostly used "partial label" (train on base, test on novel); REC is normally overlooked (Sec. 1, Sec. 2, Table 1).

## 4. The one key idea
Take DINO (a DETR-family detector whose blocks are all attention layers) and inject text at all three phases: a feature enhancer that cross-attends image and text tokens (neck, A), language-guided query selection that picks the decoder's starting anchors from the image tokens most similar to any text token (query init, B), and an extra text cross-attention in every decoder layer (head, C). Train it like GLIP, on detection + grounding + caption data, with a sub-sentence attention mask so concatenated category names do not attend to each other.

## 5. Mechanism, step by step
"dual-encoder-single-decoder architecture" (Sec. 3). Fig. 3 = framework (block 1 overall, block 2 feature enhancer layer, block 3 decoder layer). Fig. 5 (p23) marks the parts new relative to DINO in blue.
1. Text input (Sec. 3, App. B): for detection data, category names are concatenated into one prompt, "cat . person . mouse ." in Fig. 3 (period-separated). "To simulate different text inputs, we randomly sampled category names from all categories in a dataset on the fly during training" (App. B item 1). Max 256 text tokens; BERT tokenization ("BPE scheme" as the paper words it) (Sec. 4.1). For REC, the text is the expression, and the highest-scoring box is the answer (Sec. 3).
2. Text backbone (Sec. 3.1, 4.1): BERT-base (Hugging Face) -> vanilla text features, N_T < 256 tokens. Sub-sentence mask (Sec. 3.4, Fig. 4c): "we introduce attention masks to block attentions among unrelated category names". Each category name keeps per-word features but sees only its own words. Word level (Fig. 4b) = full attention across the whole concatenated prompt; sentence level (Fig. 4a) = one feature per phrase.
3. Image backbone (Sec. 3.1, 4.1): Swin-T or Swin-L -> multi-scale features. Swin-T: three backbone scales, strides 8x to 32x, plus the 32x map downsampled to 64x ("4scale" in DINO naming). Swin-L: four backbone scales, 4x to 32x. Flattened, the image tokens number N_I > 10,000 (Sec. 3.2).
4. Feature enhancer, 6 layers (Sec. 3.1, 4.1, Fig. 3 block 2). Per layer, read from Fig. 3 block 2:
   - image tokens: deformable self-attention (Q,K,V = image); text tokens: vanilla self-attention (Q,K,V = text);
   - "Image-to-text cross-attention": Q from image, K,V from text (updates the image tokens);
   - "Text-to-image cross-attention": Q from text, K,V from image (updates the text tokens);
   - one FFN on each side. Output: "Updated Image Features" and "Updated Text Features".
   Text says: "We leverage the Deformable self-attention to enhance image features and the vanilla self-attention for text feature enhancers. Inspired by GLIP, we add an image-to-text and a text-to-image cross-attention modules for feature fusion." The Q/K,V direction of each cross-attention is from the figure labels only.
   Order conflict: Fig. 3 block 2 draws self-attention first, then the two cross-attentions, then the FFNs. MM-Grounding-DINO (2401.02361v2, Algorithm 1), which rebuilt the model from the released test code, puts a `BiAttentionBlock(image, text)` first and then self-attention / deformable self-attention + FFN. Cite one and say which.
5. Language-guided query selection (Sec. 3.2, Eq. 1, Algorithm 1 p20): logits = X_I X_T^T, shape (N_I, N_T); max over text tokens -> one score per image token; top N_q = 900 indices. Only the positional part of each query is taken from the selected encoder outputs: "We formulate the positional part as dynamic anchor boxes, which are initialized with encoder outputs. The other part, the content queries, are set to be learnable during training" (DINO's "mixed query selection"). Fig. 7 (p25) shows the top-900 query boxes move with the prompt.
6. Cross-modality decoder, 6 layers (Sec. 3.3, 4.1, Fig. 3 block 3). Per layer: self-attention (Q,K,V = queries) -> image cross-attention (Q = query, K,V = image features; deformable attention, Sec. 4.1) -> text cross-attention (Q = query, K,V = text features) -> FFN. "Each decoder layer has an extra text cross-attention layer compared with the DINO decoder layer."
7. Outputs (Sec. 3.5): each final query gives a box and a logit per text token: "we dot product each query with text features to predict logits for each text token". A phrase's tokens light up -> the label is that phrase (Sec. 3, Fig. 3).
8. Matching and loss (Sec. 3.5, 4.1): Hungarian matching with cost = classification + box L1 + GIoU; final loss on matched pairs with the same three terms: focal loss on each query-token logit (contrastive classification, following GLIP), L1 + GIoU on boxes. Auxiliary losses "after each decoder layer and after the encoder outputs". Weights: see the conflict in section 7.
9. Trainable vs frozen (default pre-training): nothing is stated as frozen. Table 8 gives a learning rate to both backbones: lr 1e-4, image backbone 1e-5, text backbone 1e-5. So Swin and BERT are fine-tuned at 10x lower lr (our reading of Table 8). The only freezing in the paper is the DINO-to-Grounding-DINO transfer (App. C.1, Table 9): "We freeze the modules co-existing in DINO and Grounding DINO and fine-tune the other parameters only"; the text summarises this as "Training only text and fusion blocks using a pre-trained DINO". Per Fig. 5 the non-DINO (blue) parts are: text backbone, the feature enhancer's text self-attention, both cross-attentions and the text FFN, language-guided query selection, the decoder text cross-attention, and the contrastive loss (our reading of Fig. 5 colours).
10. Things the paper does not say: input image size (only "1.5x image sizes, i.e., with a maximum image size of 2000", Table 2 caption; a default max side of about 1333 is our reading), pre-training epochs or iterations, the number of categories sampled per prompt, how many negatives a prompt holds, the LR schedule, the text-to-256 projection layer, whether contrastive denoising from DINO is kept. The phrase-to-token target expansion is GLIP's (see `glip.md`); this paper only says "Following GLIP".

Shapes for the archflow (constants from Sec. 3.2, 4.1, Table 8; example sizes are our reading):
- d = 256 (hidden dim), 8 heads, FFN dim 2048, dropout 0.0, 6 enhancer + 6 decoder layers, N_q = 900, N_T <= 256 (max tokens 256), N_I > 10,000.
- Our reading, Swin-T 4scale at an assumed 800x1333 input: strides 8/16/32/64 -> 100x167 + 50x84 + 25x42 + 13x21 = 22,223 image tokens. Paper gives only "> 10,000".
- BERT-base hidden width 768 is a BERT fact, not stated in this paper; the mapping to d = 256 is not described (our reading: a linear projection).
- Query selection: (bs, N_I, 256) x (bs, N_T, 256) -> logits (bs, N_I, N_T) -> max -> (bs, N_I) -> topk -> (bs, 900) (Algorithm 1 comments).
- Final scores: (bs, 900, 256) x (bs, N_T, 256)^T -> (bs, 900, N_T) logits.

## 6. Equations
Custom macros (from `src/mlVecMat.tex`): `\newcommand{\Xmat}[0]{{{\bf X}}}` (line 143), `\newcommand{\Imat}{{\bf I}}` (line 127). `\textsc{R}` is used for the reals. The paper has exactly one displayed equation.

### Eq. (1) -- language-guided query selection (Sec. 3.2, p7; `\begin{equation}`, no `\label`, printed as (1))
```tex
\Imat_{N_q} = \texttt{Top}_{{N_q}}(\texttt{Max}^{(-1)}(\Xmat_{I}\Xmat_{T}^{\intercal})).
```
Paper: "The top N_q query indices for the image feature, denoted as I_{N_q}, are selected using the following expression ... Top_{N_q} represents the operation to pick the top N_q indices. The function Max^{(-1)} executes the max operation along the -1 dimension, and the symbol T denotes matrix transposition."
- `X_I` in R^{N_I x d}: enhanced image tokens (all scales flattened), one row per token; N_I > 10,000.
- `X_T` in R^{N_T x d}: enhanced text tokens; N_T < 256. d = 256.
- `X_I X_T^T`: (N_I x N_T) matrix of dot products, image token vs text token.
- `Max^{(-1)}`: max over the last (text) axis -> each image token's best match to any text token.
- `Top_{N_q}`: indices of the N_q = 900 highest-scoring image tokens. `I_{N_q}`: those indices.

### Algorithm 1 -- pseudocode (App. A.2, p20; `src/image_tex/algorithms_pseudocode.tex`, uncommented copy)
```python
"""
Input:
image_feat: (bs, num_img_tokens, ndim)
text_feat: (bs, num_text_tokens, ndim)
num_query: int

Output:
topk_idx: (bs, num_query)
"""

logits = torch.einsum("bic,btc->bit", image_feat, text_feat) # bs, num_img_tokens, num_text_tokens
logits_per_img_feat = logits.max(-1)[0]# bs, num_img_tokens
topk_idx = torch.topk(logits_per_img_feature, num_query, dim=1)[1] # bs, num_query
```
Verbatim, including the name mismatch: line 2 defines `logits_per_img_feat`, line 3 reads `logits_per_img_feature`. App. A.2: "num_query is the number of queries in the decoder, which is set to 900".

### Loss (prose only, Sec. 3.5; no equation in the paper)
"we use the L1 loss and the GIOU loss for bounding box regressions. We follow GLIP and use contrastive loss between predicted objects and language tokens for classification. Specifically, we dot product each query with text features to predict logits for each text token and then compute focal loss for each logit. Box regression and classification costs are first used for bipartite matching between predictions and ground truths. We then calculate final losses between ground truths and matched predictions with the same loss components. Following DETR-like models, we add auxiliary loss after each decoder layer and after the encoder outputs."

## 7. Config / setup table
| Item | Value | Loc |
|---|---|---|
| Variants | Grounding DINO T (Swin-T), Grounding DINO L (Swin-L) | Sec. 4.1 |
| Text backbone | BERT-base, Hugging Face | Sec. 4.1 |
| Queries | 900 ("following DINO") | Sec. 3.2, 4.1, Table 8 |
| Max text tokens | 256 | Sec. 4.1 |
| Hidden dim d | 256 | Sec. 3.2, Table 8 |
| Feature enhancer layers | 6 | Sec. 4.1; Table 8 "number of encoder layers 6" |
| Decoder layers | 6 | Sec. 4.1, Table 8 |
| Heads / FFN dim / dropout | 8 / 2048 / 0.0 | Table 8 |
| Image cross-attn | deformable attention | Sec. 4.1 |
| Image scales | T: 8x-32x from backbone + 64x extra ("4scale"); L: 4x-32x from backbone (four scales) | Sec. 4.1 |
| Optimizer | AdamW, lr 1e-4, image backbone lr 1e-5, text backbone lr 1e-5, weight decay 0.0001, clip max norm 0.1 | Table 8 |
| Hardware, batch | T: 16 V100, total batch 32. L: 64 A100, total batch 64 | Sec. 4.1 |
| Epochs, image size, schedule | not stated (only "maximum image size of 2000" for the 1.5x runs) | Table 2 caption |
| Params | T 172M, L 341M (GLIP-T 232M) | Table 4 col "Model Size" |
| GFLOPs / FPS | T 464G / 8.37; GLIP-T 488G / 6.11 (hardware and image size not stated) | Table 20, p26 |

Matching and loss weights -- the paper contradicts itself (record both, do not reconcile):
| | cls | L1 | GIoU | Loc |
|---|---|---|---|---|
| Matching cost, text | 2.0 | 5.0 | 2.0 | Sec. 4.1 "weight of classification costs, box L1 costs, and GIOU costs as 2.0, 5.0, and 2.0" |
| Final loss, text | 1.0 | 5.0 | 2.0 | Sec. 4.1 "The corresponding loss weights are 1.0, 5.0, and 2.0" |
| Matching cost, Table 8 | 1.0 (`set cost class`) | 5.0 | 2.0 | Table 8, p19 |
| Final loss, Table 8 | 2.0 (`ce loss coef`) | 5.0 | 2.0 | Table 8, p19 |
The class weights are swapped between Sec. 4.1 and Table 8.

Pre-training data (App. B; Tables 2-4):
- Detection: COCO, O365, OpenImages (OI); category names concatenated into prompts, names sampled per image on the fly.
- Grounding: GoldG (Flickr30k entities + Visual Genome, MDETR preprocessing) and RefC (RefCOCO/+/g, MDETR preprocessing).
- Caption: Cap4M, GLIP-pseudo-labelled caption data; "GLIP-T annotated caption data for Grounding DINO T, while GLIP-L annotated caption data for Grounding DINO L".
- O365v1 (~600K images) for T; O365v2 (~1.7M images, superset) for L (App. B).
- Dataset sizes of GoldG and Cap4M are not stated in this paper (see `glip.md`).
- Checkpoints per table (the "L" model differs between tables):
  - T, O365: Table 2, Table 7 (ablation), Table 9.
  - T, O365+GoldG ("OG"): Tables 2, 3, 4, 5, 6, 9, 12.
  - T, O365+GoldG+Cap4M ("OGC"): Tables 2, 3, 4, 13, 18.
  - L, O365+OI+GoldG: Table 2 (COCO 52.5) only.
  - L, O365+OI+GoldG+Cap4M+COCO+RefC: Table 2 last row (COCO 60.7, not zero-shot), Table 3 (LVIS 33.9), Table 4 (ODinW 26.1), Table 5, Table 14.

## 8. Results worth remembering
COCO 2017val, box AP (Table 2, p8; col "Zero-Shot 2017val" / "Fine-Tuning 2017val/test-dev"):
- Grounding DINO T, O365: zero-shot 46.7, fine-tune 56.9. DINO(Swin-T) O365 zero-shot (category-mapped) 46.2. GLIP-T (B) O365: 44.9.
- Grounding DINO T, O365,GoldG: 48.1 (fine-tune 57.1). GLIP-T (C) O365,GoldG: 46.7.
- Grounding DINO T, O365,GoldG,Cap4M: 48.4 (fine-tune 57.2).
- Grounding DINO L, O365,OI,GoldG: 52.5 zero-shot; fine-tune 62.6 val / 62.7 test-dev; (63.0 / 63.0) at 1.5x image size. GLIP-L (FourODs,GoldG,Cap24M): 49.8.
- Grounding DINO L with COCO,RefC added: 60.7, marked "not a real zero-shot performance".
LVIS minival (MDETR's minival, "MiniVal [18]"), box AP and APr/APc/APf; fixed-AP protocol not stated (Table 3, p10):
- Zero-shot: Grounding DINO T OG 25.6 (14.4/19.6/32.2); T OGC 27.4 (18.1/23.3/32.7); L (O365,OI,GoldG,Cap4M,COCO,RefC) 33.9 (22.2/30.7/38.8).
- Baselines zero-shot: GLIP-T (C) 24.9 (17.7/19.5/31.0); GLIP-T OGC 26.0 (20.8/21.4/31.0); DetCLIPv2 Swin-T (O365,GoldG,CC15M) 40.4 (36.0/41.7/40.0).
- Fine-tuned on LVIS: Grounding DINO T OG 52.1 (35.4/51.3/55.7) vs DetCLIPv2 50.7 (44.3/52.4/50.3).
- Paper's own reading (Sec. 4.2): better than GLIP on common, worse on rare; DETR-like models "often exhibit lower rare category AP" -- "may be a characteristic limitation of the architecture". Cap4M adds +1.8 AP for Grounding DINO vs +1.1 for GLIP.
ODinW, zero-shot, AP_average / AP_median (Table 4, p11). The per-dataset Tables 12-14 have 35 rows, and their mean/median match these values (our computation), so these are ODinW-35 statistics:
- Grounding DINO T OG 20.0 / 9.5; T OGC 22.3 / 11.9; L 26.1 / 18.4 ("new record", abstract).
- Baselines: MDETR 10.7 / 3.0; OWL-ViT L/14 18.8 / 9.8; GLIP-T 19.6 / 5.1; OmDet 19.7 / 10.8; GLIPv2-T 22.3 / 8.9; DetCLIP Swin-L 24.9 / 18.3; Florence CoSwinH 25.8 / 14.3.
- Few-shot: Grounding DINO T OG 46.4 / 51.1. Full-shot: 70.7 / 76.2 (DINO-Swin-L 68.8 / 70.7).
- ODinW-13 (GLIP's 13-dataset subset) is not printed for Grounding DINO in this paper. Our computation from Tables 12-14 is in `notes/data/odinw35-gdino.json` (marked uncertain).
Ablation, Swin-T, O365 only (Table 7, p14; cols COCO minival zero-shot / COCO fine-tune / LVIS minival zero-shot):
- #0 full model 46.7 / 56.9 / 16.1
- #1 w/o encoder fusion 45.8 / 56.1 / 13.1
- #2 static query selection 46.3 / 56.6 / 13.6
- #3 w/o text cross-attention 46.1 / 56.3 / 14.3
- #4 word-level text prompt 46.4 / 56.6 / 15.6
- Text (Sec. 4.5) attributes "+3.0 AP, +1.8 AP, and +0.5 AP" on LVIS to query selection, text cross-attention and sub-sentence prompt. From the table, query selection is 16.1 - 13.6 = 2.5 and encoder fusion is 16.1 - 13.1 = 3.0. Discrepancy in the paper; cite table values.
- The same O365 Swin-T config gives three different LVIS values: 16.1 (Table 7 #0), 16.2 (Table 9 from scratch), 15.8 with APr 9.4 (Table 15, 900 queries). The paper does not explain the spread.
DINO -> Grounding DINO transfer, Swin-T (Table 9, p21; cols COCO minival zs / LVIS minival zs / ODinW zs):
- From scratch: O365 46.7 / 16.2 / 14.5; O365,GoldG 48.1 / 25.6 / 20.0.
- From O365-pretrained DINO, shared modules frozen: O365 46.5 / 17.9 / 13.6; O365,GoldG 46.4 / 26.1 / 18.5.
- "Table 10" (p21) is a training-curve figure with a table caption (O365 training, COCO eval, no EMA): from DINO pretrained ends at 46.32 (last point at about epoch 7); from scratch ends at 38.12 (about epoch 18). Epochs are read off the x-axis. Values are point labels in `src/resources/few_show_compare.pdf`, not LaTeX.
Efficiency (Table 20, p26): params / GFLOPs / FPS: GLIP-T 232M / 488G / 6.11; Grounding DINO T 172M / 464G / 8.37.
More queries (Table 15, p31; O365 Swin-T): 900 -> COCO 46.7, LVIS 15.8 (APr 9.4); 1200 -> 46.7, 15.7 (9.0); 1500 -> 46.9, 15.8 (9.2).
Data effects, Swin-T (Table 6, p13; COCO zs / COCO ft / LVIS zs / ODinW zs): OG 48.1 / 57.1 / 25.6 / 20.0; +RefC 48.5 / 57.3 / 21.9 / 17.7; +RefC,COCO 56.1 / 57.5 / 22.3 / 17.4. RefC helps COCO, hurts LVIS and ODinW.
REC (Table 5, p12): zero-shot T OG RefCOCO val 50.41 vs GLIP-T(B) 49.96; with RefC in training (not fine-tuned) 73.98; fine-tuned 89.19; L fine-tuned 90.56.
Per-dataset failure examples (Table 18, p33; App. C.8, T OGC vs GLIP-T OGC): PlantDoc 0.36 vs 1.1 ("infrequent categories such as 'Tomato leaf mosaic virus'"); PKLot 0.06 vs 0.00; openPoetryVision 0.05 vs 0.00; OxfordPets by-breed 0.21 vs 0.40.
Closed-set COCO 1x, R50, 12 epochs (Table 11, p22): Grounding DINO 4scale 48.1 vs DINO-4scale 49.0; "the new components may make the model harder to optimize than DINO".

## 9. Limitations the paper states, and failure cases
- "cannot be used for segmentation tasks like GLIPv2" (Sec. 5).
- Less training data than the largest GLIP (Sec. 5).
- "our model will produce false positive results in some cases, which may need more techniques or data to reduce the hallucination" (Sec. 5).
- Rare LVIS categories: APr below GLIP-T at equal data (Table 3); suggested to be a DETR-like limitation (Sec. 4.2). Behind DetCLIPv2 trained on more data (Sec. 4.2).
- REC without REC data is weak for both GLIP and Grounding DINO; RefCOCO prompts map to one box while the model predicts several (App. D, Fig. 8).
- Uncommon ODinW datasets: PlantDoc and similar, where the categories are infrequent in training data (App. C.8, Table 18).
- More queries do not help much: "introducing more queries exacerbates data imbalance during training, as the model is trained on objects from sampled categories" (App. C.5).
- Larger input (1.5x) gives smaller benefit: "We suspect that the text branch enlarges the gap between models with different input images" (Sec. 4.2).

## 10. Outline feedback
- ch 9 "what is frozen": in default pre-training nothing is frozen; Table 8 trains both backbones at lr 1e-5 (vs 1e-4 elsewhere). Do not draw BERT or Swin as frozen (brown). Freezing appears only in the DINO-transfer experiment (Table 9), where DINO's shared modules are frozen and the text + fusion blocks train. Show it as an optional toggle, if at all.
- ch 9 `viz-qsel`: the selected tokens become the queries' positional part (anchor boxes) only; content queries are learned embeddings (Sec. 3.2). Saying "top-K tokens become queries" is half right; label it.
- ch 9 archflow: write "text <= 256 tokens" as "max 256 text tokens" (Sec. 4.1). N_I is "> 10,000" in the paper; any exact count needs an assumed image size (our reading).
- ch 9 loss: the matching-cost and loss weights for the class term conflict between Sec. 4.1 and Table 8. Either show L1 5.0 and GIoU 2.0 only, or show both class values with a note.
- Interlude `ch-training`: the paper gives only "randomly sampled category names from all categories in a dataset on the fly". It gives no negatives count, no epochs and no image size. Primary sources from the same family fill the gap, each labelled with its own paper: MM-G (Sec. 2.3) concatenates categories or descriptions "randomly sampled from other images as negative examples" with the positives. OV-DINO (Sec. 4.2) caps prompts at 150 per image, positives plus random negatives, at input [800, 1333]. Third-party epoch counts disagree (MM-G Table 1: 30; OV-DINO TABLE 5: 50); do not present either as Grounding DINO's own.
- ch 9 sourced numbers: the "Grounding DINO L" in COCO 52.5 (O365,OI,GoldG) is a different checkpoint from the L in LVIS 33.9 and ODinW 26.1 (adds Cap4M, COCO, RefC). The page must say which.
- ch 12 card 2: Table 4's 20.0 / 22.3 / 26.1 means and 9.5 / 11.9 / 18.4 medians are ODinW-35 (verified by recomputing from Tables 12-14). Grounding DINO does not print an ODinW-13 mean. Primary ODinW-13 values come from later papers: Grounding DINO 1.5 Table 1 gives T (OGC) 49.8 and L 56.9; MM-G Table 6 gives T (OGC) 51.4. Our recompute over Table 13 gives 49.75, consistent with 49.8. Our recompute over Table 14 (the L with COCO/RefC) gives 56.27, a different checkpoint from GD 1.5's 56.9. Use a printed value with its source; the recomputes are in the data file, flagged uncertain.
- ch 12 card 2 note: Table 4 prints the OGC median as 11.9; our recompute of Table 13 gives 11.96. Use the printed value.
- ch 13 item 1 (jargon names): App. C.8 PlantDoc "Tomato leaf mosaic virus" (0.36 AP) is a direct source; Table 18 adds per-dataset GLIP-T vs Grounding DINO T. A stronger case is MM-G Table 11: Brain Tumor, whose labels are numeric IDs, scores 0.4 zero-shot and 47.5 after fine-tuning (see `mmgdino.md`). Item 4 could cite "false positive ... hallucination" (Sec. 5); GD 1.5 ties hallucination to early fusion (see `gdino15.md`).
- ch 13 could add: DETR-style query detectors lose APr relative to GLIP (Table 3, Sec. 4.2).
- Frame diagram: Table 1 gives the paper's own A/B/C phase labels (GLIP = A, OV-DETR = B, MDETR = A,C, Grounding DINO = A,B,C). Useful for the "where text enters" hero.
- ch 10 `viz-speed`: Table 20 FPS (8.37 vs 6.11) has no stated hardware or input size; do not mix with YOLO-World's FPS axis.
